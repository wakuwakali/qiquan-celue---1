import ccxt
import pandas as pd
import numpy as np
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0: return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def call_delta(S, K, T, r, sigma):
    if T <= 0: return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def find_call_strike(S, T, r, sigma, target_delta):
    K = S
    step = S * 0.01
    for _ in range(1000):
        if call_delta(S, K, T, r, sigma) < target_delta: return K
        K += step
    return K

def fetch_binance_data():
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    symbol = 'BTC/USDT'
    start_time_str = '2020-01-01T00:00:00Z'
    since = exchange.parse8601(start_time_str)
    all_klines = []
    while True:
        klines = exchange.fetch_ohlcv(symbol, '1d', since=since, limit=1000)
        if not klines: break
        all_klines.extend(klines)
        since = klines[-1][0] + 86400000
        if len(klines) < 1000: break
    df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    df.set_index('Date', inplace=True)
    return df

def run_phase3_simulation(df):
    r = 0.0
    FEE_RATE = 0.0010  
    
    # Phase 1 Best Params
    tenor = 90
    delta_short = 0.3
    vol_filter = 0.65
    threshold = 0.05
    take_profit = 0.20
    
    # Phase 3 Param: Trend Asymmetric Hedging
    # If S_t > SMA50 (Uptrend), we hold +0.15 Delta exposure instead of 0.
    
    active_option = False
    strike = 0
    days_to_expiry = 0
    initial_premium = 0
    hedge_position = 0
    hedge_cash = 0
    total_realized_pnl = 0
    
    pnl_curve = []
    
    closes = df['close'].values
    vols = df['Vol_30d'].values
    smas = df['SMA50'].values
    
    for i in range(len(df)):
        S_t = closes[i]
        sigma_t = vols[i]
        sma_t = smas[i]
        
        # Calculate Target Delta Exposure based on Trend
        # If in uptrend, we want +0.15 delta. The option gives us negative delta (-curr_delta).
        # So hedge_pos needs to be curr_delta + 0.15
        target_exposure = 0.15 if (not np.isnan(sma_t) and S_t > sma_t) else 0.0
        
        if not active_option:
            if sigma_t > vol_filter:
                days_to_expiry = tenor
                T = days_to_expiry / 365.0
                strike = find_call_strike(S_t, T, r, sigma_t, delta_short)
                initial_premium = black_scholes_call(S_t, strike, T, r, sigma_t)
                
                curr_delta = call_delta(S_t, strike, T, r, sigma_t)
                # Hedge Position = Option Delta + Target Exposure
                hedge_position = curr_delta + target_exposure
                
                fee = abs(hedge_position) * S_t * FEE_RATE
                hedge_cash = -hedge_position * S_t - fee
                
                active_option = True
                daily_mtm = total_realized_pnl + (hedge_cash + hedge_position * S_t)
            else:
                daily_mtm = total_realized_pnl
        else:
            days_to_expiry -= 1
            T = max(0.00001, days_to_expiry / 365.0)
            
            curr_price = black_scholes_call(S_t, strike, T, r, sigma_t)
            curr_delta = call_delta(S_t, strike, T, r, sigma_t)
            
            if curr_price <= initial_premium * take_profit:
                opt_mtm = initial_premium - curr_price
                fee = abs(hedge_position) * S_t * FEE_RATE
                hedge_cash = hedge_cash + hedge_position * S_t - fee
                hedge_position = 0
                
                total_realized_pnl += (opt_mtm + hedge_cash)
                daily_mtm = total_realized_pnl
                active_option = False
                
            elif days_to_expiry > 0:
                desired_hedge = curr_delta + target_exposure
                delta_diff = desired_hedge - hedge_position
                
                if abs(delta_diff) > threshold:
                    fee = abs(delta_diff) * S_t * FEE_RATE
                    hedge_cash = hedge_cash - delta_diff * S_t - fee
                    hedge_position = desired_hedge
                
                opt_mtm = initial_premium - curr_price
                daily_mtm = total_realized_pnl + opt_mtm + (hedge_cash + hedge_position * S_t)
                
            else:
                payoff = max(0, S_t - strike)
                opt_mtm = initial_premium - payoff
                fee = abs(hedge_position) * S_t * FEE_RATE
                
                hedge_cash = hedge_cash + hedge_position * S_t - fee
                hedge_position = 0
                
                total_realized_pnl += (opt_mtm + hedge_cash)
                daily_mtm = total_realized_pnl
                active_option = False
                
        pnl_curve.append(daily_mtm)

    pnl_series = pd.Series(pnl_curve, index=df.index)
    
    peak_pnl = pnl_series.cummax()
    drawdown_usd = peak_pnl - pnl_series
    max_drawdown_usd = drawdown_usd.max()
    lowest_underwater = pnl_series.min()
    min_safe_capital = max(5000.0, max_drawdown_usd / 0.8, abs(lowest_underwater) / 0.8)
    
    equity_curve = min_safe_capital + pnl_series
    years = len(df) / 365.25
    total_return = equity_curve.iloc[-1] / min_safe_capital - 1
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    peak = equity_curve.cummax()
    drawdown = (peak - equity_curve) / peak
    max_drawdown = drawdown.max()
    
    return {
        'total_pnl_usd': pnl_series.iloc[-1],
        'min_capital_req': min_safe_capital,
        'ann_return': annualized_return,
        'max_dd': max_drawdown
    }

def main():
    print("Fetching data from Binance...")
    df = fetch_binance_data()
    df['LogRet'] = np.log(df['close'] / df['close'].shift(1))
    df['Vol_30d'] = df['LogRet'].rolling(window=30).std() * np.sqrt(365)
    # Phase 3 metric: SMA 50
    df['SMA50'] = df['close'].rolling(window=50).mean()
    df.dropna(inplace=True)
    
    print("Running Phase 3 Simulation (Trend Asymmetric Hedging)...")
    res = run_phase3_simulation(df)
    
    print("\n--- PHASE 3 RESULTS (Trend Hedging) ---")
    print(f"Total PnL: ${res['total_pnl_usd']:,.2f}")
    print(f"Annualized Return: {res['ann_return']:.2%}")
    print(f"Max Drawdown: {res['max_dd']:.2%}")
    print(f"Capital Required: ${res['min_capital_req']:,.2f}")

if __name__ == "__main__":
    main()

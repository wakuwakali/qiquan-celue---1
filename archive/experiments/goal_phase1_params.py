import ccxt
import pandas as pd
import numpy as np
from scipy.stats import norm
import os
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

def load_local_data():
    # Attempt to load the previously saved data or fetch it again to avoid API limits
    # But since fetching takes only 10s, we will just fetch it directly to be safe
    pass

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

def run_simulation(df, tenor, delta, vol_filter=0.65, threshold=0.05, take_profit=0.20):
    r = 0.0
    FEE_RATE = 0.0010  
    
    active_option = False
    strike = 0
    days_to_expiry = 0
    initial_premium = 0
    hedge_position = 0
    
    hedge_cash = 0
    total_realized_pnl = 0
    
    pnl_curve = []
    dates = df.index
    closes = df['close'].values
    vols = df['Vol_30d'].values
    
    for i in range(len(df)):
        S_t = closes[i]
        sigma_t = vols[i]
        
        if not active_option:
            if sigma_t > vol_filter:
                days_to_expiry = tenor
                T = days_to_expiry / 365.0
                strike = find_call_strike(S_t, T, r, sigma_t, delta)
                initial_premium = black_scholes_call(S_t, strike, T, r, sigma_t)
                
                target_delta = call_delta(S_t, strike, T, r, sigma_t)
                trade_qty = target_delta
                
                fee = abs(trade_qty) * S_t * FEE_RATE
                hedge_cash = -trade_qty * S_t - fee
                hedge_position = trade_qty
                
                active_option = True
                daily_total_mtm = total_realized_pnl + (hedge_cash + hedge_position * S_t)
            else:
                daily_total_mtm = total_realized_pnl
        else:
            days_to_expiry -= 1
            T = max(0.00001, days_to_expiry / 365.0)
            
            current_option_price = black_scholes_call(S_t, strike, T, r, sigma_t)
            current_delta = call_delta(S_t, strike, T, r, sigma_t)
            
            if current_option_price <= initial_premium * take_profit:
                option_mtm = initial_premium - current_option_price
                fee = abs(hedge_position) * S_t * FEE_RATE
                hedge_cash = hedge_cash + hedge_position * S_t - fee
                hedge_position = 0
                
                total_realized_pnl += (option_mtm + hedge_cash)
                daily_total_mtm = total_realized_pnl
                active_option = False
                
            elif days_to_expiry > 0:
                delta_diff = current_delta - hedge_position
                if abs(delta_diff) > threshold:
                    fee = abs(delta_diff) * S_t * FEE_RATE
                    hedge_cash = hedge_cash - delta_diff * S_t - fee
                    hedge_position = current_delta
                
                option_mtm = initial_premium - current_option_price
                daily_total_mtm = total_realized_pnl + option_mtm + (hedge_cash + hedge_position * S_t)
                
            else:
                payoff = max(0, S_t - strike)
                option_mtm = initial_premium - payoff
                fee = abs(hedge_position) * S_t * FEE_RATE
                
                hedge_cash = hedge_cash + hedge_position * S_t - fee
                hedge_position = 0
                
                total_realized_pnl += (option_mtm + hedge_cash)
                daily_total_mtm = total_realized_pnl
                active_option = False
                
        pnl_curve.append(daily_total_mtm)

    pnl_series = pd.Series(pnl_curve, index=df.index)
    
    # Calculate min capital to survive
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
        'tenor': tenor,
        'delta': delta,
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
    df.dropna(inplace=True)
    
    tenors = [14, 30, 60, 90]
    deltas = [0.1, 0.2, 0.3]
    
    print(f"\nRunning Parameter Optimization: {len(tenors) * len(deltas)} combinations...")
    results = []
    
    for t in tenors:
        for d in deltas:
            res = run_simulation(df, tenor=t, delta=d)
            results.append(res)
            print(f"Tenor: {t:2d}d | Delta: {d:.1f} | PnL: ${res['total_pnl_usd']:8.2f} | Ann.Return: {res['ann_return']:6.2%} | Max DD: {res['max_dd']:6.2%} | Cap Req: ${res['min_capital_req']:.0f}")

    res_df = pd.DataFrame(results)
    best_ret = res_df.loc[res_df['ann_return'].idxmax()]
    best_sharpe_proxy = res_df.loc[(res_df['ann_return'] / res_df['max_dd']).idxmax()]
    
    print("\n--- OPTIMIZATION RESULTS ---")
    print(f"Highest Annualized Return: Tenor {best_ret['tenor']}d, Delta {best_ret['delta']} ({best_ret['ann_return']:.2%} AnnRet, {best_ret['max_dd']:.2%} DD)")
    print(f"Best Risk-Adjusted (Return/DD): Tenor {best_sharpe_proxy['tenor']}d, Delta {best_sharpe_proxy['delta']} ({best_sharpe_proxy['ann_return']:.2%} AnnRet, {best_sharpe_proxy['max_dd']:.2%} DD)")

if __name__ == "__main__":
    main()

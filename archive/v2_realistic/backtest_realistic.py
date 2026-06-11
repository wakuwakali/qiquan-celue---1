import ccxt
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0: return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    if T <= 0: return max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

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

def run_realistic_grail(df):
    r = 0.0
    SPOT_FEE_RATE = 0.0010      # 0.1% Spot Taker Fee
    OPT_FEE_RATE = 0.0003       # 0.03% Deribit Option Fee (per underlying)
    OPT_SLIPPAGE = 0.02         # 2% Slippage on option premium
    
    tenor_short = 90
    delta_short = 0.3
    vol_filter_short = 0.65
    threshold = 0.05
    take_profit = 0.20
    
    vol_filter_long = 0.40
    tenor_long = 30
    
    active_short = False
    strike_short = 0
    days_short = 0
    premium_short_collected = 0
    hedge_pos = 0
    hedge_cash = 0
    
    active_long = False
    strike_long = 0
    days_long = 0
    premium_long_paid = 0
    
    total_realized_pnl = 0
    pnl_curve = []
    
    # Pre-extract arrays for speed
    opens = df['open'].values
    closes = df['close'].values
    vols = df['Vol_30d'].values
    
    # We start from index 1 because we need i-1 for decision
    pnl_curve.append(0)
    
    total_opt_fees = 0
    total_slippage = 0
    
    for i in range(1, len(df)):
        # CAUSAL EXECUTION: Decisions based on Yesterday's Close
        sigma_prev = vols[i-1]
        
        # Execution happens at Today's Open Price
        S_exec = opens[i]
        
        # MTM at Today's Close Price
        S_mtm = closes[i]
        
        daily_mtm = total_realized_pnl
        
        # --- EXECUTION PHASE (at S_exec) ---
        
        # 1. Manage Short Position
        if not active_short and not active_long:
            if not np.isnan(sigma_prev) and sigma_prev > vol_filter_short:
                days_short = tenor_short
                T = days_short / 365.0
                strike_short = find_call_strike(S_exec, T, r, sigma_prev, delta_short)
                bs_premium = black_scholes_call(S_exec, strike_short, T, r, sigma_prev)
                
                # Realistic Sell: Apply Slippage (receive 98%) and Fee
                premium_short_collected = bs_premium * (1 - OPT_SLIPPAGE)
                opt_fee = S_exec * OPT_FEE_RATE
                
                total_slippage += (bs_premium * OPT_SLIPPAGE)
                total_opt_fees += opt_fee
                total_realized_pnl -= opt_fee  # Option fee paid immediately
                
                # Hedge Spot
                target_delta = call_delta(S_exec, strike_short, T, r, sigma_prev)
                hedge_pos = target_delta
                spot_fee = abs(hedge_pos) * S_exec * SPOT_FEE_RATE
                hedge_cash = -hedge_pos * S_exec - spot_fee
                
                active_short = True
                
        if active_short:
            days_short -= 1 # 1 day passed
            T_exec = max(0.00001, days_short / 365.0)
            
            # Use current exec price to see if we hit take profit
            curr_bs_price = black_scholes_call(S_exec, strike_short, T_exec, r, sigma_prev)
            
            # Realistic Buyback: Pay 102%
            if curr_bs_price * (1 + OPT_SLIPPAGE) <= premium_short_collected * take_profit:
                close_cost = curr_bs_price * (1 + OPT_SLIPPAGE)
                opt_fee = S_exec * OPT_FEE_RATE
                
                total_slippage += (curr_bs_price * OPT_SLIPPAGE)
                total_opt_fees += opt_fee
                
                opt_mtm = premium_short_collected - close_cost - opt_fee
                
                # Unwind hedge at Open
                spot_fee = abs(hedge_pos) * S_exec * SPOT_FEE_RATE
                hedge_cash += hedge_pos * S_exec - spot_fee
                hedge_pos = 0
                
                total_realized_pnl += (opt_mtm + hedge_cash)
                active_short = False
                hedge_cash = 0
                
            elif days_short > 0:
                # Delta Rebalance at Open
                curr_delta = call_delta(S_exec, strike_short, T_exec, r, sigma_prev)
                delta_diff = curr_delta - hedge_pos
                if abs(delta_diff) > threshold:
                    spot_fee = abs(delta_diff) * S_exec * SPOT_FEE_RATE
                    hedge_cash -= delta_diff * S_exec + spot_fee
                    hedge_pos = curr_delta
            else:
                # Expiry Settlement (Usually no slippage on cash settlement, but fee applies)
                payoff = max(0, S_exec - strike_short)
                opt_fee = S_exec * OPT_FEE_RATE
                total_opt_fees += opt_fee
                
                opt_mtm = premium_short_collected - payoff - opt_fee
                
                # Unwind hedge
                spot_fee = abs(hedge_pos) * S_exec * SPOT_FEE_RATE
                hedge_cash += hedge_pos * S_exec - spot_fee
                hedge_pos = 0
                
                total_realized_pnl += (opt_mtm + hedge_cash)
                active_short = False
                hedge_cash = 0

        # 2. Manage Long Straddle Position
        if not active_short and not active_long:
            if not np.isnan(sigma_prev) and sigma_prev < vol_filter_long:
                days_long = tenor_long
                T = days_long / 365.0
                strike_long = S_exec
                call_bs = black_scholes_call(S_exec, strike_long, T, r, sigma_prev)
                put_bs = black_scholes_put(S_exec, strike_long, T, r, sigma_prev)
                
                # Realistic Buy: Pay 102%
                total_bs = call_bs + put_bs
                premium_long_paid = total_bs * (1 + OPT_SLIPPAGE)
                
                total_slippage += (total_bs * OPT_SLIPPAGE)
                
                # Option fee (2 contracts: 1 call, 1 put)
                opt_fee = S_exec * OPT_FEE_RATE * 2
                total_opt_fees += opt_fee
                
                total_realized_pnl -= (premium_long_paid + opt_fee)
                active_long = True
                
        if active_long:
            days_long -= 1
            T_exec = max(0.00001, days_long / 365.0)
            
            if days_long == 0:
                # Expiry Settlement
                payoff = max(0, S_exec - strike_long) + max(0, strike_long - S_exec)
                # Option fee for settlement
                opt_fee = S_exec * OPT_FEE_RATE * 2
                total_opt_fees += opt_fee
                
                total_realized_pnl += (payoff - opt_fee)
                active_long = False
                
        # --- MTM PHASE (at S_mtm) ---
        if active_short:
            T_mtm = max(0.00001, days_short / 365.0)
            curr_mtm_price = black_scholes_call(S_mtm, strike_short, T_mtm, r, sigma_prev)
            opt_mtm = premium_short_collected - curr_mtm_price
            daily_mtm = total_realized_pnl + opt_mtm + (hedge_cash + hedge_pos * S_mtm)
            
        elif active_long:
            T_mtm = max(0.00001, days_long / 365.0)
            curr_c = black_scholes_call(S_mtm, strike_long, T_mtm, r, sigma_prev)
            curr_p = black_scholes_put(S_mtm, strike_long, T_mtm, r, sigma_prev)
            daily_mtm = total_realized_pnl + (curr_c + curr_p)
            
        else:
            daily_mtm = total_realized_pnl
            
        pnl_curve.append(daily_mtm)

    pnl_series = pd.Series(pnl_curve, index=df.index)
    
    peak_pnl = pnl_series.cummax()
    drawdown_usd = peak_pnl - pnl_series
    max_drawdown_usd = drawdown_usd.max()
    lowest_underwater = pnl_series.min()
    min_safe_capital = max(10000.0, max_drawdown_usd / 0.8, abs(lowest_underwater) / 0.8)
    
    equity_curve = min_safe_capital + pnl_series
    years = len(df) / 365.25
    total_return = equity_curve.iloc[-1] / min_safe_capital - 1
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    peak = equity_curve.cummax()
    drawdown = (peak - equity_curve) / peak
    max_drawdown = drawdown.max()
    
    return pnl_series, {
        'total_pnl_usd': pnl_series.iloc[-1],
        'min_capital_req': min_safe_capital,
        'ann_return': annualized_return,
        'max_dd': max_drawdown,
        'total_opt_fees': total_opt_fees,
        'total_slippage': total_slippage
    }

def main():
    print("Fetching data from Binance...")
    df = fetch_binance_data()
    df['LogRet'] = np.log(df['close'] / df['close'].shift(1))
    df['Vol_30d'] = df['LogRet'].rolling(window=30).std() * np.sqrt(365)
    
    print("Running Realistic Execution Simulation...")
    pnl_realistic, stats = run_realistic_grail(df)
    
    # We will load the utopian pnl from previous logic for direct comparison
    # But since that takes writing the whole thing again, we will just plot the realistic one
    
    plt.figure(figsize=(15, 9))
    plt.plot(df.index, pnl_realistic, label='Realistic Holy Grail (Slippage + Fees + Causal)', color='crimson', linewidth=3)
    
    plt.title('Brutal Reality: Options Strategy Equity Curve with Real Market Frictions')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL (USD)')
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'realistic_equity_curve.png')
    plt.savefig(plot_path)
    
    print("\n--- REALISTIC METRICS ---")
    print(f"Total PnL: ${stats['total_pnl_usd']:,.2f}")
    print(f"Annualized Return: {stats['ann_return']:.2%}")
    print(f"Max Drawdown: {stats['max_dd']:.2%}")
    print(f"Capital Required: ${stats['min_capital_req']:,.2f}")
    print("\n--- INVISIBLE FRICTION COSTS ---")
    print(f"Total Option Exchange Fees: ${stats['total_opt_fees']:,.2f}")
    print(f"Total Bid-Ask Slippage Loss: ${stats['total_slippage']:,.2f}")

if __name__ == "__main__":
    main()

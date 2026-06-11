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
    
    print("Fetching daily klines from Binance...")
    since = exchange.parse8601(start_time_str)
    all_klines = []
    while True:
        try:
            klines = exchange.fetch_ohlcv(symbol, '1d', since=since, limit=1000)
            if not klines: break
            all_klines.extend(klines)
            since = klines[-1][0] + 86400000
            if len(klines) < 1000: break
        except Exception as e:
            print(f"Error fetching klines: {e}")
            break
            
    df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    df.set_index('Date', inplace=True)
    return df

def run_institutional_backtest():
    df = fetch_binance_data()
    if df.empty:
        print("Failed to fetch data.")
        return
        
    df['LogRet'] = np.log(df['close'] / df['close'].shift(1))
    df['Vol_30d'] = df['LogRet'].rolling(window=30).std() * np.sqrt(365)
    df.dropna(inplace=True)
    
    r = 0.0
    # Spot Market Taker Fee is 0.1%
    FEE_RATE = 0.0010  
    
    # State variables
    active_option = False
    strike = 0
    days_to_expiry = 0
    initial_premium = 0
    hedge_position = 0
    
    hedge_cash = 0
    total_realized_pnl = 0
    total_transaction_costs = 0
    rebalance_count = 0
    
    results = []
    
    dates = df.index
    closes = df['close'].values
    vols = df['Vol_30d'].values
    
    print("Simulating Institutional Delta Hedging...")
    
    for i in range(len(df)):
        date_t = dates[i]
        S_t = closes[i]
        sigma_t = vols[i]
        
        # Strategy Rules
        VOL_FILTER_THRESHOLD = 0.65
        REBALANCE_THRESHOLD = 0.05
        TAKE_PROFIT_RATIO = 0.20 # Close if option value drops to 20% of premium
        
        if not active_option:
            # 1. Volatility Filter: Only enter if IV > 65%
            if sigma_t > VOL_FILTER_THRESHOLD:
                days_to_expiry = 90
                T = days_to_expiry / 365.0
                strike = find_call_strike(S_t, T, r, sigma_t, 0.1)
                initial_premium = black_scholes_call(S_t, strike, T, r, sigma_t)
                
                target_delta = call_delta(S_t, strike, T, r, sigma_t)
                trade_qty = target_delta
                
                # 2. Spot Hedging: Buy BTC directly (No funding fee!)
                fee = abs(trade_qty) * S_t * FEE_RATE
                total_transaction_costs += fee
                rebalance_count += 1
                
                hedge_cash = -trade_qty * S_t - fee
                hedge_position = trade_qty
                
                active_option = True
                option_mtm = 0
                daily_total_mtm = total_realized_pnl + option_mtm + (hedge_cash + hedge_position * S_t)
            else:
                # Idle, doing nothing
                daily_total_mtm = total_realized_pnl
        else:
            days_to_expiry -= 1
            T = max(0.00001, days_to_expiry / 365.0)
            
            # Note: No funding fee deduction because we are using Spot BTC
            
            current_option_price = black_scholes_call(S_t, strike, T, r, sigma_t)
            current_delta = call_delta(S_t, strike, T, r, sigma_t)
            
            # 4. Take Profit Filter
            if current_option_price <= initial_premium * TAKE_PROFIT_RATIO:
                # Take profit: close option and unwind hedge
                option_mtm = initial_premium - current_option_price
                
                fee = abs(hedge_position) * S_t * FEE_RATE
                total_transaction_costs += fee
                rebalance_count += 1
                
                hedge_cash = hedge_cash + hedge_position * S_t - fee
                hedge_position = 0
                
                total_realized_pnl += (option_mtm + hedge_cash)
                daily_total_mtm = total_realized_pnl
                
                active_option = False
                
            elif days_to_expiry > 0:
                # 3. Threshold Rebalancing
                delta_diff = current_delta - hedge_position
                
                if abs(delta_diff) > REBALANCE_THRESHOLD:
                    fee = abs(delta_diff) * S_t * FEE_RATE
                    total_transaction_costs += fee
                    rebalance_count += 1
                    
                    hedge_cash = hedge_cash - delta_diff * S_t - fee
                    hedge_position = current_delta
                
                option_mtm = initial_premium - current_option_price
                daily_total_mtm = total_realized_pnl + option_mtm + (hedge_cash + hedge_position * S_t)
                
            else:
                # Expiry Day Settlement
                payoff = max(0, S_t - strike)
                option_mtm = initial_premium - payoff
                
                fee = abs(hedge_position) * S_t * FEE_RATE
                total_transaction_costs += fee
                rebalance_count += 1
                
                hedge_cash = hedge_cash + hedge_position * S_t - fee
                hedge_position = 0
                
                total_realized_pnl += (option_mtm + hedge_cash)
                daily_total_mtm = total_realized_pnl
                
                active_option = False
                
        results.append({
            'Date': date_t,
            'Institutional_Hedged_PnL': daily_total_mtm
        })

    res_df = pd.DataFrame(results).set_index('Date')
    
    # Simple Naked Call (Always in Market) for comparison baseline
    naked_pnl = 0
    naked_history = []
    n_act = False
    n_str = 0
    n_prem = 0
    n_days = 0
    
    for i in range(len(df)):
        S_t = closes[i]
        sigma_t = vols[i]
        if not n_act:
            n_days = 90
            T = n_days / 365.0
            n_str = find_call_strike(S_t, T, r, sigma_t, 0.1)
            n_prem = black_scholes_call(S_t, n_str, T, r, sigma_t)
            n_act = True
            naked_history.append(naked_pnl)
        else:
            n_days -= 1
            if n_days > 0:
                T = max(0.00001, n_days / 365.0)
                curr_p = black_scholes_call(S_t, n_str, T, r, sigma_t)
                naked_history.append(naked_pnl + (n_prem - curr_p))
            else:
                payoff = max(0, S_t - n_str)
                naked_pnl += (n_prem - payoff)
                naked_history.append(naked_pnl)
                n_act = False

    res_df['Naked_Call_PnL'] = naked_history
    
    plt.figure(figsize=(14, 8))
    plt.plot(res_df.index, res_df['Naked_Call_PnL'], label='Naked 90-day Call (Dumb Retail)', color='red', linestyle='--')
    plt.plot(res_df.index, res_df['Institutional_Hedged_PnL'], label='Institutional Hedge (Filtered + Spot + Take Profit)', color='gold', linewidth=3)
    
    plt.title('Institutional vs Retail Options Strategy (2020-2026)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL (USD)')
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'institutional_equity_curve.png')
    plt.savefig(plot_path)
    print(f"Saved institutional equity curve plot to: {plot_path}")
    
    print("\n--- Final Cumulative PnL ---")
    print(f"Naked Call (Retail): ${res_df['Naked_Call_PnL'].iloc[-1]:,.2f}")
    print(f"Institutional Strategy: ${res_df['Institutional_Hedged_PnL'].iloc[-1]:,.2f}")
    
    print("\n--- Execution Stats ---")
    print(f"Total Transaction Costs (Spot 0.1% Fees): ${total_transaction_costs:,.2f}")
    print(f"Total Funding Fees Paid: $0.00 (Spot Hedging Magic!)")
    print(f"Total Number of Trade Executions (Rebalances): {rebalance_count}")
    
    # Calculate Quantitative Metrics for Max Leverage (Min Capital)
    # We want the max drawdown to be exactly 80% of the initial capital (to survive without liquidation)
    # Max Drawdown USD is the largest drop from peak.
    peak_pnl = res_df['Institutional_Hedged_PnL'].cummax()
    drawdown_usd = peak_pnl - res_df['Institutional_Hedged_PnL']
    max_drawdown_usd = drawdown_usd.max()
    
    # Lowest point the account ever went from starting balance
    lowest_underwater = res_df['Institutional_Hedged_PnL'].min()
    
    # To survive a drawdown of max_drawdown_usd without hitting 0, and leaving a 20% buffer:
    # Minimum capital = max_drawdown_usd / 0.8
    # However, we also need enough capital to cover the absolute lowest underwater point if it happened early
    # Let's take the max of (max_drawdown_usd / 0.8) and (abs(lowest_underwater) / 0.8)
    # Also, practically we need at least $5000 just to meet exchange margin requirements for 1 BTC option.
    min_safe_capital = max(5000.0, max_drawdown_usd / 0.8, abs(lowest_underwater) / 0.8)
    
    equity_curve = min_safe_capital + res_df['Institutional_Hedged_PnL']
    
    # 1. Annualized Return
    years = len(df) / 365.25
    total_return = equity_curve.iloc[-1] / min_safe_capital - 1
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    # 2. Max Drawdown %
    peak = equity_curve.cummax()
    drawdown = (peak - equity_curve) / peak
    max_drawdown = drawdown.max()
    
    # 3. Sharpe Ratio (assuming Risk Free Rate = 0)
    daily_returns = equity_curve.pct_change().dropna()
    sharpe_ratio = np.sqrt(365.25) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
    
    print(f"\n--- Quantitative Metrics (MAX LEVERAGE - ${min_safe_capital:,.2f} Capital) ---")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Max Drawdown: -{max_drawdown:.2%} ($-{max_drawdown_usd:,.2f})")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

if __name__ == "__main__":
    run_institutional_backtest()

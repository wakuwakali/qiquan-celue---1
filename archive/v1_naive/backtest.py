import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0:
        return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    if T <= 0:
        return max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def call_delta(S, K, T, r, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def put_delta(S, K, T, r, sigma):
    if T <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1.0

def find_strike_for_target_call_delta(S, T, r, sigma, target_delta=0.1):
    K = S
    step = S * 0.01
    max_iters = 1000
    for _ in range(max_iters):
        delta = call_delta(S, K, T, r, sigma)
        if delta < target_delta:
            return K, delta
        K += step
    return K, call_delta(S, K, T, r, sigma)

def find_strike_for_target_put_delta(S, T, r, sigma, target_delta=-0.1):
    K = S
    step = S * 0.01
    max_iters = 1000
    for _ in range(max_iters):
        delta = put_delta(S, K, T, r, sigma)
        # put delta is negative, e.g., starts at -0.5 and goes to 0 as K goes down
        # we want delta > target_delta (e.g. delta > -0.1)
        if delta > target_delta:
            return K, delta
        K -= step
        if K <= 0:
            return step, put_delta(S, step, T, r, sigma)
    return K, put_delta(S, K, T, r, sigma)

def run_backtest():
    print("Downloading BTC daily data...")
    btc = yf.download('BTC-USD', start='2020-01-01', progress=False)
    
    if len(btc) == 0:
        print("Failed to download data.")
        return

    if isinstance(btc.columns, pd.MultiIndex):
        df = btc['Close'].copy()
        df = pd.DataFrame({'Close': df.iloc[:, 0]})
    else:
        df = pd.DataFrame({'Close': btc['Close']})

    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_30d'] = df['LogRet'].rolling(window=30).std() * np.sqrt(365)
    df.dropna(inplace=True)
    
    T = 1 / 365.0
    r = 0.0
    target_delta_c = 0.1
    target_delta_p = -0.1
    
    pnl_list = []
    trades = []
    
    print(f"Running backtest from {df.index[0].date()} to {df.index[-1].date()}...")
    
    dates = df.index
    closes = df['Close'].values
    vols = df['Vol_30d'].values
    
    for i in range(len(df) - 1):
        date_t = dates[i]
        date_next = dates[i+1]
        
        S_t = closes[i]
        sigma_t = vols[i]
        S_next = closes[i+1]
        
        # Call
        K_c, actual_delta_c = find_strike_for_target_call_delta(S_t, T, r, sigma_t, target_delta_c)
        premium_c = black_scholes_call(S_t, K_c, T, r, sigma_t)
        payoff_c = max(0, S_next - K_c)
        
        # Put
        K_p, actual_delta_p = find_strike_for_target_put_delta(S_t, T, r, sigma_t, target_delta_p)
        premium_p = black_scholes_put(S_t, K_p, T, r, sigma_t)
        payoff_p = max(0, K_p - S_next)
        
        # Total Premium & Payoff
        total_premium = premium_c + premium_p
        total_payoff = payoff_c + payoff_p
        
        pnl = total_premium - total_payoff
        pnl_list.append(pnl)
        
        trades.append({
            'Date': date_t,
            'Spot': S_t,
            'Vol': sigma_t,
            'Call_Strike': K_c,
            'Put_Strike': K_p,
            'Total_Premium': total_premium,
            'Spot_Next': S_next,
            'Total_Payoff': total_payoff,
            'PnL': pnl
        })
    
    trades_df = pd.DataFrame(trades)
    trades_df.set_index('Date', inplace=True)
    trades_df['Cumulative_PnL'] = trades_df['PnL'].cumsum()
    
    print("Backtest finished.")
    print("--- Summary Statistics ---")
    total_trades = len(trades_df)
    win_trades = len(trades_df[trades_df['PnL'] > 0])
    win_rate = win_trades / total_trades
    total_pnl = trades_df['Cumulative_PnL'].iloc[-1]
    
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Total PnL (USD per 1 BTC option): ${total_pnl:.2f}")
    print(f"Average Premium Collected: ${trades_df['Total_Premium'].mean():.2f}")
    print(f"Average PnL per Trade: ${trades_df['PnL'].mean():.2f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(trades_df.index, trades_df['Cumulative_PnL'], label='Cumulative PnL (USD)')
    plt.title('Short 0.1 Delta Strangle (Call + Put) BTC Option (1-day Expiry)')
    plt.xlabel('Date')
    plt.ylabel('PnL (USD)')
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'equity_curve.png')
    plt.savefig(plot_path)
    print(f"Saved equity curve plot to: {plot_path}")
    
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_trades.csv')
    try:
        trades_df.to_csv(csv_path)
        print(f"Saved trades to: {csv_path}")
    except Exception as e:
        print(f"Warning: Could not save CSV. Is it open? Error: {e}")

if __name__ == "__main__":
    run_backtest()

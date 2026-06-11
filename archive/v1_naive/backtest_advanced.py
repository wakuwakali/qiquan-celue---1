import pandas as pd
import numpy as np
import yfinance as yf
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

def put_delta(S, K, T, r, sigma):
    if T <= 0: return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1.0

def find_call_strike(S, T, r, sigma, target_delta):
    K = S
    step = S * 0.01
    for _ in range(500):
        if call_delta(S, K, T, r, sigma) < target_delta: return K
        K += step
    return K

def find_put_strike(S, T, r, sigma, target_delta):
    K = S
    step = S * 0.01
    for _ in range(500):
        if put_delta(S, K, T, r, sigma) > target_delta: return K
        K -= step
        if K <= 0: return step
    return K

def run_multi_backtest():
    print("Downloading BTC daily data...")
    btc = yf.download('BTC-USD', start='2020-01-01', progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        df = pd.DataFrame({'Close': btc['Close'].iloc[:, 0]})
    else:
        df = pd.DataFrame({'Close': btc['Close']})

    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_30d'] = df['LogRet'].rolling(window=30).std() * np.sqrt(365)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df.dropna(inplace=True)
    
    T = 1 / 365.0
    r = 0.0
    
    results = []
    
    dates = df.index
    closes = df['Close'].values
    vols = df['Vol_30d'].values
    sma = df['SMA20'].values
    
    print("Simulating strategies...")
    for i in range(len(df) - 1):
        date_t = dates[i]
        S_t = closes[i]
        sigma_t = vols[i]
        sma_t = sma[i]
        S_next = closes[i+1]
        
        # 1. Baseline: Hold 1 BTC
        pnl_hold = S_next - S_t
        
        # Strikes
        C_01 = find_call_strike(S_t, T, r, sigma_t, 0.1)
        C_05 = find_call_strike(S_t, T, r, sigma_t, 0.05)
        P_01 = find_put_strike(S_t, T, r, sigma_t, -0.1)
        P_05 = find_put_strike(S_t, T, r, sigma_t, -0.05)
        
        # Premiums
        prem_C_01 = black_scholes_call(S_t, C_01, T, r, sigma_t)
        prem_C_05 = black_scholes_call(S_t, C_05, T, r, sigma_t)
        prem_P_01 = black_scholes_put(S_t, P_01, T, r, sigma_t)
        prem_P_05 = black_scholes_put(S_t, P_05, T, r, sigma_t)
        
        # Payoffs
        payoff_C_01 = max(0, S_next - C_01)
        payoff_C_05 = max(0, S_next - C_05)
        payoff_P_01 = max(0, P_01 - S_next)
        payoff_P_05 = max(0, P_05 - S_next)
        
        # 2. Iron Condor
        prem_ic = prem_C_01 + prem_P_01 - prem_C_05 - prem_P_05
        payoff_ic = payoff_C_01 + payoff_P_01 - payoff_C_05 - payoff_P_05
        pnl_ic = prem_ic - payoff_ic
        
        # 3. Volatility Filter (>60%)
        if sigma_t > 0.60:
            pnl_vol = (prem_C_01 + prem_P_01) - (payoff_C_01 + payoff_P_01)
        else:
            pnl_vol = 0
            
        # 4. Covered Call
        pnl_cc = pnl_hold + (prem_C_01 - payoff_C_01)
        
        # 5. Trend Filtered
        if S_t > sma_t:
            pnl_trend = prem_P_01 - payoff_P_01
        else:
            pnl_trend = prem_C_01 - payoff_C_01
            
        results.append({
            'Date': date_t,
            'Hold_BTC': pnl_hold,
            'Iron_Condor': pnl_ic,
            'Vol_Filter': pnl_vol,
            'Covered_Call': pnl_cc,
            'Trend_Filter': pnl_trend
        })
        
    res_df = pd.DataFrame(results).set_index('Date')
    cum_pnl = res_df.cumsum()
    
    # We want Covered Call and Hold BTC to start at 0 PnL (they already do via daily diffs)
    
    plt.figure(figsize=(14, 8))
    plt.plot(cum_pnl.index, cum_pnl['Hold_BTC'], label='Baseline: Hold 1 BTC', color='black', linestyle='--')
    plt.plot(cum_pnl.index, cum_pnl['Covered_Call'], label='3. Covered Call (Hold BTC + Short 0.1 Call)', color='blue')
    plt.plot(cum_pnl.index, cum_pnl['Iron_Condor'], label='1. Iron Condor (Risk-Defined Strangle)', color='green')
    plt.plot(cum_pnl.index, cum_pnl['Trend_Filter'], label='4. Trend Filtered (Uptrend: Short Put, Downtrend: Short Call)', color='purple')
    plt.plot(cum_pnl.index, cum_pnl['Vol_Filter'], label='2. Volatility Filter (Short Strangle only when IV > 60%)', color='orange')
    
    plt.title('Comparison of Advanced BTC Option Strategies (2020 - 2026)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL (USD)')
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'advanced_equity_curve.png')
    plt.savefig(plot_path)
    print(f"Saved advanced equity curve plot to: {plot_path}")
    
    print("\n--- Final Cumulative PnL ---")
    for col in cum_pnl.columns:
        print(f"{col}: ${cum_pnl[col].iloc[-1]:,.2f}")
        
    # Calculate win rate for the pure option strategies (exclude Hold_BTC and Covered_Call which are mostly delta)
    print("\n--- Win Rates (Option Strategies) ---")
    for col in ['Iron_Condor', 'Vol_Filter', 'Trend_Filter']:
        # Only count days where a trade actually happened (PnL != 0)
        active_days = res_df[res_df[col] != 0]
        if len(active_days) > 0:
            win_rate = len(active_days[active_days[col] > 0]) / len(active_days)
            print(f"{col}: {win_rate:.2%} (over {len(active_days)} active trading days)")

if __name__ == "__main__":
    run_multi_backtest()

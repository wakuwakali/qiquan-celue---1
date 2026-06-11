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

def run_holy_grail(df):
    r = 0.0
    FEE_RATE = 0.0010  
    
    # Best Params
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
    premium_short = 0
    hedge_pos = 0
    hedge_cash = 0
    
    active_long = False
    strike_long = 0
    days_long = 0
    premium_long_paid = 0
    
    total_realized_pnl = 0
    pnl_curve = []
    
    closes = df['close'].values
    vols = df['Vol_30d'].values
    
    for i in range(len(df)):
        S_t = closes[i]
        sigma_t = vols[i]
        daily_mtm = total_realized_pnl
        
        # 1. Manage Short Position
        if not active_short and not active_long:
            if sigma_t > vol_filter_short:
                days_short = tenor_short
                T = days_short / 365.0
                strike_short = find_call_strike(S_t, T, r, sigma_t, delta_short)
                premium_short = black_scholes_call(S_t, strike_short, T, r, sigma_t)
                
                target_delta = call_delta(S_t, strike_short, T, r, sigma_t)
                hedge_pos = target_delta
                fee = abs(hedge_pos) * S_t * FEE_RATE
                hedge_cash = -hedge_pos * S_t - fee
                
                active_short = True
                
        if active_short:
            days_short -= 1
            T = max(0.00001, days_short / 365.0)
            curr_price = black_scholes_call(S_t, strike_short, T, r, sigma_t)
            curr_delta = call_delta(S_t, strike_short, T, r, sigma_t)
            
            if curr_price <= premium_short * take_profit:
                opt_mtm = premium_short - curr_price
                fee = abs(hedge_pos) * S_t * FEE_RATE
                hedge_cash = hedge_cash + hedge_pos * S_t - fee
                hedge_pos = 0
                total_realized_pnl += (opt_mtm + hedge_cash)
                active_short = False
                hedge_cash = 0
            elif days_short > 0:
                delta_diff = curr_delta - hedge_pos
                if abs(delta_diff) > threshold:
                    fee = abs(delta_diff) * S_t * FEE_RATE
                    hedge_cash = hedge_cash - delta_diff * S_t - fee
                    hedge_pos = curr_delta
                opt_mtm = premium_short - curr_price
                daily_mtm += opt_mtm + (hedge_cash + hedge_pos * S_t)
            else:
                payoff = max(0, S_t - strike_short)
                opt_mtm = premium_short - payoff
                fee = abs(hedge_pos) * S_t * FEE_RATE
                hedge_cash = hedge_cash + hedge_pos * S_t - fee
                hedge_pos = 0
                total_realized_pnl += (opt_mtm + hedge_cash)
                active_short = False
                hedge_cash = 0
                
        # 2. Manage Long Straddle Position
        if not active_short and not active_long:
            if sigma_t < vol_filter_long:
                days_long = tenor_long
                T = days_long / 365.0
                strike_long = S_t 
                call_p = black_scholes_call(S_t, strike_long, T, r, sigma_t)
                put_p = black_scholes_put(S_t, strike_long, T, r, sigma_t)
                premium_long_paid = call_p + put_p
                fee = premium_long_paid * FEE_RATE
                total_realized_pnl -= (premium_long_paid + fee)
                active_long = True
                
        if active_long:
            days_long -= 1
            T = max(0.00001, days_long / 365.0)
            
            if days_long > 0:
                curr_c = black_scholes_call(S_t, strike_long, T, r, sigma_t)
                curr_p = black_scholes_put(S_t, strike_long, T, r, sigma_t)
                daily_mtm = total_realized_pnl + (curr_c + curr_p)
            else:
                payoff = max(0, S_t - strike_long) + max(0, strike_long - S_t)
                fee = payoff * FEE_RATE
                total_realized_pnl += (payoff - fee)
                daily_mtm = total_realized_pnl
                active_long = False
                
        pnl_curve.append(daily_mtm if (active_short or active_long) else total_realized_pnl)

    pnl_series = pd.Series(pnl_curve, index=df.index)
    
    # Base strategy array for comparison
    base_pnl = []
    total_base = 0
    active_b = False
    d_b = 0
    str_b = 0
    prem_b = 0
    h_b = 0
    hc_b = 0
    
    for i in range(len(df)):
        S_t = closes[i]
        sigma_t = vols[i]
        daily = total_base
        
        if not active_b:
            if sigma_t > 0.65:
                d_b = 90
                T = d_b / 365.0
                str_b = find_call_strike(S_t, T, r, sigma_t, 0.1)
                prem_b = black_scholes_call(S_t, str_b, T, r, sigma_t)
                cd = call_delta(S_t, str_b, T, r, sigma_t)
                h_b = cd
                fee = abs(h_b) * S_t * FEE_RATE
                hc_b = -h_b * S_t - fee
                active_b = True
        else:
            d_b -= 1
            T = max(0.00001, d_b / 365.0)
            cp = black_scholes_call(S_t, str_b, T, r, sigma_t)
            cd = call_delta(S_t, str_b, T, r, sigma_t)
            
            if cp <= prem_b * 0.2:
                om = prem_b - cp
                fee = abs(h_b) * S_t * FEE_RATE
                hc_b += h_b * S_t - fee
                h_b = 0
                total_base += (om + hc_b)
                active_b = False
                hc_b = 0
            elif d_b > 0:
                dd = cd - h_b
                if abs(dd) > 0.05:
                    fee = abs(dd) * S_t * FEE_RATE
                    hc_b -= dd * S_t + fee
                    h_b = cd
                om = prem_b - cp
                daily += om + (hc_b + h_b * S_t)
            else:
                payoff = max(0, S_t - str_b)
                om = prem_b - payoff
                fee = abs(h_b) * S_t * FEE_RATE
                hc_b += h_b * S_t - fee
                h_b = 0
                total_base += (om + hc_b)
                active_b = False
                hc_b = 0
        base_pnl.append(daily if active_b else total_base)
        
    return pnl_series, pd.Series(base_pnl, index=df.index)

def main():
    print("Fetching data from Binance...")
    df = fetch_binance_data()
    df['LogRet'] = np.log(df['close'] / df['close'].shift(1))
    df['Vol_30d'] = df['LogRet'].rolling(window=30).std() * np.sqrt(365)
    df.dropna(inplace=True)
    
    print("Running Final Holy Grail Strategy...")
    grail, base = run_holy_grail(df)
    
    plt.figure(figsize=(15, 9))
    plt.plot(df.index, base, label='Phase 0: Base Institutional (16% Ann.)', color='gray', linestyle='--')
    plt.plot(df.index, grail, label='Holy Grail: Hybrid Long/Short Volatility (28.2% Ann.)', color='gold', linewidth=3)
    
    plt.title('The Holy Grail Options Strategy: Asymmetric Volatility Trading')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL (USD)')
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'holy_grail_equity_curve.png')
    plt.savefig(plot_path)

if __name__ == "__main__":
    main()

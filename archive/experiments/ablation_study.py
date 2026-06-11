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
    
    all_rates = []
    since = exchange.parse8601(start_time_str)
    while True:
        rates = exchange.fapiPublicGetFundingRate({'symbol': 'BTCUSDT', 'startTime': since, 'limit': 1000})
        if not rates: break
        all_rates.extend(rates)
        since = int(rates[-1]['fundingTime']) + 1
        if len(rates) < 1000: break
            
    df_rates = pd.DataFrame(all_rates)
    df_rates['Date'] = pd.to_datetime(pd.to_numeric(df_rates['fundingTime']), unit='ms').dt.date
    df_rates['fundingRate'] = pd.to_numeric(df_rates['fundingRate'])
    
    daily_funding = df_rates.groupby('Date')['fundingRate'].sum().reset_index()
    
    df = pd.merge(df, daily_funding, on='Date', how='left')
    df['fundingRate'] = df['fundingRate'].fillna(0)
    df.set_index('Date', inplace=True)
    return df

def run_simulation(df, vol_filter, use_spot, threshold, take_profit):
    r = 0.0
    FEE_RATE = 0.0010  
    
    active_option = False
    strike = 0
    days_to_expiry = 0
    initial_premium = 0
    hedge_position = 0
    
    hedge_cash = 0
    total_realized_pnl = 0
    total_transaction_costs = 0
    total_funding_fees_paid = 0
    
    pnl_curve = []
    
    dates = df.index
    closes = df['close'].values
    vols = df['Vol_30d'].values
    funding_rates = df['fundingRate'].values
    
    for i in range(len(df)):
        S_t = closes[i]
        sigma_t = vols[i]
        daily_funding_rate = funding_rates[i] if not use_spot else 0.0
        
        if not active_option:
            if sigma_t > vol_filter:
                days_to_expiry = 90
                T = days_to_expiry / 365.0
                strike = find_call_strike(S_t, T, r, sigma_t, 0.1)
                initial_premium = black_scholes_call(S_t, strike, T, r, sigma_t)
                
                target_delta = call_delta(S_t, strike, T, r, sigma_t)
                trade_qty = target_delta
                
                fee = abs(trade_qty) * S_t * FEE_RATE
                total_transaction_costs += fee
                
                hedge_cash = -trade_qty * S_t - fee
                hedge_position = trade_qty
                
                active_option = True
                option_mtm = 0
                daily_total_mtm = total_realized_pnl + option_mtm + (hedge_cash + hedge_position * S_t)
            else:
                daily_total_mtm = total_realized_pnl
        else:
            days_to_expiry -= 1
            T = max(0.00001, days_to_expiry / 365.0)
            
            funding_fee = hedge_position * S_t * daily_funding_rate
            total_funding_fees_paid += funding_fee
            hedge_cash -= funding_fee
            
            current_option_price = black_scholes_call(S_t, strike, T, r, sigma_t)
            current_delta = call_delta(S_t, strike, T, r, sigma_t)
            
            if take_profit > 0 and current_option_price <= initial_premium * take_profit:
                option_mtm = initial_premium - current_option_price
                fee = abs(hedge_position) * S_t * FEE_RATE
                total_transaction_costs += fee
                
                hedge_cash = hedge_cash + hedge_position * S_t - fee
                hedge_position = 0
                total_realized_pnl += (option_mtm + hedge_cash)
                daily_total_mtm = total_realized_pnl
                active_option = False
                
            elif days_to_expiry > 0:
                delta_diff = current_delta - hedge_position
                
                if abs(delta_diff) > threshold or threshold == 0:
                    fee = abs(delta_diff) * S_t * FEE_RATE
                    total_transaction_costs += fee
                    hedge_cash = hedge_cash - delta_diff * S_t - fee
                    hedge_position = current_delta
                
                option_mtm = initial_premium - current_option_price
                daily_total_mtm = total_realized_pnl + option_mtm + (hedge_cash + hedge_position * S_t)
                
            else:
                payoff = max(0, S_t - strike)
                option_mtm = initial_premium - payoff
                fee = abs(hedge_position) * S_t * FEE_RATE
                total_transaction_costs += fee
                
                hedge_cash = hedge_cash + hedge_position * S_t - fee
                hedge_position = 0
                total_realized_pnl += (option_mtm + hedge_cash)
                daily_total_mtm = total_realized_pnl
                active_option = False
                
        pnl_curve.append(daily_total_mtm)

    return pd.Series(pnl_curve, index=df.index), total_transaction_costs, total_funding_fees_paid

def run_ablation_study():
    print("Fetching data from Binance...")
    df = fetch_binance_data()
    df['LogRet'] = np.log(df['close'] / df['close'].shift(1))
    df['Vol_30d'] = df['LogRet'].rolling(window=30).std() * np.sqrt(365)
    df.dropna(inplace=True)
    
    print("Running Baseline (All Rules ON)...")
    base_pnl, base_tx, base_fun = run_simulation(df, vol_filter=0.65, use_spot=True, threshold=0.05, take_profit=0.20)
    
    print("Running Test 1 (NO Vol Filter)...")
    no_vol_pnl, no_vol_tx, no_vol_fun = run_simulation(df, vol_filter=0.0, use_spot=True, threshold=0.05, take_profit=0.20)
    
    print("Running Test 2 (NO Spot Hedge / Pay Funding)...")
    no_spot_pnl, no_spot_tx, no_spot_fun = run_simulation(df, vol_filter=0.65, use_spot=False, threshold=0.05, take_profit=0.20)
    
    print("Running Test 3 (NO Threshold / Daily Rebalance)...")
    no_thresh_pnl, no_thresh_tx, no_thresh_fun = run_simulation(df, vol_filter=0.65, use_spot=True, threshold=0.0, take_profit=0.20)
    
    print("Running Test 4 (NO Take Profit / Hold to Expiry)...")
    no_tp_pnl, no_tp_tx, no_tp_fun = run_simulation(df, vol_filter=0.65, use_spot=True, threshold=0.05, take_profit=0.0)
    
    # Plotting
    plt.figure(figsize=(15, 9))
    plt.plot(df.index, base_pnl, label='Baseline (All Rules ON)', color='gold', linewidth=3)
    plt.plot(df.index, no_vol_pnl, label='T1: NO Vol Filter (Always in Market)', linestyle='--')
    plt.plot(df.index, no_spot_pnl, label='T2: NO Spot Hedge (Pay Funding)', linestyle='--')
    plt.plot(df.index, no_thresh_pnl, label='T3: NO Threshold (Daily Rebalance)', linestyle='--')
    plt.plot(df.index, no_tp_pnl, label='T4: NO Take Profit (Hold to Expiry)', linestyle='--')
    
    plt.title('Ablation Study: Institutional Options Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL (USD)')
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ablation_equity_curve.png')
    plt.savefig(plot_path)
    
    print("\n========== ABLATION STUDY RESULTS ==========")
    print(f"BASELINE: PnL = ${base_pnl.iloc[-1]:,.2f} | Tx Cost = ${base_tx:,.2f} | Funding = ${base_fun:,.2f}")
    
    print(f"\nTEST 1 (NO Vol Filter): PnL = ${no_vol_pnl.iloc[-1]:,.2f} -> Diff vs Base = ${(no_vol_pnl.iloc[-1] - base_pnl.iloc[-1]):,.2f}")
    print(f"TEST 2 (NO Spot Hedge): PnL = ${no_spot_pnl.iloc[-1]:,.2f} -> Diff vs Base = ${(no_spot_pnl.iloc[-1] - base_pnl.iloc[-1]):,.2f}")
    print(f"TEST 3 (NO Threshold): PnL = ${no_thresh_pnl.iloc[-1]:,.2f} -> Diff vs Base = ${(no_thresh_pnl.iloc[-1] - base_pnl.iloc[-1]):,.2f}")
    print(f"TEST 4 (NO Take Profit): PnL = ${no_tp_pnl.iloc[-1]:,.2f} -> Diff vs Base = ${(no_tp_pnl.iloc[-1] - base_pnl.iloc[-1]):,.2f}")

if __name__ == "__main__":
    run_ablation_study()

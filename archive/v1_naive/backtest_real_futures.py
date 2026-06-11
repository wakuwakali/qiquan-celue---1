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
    
    print("Fetching funding rates from Binance...")
    all_rates = []
    since = exchange.parse8601(start_time_str)
    while True:
        try:
            rates = exchange.fapiPublicGetFundingRate({'symbol': 'BTCUSDT', 'startTime': since, 'limit': 1000})
            if not rates: break
            all_rates.extend(rates)
            since = int(rates[-1]['fundingTime']) + 1
            if len(rates) < 1000: break
        except Exception as e:
            print(f"Error fetching funding rates: {e}")
            break
            
    df_rates = pd.DataFrame(all_rates)
    df_rates['Date'] = pd.to_datetime(pd.to_numeric(df_rates['fundingTime']), unit='ms').dt.date
    df_rates['fundingRate'] = pd.to_numeric(df_rates['fundingRate'])
    
    # Aggregate funding rates by day
    daily_funding = df_rates.groupby('Date')['fundingRate'].sum().reset_index()
    
    # Merge klines and funding rates
    df = pd.merge(df, daily_funding, on='Date', how='left')
    df['fundingRate'] = df['fundingRate'].fillna(0) # In case some days missing
    df.set_index('Date', inplace=True)
    
    return df

def run_real_futures_backtest():
    df = fetch_binance_data()
    if df.empty:
        print("Failed to fetch data.")
        return
        
    df['LogRet'] = np.log(df['close'] / df['close'].shift(1))
    df['Vol_30d'] = df['LogRet'].rolling(window=30).std() * np.sqrt(365)
    df.dropna(inplace=True)
    
    r = 0.0
    FEE_RATE = 0.0005  # 0.05%
    
    # State variables
    active_option = False
    strike = 0
    days_to_expiry = 0
    initial_premium = 0
    hedge_position = 0
    
    hedge_cash = 0
    total_realized_pnl = 0
    total_transaction_costs = 0
    total_funding_fees_paid = 0
    
    results = []
    
    dates = df.index
    closes = df['close'].values
    vols = df['Vol_30d'].values
    funding_rates = df['fundingRate'].values
    
    print("Simulating Dynamic Delta Hedging (Real Binance Perpetual Data)...")
    
    for i in range(len(df)):
        date_t = dates[i]
        S_t = closes[i]
        sigma_t = vols[i]
        daily_funding_rate = funding_rates[i]
        
        if not active_option:
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
            days_to_expiry -= 1
            T = max(0.00001, days_to_expiry / 365.0)
            
            # Real funding fee deduction based on CURRENT held position
            funding_fee = hedge_position * S_t * daily_funding_rate
            total_funding_fees_paid += funding_fee
            hedge_cash -= funding_fee
            
            if days_to_expiry > 0:
                # Daily Rebalance
                current_option_price = black_scholes_call(S_t, strike, T, r, sigma_t)
                current_delta = call_delta(S_t, strike, T, r, sigma_t)
                
                delta_diff = current_delta - hedge_position
                fee = abs(delta_diff) * S_t * FEE_RATE
                total_transaction_costs += fee
                
                hedge_cash = hedge_cash - delta_diff * S_t - fee
                hedge_position = current_delta
                
                option_mtm = initial_premium - current_option_price
                daily_total_mtm = total_realized_pnl + option_mtm + (hedge_cash + hedge_position * S_t)
                
            else:
                # Expiry Day Settlement
                payoff = max(0, S_t - strike)
                option_mtm = initial_premium - payoff
                
                # Unwind hedge
                fee = abs(hedge_position) * S_t * FEE_RATE
                total_transaction_costs += fee
                
                hedge_cash = hedge_cash + hedge_position * S_t - fee
                hedge_position = 0
                
                total_realized_pnl += (option_mtm + hedge_cash)
                daily_total_mtm = total_realized_pnl
                
                active_option = False
                
        results.append({
            'Date': date_t,
            'Cumulative_PnL': daily_total_mtm
        })

    res_df = pd.DataFrame(results).set_index('Date')
    
    # Naked call comparison using real futures data
    naked_pnl = 0
    naked_pnl_history = []
    n_active = False
    n_strike = 0
    n_premium = 0
    n_days = 0
    
    for i in range(len(df)):
        S_t = closes[i]
        sigma_t = vols[i]
        if not n_active:
            n_days = 90
            T = n_days / 365.0
            n_strike = find_call_strike(S_t, T, r, sigma_t, 0.1)
            n_premium = black_scholes_call(S_t, n_strike, T, r, sigma_t)
            n_active = True
            naked_pnl_history.append(naked_pnl)
        else:
            n_days -= 1
            if n_days > 0:
                T = max(0.00001, n_days / 365.0)
                curr_price = black_scholes_call(S_t, n_strike, T, r, sigma_t)
                naked_pnl_history.append(naked_pnl + (n_premium - curr_price))
            else:
                payoff = max(0, S_t - n_strike)
                naked_pnl += (n_premium - payoff)
                naked_pnl_history.append(naked_pnl)
                n_active = False

    res_df['Naked_90d_Call_PnL'] = naked_pnl_history
    
    plt.figure(figsize=(14, 8))
    plt.plot(res_df.index, res_df['Naked_90d_Call_PnL'], label='Naked 90-day 0.1 Delta Call (No Hedge)', color='red', linestyle='--')
    plt.plot(res_df.index, res_df['Cumulative_PnL'], label='Real Binance Futures Hedged (Fees + Real Funding Rates)', color='darkblue', linewidth=2)
    
    plt.title('Dynamic Delta Hedging: 100% Real Binance Perpetual Data')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL (USD)')
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real_futures_equity_curve.png')
    plt.savefig(plot_path)
    print(f"Saved real futures equity curve plot to: {plot_path}")
    
    print("\n--- Final Cumulative PnL ---")
    print(f"Naked 90-day Call: ${res_df['Naked_90d_Call_PnL'].iloc[-1]:,.2f}")
    print(f"Real Binance Futures Hedged: ${res_df['Cumulative_PnL'].iloc[-1]:,.2f}")
    
    print("\n--- Real Hedging Cost Breakdown ---")
    print(f"Total Transaction Costs (0.05% Fees): ${total_transaction_costs:,.2f}")
    print(f"Total REAL Binance Funding Fees Paid: ${total_funding_fees_paid:,.2f}")
    print(f"Total Friction Costs: ${(total_transaction_costs + total_funding_fees_paid):,.2f}")

if __name__ == "__main__":
    run_real_futures_backtest()

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

def run_delta_hedge_backtest():
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
    
    r = 0.0
    # Update: 0.05% fee + 0.05% slippage = 0.1% transaction cost
    FEE_RATE = 0.0010  
    
    # State variables
    active_option = False
    strike = 0
    days_to_expiry = 0
    initial_premium = 0
    hedge_position = 0
    
    # Tracking With Realistic Fees & Funding Rate
    hedge_cash_real = 0
    total_realized_pnl_real = 0
    total_transaction_costs = 0
    total_funding_fees_paid = 0
    
    results = []
    
    dates = df.index
    closes = df['Close'].values
    vols = df['Vol_30d'].values
    sma = df['SMA20'].values
    
    print("Simulating Dynamic Delta Hedging (With Funding Rates & Slippage)...")
    
    for i in range(len(df)):
        date_t = dates[i]
        S_t = closes[i]
        sigma_t = vols[i]
        sma_t = sma[i]
        
        # Calculate daily funding rate based on trend
        # If bull market (price > SMA20), longs pay 0.03% daily.
        # If bear market (price <= SMA20), shorts pay 0.01% daily (longs receive 0.01%).
        daily_funding_rate = 0.0003 if S_t > sma_t else -0.0001
        
        # 1. Check if we need to open a new 90-day trade
        if not active_option:
            days_to_expiry = 90
            T = days_to_expiry / 365.0
            strike = find_call_strike(S_t, T, r, sigma_t, 0.1)
            initial_premium = black_scholes_call(S_t, strike, T, r, sigma_t)
            
            target_delta = call_delta(S_t, strike, T, r, sigma_t)
            trade_qty = target_delta
            
            fee = abs(trade_qty) * S_t * FEE_RATE
            total_transaction_costs += fee
            
            hedge_cash_real = -trade_qty * S_t - fee
            hedge_position = trade_qty
            
            active_option = True
            option_mtm = 0
            daily_total_mtm_real = total_realized_pnl_real + option_mtm + (hedge_cash_real + hedge_position * S_t)
            
        else:
            days_to_expiry -= 1
            T = max(0.00001, days_to_expiry / 365.0)
            
            # Daily funding fee deduction based on CURRENT held position
            funding_fee = hedge_position * S_t * daily_funding_rate
            total_funding_fees_paid += funding_fee
            hedge_cash_real -= funding_fee
            
            if days_to_expiry > 0:
                # 2. Daily Rebalance
                current_option_price = black_scholes_call(S_t, strike, T, r, sigma_t)
                current_delta = call_delta(S_t, strike, T, r, sigma_t)
                
                delta_diff = current_delta - hedge_position
                fee = abs(delta_diff) * S_t * FEE_RATE
                total_transaction_costs += fee
                
                hedge_cash_real = hedge_cash_real - delta_diff * S_t - fee
                hedge_position = current_delta
                
                option_mtm = initial_premium - current_option_price
                daily_total_mtm_real = total_realized_pnl_real + option_mtm + (hedge_cash_real + hedge_position * S_t)
                
            else:
                # 3. Expiry Day Settlement
                payoff = max(0, S_t - strike)
                option_mtm = initial_premium - payoff
                
                # Unwind hedge
                fee = abs(hedge_position) * S_t * FEE_RATE
                total_transaction_costs += fee
                
                hedge_cash_real = hedge_cash_real + hedge_position * S_t - fee
                hedge_position = 0
                
                total_realized_pnl_real += (option_mtm + hedge_cash_real)
                daily_total_mtm_real = total_realized_pnl_real
                
                # Reset for next trade
                active_option = False
                
        results.append({
            'Date': date_t,
            'Cumulative_PnL_Real': daily_total_mtm_real
        })

    res_df = pd.DataFrame(results).set_index('Date')
    
    # Naked call comparison
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
    plt.plot(res_df.index, res_df['Cumulative_PnL_Real'], label='Realistic Delta Hedged (Fees + Slippage + Funding Rates)', color='purple', linewidth=2)
    
    plt.title('Realistic Dynamic Delta Hedging in Crypto Markets')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL (USD)')
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delta_hedge_equity_curve_real.png')
    plt.savefig(plot_path)
    print(f"Saved realistic delta hedge equity curve plot to: {plot_path}")
    
    print("\n--- Final Cumulative PnL ---")
    print(f"Naked 90-day Call: ${res_df['Naked_90d_Call_PnL'].iloc[-1]:,.2f}")
    print(f"Realistic Hedged:  ${res_df['Cumulative_PnL_Real'].iloc[-1]:,.2f}")
    
    print("\n--- Hedging Cost Breakdown ---")
    print(f"Total Transaction Costs (Fees + Slippage): ${total_transaction_costs:,.2f}")
    print(f"Total Funding Fees Paid (Longs pay Shorts): ${total_funding_fees_paid:,.2f}")
    print(f"Total Friction Costs: ${(total_transaction_costs + total_funding_fees_paid):,.2f}")

if __name__ == "__main__":
    run_delta_hedge_backtest()

import ccxt
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
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

def find_call_strike_exact(S, T, r, sigma, target_delta):
    def f(K):
        return call_delta(S, K, T, r, sigma) - target_delta
    try:
        return brentq(f, S * 0.5, S * 5.0)
    except ValueError:
        return S * 1.5 

def find_put_strike_exact(S, T, r, sigma, target_delta):
    def f(K):
        return put_delta(S, K, T, r, sigma) - target_delta
    try:
        return brentq(f, S * 0.1, S * 1.5)
    except ValueError:
        return S * 0.5 

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

def run_iron_condor(df):
    r = 0.0
    OPT_FEE_RATE = 0.0003
    OPT_SLIPPAGE = 0.02
    
    INITIAL_EQUITY = 100000.0
    equity = INITIAL_EQUITY
    
    tenor = 90
    vol_filter = 0.65
    take_profit = 0.50 # Take profit if condor value drops 50%
    
    active_ic = False
    qty = 0.0
    days_left = 0
    
    # Strikes
    k_call_short = 0
    k_call_long = 0
    k_put_short = 0
    k_put_long = 0
    
    # Entry prices
    p_call_short_open = 0
    p_call_long_open = 0
    p_put_short_open = 0
    p_put_long_open = 0
    
    net_premium_received = 0.0
    
    pnl_curve = []
    pnl_curve.append(equity)
    
    opens = df['open'].values
    closes = df['close'].values
    vols = df['Vol_30d'].values
    
    for i in range(1, len(df)):
        sigma_prev = vols[i-1]
        if np.isnan(sigma_prev):
            pnl_curve.append(equity)
            continue
            
        S_exec = opens[i]
        S_mtm = closes[i]
        
        IV_open = sigma_prev * 1.05
        IV_mtm = sigma_prev * 1.05
        
        if active_ic:
            days_left -= 1
            T_exec = max(0.00001, days_left / 365.0)
            
            # Current value of the 4 legs (Theoretical)
            c_call_short = black_scholes_call(S_exec, k_call_short, T_exec, r, IV_mtm)
            c_call_long = black_scholes_call(S_exec, k_call_long, T_exec, r, IV_mtm)
            c_put_short = black_scholes_put(S_exec, k_put_short, T_exec, r, IV_mtm)
            c_put_long = black_scholes_put(S_exec, k_put_long, T_exec, r, IV_mtm)
            
            # To close the short legs, we buy them back (pay 102%)
            close_short_cost = (c_call_short + c_put_short) * (1 + OPT_SLIPPAGE)
            # To close the long legs, we sell them (receive 98%)
            close_long_revenue = (c_call_long + c_put_long) * (1 - OPT_SLIPPAGE)
            
            current_condor_liability = close_short_cost - close_long_revenue
            
            # Take Profit: if the cost to buy back the condor is much less than what we received
            if current_condor_liability <= (net_premium_received / qty) * (1 - take_profit) and days_left > 0:
                opt_fee = 4 * S_exec * OPT_FEE_RATE * qty
                equity += net_premium_received - (current_condor_liability * qty) - opt_fee
                active_ic = False
                qty = 0
                
            elif days_left == 0:
                # Expiry Payoff
                payoff_call_short = max(0, S_exec - k_call_short)
                payoff_call_long = max(0, S_exec - k_call_long)
                payoff_put_short = max(0, k_put_short - S_exec)
                payoff_put_long = max(0, k_put_long - S_exec)
                
                net_payoff = (payoff_call_short + payoff_put_short) - (payoff_call_long + payoff_put_long)
                opt_fee = 4 * S_exec * OPT_FEE_RATE * qty
                
                equity += net_premium_received - (net_payoff * qty) - opt_fee
                active_ic = False
                qty = 0

        if not active_ic:
            if sigma_prev > vol_filter:
                days_left = tenor
                T = days_left / 365.0
                
                k_call_short = find_call_strike_exact(S_exec, T, r, IV_open, 0.30)
                k_call_long = find_call_strike_exact(S_exec, T, r, IV_open, 0.10)
                k_put_short = find_put_strike_exact(S_exec, T, r, IV_open, -0.30)
                k_put_long = find_put_strike_exact(S_exec, T, r, IV_open, -0.10)
                
                bs_call_s = black_scholes_call(S_exec, k_call_short, T, r, IV_open)
                bs_call_l = black_scholes_call(S_exec, k_call_long, T, r, IV_open)
                bs_put_s = black_scholes_put(S_exec, k_put_short, T, r, IV_open)
                bs_put_l = black_scholes_put(S_exec, k_put_long, T, r, IV_open)
                
                rev_call_s = bs_call_s * (1 - OPT_SLIPPAGE)
                cost_call_l = bs_call_l * (1 + OPT_SLIPPAGE)
                rev_put_s = bs_put_s * (1 - OPT_SLIPPAGE)
                cost_put_l = bs_put_l * (1 + OPT_SLIPPAGE)
                
                net_unit_premium = (rev_call_s + rev_put_s) - (cost_call_l + cost_put_l)
                
                # If we don't even collect a net premium due to massive spread, skip
                if net_unit_premium > 0:
                    max_call_risk = (k_call_long - k_call_short)
                    max_put_risk = (k_put_short - k_put_long)
                    max_loss_per_unit = max(max_call_risk, max_put_risk) - net_unit_premium
                    
                    if max_loss_per_unit > 0:
                        # Risk 20% of Equity
                        target_risk = equity * 0.20
                        qty = target_risk / max_loss_per_unit
                        
                        net_premium_received = net_unit_premium * qty
                        opt_fee = 4 * S_exec * OPT_FEE_RATE * qty
                        
                        equity -= opt_fee
                        active_ic = True

        # MTM Phase
        mtm_equity = equity
        if active_ic:
            T_mtm = max(0.00001, days_left / 365.0)
            c_cs = black_scholes_call(S_mtm, k_call_short, T_mtm, r, IV_mtm)
            c_cl = black_scholes_call(S_mtm, k_call_long, T_mtm, r, IV_mtm)
            c_ps = black_scholes_put(S_mtm, k_put_short, T_mtm, r, IV_mtm)
            c_pl = black_scholes_put(S_mtm, k_put_long, T_mtm, r, IV_mtm)
            
            mtm_liability = (c_cs + c_ps) - (c_cl + c_pl)
            opt_mtm = net_premium_received - (mtm_liability * qty)
            mtm_equity += opt_mtm
            
        pnl_curve.append(mtm_equity)

    pnl_series = pd.Series(pnl_curve, index=df.index)
    peak = pnl_series.cummax()
    drawdown = (peak - pnl_series) / peak
    max_drawdown = drawdown.max()
    
    years = len(df) / 365.25
    total_return = pnl_series.iloc[-1] / INITIAL_EQUITY - 1
    annualized_return = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1
    
    return pnl_series, {
        'final_equity': pnl_series.iloc[-1],
        'ann_return': annualized_return,
        'max_dd': max_drawdown
    }

def main():
    print("Fetching data from Binance...")
    df = fetch_binance_data()
    df['LogRet'] = np.log(df['close'] / df['close'].shift(1))
    df['Vol_30d'] = df['LogRet'].rolling(window=30).std() * np.sqrt(365)
    
    print("Running V3 Iron Condor Simulation...")
    pnl_v3, stats = run_iron_condor(df)
    
    plt.figure(figsize=(15, 9))
    plt.plot(df.index, pnl_v3, label='V3 Iron Condor (Strict Causal + Slippage + No Liquidation Risk)', color='teal', linewidth=3)
    
    plt.title('V3 Iron Condor: The Defined-Risk Crypto Option Strategy')
    plt.xlabel('Date')
    plt.ylabel('Equity (USD)')
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v3_equity_curve.png')
    plt.savefig(plot_path)
    
    print("\n--- V3 IRON CONDOR METRICS ---")
    print(f"Final Equity: ${stats['final_equity']:,.2f} (from $100k)")
    print(f"Annualized Return: {stats['ann_return']:.2%}")
    print(f"Max Drawdown: {stats['max_dd']:.2%}")

if __name__ == "__main__":
    main()

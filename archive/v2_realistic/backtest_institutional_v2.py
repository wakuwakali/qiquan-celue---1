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

def find_call_strike_exact(S, T, r, sigma, target_delta):
    def f(K):
        return call_delta(S, K, T, r, sigma) - target_delta
    try:
        K_root = brentq(f, S * 0.5, S * 5.0)
        return K_root
    except ValueError:
        return S * 1.5 

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

def run_v2_grail(df):
    r = 0.0
    SPOT_FEE_RATE = 0.0010
    OPT_FEE_RATE = 0.0003
    OPT_SLIPPAGE = 0.02
    
    INITIAL_EQUITY = 100000.0
    equity = INITIAL_EQUITY
    
    tenor_short = 90
    delta_short = 0.3
    vol_filter_short = 0.65
    threshold = 0.05
    take_profit = 0.20
    
    vol_filter_long = 0.40
    tenor_long = 30
    
    active_short = False
    qty_short = 0.0
    strike_short = 0
    days_short = 0
    premium_short_collected = 0.0
    hedge_pos = 0.0
    
    active_long = False
    qty_long = 0.0
    strike_long = 0
    days_long = 0
    premium_long_paid = 0.0
    
    pnl_curve = []
    pnl_curve.append(equity)
    
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    vols = df['Vol_30d'].values
    
    liquidations = 0
    
    for i in range(1, len(df)):
        sigma_prev = vols[i-1]
        if np.isnan(sigma_prev):
            pnl_curve.append(equity)
            continue
            
        S_exec = opens[i]
        S_high = highs[i]
        S_low = lows[i]
        S_mtm = closes[i]
        
        # Heuristic IV Modeling
        # For opening short, conservatively assume IV = HV
        IV_sell_open = sigma_prev
        # For MTM of short, assume IV remains somewhat high
        IV_sell_mtm = sigma_prev * 1.05
        
        # For opening long, we pay VRP premium
        IV_buy_open = max(sigma_prev * 1.15, 0.40)
        # For MTM of long, we assume IV crushes back to HV
        IV_buy_mtm = sigma_prev * 0.90
        
        # --- 1. Manage Short Position ---
        if active_short:
            # Intraday Liquidation Check (using High of the day, worst case for Short Call)
            T_exec = max(0.00001, days_short / 365.0)
            intraday_option_price = black_scholes_call(S_high, strike_short, T_exec, r, IV_sell_mtm)
            intraday_opt_loss = (intraday_option_price - (premium_short_collected / qty_short)) * qty_short
            
            # Hedge PnL at high
            intraday_hedge_pnl = hedge_pos * (S_high - S_exec)  # Approximation from day open
            intraday_equity = equity - intraday_opt_loss + intraday_hedge_pnl
            
            # Deribit Maintenance Margin for Short Call (approx 7.5%)
            MM_req = (0.075 * S_high + intraday_option_price) * qty_short
            
            if intraday_equity < MM_req:
                # LIQUIDATED
                liquidations += 1
                close_cost = intraday_option_price * qty_short * (1 + OPT_SLIPPAGE)
                opt_fee = S_high * OPT_FEE_RATE * qty_short
                liq_penalty = 0.005 * S_high * qty_short # 0.5% liquidation fee
                
                spot_fee = abs(hedge_pos) * S_high * SPOT_FEE_RATE
                
                # Settle equity
                equity += premium_short_collected - close_cost - opt_fee - liq_penalty
                equity += hedge_pos * S_high - spot_fee # Unwind hedge
                
                active_short = False
                qty_short = 0
                hedge_pos = 0
            else:
                # Normal Execution at Open
                days_short -= 1
                curr_bs_price = black_scholes_call(S_exec, strike_short, T_exec, r, IV_sell_mtm)
                
                if curr_bs_price * (1 + OPT_SLIPPAGE) <= (premium_short_collected / qty_short) * take_profit:
                    close_cost = curr_bs_price * qty_short * (1 + OPT_SLIPPAGE)
                    opt_fee = S_exec * OPT_FEE_RATE * qty_short
                    
                    spot_fee = abs(hedge_pos) * S_exec * SPOT_FEE_RATE
                    
                    equity += premium_short_collected - close_cost - opt_fee
                    equity += hedge_pos * S_exec - spot_fee
                    
                    active_short = False
                    qty_short = 0
                    hedge_pos = 0
                elif days_short > 0:
                    curr_delta = call_delta(S_exec, strike_short, T_exec, r, IV_sell_mtm)
                    target_hedge_pos = curr_delta * qty_short
                    delta_diff = target_hedge_pos - hedge_pos
                    if abs(delta_diff) > threshold * qty_short:
                        spot_fee = abs(delta_diff) * S_exec * SPOT_FEE_RATE
                        equity -= delta_diff * S_exec + spot_fee
                        hedge_pos = target_hedge_pos
                else:
                    # Expiry
                    payoff = max(0, S_exec - strike_short) * qty_short
                    opt_fee = S_exec * OPT_FEE_RATE * qty_short
                    
                    spot_fee = abs(hedge_pos) * S_exec * SPOT_FEE_RATE
                    
                    equity += premium_short_collected - payoff - opt_fee
                    equity += hedge_pos * S_exec - spot_fee
                    
                    active_short = False
                    qty_short = 0
                    hedge_pos = 0

        # --- 2. Manage Long Straddle Position ---
        elif active_long:
            days_long -= 1
            T_exec = max(0.00001, days_long / 365.0)
            
            if days_long == 0:
                payoff = (max(0, S_exec - strike_long) + max(0, strike_long - S_exec)) * qty_long
                opt_fee = S_exec * OPT_FEE_RATE * 2 * qty_long
                equity += (payoff - opt_fee)
                active_long = False
                qty_long = 0

        # --- 3. Open New Positions ---
        if not active_short and not active_long:
            if sigma_prev > vol_filter_short:
                # Open Short
                days_short = tenor_short
                T = days_short / 365.0
                strike_short = find_call_strike_exact(S_exec, T, r, IV_sell_open, delta_short)
                bs_premium = black_scholes_call(S_exec, strike_short, T, r, IV_sell_open)
                
                # Sizing: Target Margin = 30% of Equity. IM approx 15% of Spot.
                target_margin = equity * 0.30
                qty_short = target_margin / (0.15 * S_exec)
                
                premium_short_collected = bs_premium * qty_short * (1 - OPT_SLIPPAGE)
                opt_fee = S_exec * OPT_FEE_RATE * qty_short
                equity -= opt_fee  # Pay fees immediately
                
                # Initial Hedge
                target_delta = call_delta(S_exec, strike_short, T, r, IV_sell_open)
                hedge_pos = target_delta * qty_short
                spot_fee = abs(hedge_pos) * S_exec * SPOT_FEE_RATE
                equity -= hedge_pos * S_exec + spot_fee
                
                active_short = True
                
            elif sigma_prev < vol_filter_long:
                # Open Long Straddle
                days_long = tenor_long
                T = days_long / 365.0
                strike_long = S_exec
                call_bs = black_scholes_call(S_exec, strike_long, T, r, IV_buy_open)
                put_bs = black_scholes_put(S_exec, strike_long, T, r, IV_buy_open)
                
                total_bs = call_bs + put_bs
                
                # Sizing: Risk 5% of Equity on premium
                target_premium = equity * 0.05
                qty_long = target_premium / (total_bs * (1 + OPT_SLIPPAGE))
                
                premium_long_paid = total_bs * qty_long * (1 + OPT_SLIPPAGE)
                opt_fee = S_exec * OPT_FEE_RATE * 2 * qty_long
                
                equity -= (premium_long_paid + opt_fee)
                active_long = True

        # --- MTM PHASE ---
        mtm_equity = equity
        if active_short:
            T_mtm = max(0.00001, days_short / 365.0)
            curr_mtm_price = black_scholes_call(S_mtm, strike_short, T_mtm, r, IV_sell_mtm)
            # Add premium back, subtract current liability
            opt_mtm = premium_short_collected - (curr_mtm_price * qty_short)
            mtm_equity += opt_mtm + (hedge_pos * S_mtm)
            
        elif active_long:
            T_mtm = max(0.00001, days_long / 365.0)
            curr_c = black_scholes_call(S_mtm, strike_long, T_mtm, r, IV_buy_mtm)
            curr_p = black_scholes_put(S_mtm, strike_long, T_mtm, r, IV_buy_mtm)
            mtm_equity += (curr_c + curr_p) * qty_long
            
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
        'max_dd': max_drawdown,
        'liquidations': liquidations
    }

def main():
    print("Fetching data from Binance...")
    df = fetch_binance_data()
    df['LogRet'] = np.log(df['close'] / df['close'].shift(1))
    df['Vol_30d'] = df['LogRet'].rolling(window=30).std() * np.sqrt(365)
    
    print("Running V2 Institutional Execution Simulation...")
    pnl_v2, stats = run_v2_grail(df)
    
    plt.figure(figsize=(15, 9))
    plt.plot(df.index, pnl_v2, label='V2 Holy Grail (Strict Logic + Margin Engine)', color='darkred', linewidth=3)
    
    plt.title('V2 Post-Audit Equity Curve: $100,000 Initial Capital')
    plt.xlabel('Date')
    plt.ylabel('Equity (USD)')
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v2_equity_curve.png')
    plt.savefig(plot_path)
    
    print("\n--- V2 POST-AUDIT METRICS ---")
    print(f"Final Equity: ${stats['final_equity']:,.2f} (from $100k)")
    print(f"Annualized Return: {stats['ann_return']:.2%}")
    print(f"Max Drawdown: {stats['max_dd']:.2%}")
    print(f"Liquidations Hit: {stats['liquidations']}")

if __name__ == "__main__":
    main()

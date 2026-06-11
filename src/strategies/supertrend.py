import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vectorized_engine import VectorizedBacktester

def calculate_supertrend(df, period=10, multiplier=3.0):
    """
    Calculates the Supertrend indicator.
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros(len(df))
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(df)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
    hl2 = (high + low) / 2
    basic_ub = hl2 + (multiplier * atr)
    basic_lb = hl2 - (multiplier * atr)
    
    final_ub = np.zeros(len(df))
    final_lb = np.zeros(len(df))
    supertrend = np.zeros(len(df))
    trend_dir = np.zeros(len(df)) # 1 for up, -1 for down
    
    for i in range(1, len(df)):
        # Final Upper Band
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        # Final Lower Band
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
            
        # Supertrend
        if supertrend[i-1] == final_ub[i-1]:
            if close[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
                trend_dir[i] = -1
            else:
                supertrend[i] = final_lb[i]
                trend_dir[i] = 1
        elif supertrend[i-1] == final_lb[i-1]:
            if close[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
                trend_dir[i] = 1
            else:
                supertrend[i] = final_ub[i]
                trend_dir[i] = -1
        else:
            # Initialization
            supertrend[i] = final_ub[i]
            trend_dir[i] = -1
            
    df['supertrend'] = supertrend
    df['trend_dir'] = trend_dir
    return df

def supertrend_strategy(df_1h, period=10, multiplier=3.0, ema_period=200, atr_window=14, atr_sma_window=50):
    """
    Supertrend Strategy with Long-term EMA filter and ATR Regime Filter.
    """
    df_1h = calculate_supertrend(df_1h, period, multiplier)
    
    # 1. Macro Trend Filter
    df_1h['ema_macro'] = df_1h['close'].ewm(span=ema_period, adjust=False).mean()
    
    # 2. Volatility Regime Filter (Same as Dual Thrust)
    high_low = df_1h['high'] - df_1h['low']
    high_close = np.abs(df_1h['high'] - df_1h['close'].shift())
    low_close = np.abs(df_1h['low'] - df_1h['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    atr = tr.rolling(window=atr_window).mean()
    atr_sma = atr.rolling(window=atr_sma_window).mean()
    regime_valid = atr > atr_sma  # Only trade when volatility is expanding
    
    # 3. Generate Signals
    target_pos = np.zeros(len(df_1h))
    
    # Long Condition: Supertrend is bullish (1) AND price above EMA200 AND Volatility is expanding
    long_cond = (df_1h['trend_dir'] == 1) & (df_1h['close'] > df_1h['ema_macro']) & regime_valid
    
    # Short Condition: Supertrend is bearish (-1) AND price below EMA200 AND Volatility is expanding
    short_cond = (df_1h['trend_dir'] == -1) & (df_1h['close'] < df_1h['ema_macro']) & regime_valid
    
    # Exit conditions: If Supertrend flips against our position, we exit to 0
    # Or if macro trend breaks. To keep it simple, we exit if Supertrend flips.
    exit_long = (df_1h['trend_dir'] == -1)
    exit_short = (df_1h['trend_dir'] == 1)
    
    # State Machine
    current_pos = 0
    
    # We use a loop for state tracking
    long_cond_arr = long_cond.values
    short_cond_arr = short_cond.values
    exit_long_arr = exit_long.values
    exit_short_arr = exit_short.values
    
    for i in range(len(df_1h)):
        if current_pos == 0:
            if long_cond_arr[i]:
                current_pos = 1
            elif short_cond_arr[i]:
                current_pos = -1
        elif current_pos == 1:
            if exit_long_arr[i]:
                current_pos = 0
        elif current_pos == -1:
            if exit_short_arr[i]:
                current_pos = 0
                
        target_pos[i] = current_pos
        
    df_1h['target_position'] = target_pos
    return df_1h['target_position']

if __name__ == '__main__':
    print("Testing Filtered Supertrend Strategy...")
    tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z', fee_rate=0.0004, slippage=0.001)
    
    # Test different Multipliers
    best_df = None
    for mult in [3.0, 4.0, 5.0]:
        df, stats = tester.run_strategy(supertrend_strategy, period=10, multiplier=mult)
        print(f"\n--- Supertrend Multiplier = {mult} ---")
        for key, v in stats.items():
            print(f"{key}: {v}")
        if mult == 4.0:
            best_df = df
            
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(best_df.index, best_df['equity_curve'], label='Supertrend (P=10, M=4.0)', color='red', linewidth=2)
    plt.title('Supertrend Strategy - BTC/USDT')
    plt.ylabel('Cumulative Return (1.0 = Initial Capital)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'plots', 'supertrend_equity_curve.png'))

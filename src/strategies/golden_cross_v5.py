import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vectorized_engine import VectorizedBacktester

def golden_cross_v5_strategy(df_1h, fast_h=1200, slow_h=4800, signal_h=480, confirm_hours=48):
    """
    Golden Cross V5 (Holy Grail - Anti-Whipsaw Edition)
    Uses Time-Confirmation (Hysteresis) to eliminate small losses.
    Requires the Histogram to stay positive/negative for X consecutive hours before acting.
    """
    
    fast_ema = df_1h['close'].ewm(span=fast_h, adjust=False).mean()
    slow_ema = df_1h['close'].ewm(span=slow_h, adjust=False).mean()
    
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal_h, adjust=False).mean()
    histogram = macd - signal_line
    
    # Raw Conditions
    raw_long_cond = (macd > 0) & (histogram > 0)
    raw_exit_cond = (macd < 0) | (histogram < 0)
    
    # Time-Confirmation (Rolling Min/Max)
    # If raw_long_cond is True for 48 consecutive hours, rolling_min(48) will be 1
    long_cond = raw_long_cond.rolling(window=confirm_hours).min() == 1
    exit_long = raw_exit_cond.rolling(window=confirm_hours).min() == 1
    
    # State Machine
    target_pos = np.zeros(len(df_1h))
    current_pos = 0
    
    long_cond_arr = long_cond.fillna(False).values
    exit_long_arr = exit_long.fillna(False).values
    
    for i in range(len(df_1h)):
        if current_pos == 0:
            if long_cond_arr[i]:
                current_pos = 1
        elif current_pos == 1:
            if exit_long_arr[i]:
                current_pos = 0
                
        target_pos[i] = current_pos
        
    df_1h['target_position'] = target_pos
    return df_1h['target_position']

if __name__ == '__main__':
    print("Testing Golden Cross V5 (Anti-Whipsaw Time Confirmation)...")
    tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z', fee_rate=0.0004, slippage=0.001)
    
    df, stats = tester.run_strategy(golden_cross_v5_strategy, confirm_hours=48)
    print(f"\n--- Golden Cross V5 ---")
    for key, v in stats.items():
        print(f"{key}: {v}")
        
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['equity_curve'], label='Golden Cross V5 (Anti-Whipsaw)', color='gold', linewidth=2)
    plt.title('Golden Cross V5 Strategy - BTC/USDT')
    plt.ylabel('Cumulative Return (1.0 = Initial Capital)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'plots', 'golden_cross_v5_equity_curve.png'))

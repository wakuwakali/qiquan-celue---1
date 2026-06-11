import sys, os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vectorized_engine import VectorizedBacktester

def golden_cross_v9_fast_strategy(df_1h, fast_h=480, slow_h=1200, signal_h=216):
    """
    Golden Cross V9 (Fast Entry)
    Uses 20D/50D MACD instead of 50D/200D to catch the bottom earlier,
    relying on the V7 Profit Lock and Histogram exits to manage the increased noise.
    """
    
    fast_ema = df_1h['close'].ewm(span=fast_h, adjust=False).mean()
    slow_ema = df_1h['close'].ewm(span=slow_h, adjust=False).mean()
    
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal_h, adjust=False).mean()
    histogram = macd - signal_line
    
    long_cond = (macd > 0) & (histogram > 0)
    exit_primary = (macd < 0) | (histogram < 0)
    
    target_pos = np.zeros(len(df_1h))
    current_pos = 0
    highest_high = 0
    entry_price = 0
    
    long_cond_arr = long_cond.values
    exit_primary_arr = exit_primary.values
    close_arr = df_1h['close'].values
    high_arr = df_1h['high'].values
    
    for i in range(len(df_1h)):
        if current_pos == 0:
            if long_cond_arr[i]:
                current_pos = 1
                entry_price = close_arr[i]
                highest_high = high_arr[i]
        elif current_pos == 1:
            if high_arr[i] > highest_high:
                highest_high = high_arr[i]
                
            profit_lock = False
            if (highest_high / entry_price) > 1.20:
                trailing_stop = highest_high * 0.88
                if close_arr[i] < trailing_stop:
                    profit_lock = True
                    
            if exit_primary_arr[i] or profit_lock:
                current_pos = 0
                
        target_pos[i] = current_pos
        
    df_1h['target_position'] = target_pos
    return df_1h['target_position']

if __name__ == '__main__':
    tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z', fee_rate=0.0004, slippage=0.001)
    df, stats = tester.run_strategy(golden_cross_v9_fast_strategy)
    
    print("\n--- Golden Cross V9 (Fast Entry 20D/50D) ---")
    for key, v in stats.items():
        print(f"{key}: {v}")

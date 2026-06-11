import sys, os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vectorized_engine import VectorizedBacktester

def golden_cross_v8_ls_strategy(df_1h, fast_h=1200, slow_h=4800, signal_h=216):
    """
    Golden Cross V8 (Long-Short System)
    Attempts to capture downside alpha by shorting during macro bear markets.
    """
    
    fast_ema = df_1h['close'].ewm(span=fast_h, adjust=False).mean()
    slow_ema = df_1h['close'].ewm(span=slow_h, adjust=False).mean()
    
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal_h, adjust=False).mean()
    histogram = macd - signal_line
    
    # Long Conditions
    long_cond = (macd > 0) & (histogram > 0)
    exit_long = (macd < 0) | (histogram < 0)
    
    # Short Conditions
    short_cond = (macd < 0) & (histogram < 0)
    exit_short = (macd > 0) | (histogram > 0)
    
    target_pos = np.zeros(len(df_1h))
    current_pos = 0
    
    long_cond_arr = long_cond.values
    exit_long_arr = exit_long.values
    short_cond_arr = short_cond.values
    exit_short_arr = exit_short.values
    
    # Profit Lock vars
    highest_high = 0
    lowest_low = float('inf')
    entry_price = 0
    close_arr = df_1h['close'].values
    high_arr = df_1h['high'].values
    low_arr = df_1h['low'].values
    
    for i in range(len(df_1h)):
        if current_pos == 0:
            if long_cond_arr[i]:
                current_pos = 1
                entry_price = close_arr[i]
                highest_high = high_arr[i]
            elif short_cond_arr[i]:
                current_pos = -1
                entry_price = close_arr[i]
                lowest_low = low_arr[i]
                
        elif current_pos == 1:
            if high_arr[i] > highest_high:
                highest_high = high_arr[i]
                
            profit_lock = False
            if (highest_high / entry_price) > 1.20:
                trailing_stop = highest_high * 0.88
                if close_arr[i] < trailing_stop:
                    profit_lock = True
                    
            if exit_long_arr[i] or profit_lock:
                current_pos = 0
                
        elif current_pos == -1:
            if low_arr[i] < lowest_low:
                lowest_low = low_arr[i]
                
            profit_lock = False
            # Short profit calculation
            if (entry_price / lowest_low) > 1.20:
                # If we dropped 20%, activate trailing stop 12% above lowest low
                trailing_stop = lowest_low * 1.12
                if close_arr[i] > trailing_stop:
                    profit_lock = True
                    
            if exit_short_arr[i] or profit_lock:
                current_pos = 0
                
        target_pos[i] = current_pos
        
    df_1h['target_position'] = target_pos
    return df_1h['target_position']

if __name__ == '__main__':
    # Using 0.0004 fee and 0.001 slippage. 
    # For a real short system, we should technically add funding rate costs,
    # but the engine handles this via fixed slippage/fees per trade for now.
    tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z', fee_rate=0.0004, slippage=0.001)
    df, stats = tester.run_strategy(golden_cross_v8_ls_strategy)
    
    print("\n--- Golden Cross V8 (Long-Short System) ---")
    for key, v in stats.items():
        print(f"{key}: {v}")

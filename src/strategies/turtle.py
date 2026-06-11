import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vectorized_engine import VectorizedBacktester

def turtle_strategy(df_1h, entry_window=480, exit_window=240):
    """
    Classic Turtle Trading (Donchian Channel Breakout)
    Adapted for 1h crypto data.
    """
    
    # Calculate Donchian Channels
    # Shift by 1 to not include the current bar in the historical high/low
    high_entry = df_1h['high'].rolling(entry_window).max().shift(1)
    low_entry = df_1h['low'].rolling(entry_window).min().shift(1)
    
    high_exit = df_1h['high'].rolling(exit_window).max().shift(1)
    low_exit = df_1h['low'].rolling(exit_window).min().shift(1)
    
    # Generate Signals
    target_pos = np.zeros(len(df_1h))
    
    long_cond = df_1h['close'] > high_entry
    short_cond = df_1h['close'] < low_entry
    
    exit_long = df_1h['close'] < low_exit
    exit_short = df_1h['close'] > high_exit
    
    current_pos = 0
    
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
    print("Testing Turtle Breakout Strategy...")
    tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z', fee_rate=0.0004, slippage=0.001)
    
    best_df = None
    for ew, xw in [(480, 240), (720, 360)]:
        df, stats = tester.run_strategy(turtle_strategy, entry_window=ew, exit_window=xw)
        print(f"\n--- Entry {ew}h, Exit {xw}h ---")
        for key, v in stats.items():
            print(f"{key}: {v}")
        if ew == 480:
            best_df = df
            
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(best_df.index, best_df['equity_curve'], label='Turtle (20D/10D)', color='teal', linewidth=2)
    plt.title('Turtle Strategy - BTC/USDT')
    plt.ylabel('Cumulative Return (1.0 = Initial Capital)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'plots', 'turtle_equity_curve.png'))

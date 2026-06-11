import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vectorized_engine import VectorizedBacktester

def golden_cross_v2_strategy(df_1h, fast_h=1200, slow_h=4800, trailing_atr_period=336, atr_mult=3.0):
    """
    Improved Golden Cross (V2)
    Entry: 50-day EMA > 200-day EMA (Macro Bull Market) AND Price crosses above 50-day EMA.
    Exit: 
    1. Death Cross (50-day < 200-day) - The ultimate macro exit.
    2. Price closes below 50-day EMA - Faster exit to protect profits.
    """
    
    fast_ema = df_1h['close'].ewm(span=fast_h, adjust=False).mean()
    slow_ema = df_1h['close'].ewm(span=slow_h, adjust=False).mean()
    
    # Generate Conditions
    macro_bull = fast_ema > slow_ema
    price_above_fast = df_1h['close'] > fast_ema
    
    long_cond = macro_bull & price_above_fast
    
    # We exit if the macro trend breaks OR if the price falls below the fast EMA (loss of short-term momentum)
    exit_long = (~macro_bull) | (df_1h['close'] < fast_ema)
    
    # State Machine for Signal Generation
    target_pos = np.zeros(len(df_1h))
    current_pos = 0
    
    long_cond_arr = long_cond.values
    exit_long_arr = exit_long.values
    
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
    print("Testing Improved Golden Cross V2 (Profit Protection)...")
    tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z', fee_rate=0.0004, slippage=0.001)
    
    df, stats = tester.run_strategy(golden_cross_v2_strategy)
    print(f"\n--- Golden Cross V2 ---")
    for key, v in stats.items():
        print(f"{key}: {v}")
        
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['equity_curve'], label='Golden Cross V2 (Fast Exit)', color='orange', linewidth=2)
    plt.title('Golden Cross V2 Strategy - BTC/USDT')
    plt.ylabel('Cumulative Return (1.0 = Initial Capital)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'plots', 'golden_cross_v2_equity_curve.png'))

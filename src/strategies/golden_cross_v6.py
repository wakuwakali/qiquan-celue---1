import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vectorized_engine import VectorizedBacktester

def golden_cross_v6_strategy(df_1h, fast_h=1200, slow_h=4800, signal_h=216):
    """
    Golden Cross V6 (The Holy Grail)
    Optimized MACD Signal Line to 9 Days (216 hours) for the ultimate balance
    of Drawdown Reduction and Profit Maximization.
    """
    
    fast_ema = df_1h['close'].ewm(span=fast_h, adjust=False).mean()
    slow_ema = df_1h['close'].ewm(span=slow_h, adjust=False).mean()
    
    macd = fast_ema - slow_ema
    # 9-Day Signal Line for faster exits at the absolute top
    signal_line = macd.ewm(span=signal_h, adjust=False).mean()
    histogram = macd - signal_line
    
    # Entry: Macro Bull Market (50D > 200D) AND Momentum is accelerating
    long_cond = (macd > 0) & (histogram > 0)
    
    # Exit: Macro trend breaks OR Momentum rolls over (Histogram < 0)
    exit_long = (macd < 0) | (histogram < 0)
    
    # State Machine
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
    print("Testing Golden Cross V6 (Holy Grail)...")
    tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z', fee_rate=0.0004, slippage=0.001)
    
    df, stats = tester.run_strategy(golden_cross_v6_strategy)
    print(f"\n--- Golden Cross V6 ---")
    for key, v in stats.items():
        print(f"{key}: {v}")
        
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['equity_curve'], label='Golden Cross V6 (Holy Grail)', color='magenta', linewidth=2)
    plt.title('Golden Cross V6 (Holy Grail) Strategy - BTC/USDT')
    plt.ylabel('Cumulative Return (1.0 = Initial Capital)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'plots', 'holy_grail_equity_curve.png'))

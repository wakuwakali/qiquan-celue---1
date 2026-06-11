import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vectorized_engine import VectorizedBacktester

def golden_cross_strategy(df_1h, fast_h=1200, slow_h=4800):
    """
    Classic Golden Cross (e.g. 50-day EMA vs 200-day EMA)
    1 day = 24 hours, so 50 days = 1200 hours, 200 days = 4800 hours.
    """
    
    fast_ema = df_1h['close'].ewm(span=fast_h, adjust=False).mean()
    slow_ema = df_1h['close'].ewm(span=slow_h, adjust=False).mean()
    
    target_pos = np.where(fast_ema > slow_ema, 1.0, 0.0) # Long only!
    
    df_1h['target_position'] = target_pos
    return df_1h['target_position']

if __name__ == '__main__':
    print("Testing Golden Cross (Long Only) Strategy...")
    tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z', fee_rate=0.0004, slippage=0.001)
    
    df, stats = tester.run_strategy(golden_cross_strategy)
    print(f"\n--- Golden Cross (50D/200D) ---")
    for key, v in stats.items():
        print(f"{key}: {v}")
        
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['equity_curve'], label='Golden Cross', color='gold', linewidth=2)
    plt.title('Golden Cross Strategy - BTC/USDT')
    plt.ylabel('Cumulative Return (1.0 = Initial Capital)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'plots', 'golden_cross_equity_curve.png'))

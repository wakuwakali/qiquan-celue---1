import sys, os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vectorized_engine import VectorizedBacktester

def golden_cross_v7_strategy(df_1h, fast_h=1200, slow_h=4800, signal_h=216):
    """
    Golden Cross V7 (Profit Lock)
    Added a dynamic trailing stop that ONLY activates when we have massive unrealized profits (>25%),
    protecting us from 'shark fin' micro-bubbles giving back all profits before the MACD can react.
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
                
            # Profit Lock Logic: If we are up more than 20% from entry, 
            # we trail a 12% stop loss behind the highest high.
            profit_lock_triggered = False
            if (highest_high / entry_price) > 1.20:
                trailing_stop = highest_high * 0.88 # 12% drop from absolute peak
                if close_arr[i] < trailing_stop:
                    profit_lock_triggered = True
                    
            if exit_primary_arr[i] or profit_lock_triggered:
                current_pos = 0
                
        target_pos[i] = current_pos
        
    df_1h['target_position'] = target_pos
    return df_1h['target_position']

if __name__ == '__main__':
    tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z', fee_rate=0.0004, slippage=0.001)
    df, stats = tester.run_strategy(golden_cross_v7_strategy)
    
    print("\n--- Golden Cross V7 (Profit Lock) ---")
    for key, v in stats.items():
        print(f"{key}: {v}")
        
    # Analyze early 2021
    subset = df.loc['2021-01-01':'2021-03-01']
    if not subset.empty:
        peak_val = subset['equity_curve'].max()
        trough_val = subset['equity_curve'][subset['equity_curve'].idxmax():].min()
        dd = (peak_val - trough_val) / peak_val
        print(f"Early 2021 Drawdown with V7: {dd:.2%}")
        
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(df.index, df['equity_curve'], color='cyan', linewidth=2)
    ax1.set_title('Golden Cross V7 (Profit Lock) - Fixing the Shark Fin', fontsize=16)
    ax1.set_ylabel('Cumulative Return')
    ax1.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'plots', 'v7_profit_lock_equity_curve.png'))

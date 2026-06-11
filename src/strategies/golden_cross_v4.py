import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vectorized_engine import VectorizedBacktester

def golden_cross_v4_strategy(df_1h, fast_h=1200, slow_h=4800, signal_h=480, atr_p=336, atr_sma_p=1200):
    """
    Golden Cross V4 (The Holy Grail)
    Added Volatility Regime Filter to prevent whipsaws in ranging markets.
    Added Chandelier / Catastrophe Trailing Stop.
    """
    
    fast_ema = df_1h['close'].ewm(span=fast_h, adjust=False).mean()
    slow_ema = df_1h['close'].ewm(span=slow_h, adjust=False).mean()
    
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal_h, adjust=False).mean()
    histogram = macd - signal_line
    
    # Macro Volatility Regime Filter (14 days vs 50 days)
    high_low = df_1h['high'] - df_1h['low']
    high_close = np.abs(df_1h['high'] - df_1h['close'].shift())
    low_close = np.abs(df_1h['low'] - df_1h['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    atr = tr.rolling(window=atr_p).mean()
    atr_sma = atr.rolling(window=atr_sma_p).mean()
    regime_valid = atr > atr_sma # Only enter when volatility is expanding
    
    # Conditions
    long_cond = (macd > 0) & (histogram > 0) & regime_valid
    
    # Primary Exit: Momentum rollover
    exit_cond_primary = (macd < 0) | (histogram < 0)
    
    # State Machine for Signal Generation with Trailing Stop
    target_pos = np.zeros(len(df_1h))
    current_pos = 0
    highest_high = 0
    
    long_cond_arr = long_cond.values
    exit_cond_primary_arr = exit_cond_primary.values
    close_arr = df_1h['close'].values
    high_arr = df_1h['high'].values
    atr_arr = atr.fillna(0).values
    
    for i in range(len(df_1h)):
        if current_pos == 0:
            if long_cond_arr[i]:
                current_pos = 1
                highest_high = high_arr[i]
        elif current_pos == 1:
            # Update highest high
            if high_arr[i] > highest_high:
                highest_high = high_arr[i]
                
            # Secondary Exit: Trailing Stop Loss (e.g. 5 ATRs from highest high)
            trailing_stop_price = highest_high - (5.0 * atr_arr[i])
            
            if exit_cond_primary_arr[i] or close_arr[i] < trailing_stop_price:
                current_pos = 0
                
        target_pos[i] = current_pos
        
    df_1h['target_position'] = target_pos
    return df_1h['target_position']

if __name__ == '__main__':
    print("Testing Golden Cross V4 (Regime Filter & Trailing Stop)...")
    tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z', fee_rate=0.0004, slippage=0.001)
    
    df, stats = tester.run_strategy(golden_cross_v4_strategy)
    print(f"\n--- Golden Cross V4 ---")
    for key, v in stats.items():
        print(f"{key}: {v}")
        
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['equity_curve'], label='Golden Cross V4 (Holy Grail)', color='purple', linewidth=2)
    plt.title('Golden Cross V4 Strategy - BTC/USDT')
    plt.ylabel('Cumulative Return (1.0 = Initial Capital)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'plots', 'golden_cross_v4_equity_curve.png'))

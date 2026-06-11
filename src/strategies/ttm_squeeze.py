import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vectorized_engine import VectorizedBacktester

def ttm_squeeze_strategy(df_1h, period=20, bb_mult=2.0, kc_mult=1.5):
    """
    TTM Squeeze Breakout Strategy
    """
    # 1. Bollinger Bands
    sma = df_1h['close'].rolling(period).mean()
    std = df_1h['close'].rolling(period).std()
    bb_upper = sma + bb_mult * std
    bb_lower = sma - bb_mult * std
    
    # 2. Keltner Channels
    high_low = df_1h['high'] - df_1h['low']
    high_close = np.abs(df_1h['high'] - df_1h['close'].shift())
    low_close = np.abs(df_1h['low'] - df_1h['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    kc_upper = sma + kc_mult * atr
    kc_lower = sma - kc_mult * atr
    
    # 3. Squeeze Condition
    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    squeeze_off = ~squeeze_on
    
    # Squeeze "Fires" (Transitions from On to Off)
    squeeze_fires = squeeze_off & squeeze_on.shift(1)
    
    # 4. Momentum Direction (Linear Regression of close against period, or just close > SMA)
    # Simple Momentum: Price > SMA is bullish, Price < SMA is bearish
    bullish = df_1h['close'] > sma
    bearish = df_1h['close'] < sma
    
    # 5. Generate Signals
    target_pos = np.zeros(len(df_1h))
    
    long_cond = squeeze_fires & bullish
    short_cond = squeeze_fires & bearish
    
    # Exit conditions: Cross back to SMA
    exit_long = df_1h['close'] <= sma
    exit_short = df_1h['close'] >= sma
    
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
    print("Testing TTM Squeeze Strategy...")
    tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z', fee_rate=0.0004, slippage=0.001)
    
    best_df = None
    for mult in [1.5, 2.0]:
        df, stats = tester.run_strategy(ttm_squeeze_strategy, period=20, bb_mult=2.0, kc_mult=mult)
        print(f"\n--- KC Multiplier = {mult} ---")
        for key, v in stats.items():
            print(f"{key}: {v}")
        if mult == 1.5:
            best_df = df
            
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(best_df.index, best_df['equity_curve'], label='TTM Squeeze', color='blue', linewidth=2)
    plt.title('TTM Squeeze Strategy - BTC/USDT')
    plt.ylabel('Cumulative Return (1.0 = Initial Capital)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'plots', 'ttm_squeeze_equity_curve.png'))

import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vectorized_engine import VectorizedBacktester

def bollinger_rsi_mean_reversion(df_1h, bb_window=20, bb_std=2.0, rsi_window=14, atr_window=14, atr_sma_window=50):
    """
    Bollinger Band + RSI Mean Reversion Strategy (with Low Volatility Regime Filter)
    
    Logic:
    1. In a low volatility regime (ATR < ATR_SMA), price tends to mean revert.
    2. Buy when price touches lower BB AND RSI is oversold (<30).
    3. Sell when price touches upper BB AND RSI is overbought (>70).
    4. Exit trades when price crosses the Middle BB (Mean).
    """
    
    # 1. Calculate Bollinger Bands on 1h timeframe
    df_1h['bb_middle'] = df_1h['close'].rolling(window=bb_window).mean()
    bb_std_dev = df_1h['close'].rolling(window=bb_window).std()
    df_1h['bb_upper'] = df_1h['bb_middle'] + bb_std * bb_std_dev
    df_1h['bb_lower'] = df_1h['bb_middle'] - bb_std * bb_std_dev
    
    # 2. Calculate RSI
    delta = df_1h['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    df_1h['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. Calculate ATR & Low Volatility Filter
    high_low = df_1h['high'] - df_1h['low']
    high_close = np.abs(df_1h['high'] - df_1h['close'].shift())
    low_close = np.abs(df_1h['low'] - df_1h['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    atr = tr.rolling(window=atr_window).mean()
    atr_sma = atr.rolling(window=atr_sma_window).mean()
    
    # REGIME: We only mean-revert in LOW volatility environments (ATR <= ATR_SMA)
    # Mean reversion in high volatility gets you run over by trends.
    regime_valid = atr <= atr_sma
    
    # 4. Generate Raw Signals
    # Important: Signals are evaluated on Close(t) and executed on Open(t+1) by the engine.
    long_condition = (df_1h['close'] < df_1h['bb_lower']) & (df_1h['rsi'] < 30) & regime_valid
    short_condition = (df_1h['close'] > df_1h['bb_upper']) & (df_1h['rsi'] > 70) & regime_valid
    
    # Exit conditions: Cross the mean
    exit_long = df_1h['close'] >= df_1h['bb_middle']
    exit_short = df_1h['close'] <= df_1h['bb_middle']
    
    # State Machine to resolve hold positions
    # 1 for Long, -1 for Short, 0 for Flat
    positions = np.zeros(len(df_1h))
    current_pos = 0
    
    # We must loop through to manage the exits properly, since it's state dependent.
    # Numba could speed this up, but Python loop is okay for 50k rows.
    # To vectorize state perfectly without loops, we use forward fill tricks.
    
    sig = np.zeros(len(df_1h))
    sig[long_condition] = 1
    sig[short_condition] = -1
    sig[(exit_long) & (sig == 0)] = 2 # magic number for flat
    sig[(exit_short) & (sig == 0)] = 2
    
    # This loop is fast enough for 50k rows.
    for i in range(len(df_1h)):
        if sig[i] == 1:
            current_pos = 1
        elif sig[i] == -1:
            current_pos = -1
        elif sig[i] == 2:
            current_pos = 0
        positions[i] = current_pos
        
    df_1h['target_position'] = positions
    return df_1h['target_position']

if __name__ == '__main__':
    print("Testing Bollinger+RSI Mean Reversion Strategy...")
    # Instantiate backtester with 1h data and standard frictions
    tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z', fee_rate=0.0004, slippage=0.001)
    
    # Test different BB windows and RSI setups
    for bb_w in [20, 30]:
        df, stats = tester.run_strategy(bollinger_rsi_mean_reversion, bb_window=bb_w, bb_std=2.0, rsi_window=14)
        print(f"\n--- BB Window = {bb_w} ---")
        for key, v in stats.items():
            print(f"{key}: {v}")
            
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['equity_curve'], label=f'Mean Reversion (BB={bb_w})', color='purple', linewidth=2)
        plt.title('Regime-Filtered Mean Reversion Strategy - BTC/USDT')
        plt.ylabel('Cumulative Return (1.0 = Initial Capital)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'plots', f'mean_reversion_equity_curve_bb{bb_w}.png'))

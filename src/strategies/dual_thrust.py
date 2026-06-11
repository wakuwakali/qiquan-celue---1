import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from vectorized_engine import VectorizedBacktester

def regime_filtered_dual_thrust(df_1h, n_days=5, k1=0.6, k2=0.6, atr_period=14, atr_sma_period=50):
    df_1d = df_1h.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    # Dual Thrust calculations
    hh = df_1d['high'].rolling(n_days).max()
    hc = df_1d['close'].rolling(n_days).max()
    lc = df_1d['close'].rolling(n_days).min()
    ll = df_1d['low'].rolling(n_days).min()
    
    range_1d = np.maximum(hh - lc, hc - ll).shift(1)
    range_1h = range_1d.reindex(df_1h.index, method='ffill')
    daily_open = df_1h.groupby(df_1h.index.date)['open'].transform('first')
    
    buy_line = daily_open + k1 * range_1h
    sell_line = daily_open - k2 * range_1h
    
    # ATR calculations on 1h timeframe
    high_low = df_1h['high'] - df_1h['low']
    high_close = np.abs(df_1h['high'] - df_1h['close'].shift())
    low_close = np.abs(df_1h['low'] - df_1h['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    atr = tr.rolling(atr_period).mean()
    atr_sma = atr.rolling(atr_sma_period).mean()
    
    # Regime condition: Volatility is expanding
    regime_valid = atr > atr_sma
    
    # Generate Signals
    signals = pd.Series(0, index=df_1h.index)
    
    # Entry conditions
    long_cond = (df_1h['close'] > buy_line) & regime_valid
    short_cond = (df_1h['close'] < sell_line) & regime_valid
    
    # We also need an exit condition if the trend dies or reverses.
    # To keep it simple in a vectorized framework, we hold until the opposite signal triggers,
    # OR we exit to neutral if regime becomes invalid? No, trend following needs to let profits run.
    # We will hold the position until a reversal signal triggers.
    
    signals[long_cond] = 1
    signals[short_cond] = -1
    
    # Forward fill
    signals = signals.replace(0, np.nan).ffill().fillna(0)
    
    return signals

if __name__ == "__main__":
    print("Testing Regime-Filtered Dual Thrust...")
    tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z', fee_rate=0.0004, slippage=0.001)
    
    # Test different K values
    for k in [0.5]:
        df, stats = tester.run_strategy(regime_filtered_dual_thrust, k1=k, k2=k)
        print(f"\n--- K = {k} ---")
        for key, v in stats.items():
            print(f"{key}: {v}")
            
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['equity_curve'], label=f'Regime-Filtered Dual Thrust (K={k})', color='orange', linewidth=2)
        plt.title('Regime-Filtered Dual Thrust Strategy - BTC/USDT')
        plt.ylabel('Cumulative Return (1.0 = Initial Capital)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'plots', 'dual_thrust_equity_curve.png'))

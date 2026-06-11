import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from vectorized_engine import VectorizedBacktester

def vol_scaled_tsmom(df_1h, fast_ema=10, slow_ema=50, vol_window=30, target_vol=0.40):
    # Calculate daily returns to measure annualized volatility
    df_1d = df_1h.resample('D').agg({'close': 'last'})
    df_1d['return'] = df_1d['close'].pct_change()
    df_1d['realized_vol'] = df_1d['return'].rolling(vol_window).std() * np.sqrt(365)
    
    # We must shift realized_vol by 1 so we don't look ahead into today's close!
    df_1d['realized_vol'] = df_1d['realized_vol'].shift(1)
    realized_vol_1h = df_1d['realized_vol'].reindex(df_1h.index, method='ffill')
    
    # Calculate EMA signals on the daily timeframe to avoid intraday whipsaws,
    # or just use hourly EMA? The subagent recommended 10 and 50. Let's assume daily EMA.
    fast = df_1d['close'].ewm(span=fast_ema, adjust=False).mean()
    slow = df_1d['close'].ewm(span=slow_ema, adjust=False).mean()
    
    signal_1d = np.where(fast > slow, 1.0, -1.0)
    signal_1d = pd.Series(signal_1d, index=df_1d.index)
    
    # We must shift signal_1d by 1 so we don't look ahead into today's close!
    signal_1d = signal_1d.shift(1)
    signal_1h = signal_1d.reindex(df_1h.index, method='ffill')
    
    # Calculate target weight: Signal * (Target Volatility / Realized Volatility)
    # Cap maximum leverage to 2x for safety
    raw_weight = signal_1h * (target_vol / realized_vol_1h)
    weight = raw_weight.clip(-2.0, 2.0)
    
    # Forward fill any NaNs at the beginning
    weight = weight.fillna(0)
    
    return weight

if __name__ == "__main__":
    print("Testing Volatility-Scaled TSMOM Strategy...")
    tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z', fee_rate=0.0004, slippage=0.001)
    
    for target in [0.4]:
        df, stats = tester.run_strategy(vol_scaled_tsmom, fast_ema=10, slow_ema=50, vol_window=30, target_vol=target)
        print(f"\n--- Target Vol = {target} ---")
        for key, v in stats.items():
            print(f"{key}: {v}")
            
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['equity_curve'], label=f'TSMOM Target Vol {target}', color='green', linewidth=2)
        plt.title('Volatility-Scaled Time-Series Momentum (TSMOM) - BTC/USDT')
        plt.ylabel('Cumulative Return (1.0 = Initial Capital)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'plots', 'tsmom_equity_curve.png'))

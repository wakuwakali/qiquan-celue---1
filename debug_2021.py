import sys, os
import pandas as pd
sys.path.append('src')
from vectorized_engine import VectorizedBacktester
from strategies.golden_cross_v6 import golden_cross_v6_strategy

tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z')
df, _ = tester.run_strategy(golden_cross_v6_strategy)

# Look at Jan - Feb 2021
subset = df.loc['2021-01-01':'2021-03-01']
peak_val = subset['equity_curve'].max()
peak_idx = subset['equity_curve'].idxmax()

trough_val = subset['equity_curve'][peak_idx:].min()
trough_idx = subset['equity_curve'][peak_idx:].idxmin()

dd = (peak_val - trough_val) / peak_val

print(f"Early 2021 Peak: {peak_val:.4f} at {peak_idx}")
print(f"Early 2021 Trough: {trough_val:.4f} at {trough_idx}")
print(f"Drawdown in this period: {dd:.2%}")

# Look at target positions
print("\nPositions during this period:")
changes = subset[subset['actual_position'].diff() != 0]
print(changes[['close', 'actual_position', 'equity_curve']])

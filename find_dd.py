import sys
import os
sys.path.append('src')
from vectorized_engine import VectorizedBacktester
from strategies.golden_cross_v6 import golden_cross_v6_strategy

tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z')
df, _ = tester.run_strategy(golden_cross_v6_strategy)

peak = df['equity_curve'].cummax()
dd = (peak - df['equity_curve']) / peak

max_dd_date = dd.idxmax()
max_dd_val = dd.max()
max_dd_peak_date = df['equity_curve'][:max_dd_date].idxmax()

print(f"Overall Max DD: {max_dd_val:.2%} occurred at {max_dd_date}")
print(f"This drawdown started from the peak on {max_dd_peak_date}")

early_dd = dd.loc['2021-01-01':'2021-08-01']
print(f"Early 2021 Max DD: {early_dd.max():.2%} occurred at {early_dd.idxmax()}")

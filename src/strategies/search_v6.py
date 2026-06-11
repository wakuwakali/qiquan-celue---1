import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vectorized_engine import VectorizedBacktester
from strategies.golden_cross_v3 import golden_cross_v3_strategy

if __name__ == '__main__':
    tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z', fee_rate=0.0004, slippage=0.001)
    tester.fetch_data()
    
    results = []
    # Test different Signal Line lengths (Fast exit parameter)
    # Signal line length in hours: 120 (5D), 216 (9D), 360 (15D), 480 (20D), 720 (30D)
    for sig in [120, 216, 360, 480, 720]:
        def strategy(df):
            return golden_cross_v3_strategy(df, fast_h=1200, slow_h=4800, signal_h=sig)
            
        _, stats = tester.run_strategy(strategy)
        results.append({
            'Signal Line (Hours)': sig,
            'Total Return': stats['Total Return'],
            'Max Drawdown': stats['Max Drawdown'],
            'Sharpe': stats['Sharpe Ratio'],
            'Trades': stats['Total Trades']
        })
        
    print(pd.DataFrame(results).to_markdown(index=False))

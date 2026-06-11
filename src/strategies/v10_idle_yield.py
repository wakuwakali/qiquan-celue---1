import sys, os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vectorized_engine import VectorizedBacktester
from strategies.golden_cross_v7 import golden_cross_v7_strategy

def test_idle_yield(annual_yield=0.08):
    """
    Test V7 strategy but add an APY to the capital when it is sitting in Cash (USDT).
    """
    tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z', fee_rate=0.0004, slippage=0.001)
    df, stats = tester.run_strategy(golden_cross_v7_strategy)
    
    hourly_yield = (1 + annual_yield) ** (1 / (365 * 24)) - 1
    
    # Calculate custom equity curve
    actual_pos = df['actual_position'].values
    returns = df['close'].pct_change().fillna(0).values
    
    equity = np.ones(len(df))
    equity[0] = 1.0
    
    for i in range(1, len(df)):
        if actual_pos[i-1] == 1:
            step_return = returns[i]
        else:
            step_return = hourly_yield # Earn idle yield when in cash
            
        # apply friction if position changed
        friction = 0
        if actual_pos[i] != actual_pos[i-1]:
            friction = tester.fee_rate + tester.slippage
            
        equity[i] = equity[i-1] * (1 + step_return) * (1 - friction)
        
    df['yield_equity_curve'] = equity
    
    total_return = equity[-1] - 1
    peak = df['yield_equity_curve'].cummax()
    dd = (peak - df['yield_equity_curve']) / peak
    max_dd = dd.max()
    
    print(f"\n--- Golden Cross V7 + {annual_yield*100}% Idle Yield ---")
    print(f"Total Return: {total_return:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")
    
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(df.index, df['yield_equity_curve'], color='gold', linewidth=2, label=f'V7 + {annual_yield*100}% Yield')
    ax1.plot(df.index, df['equity_curve'], color='cyan', linewidth=1, alpha=0.5, label='V7 Standard')
    ax1.set_title(f'Golden Cross V7 + Idle Capital Yield ({annual_yield*100}% APY)', fontsize=16)
    ax1.legend()
    ax1.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'plots', 'v10_idle_yield.png'))

if __name__ == '__main__':
    test_idle_yield(0.08) # 8% APY

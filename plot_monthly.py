import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append('src')
from vectorized_engine import VectorizedBacktester
from strategies.golden_cross_v7 import golden_cross_v7_strategy

if __name__ == '__main__':
    tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z', fee_rate=0.0004, slippage=0.001)
    df, _ = tester.run_strategy(golden_cross_v7_strategy)

    # Setup the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 14), gridspec_kw={'height_ratios': [3, 1.5]})

    # --- Top Panel: Equity Curve ---
    ax1.plot(df.index, df['equity_curve'], color='cyan', linewidth=2.5)
    ax1.set_title('Golden Cross V7 (Profit Lock) - Detailed Monthly Equity Curve & Returns', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Cumulative Return (1.0 = Initial)', fontsize=16)
    
    # Configure precise monthly grid
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2)) # Every 2 months major tick
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=1)) # Every 1 month minor grid
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.tick_params(axis='x', labelsize=12, rotation=45)
    ax1.tick_params(axis='y', labelsize=14)
    
    # Enable grids
    ax1.grid(True, which='major', color='black', linestyle='-', alpha=0.3)
    ax1.grid(True, which='minor', color='gray', linestyle='--', alpha=0.2)

    # --- Bottom Panel: Monthly Returns ---
    # Resample to end of month equity to compute monthly return
    monthly_equity = df['equity_curve'].resample('ME').last()
    # If the first month doesn't have a previous month, use 1.0 as base
    monthly_ret = (monthly_equity / monthly_equity.shift(1).fillna(1.0)) - 1.0
    monthly_ret_pct = monthly_ret * 100

    colors = ['#2ca02c' if r > 0 else '#d62728' for r in monthly_ret_pct]
    ax2.bar(monthly_ret_pct.index, monthly_ret_pct, color=colors, width=25, alpha=0.8)
    
    ax2.set_title('Monthly Profit / Loss (%)', fontsize=16)
    ax2.set_ylabel('Return %', fontsize=14)
    
    # Configure precise monthly grid for bottom panel
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.tick_params(axis='x', labelsize=12, rotation=45)
    ax2.tick_params(axis='y', labelsize=14)
    
    ax2.grid(True, which='major', color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, which='minor', color='gray', linestyle='--', alpha=0.2)
    ax2.axhline(0, color='black', linewidth=1)

    plt.tight_layout()
    os.makedirs(os.path.join('results', 'plots'), exist_ok=True)
    plot_path = os.path.join('results', 'plots', 'v6_monthly_detailed.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved successfully to {plot_path}")

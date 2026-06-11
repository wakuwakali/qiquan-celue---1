import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vectorized_engine import VectorizedBacktester
from strategies.golden_cross_v3 import golden_cross_v3_strategy

def simulate_mode(df_base, leverage, fee_rate, slippage, hourly_funding_rate=0.0):
    """
    Simulates a specific trading instrument mode.
    """
    df = df_base.copy()
    
    # Apply Leverage
    df['actual_position'] = df['actual_position'] * leverage
    
    # Recalculate Returns
    df['intra_return'] = df['close'] / df['open'] - 1
    df['gap_return'] = df['open'] / df['close'].shift(1) - 1
    df['strategy_return'] = df['actual_position'].shift(1) * df['gap_return'] + df['actual_position'] * df['intra_return']
    
    # Recalculate Friction
    df['trade'] = df['actual_position'].diff().fillna(0)
    cost_per_trade = fee_rate + slippage
    df['friction_cost'] = df['trade'].abs() * cost_per_trade
    
    # Funding Rate (paid on nominal position size)
    # If holding Long (actual_position > 0), pay funding
    df['funding_cost'] = np.where(df['actual_position'] > 0, df['actual_position'] * hourly_funding_rate, 0)
    
    # Net Return
    df['net_return'] = df['strategy_return'] - df['friction_cost'] - df['funding_cost']
    
    # Initial Capital $100,000
    df['equity_curve'] = 100000 * (1 + df['net_return']).cumprod()
    
    return df

if __name__ == '__main__':
    print("Fetching data and running base V3 strategy...")
    tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z')
    df_base, _ = tester.run_strategy(golden_cross_v3_strategy)
    
    print("\nSimulating 3 Execution Modes from $100,000 Initial Capital:")
    
    # Mode 1: Spot (1x)
    # Fees: 0.04%, Slippage 0.1%, Funding 0
    df_spot = simulate_mode(df_base, leverage=1.0, fee_rate=0.0004, slippage=0.001, hourly_funding_rate=0.0)
    
    # Mode 2: Perpetual Futures (2x Leverage)
    # Fees: 0.04%, Slippage 0.1%, Funding 0.01% per 8h = 0.0000125 per hour
    df_futures = simulate_mode(df_base, leverage=2.0, fee_rate=0.0004, slippage=0.001, hourly_funding_rate=0.0000125)
    
    # Mode 3: Options Synthetic Long (3x Leverage)
    # Options have higher spread/fees but NO funding rate.
    # Fees: 0.1%, Slippage 0.2%, Funding 0
    df_options = simulate_mode(df_base, leverage=3.0, fee_rate=0.001, slippage=0.002, hourly_funding_rate=0.0)
    
    modes = [
        ('Spot (1x)', df_spot),
        ('Perp Futures (2x)', df_futures),
        ('Options Synthetic (3x)', df_options)
    ]
    
    results = []
    for name, df in modes:
        final_equity = df['equity_curve'].iloc[-1]
        roi = (final_equity - 100000) / 100000
        peak = df['equity_curve'].cummax()
        max_dd = ((peak - df['equity_curve']) / peak).max()
        
        results.append({
            'Mode': name,
            'Final Equity': f"${final_equity:,.0f}",
            'Total ROI': f"{roi:.2%}",
            'Max Drawdown': f"{max_dd:.2%}"
        })
        
    # Print Table
    print(pd.DataFrame(results).to_markdown(index=False))
    
    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(df_spot.index, df_spot['equity_curve'], label='Spot (1x)', color='blue')
    plt.plot(df_futures.index, df_futures['equity_curve'], label='Perp Futures (2x)', color='red')
    plt.plot(df_options.index, df_options['equity_curve'], label='Options Synthetic Long (3x)', color='green')
    
    plt.title('Golden Cross V3: $100k Capital across Different Execution Modes (2021-2026)')
    plt.ylabel('Equity Balance ($)')
    plt.grid(True)
    plt.legend()
    plt.yscale('log') # Use log scale because 3x leverage grows very large
    
    plot_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'plots', 'execution_modes_comparison.png')
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")

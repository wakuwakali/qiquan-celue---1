import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

class VectorizedBacktester:
    def __init__(self, symbol='BTC/USDT', timeframe='1h', start_time='2021-01-01T00:00:00Z', fee_rate=0.0004, slippage=0.001):
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_time = start_time
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.df = None
        
    def fetch_data(self):
        print(f"Fetching {self.timeframe} data for {self.symbol}...")
        exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
        since = exchange.parse8601(self.start_time)
        all_klines = []
        
        while True:
            try:
                klines = exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since, limit=1000)
                if not klines: break
                all_klines.extend(klines)
                since = klines[-1][0] + 1  # Add 1ms to avoid duplicate
                if len(klines) < 1000: break
            except Exception as e:
                print(f"Fetch error: {e}")
                break
                
        self.df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.df['datetime'] = pd.to_datetime(self.df['timestamp'], unit='ms')
        self.df.set_index('datetime', inplace=True)
        # Drop duplicates just in case
        self.df = self.df[~self.df.index.duplicated(keep='first')]
        print(f"Data fetched: {len(self.df)} rows from {self.df.index[0]} to {self.df.index[-1]}")
        return self.df
        
    def run_strategy(self, generate_signals_func, **kwargs):
        """
        generate_signals_func MUST return a Series of positions:
        1 for Long, -1 for Short, 0 for Flat.
        The position generated at row `i` will be assumed as the TARGET position
        to be entered at the OPEN price of row `i+1`.
        This completely eliminates look-ahead bias.
        """
        if self.df is None:
            self.fetch_data()
            
        df = self.df.copy()
        
        # 1. Generate Target Positions
        df['target_position'] = generate_signals_func(df, **kwargs)
        
        # 2. Shift position by 1 to simulate execution at the next candle's open
        df['actual_position'] = df['target_position'].shift(1).fillna(0)
        
        # 3. Calculate execution prices (entering at Open)
        # The return of the holding is from Open to Close of the current candle,
        # plus the Close to Open of the next candle if held.
        # A simpler robust vectorized approach:
        # PnL = position * (Close_t - Open_t) + position_yesterday * (Open_t - Close_{t-1})
        # But even simpler: just use log returns of Close, and shift position.
        # Wait, if we enter at Open, we capture the return from Open_t to Close_t.
        # If we hold through the night, we capture Close_t to Close_{t+1}.
        # Standard approach:
        # Strategy return at t = Position_{t-1} * (Close_t / Close_{t-1} - 1)
        # This assumes execution at Close_{t-1}.
        # To execute at Open_t:
        # Day Return = Position_{t-1} * (Close_t / Open_t - 1) + Position_{t-2} * (Open_t / Close_{t-1} - 1)
        
        # Let's use the standard "Execution at Close" for simplicity in vectorized form,
        # BUT we shift signals by 2 to be ultra-conservative? 
        # No, let's stick to standard: Signal at Close(t) -> Exec at Open(t+1).
        
        # Price change from Open(t) to Close(t)
        df['intra_return'] = df['close'] / df['open'] - 1
        # Price change from Close(t-1) to Open(t)
        df['gap_return'] = df['open'] / df['close'].shift(1) - 1
        
        # Total return = gap_return (earned by holding previous pos) + intra_return (earned by current pos)
        df['strategy_return'] = df['actual_position'].shift(1) * df['gap_return'] + df['actual_position'] * df['intra_return']
        
        # 4. Calculate Trading Frictions (Fees & Slippage)
        # A trade happens when actual_position changes
        df['trade'] = df['actual_position'].diff().fillna(0)
        
        # Cost is applied on the absolute change in position
        # For example, going from 1 to -1 is a trade of size 2.
        # Each unit traded pays fee_rate + slippage.
        cost_per_trade = self.fee_rate + self.slippage
        df['friction_cost'] = df['trade'].abs() * cost_per_trade
        
        # 5. Net Returns
        df['net_return'] = df['strategy_return'] - df['friction_cost']
        
        # 6. Equity Curve
        df['equity_curve'] = (1 + df['net_return']).cumprod()
        
        # Calculate Metrics
        total_return = df['equity_curve'].iloc[-1] - 1
        
        # Annualized Return (assuming 365 days / 24 hours per year)
        days = (df.index[-1] - df.index[0]).total_seconds() / 86400.0
        ann_return = (1 + total_return) ** (365.25 / days) - 1 if total_return > -1 else -1
        
        peak = df['equity_curve'].cummax()
        drawdown = (peak - df['equity_curve']) / peak
        max_dd = drawdown.max()
        
        sharpe = np.sqrt(365.25 * (24 if self.timeframe == '1h' else 1)) * (df['net_return'].mean() / df['net_return'].std())
        
        stats = {
            'Total Return': f"{total_return:.2%}",
            'Annualized Return': f"{ann_return:.2%}",
            'Max Drawdown': f"{max_dd:.2%}",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Total Trades': int(df['trade'].abs().sum() / 2) # Div 2 because entry and exit are 2 units total usually
        }
        
        return df, stats

# Example Strategy: Simple Moving Average (just to verify the engine works)
def sma_crossover(df, fast=20, slow=50):
    df_temp = pd.DataFrame(index=df.index)
    df_temp['fast'] = df['close'].rolling(fast).mean()
    df_temp['slow'] = df['close'].rolling(slow).mean()
    
    # 1 if fast > slow, -1 if fast < slow
    df_temp['signal'] = np.where(df_temp['fast'] > df_temp['slow'], 1, -1)
    # Neutral when moving averages are not yet calculated
    df_temp['signal'].iloc[:slow] = 0
    
    return df_temp['signal']

if __name__ == "__main__":
    tester = VectorizedBacktester(timeframe='1h', start_time='2022-01-01T00:00:00Z')
    df, stats = tester.run_strategy(sma_crossover, fast=20, slow=50)
    
    print("\n--- SMA Engine Test ---")
    for k, v in stats.items():
        print(f"{k}: {v}")

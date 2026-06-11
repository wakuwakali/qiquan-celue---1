import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 终极圣杯量化交易引擎 (Holy Grail Engine V10)
# 包含：V7宏观动能进出场 + Profit Lock强制锁润 + 闲置资金生息
# 适用标的：BTC现货、永续合约、期权合成多头
# ==========================================

def holy_grail_strategy(df_1h, fast_h=1200, slow_h=4800, signal_h=216):
    """
    核心信号生成模块
    - 50D/200D 宏观均线过滤垃圾震荡
    - MACD Histogram (9天) 极速逃顶
    - 20%浮盈触发 12%吊灯强制止盈 (Profit Lock)
    """
    fast_ema = df_1h['close'].ewm(span=fast_h, adjust=False).mean()
    slow_ema = df_1h['close'].ewm(span=slow_h, adjust=False).mean()
    
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal_h, adjust=False).mean()
    histogram = macd - signal_line
    
    long_cond = (macd > 0) & (histogram > 0)
    exit_primary = (macd < 0) | (histogram < 0)
    
    target_pos = np.zeros(len(df_1h))
    current_pos = 0
    highest_high = 0
    entry_price = 0
    
    long_cond_arr = long_cond.values
    exit_primary_arr = exit_primary.values
    close_arr = df_1h['close'].values
    high_arr = df_1h['high'].values
    
    for i in range(len(df_1h)):
        if current_pos == 0:
            if long_cond_arr[i]:
                current_pos = 1
                entry_price = close_arr[i]
                highest_high = high_arr[i]
        elif current_pos == 1:
            if high_arr[i] > highest_high:
                highest_high = high_arr[i]
                
            # Profit Lock Logic: If un-realized profit > 20%, activate 12% trailing stop
            profit_lock = False
            if (highest_high / entry_price) > 1.20:
                trailing_stop = highest_high * 0.88 
                if close_arr[i] < trailing_stop:
                    profit_lock = True
                    
            if exit_primary_arr[i] or profit_lock:
                current_pos = 0
                
        target_pos[i] = current_pos
        
    df_1h['target_position'] = target_pos
    return df_1h['target_position']

def simulate_full_ecosystem(df_1h, fee_rate=0.0004, slippage=0.001, idle_apy=0.08):
    """
    全生态系统回测引擎
    - 计入双边交易摩擦 (0.14%)
    - 计入空仓期间的 USDT 活期理财复利 (8% APY)
    """
    signals = holy_grail_strategy(df_1h)
    actual_pos = signals.shift(1).fillna(0).values # Executed next hour
    
    returns = df_1h['close'].pct_change().fillna(0).values
    hourly_yield = (1 + idle_apy) ** (1 / (365 * 24)) - 1
    
    equity = np.ones(len(df_1h))
    equity[0] = 1.0
    trades = 0
    
    for i in range(1, len(df_1h)):
        if actual_pos[i-1] == 1:
            step_return = returns[i]
        else:
            step_return = hourly_yield # Idle Yield Active
            
        friction = 0
        if actual_pos[i] != actual_pos[i-1]:
            friction = fee_rate + slippage
            if actual_pos[i] == 1:
                trades += 1
            
        equity[i] = equity[i-1] * (1 + step_return) * (1 - friction)
        
    df_1h['equity_curve'] = equity
    
    # Calculate Stats
    total_ret = equity[-1] - 1
    peak = df_1h['equity_curve'].cummax()
    dd = (peak - df_1h['equity_curve']) / peak
    max_dd = dd.max()
    
    print("=========================================")
    print(f"Holy Grail V10 (V7 Logic + {idle_apy*100}% Idle Yield)")
    print("=========================================")
    print(f"Total Return: {total_ret:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Total Trades: {trades}")
    print("=========================================")
    
    return df_1h

if __name__ == '__main__':
    print("This is the core engine module. Run it with your data feeder.")

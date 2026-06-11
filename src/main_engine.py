import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from scipy.optimize import brentq
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import delta as bs_delta

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Module A: OptionInstrument
# ---------------------------------------------------------
class OptionInstrument:
    def __init__(self, flag, K, T, r, sigma):
        self.flag = flag
        self.K = K
        self.T = max(T, 0.00001)
        self.r = r
        self.sigma = sigma
        
    def price(self, S):
        return bs(self.flag, S, self.K, self.T, self.r, self.sigma)
        
    def delta(self, S):
        return bs_delta(self.flag, S, self.K, self.T, self.r, self.sigma)

# ---------------------------------------------------------
# Module B: OptionPortfolio (Iron Condor)
# ---------------------------------------------------------
class IronCondorPortfolio:
    def __init__(self, S_open, T, r, IV_open):
        self.S_open = S_open
        self.T = T
        self.r = r
        self.IV_open = IV_open
        self.legs = []
        
        self.k_call_s = self._find_strike('c', S_open, 0.30)
        self.k_call_l = self._find_strike('c', S_open, 0.10)
        self.k_put_s  = self._find_strike('p', S_open, -0.30)
        self.k_put_l  = self._find_strike('p', S_open, -0.10)
        
        self.legs.append({'type': 'short', 'opt': OptionInstrument('c', self.k_call_s, T, r, IV_open)})
        self.legs.append({'type': 'long',  'opt': OptionInstrument('c', self.k_call_l, T, r, IV_open)})
        self.legs.append({'type': 'short', 'opt': OptionInstrument('p', self.k_put_s,  T, r, IV_open)})
        self.legs.append({'type': 'long',  'opt': OptionInstrument('p', self.k_put_l,  T, r, IV_open)})
        
    def _find_strike(self, flag, S, target_delta):
        def f(K):
            opt = OptionInstrument(flag, K, self.T, self.r, self.IV_open)
            return opt.delta(S) - target_delta
        try:
            if flag == 'c':
                return brentq(f, S * 0.5, S * 5.0)
            else:
                return brentq(f, S * 0.1, S * 1.5)
        except ValueError:
            return S * 1.5 if flag == 'c' else S * 0.5

    def get_unit_premium(self, S, slippage=0.0):
        premium = 0
        for leg in self.legs:
            p = leg['opt'].price(S)
            if leg['type'] == 'short':
                premium += p * (1 - slippage)
            else:
                premium -= p * (1 + slippage)
        return premium
        
    def get_unit_liability(self, S, T_remain, IV_mtm, slippage=0.0):
        liability = 0
        for leg in self.legs:
            leg['opt'].T = T_remain
            leg['opt'].sigma = IV_mtm
            p = leg['opt'].price(S)
            if leg['type'] == 'short':
                liability += p * (1 + slippage)
            else:
                liability -= p * (1 - slippage)
        return liability

    def max_loss_per_unit(self, net_premium_received):
        max_call_risk = self.k_call_l - self.k_call_s
        max_put_risk = self.k_put_s - self.k_put_l
        return max(max_call_risk, max_put_risk) - net_premium_received
        
    def get_expiry_payoff(self, S):
        payoff = 0
        for leg in self.legs:
            K = leg['opt'].K
            flag = leg['opt'].flag
            p = max(0, S - K) if flag == 'c' else max(0, K - S)
            if leg['type'] == 'short':
                payoff -= p
            else:
                payoff += p
        return payoff

# ---------------------------------------------------------
# Module C: MarginAccount
# ---------------------------------------------------------
class MarginAccount:
    def __init__(self, initial_equity):
        self.equity = initial_equity
        self.cash = initial_equity
        
    def pay_fee(self, amount):
        self.equity -= amount
        
    def add_pnl(self, pnl):
        self.equity += pnl

# ---------------------------------------------------------
# Module D: BacktestEngine
# ---------------------------------------------------------
class BacktestEngine:
    def __init__(self, df):
        self.df = df
        self.account = MarginAccount(100000.0)
        self.r = 0.0
        self.OPT_FEE_RATE = 0.0003
        self.OPT_SLIPPAGE = 0.02
        self.tenor = 90
        self.vol_filter = 0.65
        self.take_profit = 0.50
        
        self.active_ic = None
        self.qty = 0.0
        self.days_left = 0
        self.net_premium_received = 0.0
        self.pnl_curve = [self.account.equity]
        
    def run(self):
        opens = self.df['open'].values
        closes = self.df['close'].values
        vols = self.df['Vol_30d'].values
        
        for i in range(1, len(self.df)):
            sigma_prev = vols[i-1]
            if np.isnan(sigma_prev):
                self.pnl_curve.append(self.account.equity)
                continue
                
            S_exec = opens[i]
            S_mtm = closes[i]
            IV_open = sigma_prev * 1.05
            IV_mtm = sigma_prev * 1.05
            
            if self.active_ic is not None:
                self.days_left -= 1
                T_exec = max(0.00001, self.days_left / 365.0)
                
                curr_liability = self.active_ic.get_unit_liability(S_exec, T_exec, IV_mtm, self.OPT_SLIPPAGE)
                
                if curr_liability <= (self.net_premium_received / self.qty) * (1 - self.take_profit) and self.days_left > 0:
                    opt_fee = 4 * S_exec * self.OPT_FEE_RATE * self.qty
                    self.account.add_pnl(self.net_premium_received - (curr_liability * self.qty) - opt_fee)
                    self.active_ic = None
                    self.qty = 0
                    
                elif self.days_left == 0:
                    net_payoff = self.active_ic.get_expiry_payoff(S_exec)
                    opt_fee = 4 * S_exec * self.OPT_FEE_RATE * self.qty
                    self.account.add_pnl(self.net_premium_received + (net_payoff * self.qty) - opt_fee)
                    self.active_ic = None
                    self.qty = 0

            if self.active_ic is None and sigma_prev > self.vol_filter:
                ic = IronCondorPortfolio(S_exec, self.tenor / 365.0, self.r, IV_open)
                net_unit_premium = ic.get_unit_premium(S_exec, self.OPT_SLIPPAGE)
                
                if net_unit_premium > 0:
                    max_loss_per_unit = ic.max_loss_per_unit(net_unit_premium)
                    if max_loss_per_unit > 0:
                        target_risk = self.account.equity * 0.20
                        self.qty = target_risk / max_loss_per_unit
                        self.net_premium_received = net_unit_premium * self.qty
                        opt_fee = 4 * S_exec * self.OPT_FEE_RATE * self.qty
                        
                        self.account.pay_fee(opt_fee)
                        self.active_ic = ic
                        self.days_left = self.tenor

            mtm_equity = self.account.equity
            if self.active_ic is not None:
                T_mtm = max(0.00001, self.days_left / 365.0)
                mtm_liability = self.active_ic.get_unit_liability(S_mtm, T_mtm, IV_mtm, slippage=0)
                opt_mtm = self.net_premium_received - (mtm_liability * self.qty)
                mtm_equity += opt_mtm
                
            self.pnl_curve.append(mtm_equity)
            
        pnl_series = pd.Series(self.pnl_curve, index=self.df.index)
        return pnl_series

def fetch_binance_data():
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    symbol = 'BTC/USDT'
    start_time_str = '2020-01-01T00:00:00Z'
    since = exchange.parse8601(start_time_str)
    all_klines = []
    while True:
        klines = exchange.fetch_ohlcv(symbol, '1d', since=since, limit=1000)
        if not klines: break
        all_klines.extend(klines)
        since = klines[-1][0] + 86400000
        if len(klines) < 1000: break
    df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    df.set_index('Date', inplace=True)
    return df

def main():
    print("Fetching data from Binance...")
    df = fetch_binance_data()
    df['LogRet'] = np.log(df['close'] / df['close'].shift(1))
    df['Vol_30d'] = df['LogRet'].rolling(window=30).std() * np.sqrt(365)
    
    print("Running OOP Backtest Engine with py_vollib...")
    engine = BacktestEngine(df)
    pnl_series = engine.run()
    
    peak = pnl_series.cummax()
    max_drawdown = ((peak - pnl_series) / peak).max()
    years = len(df) / 365.25
    total_return = pnl_series.iloc[-1] / 100000.0 - 1
    ann_return = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1
    
    plt.figure(figsize=(15, 9))
    plt.plot(df.index, pnl_series, label='OOP py_vollib Iron Condor', color='purple', linewidth=3)
    plt.title('Institutional Architecture: Object-Oriented py_vollib Engine')
    plt.xlabel('Date')
    plt.ylabel('Equity (USD)')
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'oop_equity_curve.png')
    plt.savefig(plot_path)
    
    print("\n--- OOP ARCHITECTURE METRICS ---")
    print(f"Final Equity: ${pnl_series.iloc[-1]:,.2f} (from $100k)")
    print(f"Annualized Return: {ann_return:.2%}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

if __name__ == "__main__":
    main()

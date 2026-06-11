# 🔮 加密货币期权与现货量化交易框架 (Quant Framework)

## 📖 1. 框架概述与目录规范
这是一个专为加密货币（特别是比特币）打造的、**原生支持双边摩擦成本计算与期权合成多头逻辑**的量化回测与实盘研发框架。整个框架的目录结构被设计为极具扩展性，方便未来接入数百个不同流派的精选策略。

### 核心目录结构
```text
e:\项目\期权回测\
├── selected_strategies/       # 🌟 核心资产：最后确定的精选策略库（每个策略独立建档）
│   ├── 01_Golden_Cross_Holy_Grail/  # V10 终极圣杯系统（长线趋势、空窗期生息）
│   │   ├── Holy_Grail_Engine.py
│   │   ├── Holy_Grail_Manual.md
│   │   └── *.png (收益图表)
│   ├── 02_Dual_Thrust_Trend/        # Dual Thrust 系统（中线波动率突破）
│   └── (未来预留的更多策略空间)
├── src/                       # ⚙️ 框架引擎代码
│   ├── vectorized_engine.py         # 框架核心回测引擎 (VectorizedBacktester)
│   ├── fetch_data.py                # Binance 数据获取模块
│   └── strategies/                  # 日常研发、试错、网格测算的草稿箱
├── data/                      # 📊 历史 K 线数据存储
├── results/                   # 📈 临时回测图表、日志输出
└── QUANT_FRAMEWORK_GUIDE.md   # 本文档（全局使用说明）
```

---

## 🛠️ 2. 最初的设置与使用方式

### 2.1 核心引擎架构
整个框架的基石是位于 `src/vectorized_engine.py` 的 `VectorizedBacktester` 类。它具有以下工业级特性：
- **防未来函数：** 引擎强制使用 `.shift(1)` 将所有的交易信号向后推迟一根 K 线执行，确保在 `t` 时刻产生的信号，只能在 `t+1` 时刻的开盘价（或收盘价）成交，绝对杜绝“看未来的价格做决定”的逻辑陷阱。
- **真实摩擦模拟：** 默认设置了 `0.0004` (千分之0.4) 的手续费和 `0.001` (千分之1) 的滑点。任何一进一出的交易都会被强制扣除双边摩擦，这对于高频策略起到了极强的“去伪存真”作用。

### 2.2 如何将新策略接入框架？
如果您未来想研发新的策略并放入精选库，只需遵循以下格式：

1. **编写策略函数**：函数必须接收一个包含 K 线数据的 Pandas DataFrame (包含 `open, high, low, close, volume`)。
2. **返回信号数组**：您的函数必须在最后计算出一列 `target_position`（`1` 代表做多，`0` 代表空仓，`-1` 代表做空），并返回它。
3. **调用引擎回测**：
```python
from vectorized_engine import VectorizedBacktester

def my_new_strategy(df):
    # ... 计算您的牛逼指标 ...
    df['target_position'] = np.where(buy_cond, 1, 0)
    return df['target_position']

tester = VectorizedBacktester(timeframe='1h', start_time='2021-01-01T00:00:00Z')
df_result, stats = tester.run_strategy(my_new_strategy)
```

---

## 🚀 3. 未来的改进与展望 (Future Outlook)

虽然目前的框架已经成功孵化出了“V10 终极圣杯”这样的战神级系统，但为了适应未来更加内卷的加密货币市场，整个框架在未来可以向以下几个维度进行深度升级：

### 展望一：接入期权定价模型 (Black-Scholes Options Engine)
- **现状：** 目前框架使用的是“合成多头（现货曲线 + 资金费率减免）”的等效模拟逻辑。
- **改进：** 未来可引入 `py_vollib` 等期权定价库，输入历史的 IV（隐含波动率）面数据，真实回测“备兑卖出 (Covered Call)” 收到多少期权费，以及“深度虚值期权 (OTM LEAPS)”的 Gamma 爆炸收益。

### 展望二：多因子动态组合 (Portfolio & Capital Allocation)
- **现状：** 目前各个精选策略（01_Holy_Grail, 02_Dual_Thrust）是独立运行的。
- **改进：** 在外层加一个 `PortfolioManager`，当 `01策略` 处于大熊市空窗期（持有大量 USDT）时，这部分资金可以被动态分配给 `02策略` 去做中短线的突破交易，从而实现资金利用率的极致最大化。

### 展望三：实盘 API 自动化对接
- **现状：** 目前为纯研究（Research）阶段的回测框架。
- **改进：** 为 `target_position` 数组编写一个实盘监听器 (Live Trader)。每当最新一根 K 线的 `t-1` 时刻 position 发生变化时，自动调用 Binance / Deribit 的 API 执行市价单，实现**“回测即实盘”**的无缝部署。

### 展望四：AI 与机器学习因子的引入
- 随着传统技术指标的拥挤，未来可以向框架的 `df` 中喂入外部数据源，例如：“链上大户转移数据 (On-chain data)”、“美联储利率决议时间特征”、“恐惧贪婪指数”，将其作为过滤假突破的 AI 因子。

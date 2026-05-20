# BTC-Pulse 项目概述与技术基础

---

## 一、项目概述

**项目背景：** BTC-Pulse 是一个开源的比特币趋势预测系统（v2.0，MIT 协议），旨在将量化投资方法论应用于加密货币市场分析。项目架构遵循经典的量化流水线：数据采集 → 存储（SQLite）→ 特征工程 → 模型/信号生成 → 可视化展示（Streamlit）。

**核心问题：** 加密货币市场噪声大、情绪驱动强、散户参与度高，传统单一指标难以有效捕捉趋势转折。BTC-Pulse 试图通过**多维度数据融合**解决这一问题——综合链上数据、市场情绪（恐惧贪婪指数）、衍生品指标（资金费率、未平仓量）、技术指标（RSI、布林带）和新闻舆情，构建更全面的市场判断框架。

**主要功能：**
1. **多源数据采集**：自动抓取 Binance 行情、Alternative.me 恐贪指数、衍生品数据和新闻情绪
2. **复合信号引擎**：60% 规则模型 + 40% XGBoost 机器学习，输出三类信号（看涨/中性/看跌）
3. **Freqtrade 风格回测**：支持 ROI 表、移动止损、动态仓位管理（Kelly/ATR/固定比例）
4. **统计严谨性验证**：含 Bootstrap 置信区间、缩水夏普比率（DSR）、随机入场基准蒙特卡洛检验
5. **交易成本建模**：0.1%/单边手续费，次根K线开盘价入场（防止前瞻偏差）
6. **可视化仪表盘**：Streamlit 实时展示信号、回测绩效和异常检测结果

**目标用户：**
- 对量化交易感兴趣的**个人投资者和学习者**
- 希望快速搭建 BTC 分析框架的**独立交易员**
- 学习回测方法论和策略评估的**量化入门者**

**应用价值：** 该项目的核心价值不在于提供"盈利策略"，而在于提供一套**方法论完整的量化分析框架**——从数据采集到统计检验，展示了如何严谨地评估一个交易策略是否具有真实优势（而非数据挖掘偏差）。特别是缩水夏普比率和随机基准检验的引入，帮助用户区分"真实alpha"与"运气"，具有较强的教育和实践参考意义。

---

## 二、场景、模块、技术路线与应用落地

### 2.1 面向场景

BTC-Pulse 面向**个人量化交易者和加密货币研究者**在以下场景中的需求：

1. **趋势研判场景**：比特币市场情绪极端（恐贪指数<20或>80）时，辅助判断是否为反转信号窗口。
2. **杠杆风险预警场景**：通过累计资金费率和未平仓量变化，提前识别衍生品市场过度拥挤（leverage purge）风险。
3. **回测验证场景**：量化策略开发者需要严谨的回测框架（含交易成本、前瞻偏差防护、统计显著性检验），避免"纸上谈兵"。
4. **参数优化场景**：策略参数（止损位、ROI目标、RSI阈值等）需要系统化搜索，而非手动调参。

### 2.2 核心功能模块（架构图）

```
┌─────────────────────────────────────────────────────────────────┐
│                        BTC-Pulse 系统架构                        │
├────────────┬────────────┬──────────────┬────────────┬───────────┤
│ 数据采集层  │  存储层     │  特征工程层   │  模型/信号层 │  展示层    │
│            │            │              │            │           │
│ market.py  │            │              │ signals.py │           │
│ (Bitcoinity│ init_db.py │  engine.py   │ (规则信号)  │  app.py   │
│  + CCXT)   │ (SQLite)   │ (技术指标+   │            │ (Streamlit│
│ sentiment  │            │  衍生品特征+  │ xgboost_   │  仪表盘)  │
│  .py (FNG) │            │  情绪特征+   │ model.py   │           │
│ derivatives│            │  前向标签)   │ (ML预测)   │           │
│  .py       │            │              │            │           │
│ news.py    │            │              │ anomaly.py │           │
│ (RSS)      │            │              │(异常检测)   │           │
└─────┬──────┴─────┬──────┴──────┬───────┴─────┬──────┴─────┬─────┘
      │            │             │             │            │
      ▼            ▼             ▼             ▼            ▼
  数据入库 ──→ SQLite表 ──→ 特征矩阵 ──→ 复合信号(60%规则+40%ML) ──→ 可视化
                                              │
                                    ┌─────────┴─────────┐
                                    ▼                   ▼
                            freqtrade_bridge/     models/backtest.py
                            ├─ strategy.py        (简单事件驱动回测)
                            ├─ backtester.py
                            │  (专业级回测引擎)
                            ├─ risk_manager.py
                            │  (Kelly/ATR仓位)
                            ├─ hyperopt.py
                            │  (Optuna贝叶斯优化)
                            └─ data_provider.py
```

**六大模块功能说明：**

| 模块 | 核心职责 | 关键技术点 |
|------|---------|-----------|
| **数据采集层**（4个collector） | 从免费公开源抓取行情、情绪、衍生品、新闻 | CCXT多交易所容灾（Binance→OKX→Bybit→Kraken）|
| **存储层** | SQLite 本地持久化，增量更新 | 表名白名单防SQL注入，`get_last_timestamp()`断点续传 |
| **特征工程层** | 计算MA/RSI/BB/VWAP/资金费率累计等50+特征 | 前向标签（forward_return_30d）用于ML训练 |
| **信号生成层** | 三模块规则信号 + XGBoost三分类 + Isolation Forest异常 | 复合评分 = 0.6×规则 + 0.4×ML，支持ML不可用时降级 |
| **回测引擎** | 两套引擎：简单版 + Freqtrade专业版 | 次根K线开盘入场、0.1%/边手续费、缩水夏普比率 |
| **展示层** | Streamlit实时仪表盘 | K线图+风险雷达图+ML置信度+历史相似度匹配 |

### 2.3 技术路线

**数据流水线：**
```
定时触发 run_ingest.py
    → Bitcoinity CSV下载（2010年至今日线）
    → Binance REST API（OHLCV + 资金费率 + OI）
    → Alternative.me API（恐贪指数）
    → RSS解析（CoinDesk/CoinTelegraph标题）
    → 写入SQLite（5张表：ohlcv/fng/funding/oi/news）
```

**信号生成流水线：**
```
FeatureEngine.run_pipeline()
    → 读取SQLite → 合并多表 → 计算技术指标（RSI/BB/MACD/ADX/StochRSI/OBV）
    → 计算衍生品特征（累计资金费率30d、OI价格背离）
    → 计算情绪特征（FNG移动平均、极端区间标记）
    → XGBoost walk-forward训练（滚动窗口，防止数据泄漏）
    → 三信号模块评分 → 加权融合 → 输出评级（Strong Buy ~ Strong Sell）
```

**回测验证流水线（Freqtrade风格）：**
```
BTCPulseStrategy.run_pipeline()
    → populate_indicators()（向量化计算全部指标）
    → populate_entry_trend()（5类入场条件 + 标签）
    → populate_exit_trend()（5类出场条件 + 标签）
    → FreqtradeBacktester.run()
        → 逐K线遍历，信号产生后次根K线开盘价入场
        → 入场扣0.1%手续费，RiskManager计算仓位（Kelly/ATR）
        → 每根K线检查：ROI表止盈 → 移动止损 → 动态止损 → 策略出场信号
        → 出场扣0.1%手续费
        → 构建日频权益曲线 → 计算Sharpe/Sortino/Calmar/最大回撤
        → Bootstrap 95%置信区间 → 缩水夏普比率(DSR) → 随机入场蒙特卡洛p值
```

**超参优化流水线：**
```
StrategyHyperopt.optimize(n_trials=100)
    → Optuna TPE采样器 → 建议参数组合（止损/ROI/RSI阈值/BB周期等）
    → 每组参数实例化策略 → 完整回测 → 返回目标值（Sharpe/Sortino/Calmar）
    → 贝叶斯更新 → 收敛到最优参数
    → DSR根据总试验次数自动调整（防止过拟合假象）
```

### 2.4 业务流程

```
用户日常使用流程：

[每日] run_ingest.py → 更新数据
           ↓
[每日] run_dashboard.py → 查看仪表盘
           ↓
       ┌── 信号为 Strong Buy/Sell？──→ 参考决策
       │
       └── 需要验证策略？
                ↓
           run_strategy_backtest.py → 查看回测报告
                ↓
           DSR > 0.95 且 p-value < 0.05？
                ↓                    ↓
              是：策略有统计边际    否：策略可能是数据挖掘产物
                ↓
           run_strategy_backtest.py --hyperopt → 参数优化
                ↓
           apply_best_params() → 部署优化后策略
```

### 2.5 预期成果形式

| 成果类型 | 具体形式 | 说明 |
|---------|---------|------|
| **原型系统** | Streamlit Web仪表盘 | 可本地部署运行，实时展示BTC多维度分析面板，含K线图、风险雷达、ML预测、异常检测 |
| **算法模型** | XGBoost三分类器 + Isolation Forest | 滚动训练的趋势预测模型，输出看涨/中性/看跌概率；异常检测模型输出市场regime变化警报 |
| **回测分析报告** | BacktestResult结构化输出 | 含总收益、CAGR、Sharpe（附95% CI）、最大回撤、胜率（附CI）、逐笔交易明细（含入场/出场原因标签）、缩水夏普比率通过/失败判定、随机基准p值 |
| **策略优化方案** | HyperoptResult JSON文件 | 最优参数组合（止损值、ROI表、RSI/FNG阈值）、Top-10试验对比、可直接加载复用 |
| **解决方案框架** | 完整Python代码仓库（MIT协议） | 从数据采集到统计检验的端到端量化分析框架，可扩展至ETH/SOL等多币种，可作为教学或二次开发基础 |
| **展示材料** | run_compare_strategies.py对比报告 | 三策略横评（纯规则 vs 纯ML vs 复合信号），附随机入场基准对照，直观展示信号融合的增益效果 |

---

## 三、团队已有的数据技术基础

### 3.1 数据采集与工程化能力

团队已构建完整的多源异构数据采集框架：通过 CCXT 统一接口对接 Binance/OKX/Bybit/Kraken 四大交易所（含自动容灾降级），集成 Bitcoinity CSV 历史数据回溯至 2010 年，接入 Alternative.me 情绪 API 和 CoinDesk/CoinTelegraph RSS 新闻源。采集层具备断点续传（`get_last_timestamp` 增量更新）、指数退避重试（`utils/retry.py`）、表名白名单防注入等工程化实践，数据落地至 SQLite 本地库，具备从原始数据到可分析状态的端到端 pipeline 能力。

### 3.2 特征工程与信号建模能力

特征工程模块（`features/engine.py`）可批量计算 50+ 技术/衍生品/情绪特征（MA/RSI/BB/MACD/ADX/StochRSI/OBV/VWAP偏离/累计资金费率/OI价格背离等），支持多周期融合（日线+周线 forward-fill 防前瞻偏差）。信号层实现了三模块规则引擎（反向情绪、杠杆清洗、机构基准）与 XGBoost 三分类器的加权融合（60%规则 + 40%ML），具备 ML 不可用时的优雅降级机制。团队还集成了 Isolation Forest 异常检测和 LLM 新闻情绪分析（支持 DeepSeek/Ollama），展现了传统统计方法与深度学习的综合运用能力。

### 3.3 量化回测与统计验证能力

团队具备专业级回测基础设施：实现了 Freqtrade 风格的逐K线 walk-forward 回测引擎，内含 ROI 表止盈、移动/动态止损、交易成本建模（0.1%/边）、次根K线开盘入场（消除前瞻偏差）、Kelly/ATR/固定比例三种仓位管理及回撤熔断机制。统计验证层（`utils/statistics.py`）实现了日频权益曲线风险指标（Sharpe/Sortino/Calmar）、Bootstrap 10,000 次重采样置信区间、缩水夏普比率（DSR，Bailey & López de Prado, 2014）和随机入场蒙特卡洛假设检验，能够区分真实 alpha 与数据挖掘偏差。

### 3.4 工具链与协作基础

项目采用 Python 生态主流工具栈（Pandas/NumPy/SciPy/Optuna/XGBoost/Loguru/Streamlit），配备 pytest 测试套件（含 23 项统计模块单测），使用 Git 版本管理，代码结构清晰（采集/存储/特征/模型/回测/展示六层分离），具备良好的可扩展性和团队协作基础。

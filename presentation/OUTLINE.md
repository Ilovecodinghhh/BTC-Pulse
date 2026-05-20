# BTC-Pulse 路演 PPT 大纲

> 12页 · 对标五大评审维度 · 暗色简洁风格

---

## Slide 1 — 封面 `slide_01_title.png`

- **标题：** BTC-Pulse — 多维度比特币趋势预测系统
- **副标题：** 多Agent协作 · 量化回测 · 统计验证 · 隐私合规
- **版本信息：** v2.0 | MIT License | 100% 免费数据源
- **讲述要点：** 一句话定位 — "用严谨的量化方法论分析比特币市场"

---

## Slide 2 — 痛点分析 `slide_02_problem.png`

**对应维度：创新性（场景新颖度）**

- **痛点一 · 情绪驱动：** 散户恐慌/贪婪时非理性决策，单一指标无法全面判断
- **痛点二 · 回测失真：** 忽略交易成本和前瞻偏差，策略上线后表现远低于预期
- **痛点三 · 过拟合幻觉：** 参数优化后Sharpe很高，但无法区分alpha与数据挖掘
- **结论引出：** BTC-Pulse = 多维数据融合 + 严谨统计验证 + 全链路闭环

---

## Slide 3 — 系统架构 `slide_03_architecture.png`

**对应维度：展示表达（架构清晰度）**

- **五层流水线：** 数据采集 → 存储 → 特征工程 → 模型信号 → 展示
- 数据采集：Bitcoinity · CCXT · FNG · RSS（全免费）
- 存储：SQLite增量更新
- 特征工程：50+技术/衍生品/情绪指标
- 模型信号：规则60% + ML40%，XGBoost + LLM
- 展示：Streamlit实时仪表盘
- **底层支撑：** Freqtrade Bridge 专业回测层（Strategy · Backtester · RiskManager · Hyperopt · DataProvider）

---

## Slide 4 — 创新性 `slide_04_innovation.png`

**对应维度：创新性 25%**

- **创新点1：** 多维数据融合 — 链上+情绪+衍生品+技术面+新闻NLP，打破单一指标局限
- **创新点2：** 规则+ML混合信号 — 60%可解释规则+40%XGBoost泛化，ML不可用时优雅降级
- **创新点3：** 统计严谨性验证 — 缩水Sharpe + Bootstrap CI + 随机基准MC，从"看起来赚钱"到"统计上显著"
- **创新点4：** 开发流程创新 — Coder+Reviewer双Agent对抗协作，7轮迭代评分≥9/10方可通过
- **核心差异化：** 传统回测只报收益率，BTC-Pulse要求 DSR>0.95 + p<0.05 + CI不含零 才承认"可能有真实edge"

---

## Slide 5 — Agent使用深度 `slide_05_agent_depth.png`

**对应维度：Agent使用深度 25%**

- **Layer 1 开发智能体：** Coder ↔ Reviewer 对抗迭代，评分≥9/10才通过，共7轮
- **Layer 2 分析智能体：** LLM情感 + XGBoost预测 + Isolation Forest异常，多信号加权融合
- **Layer 3 决策智能体：** Kelly仓位 + ATR止损 + 回撤熔断，Optuna贝叶斯优化
- **Layer 4 验证智能体：** Bootstrap CI + DSR校正 + MC基准，三重统计把关
- **关键区分：**
  - ✗ 浅层 = "让AI写一段代码"
  - ✓ 深层 = Agent间对抗协作 · 多模型融合决策 · 统计验证闭环 · 全流程覆盖

---

## Slide 6 — 实用价值 `slide_06_practical_value.png`

**对应维度：实用价值 25%**

- **解决的真实问题：**
  - 零成本数据（全免费公开API）
  - 一键部署（pip install + run_ingest + run_dashboard）
  - 回测可信（交易成本 + 防前瞻 + 统计检验 = 真实绩效）
  - 智能融合（规则可解释 + ML泛化 + LLM语义理解）
  - 参数优化（Optuna自动搜索 + DSR防过拟合校正）
- **孵化潜力：**
  - SaaS化 → 订阅制信号推送（Telegram/Discord）
  - 多币种 → ETH/SOL扩展，跨资产相关性信号
  - API化 → REST API对外输出信号
  - 教育 → 量化投资教学平台
  - 社区 → 策略市场

---

## Slide 7 — 核心功能演示 `slide_07_signal_pipeline.png`

**对应维度：展示表达（Demo完整性）**

- **数据流：** 价格/FNG/资金费率/新闻 → 特征工程 → 双模型（规则60%+XGBoost40%）→ 复合信号 → 统计验证
- **统计验证层：** Freqtrade回测引擎 · 0.1%/边手续费 · 次根K线入场 · Bootstrap CI · DSR · MC p-value
- **核心原则：** 每个信号经过 融合→回测→统计检验 三重验证，无单一Agent输出被直接信任

---

## Slide 8 — 回测报告Demo `slide_08_backtest_demo.png`

**对应维度：展示表达 + 创新性**

- **左侧 · 绩效指标：** 总交易数/胜率/胜率CI/总收益/Sharpe/Sharpe CI/最大回撤/Calmar/手续费
- **右侧 · 统计验证（本项目独有）：**
  - 缩水Sharpe (DSR) = 0.972 → > 0.95 = PASS
  - 随机基准 p-value = 0.023 → < 0.05 = Significant
  - 平均收益 CI = [0.8%, 2.1%] → 不含零 = Significant
- **对比：** 传统回测只报告收益率和Sharpe，本系统额外提供3项统计显著性检验

---

## Slide 9 — 技术栈 `slide_09_tech_stack.png`

**对应维度：展示表达（技术完备性）**

- **数据层：** CCXT · Bitcoinity · feedparser · SQLite
- **计算层：** Pandas/NumPy · SciPy · XGBoost · scikit-learn
- **优化层：** Optuna · Bootstrap(10K) · DSR(Bailey 2014) · Monte Carlo
- **应用层：** Streamlit · Loguru · pytest(23测试) · Git
- 全栈Python · 100%免费数据源 · MIT开源

---

## Slide 10 — 隐私合规 `slide_10_privacy.png`

**对应维度：隐私合规意识 10%**

- **设计原则：** 数据最小化 · 本地优先 · 密钥隔离 · 可审计
- **风险识别与应对：**
  - LLM API传输 → 仅传公开新闻标题，支持本地Ollama
  - API Key泄露 → .gitignore + example模板 + 环境变量
  - 数据库暴露 → 本地文件 + 表名白名单防注入
  - 日志敏感信息 → Loguru不记录密钥或个人信息
- **数据分类表：** 全部4类数据均为公开市场数据，零PII

---

## Slide 11 — 发展路线 `slide_11_roadmap.png`

**对应维度：实用价值（孵化潜力）**

- **已完成：** 多源采集 · 复合信号 · 回测引擎 · 统计验证 · 仪表盘 · 双Agent开发
- **短期目标：** Telegram信号推送 · FreqAI重训练 · Docker部署 · Paper Trading
- **长期愿景：** 多币种扩展 · REST API · 强化学习Agent · 社区策略市场

---

## Slide 12 — 总结 `slide_12_summary.png`

- **Slogan：** 从"看起来赚钱"到"统计上显著"
- **五维总结：**
  - 创新性 → 多维融合 + 三重统计验证闭环
  - Agent深度 → 四层Agent协作（开发→分析→决策→验证）
  - 实用价值 → 零成本部署 · 真实回测 · 可孵化SaaS
  - 展示表达 → 完整Demo · 架构图 · 回测报告
  - 隐私合规 → 零PII设计 · 本地优先 · 密钥隔离
- **GitHub：** github.com/Ilovecodinghhh/BTC-Pulse

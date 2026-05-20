# BTC-Pulse Agent 设计深度说明

> 本文档阐述 BTC-Pulse 中 AI Agent 的多层次调用设计

---

## Agent 架构总览

BTC-Pulse 的 Agent 体系不是单点调用，而是**多层级、多角色协作**的智能体系统：

```
┌────────────────────────────────────────────────────┐
│              Agent 协作架构                          │
│                                                    │
│  Layer 1: 开发智能体 (Development Agents)           │
│  ┌──────────┐    ┌──────────┐                      │
│  │ Coder    │◄──►│ Reviewer │  反复迭代至9/10分     │
│  │ Agent    │    │ Agent    │  才允许合并代码        │
│  └──────────┘    └──────────┘                      │
│       │                                            │
│       ▼                                            │
│  Layer 2: 分析智能体 (Analysis Agents)              │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐          │
│  │ LLM情感  │ │ XGBoost  │ │ Isolation │          │
│  │ 分析Agent│ │ 预测Agent│ │ Forest    │          │
│  └─────┬────┘ └─────┬────┘ │ 异常Agent │          │
│        │            │      └─────┬─────┘          │
│        ▼            ▼            ▼                 │
│  ┌─────────────────────────────────────┐           │
│  │ 复合信号融合 (60% Rules + 40% ML)   │           │
│  └─────────────────────────────────────┘           │
│       │                                            │
│       ▼                                            │
│  Layer 3: 决策智能体 (Decision Agents)              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Risk     │ │ Position │ │ Hyperopt │           │
│  │ Manager  │ │ Sizer    │ │ Optimizer│           │
│  └──────────┘ └──────────┘ └──────────┘           │
│                                                    │
│  Layer 4: 验证智能体 (Validation Agents)            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Bootstrap│ │ Deflated │ │ Monte    │           │
│  │ CI Agent │ │ Sharpe   │ │ Carlo    │           │
│  └──────────┘ └──────────┘ └──────────┘           │
└────────────────────────────────────────────────────┘
```

---

## 深度调用而非浅层使用

### 1. 开发阶段：双Agent代码质量控制

项目采用 **Coder + Reviewer 双Agent协作模式**进行代码开发：
- **Coder Agent**：根据量化专家审查意见，编写修复代码
- **Reviewer Agent**：对每份代码打分（满分10分），评估正确性、完整性、风格
- **质量门禁**：只有评分 ≥ 9/10 才允许通过，否则返回 Coder 重写
- 实际开发中进行了 **7轮迭代**，2个文件因评分不足被打回重做

这不是简单的"让AI写代码"，而是**Agent间的对抗性协作**——Reviewer 的严格标准迫使 Coder 输出高质量代码。

### 2. 运行时：多模型多角色融合

| Agent角色 | 实现模块 | 能力调用深度 |
|-----------|---------|-------------|
| **LLM情感分析** | `models/llm_sentiment.py` | 结构化 prompt → JSON 输出 → 情感分数/叙事标签/置信度 |
| **XGBoost预测** | `models/xgboost_model.py` | Walk-forward训练 → 三分类概率输出 → 与规则信号加权融合 |
| **异常检测** | `models/anomaly.py` | Isolation Forest 无监督学习 → regime变化预警 |
| **风险管理** | `freqtrade_bridge/risk_manager.py` | Kelly公式/ATR自适应仓位 → 回撤熔断 → 动态止损 |
| **参数优化** | `freqtrade_bridge/hyperopt.py` | Optuna贝叶斯搜索 → DSR过拟合校正 → 最优参数推荐 |

### 3. 验证阶段：统计Agent确保结论可靠

- **Bootstrap Agent**：10,000次重采样生成Sharpe/胜率的95%置信区间
- **DSR Agent**：根据总试验次数校正Sharpe（防止hyperopt过拟合假象）
- **Monte Carlo Agent**：生成10,000个随机入场策略，计算p值判断策略是否有真实edge

### 4. 关键设计：Agent间的信息流与协作

```
新闻标题 ──→ LLM Agent ──→ 情感分数 ──┐
市场数据 ──→ 规则Agent ──→ 规则信号 ──┤
特征矩阵 ──→ XGBoost Agent ──→ ML信号 ──┼→ 复合评分 ──→ 回测Agent
异常特征 ──→ IF Agent ──→ 异常标记 ──┘        │
                                              ▼
                                    统计验证Agent群
                                    (Bootstrap + DSR + MC)
                                              │
                                              ▼
                                    最终结论：策略是否有edge？
```

这种设计确保了**没有任何单一Agent的输出被直接信任**——每个信号都经过融合、回测和统计检验的三重验证。

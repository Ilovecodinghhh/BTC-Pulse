# BTC-Pulse — Project Proposal

## Multi-Dimensional Bitcoin Trend Prediction System

**Repository:** [github.com/Ilovecodinghhh/BTC-Pulse](https://github.com/Ilovecodinghhh/BTC-Pulse)
**Version:** 2.0
**License:** MIT
**Date:** May 12, 2026

---

## 1. Executive Summary

BTC-Pulse is an open-source Bitcoin analysis and signal system that fuses **on-chain data, derivatives metrics, sentiment analysis, and machine learning** into actionable trend ratings. It does not execute trades — it surfaces when market conditions are interesting and explains *why*, empowering traders and researchers with transparent, multi-dimensional intelligence.

Version 2.0 integrates battle-tested architectural patterns from [Freqtrade](https://github.com/freqtrade/freqtrade), adding a proper strategy framework, professional-grade walk-forward backtesting, and Bayesian hyperparameter optimization — all as standalone components with no Freqtrade dependency.

---

## 2. Problem Statement

Retail and independent Bitcoin traders face several challenges:

1. **Information overload** — Price action, on-chain metrics, derivatives data, sentiment, and news each tell part of the story, but synthesizing them manually is slow and error-prone.
2. **Backtesting fragility** — Most DIY signal systems lack rigorous backtesting, leading to overfitted or untested strategies going live.
3. **Opaque signals** — Commercial signal services are black boxes. Users pay for alerts without understanding the reasoning.
4. **Cost barriers** — Professional-grade data and tooling often require expensive API subscriptions.

BTC-Pulse addresses all four by combining free public data sources, transparent rule+ML blending, and Freqtrade-caliber backtesting into a single, hackable system.

---

## 3. Objectives

| # | Objective | Success Metric |
|---|-----------|---------------|
| 1 | Provide a **unified composite signal** blending rules (60%) and ML (40%) | Composite signal generated daily with confidence score |
| 2 | Deliver **professional backtesting** with walk-forward validation | Sharpe, Sortino, Calmar, max drawdown, per-trade tagging |
| 3 | Enable **automated hyperparameter optimization** | Optuna-driven search over stoploss, ROI, thresholds |
| 4 | Keep **all core data free** — zero mandatory paid APIs | Full functionality with only public endpoints |
| 5 | Offer a **real-time Streamlit dashboard** for visualization | Interactive charts, risk radar, anomaly detection panel |
| 6 | Maintain **full transparency** — every signal is tagged and explainable | Entry/exit tags trace back to specific rule conditions |

---

## 4. System Architecture

### 4.1 Data Layer (Collectors)

| Module | Source | Data | Cost |
|--------|--------|------|------|
| **Market Data** | Bitcoinity CSV (primary) + Exchange OHLCV via CCXT (secondary) | Daily price & volume back to 2010 | Free |
| **Derivatives** | Binance + OKX + Bybit (CCXT public) | Funding rate (multi-exchange avg), open interest | Free |
| **Sentiment** | Alternative.me Fear & Greed Index | Contrarian extremes detection | Free |
| **News Headlines** | CoinDesk + CoinTelegraph RSS | Narrative analysis via LLM | Free |
| **AI Layer** | XGBoost + DeepSeek/Ollama LLM | ML trend prediction + narrative sentiment | Free (Ollama) or ~$0.001/call (DeepSeek) |
| **Anomaly Detection** | Isolation Forest | Regime change / black swan alerts | Free |

**Geo-resilience:** If Binance is unavailable, the system automatically falls back through OKX → Bybit → Kraken. Bitcoinity data is always available regardless of region.

### 4.2 Feature Engineering Pipeline

- Moving averages: 7, 14, 30, 90, 200-day
- RSI, Bollinger Bands (with squeeze detection), MACD, ADX, Stochastic RSI, OBV
- EMA crossovers (30/90)
- VWAP deviation as institutional benchmark
- Cumulative funding rate analysis
- Multi-timeframe merging (daily + weekly) with forward-fill (no look-ahead bias)

### 4.3 Signal Generation

**Composite signal = Rule-based modules (60%) + XGBoost ML prediction (40%)**

If the ML model is unavailable, the system gracefully degrades to rules-only mode.

#### Entry Signals (Buy)

| # | Signal | Key Conditions | Tag |
|---|--------|---------------|-----|
| 1 | Fear Reversal | FNG < 20 + RSI < 35 + price below BB lower + MACD hist rising | `fear_oversold_reversal` |
| 2 | Squeeze Breakout | BB squeeze (bottom 10% width) + price above mid + ADX > 20 + volume surge | `squeeze_breakout` |
| 3 | EMA Cross | EMA30 crosses above EMA90 + volume > 1.3x avg + healthy RSI | `ema_cross_bullish` |
| 4 | Funding Squeeze | Negative funding + RSI oversold + StochRSI low + OBV above avg | `negative_funding_squeeze` |
| 5 | VWAP Breakout | Price crosses above monthly VWAP + volume confirmation + trending ADX | `vwap_breakout` |

#### Exit Signals (Sell)

| # | Signal | Key Conditions | Tag |
|---|--------|---------------|-----|
| 1 | Greed Overbought | FNG > 85 + RSI > 75 + price above BB upper | `greed_overbought` |
| 2 | Bearish EMA Cross | EMA30 crosses below EMA90 | `ema_cross_bearish` |
| 3 | MACD Divergence | MACD crosses below signal + RSI > 60 | `macd_bearish_cross` |
| 4 | OI Divergence | Price up but open interest dropping | `oi_divergence` |
| 5 | Leverage Overweight | Cumulative funding > 2% + RSI elevated + weak volume | `leverage_overweight` |

### 4.4 Risk Management

| Method | Description |
|--------|-------------|
| Fixed Fractional | Risk 2% of capital per trade |
| Kelly Criterion | Optimal sizing based on win rate and payoff ratio (half-Kelly for safety) |
| ATR-Based | Volatility-adaptive — smaller positions in volatile markets |
| Drawdown Circuit Breaker | Halt trading at -15% drawdown from peak |
| Dynamic Stoploss | ATR-based stops that tighten as profit grows |
| Trailing Stop | Lock in gains once 5% profit reached (trail at 2.5%) |

### 4.5 Freqtrade-Inspired Components

BTC-Pulse does not run Freqtrade — it reimplements proven patterns as standalone modules in `freqtrade_bridge/`:

| Freqtrade Concept | BTC-Pulse Implementation | Rationale |
|---|---|---|
| `IStrategy` interface | `BaseStrategy` + `BTCPulseStrategy` | Clean, testable strategy pattern |
| `populate_indicators()` | Vectorized indicator pipeline | Battle-tested crypto indicator patterns |
| `populate_entry/exit_trend()` | Multi-condition entry/exit with tags | Know exactly why each trade triggered |
| `minimal_roi` table | Time-decay profit targets | Auto-exit when target met for trade duration |
| Trailing stoploss | Configurable trailing with offset | Lock in gains without premature exits |
| `custom_stoploss()` | ATR-based dynamic stops | Adapt to current volatility |
| Hyperopt | Optuna Bayesian optimization | Find optimal params automatically |
| DataProvider | Centralized data access + multi-timeframe | Weekly context for daily signals |
| Backtesting engine | Walk-forward with per-trade metrics | Sharpe, Sortino, Calmar, max drawdown |
| Position sizing | RiskManager (Kelly/ATR/fixed) + circuit breaker | Realistic backtesting simulations |

---

## 5. Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ML | XGBoost, scikit-learn (Isolation Forest) |
| LLM | DeepSeek API or Ollama (local) |
| Optimization | Optuna (Bayesian hyperparameter search) |
| Exchange Data | CCXT (unified crypto exchange library) |
| Dashboard | Streamlit |
| Database | SQLite |
| Configuration | YAML |
| Logging | Loguru |
| Testing | pytest |
| Package Management | pip / pyproject.toml |

---

## 6. Project Structure

```
BTC-Pulse/
├── collectors/          # Data ingestion (WRITE layer)
│   ├── market.py        # Bitcoinity + Exchange OHLCV
│   ├── sentiment.py     # Fear & Greed Index
│   ├── derivatives.py   # Funding rate, OI, liquidations
│   └── news.py          # RSS headlines
├── database/
│   └── init_db.py       # SQLite schema + connection management
├── features/
│   └── engine.py        # Feature engineering pipeline
├── models/
│   ├── signals.py       # Blended signal generator (rules + ML)
│   ├── xgboost_model.py # XGBoost trend classifier
│   ├── anomaly.py       # Isolation Forest anomaly detection
│   ├── llm_sentiment.py # LLM-powered narrative analysis
│   └── backtest.py      # Simple event-driven backtester
├── freqtrade_bridge/    # Freqtrade-inspired standalone modules
│   ├── strategy.py      # BaseStrategy + BTCPulseStrategy
│   ├── backtester.py    # Walk-forward backtester
│   ├── hyperopt.py      # Optuna-based optimization
│   ├── risk_manager.py  # Position sizing (Kelly/ATR/fixed)
│   └── data_provider.py # Multi-timeframe READ layer + caching
├── dashboard/
│   └── app.py           # Streamlit dashboard
├── tests/
│   └── test_core.py     # pytest test suite
├── utils/
│   ├── config.py        # YAML config loader
│   ├── logging.py       # Loguru setup
│   └── retry.py         # Exponential backoff retry
├── run_ingest.py        # Daily data collection pipeline
├── run_backtest.py      # Original backtest + XGBoost training
├── run_strategy_backtest.py  # Freqtrade-style backtest + hyperopt
├── run_dashboard.py     # Launch Streamlit dashboard
├── pyproject.toml       # Package metadata + CLI entry points
├── config.yaml.example  # Configuration template
└── requirements.txt     # Python dependencies
```

---

## 7. Installation & Deployment

### 7.1 Local Setup

```bash
git clone https://github.com/Ilovecodinghhh/BTC-Pulse.git
cd BTC-Pulse
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Or: pip install -e .
```

### 7.2 Configuration

```bash
cp config.yaml.example config.yaml
# Edit config.yaml — all core data is FREE, no paid keys needed
# Optional: set ai_provider to "deepseek" or "ollama" for LLM sentiment
```

### 7.3 Data Ingestion

```bash
python run_ingest.py
# Fetches historical + latest data, sets up SQLite automatically
```

### 7.4 Future: Docker Compose

One-command deployment via Docker Compose is on the roadmap for simplified setup.

---

## 8. Roadmap

### Phase 1 — Core (✅ Complete)
- Multi-source data collection (market, derivatives, sentiment, news)
- Feature engineering pipeline
- XGBoost trend classifier with rules+ML blending
- Isolation Forest anomaly detection
- LLM-powered news sentiment
- Streamlit dashboard

### Phase 2 — Strategy Framework (✅ Complete)
- Freqtrade-inspired strategy interface
- Walk-forward backtester with ROI/trailing stoploss
- Optuna hyperparameter optimization
- Risk management (Kelly, ATR, circuit breaker)
- Multi-timeframe data merging
- Entry/exit tagging

### Phase 3 — Alerts & Live Operation (Planned)
- Live signal alerts via Telegram/Discord
- Paper trading mode (forward-test without capital)
- FreqAI-style self-adaptive retraining (rolling-window XGBoost)
- REST API for signal integration

### Phase 4 — Advanced Intelligence (Planned)
- Multi-asset support (ETH, SOL) with cross-asset correlation
- On-chain data integration (Glassnode/CryptoQuant: exchange flows, MVRV, NUPL, whale wallets)
- Reinforcement learning agent to replace/complement rule-based logic
- Orderbook imbalance signals
- Ensemble model voting with consensus gating
- Volatility regime classification (trending/ranging/crisis mode switching)

### Phase 5 — Community & Ecosystem (Planned)
- Docker Compose deployment
- Backtesting report export (HTML/PDF)
- Community strategy repository (share and import strategies)
- Walk-forward cross-validation for overfitting detection

---

## 9. Target Users

| Persona | Use Case |
|---------|----------|
| **Independent traders** | Daily signal dashboard for BTC position management |
| **Quantitative researchers** | Backtesting framework for strategy development and validation |
| **Crypto students / educators** | Learning tool for multi-factor analysis, ML in finance |
| **Algorithmic traders** | Strategy prototyping with Freqtrade-familiar patterns |
| **Data scientists** | Feature engineering and model experimentation on crypto data |

---

## 10. Competitive Advantages

1. **Completely free data pipeline** — No mandatory paid API subscriptions; all core data from public endpoints.
2. **Transparent signal logic** — Every entry/exit is tagged with the exact rule that triggered it; no black box.
3. **Graceful degradation** — ML unavailable? Falls back to rules-only. Binance blocked? Falls back to OKX/Bybit/Kraken.
4. **Professional backtesting without Freqtrade** — Gets the architecture benefits without the dependency or complexity.
5. **Hackable and extensible** — Clean strategy interface makes it easy to add new signals, data sources, or ML models.
6. **Multi-dimensional** — Combines technical, sentiment, derivatives, and narrative data rather than relying on price alone.

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| API endpoint changes (Bitcoinity, exchanges) | Data pipeline breaks | Multi-source fallback chain; CCXT abstracts exchange differences |
| Overfitting in ML model | False confidence in backtest results | Walk-forward validation, hyperopt with out-of-sample testing |
| Regulatory changes to exchange APIs | Data access restricted | Geo-resilient fallback (Binance → OKX → Bybit → Kraken) |
| Users treating signals as financial advice | Financial loss, liability | Clear disclaimers; educational framing; MIT license |
| LLM sentiment hallucinations | Noisy signal component | ML weight capped at 40%; graceful fallback to rules-only |

---

## 12. Disclaimer

BTC-Pulse is for **research and educational purposes only**. It is not financial advice. Cryptocurrency trading carries significant risk. Past performance of any signal system does not guarantee future results. Always do your own research.

---

## 13. References

- [Freqtrade](https://github.com/freqtrade/freqtrade) — Strategy interface, backtesting patterns, hyperopt concepts
- [CCXT](https://github.com/ccxt/ccxt) — Unified crypto exchange API
- [Optuna](https://github.com/optuna/optuna) — Bayesian hyperparameter optimization
- [Alternative.me](https://alternative.me/crypto/fear-and-greed-index/) — Fear & Greed Index
- [XGBoost](https://github.com/dmlc/xgboost) — Gradient boosting framework
- [Streamlit](https://streamlit.io/) — Dashboard framework

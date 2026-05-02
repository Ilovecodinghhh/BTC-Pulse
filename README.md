# 📊 BTC-Pulse 2.0

**Multi-Dimensional Bitcoin Trend Prediction System**
*Now with Freqtrade-inspired strategy engine, professional backtesting, and hyperparameter optimization.*

---

## What Is BTC-Pulse?

BTC-Pulse is an open-source Bitcoin analysis and signal system that fuses **on-chain data, derivatives metrics, sentiment analysis, and machine learning** into actionable trend ratings. It doesn't trade for you — it tells you *when conditions are interesting* and *why*.

Version 2.0 integrates battle-tested concepts from [Freqtrade](https://github.com/freqtrade/freqtrade) — the leading open-source crypto trading bot — giving BTC-Pulse a proper strategy framework, professional-grade backtesting, and automated parameter optimization.

---

## ✨ Key Features

### Core Analysis Engine — 100% Free Data Sources
| Module | Source | What It Does | Cost |
|--------|--------|-------------|------|
| **Market Data** | Binance OHLCV (CCXT public) | Price, volume, moving averages, VWAP | **Free** |
| **Derivatives** | Binance + OKX + Bybit (CCXT public) | Funding rate (multi-exchange avg), open interest | **Free** |
| **Sentiment** | Alternative.me Fear & Greed | Contrarian extremes detection | **Free** |
| **AI Layer** | XGBoost + DeepSeek/Ollama LLM | ML trend prediction + narrative sentiment | **Free** (Ollama) or ~$0.001/call (DeepSeek) |
| **Anomaly Detection** | Isolation Forest | Regime change / black swan alerts | **Free** |

### Freqtrade-Inspired Strategy Engine *(NEW in v2.0)*
| Feature | Inspired By | Description |
|---------|-------------|-------------|
| **Strategy Interface** | Freqtrade `IStrategy` | Clean `populate_indicators()` → `populate_entry_trend()` → `populate_exit_trend()` pipeline |
| **Technical Indicators** | Freqtrade + TA-Lib patterns | RSI, Bollinger Bands (squeeze detection), MACD, ADX, Stochastic RSI, OBV, EMA crossovers |
| **Walk-Forward Backtester** | Freqtrade Backtesting | ROI table, trailing stoploss, dynamic stoploss, per-trade tagging, equity curve |
| **Hyperparameter Optimization** | Freqtrade Hyperopt | Optuna-powered Bayesian search over strategy params (stoploss, ROI, thresholds) |
| **Risk Management** | Freqtrade stake/position sizing | Kelly Criterion, ATR-based sizing, drawdown circuit breaker |
| **Multi-Timeframe** | Freqtrade informative pairs | Daily + weekly data merging with forward-fill (no look-ahead) |
| **Data Provider** | Freqtrade DataProvider | Centralized data access, caching, staleness checks, orderbook data |
| **Entry/Exit Tagging** | Freqtrade enter/exit tags | Know *why* each trade was entered/exited for strategy refinement |

### Signal Modules (Original BTC-Pulse)
- **🎭 Contrarian Sentiment** — Buy extreme fear, sell extreme greed (validated historically)
- **⚡ Leverage Purge** — Detect overweight positioning via cumulative funding
- **🏛️ Institutional Benchmark** — VWAP deviation as smart-money proxy

### Dashboard
- Streamlit-powered real-time dashboard
- Candlestick charts with MAs and VWAP overlay
- Risk radar (leverage, sentiment, volatility, VWAP deviation, MA deviation)
- Historical similarity analysis (find past months matching current conditions)
- Prediction history log

---

## 🏗️ Architecture

```
BTC-Pulse/
├── collectors/              # Data ingestion
│   ├── market.py            # Binance OHLCV
│   ├── sentiment.py         # Fear & Greed Index
│   └── derivatives.py       # Funding rate, OI, liquidations
├── database/
│   └── init_db.py           # SQLite schema + connection management
├── features/
│   └── engine.py            # Feature engineering pipeline
├── models/
│   ├── signals.py           # Rule-based signal generator
│   ├── xgboost_model.py     # XGBoost trend classifier
│   ├── anomaly.py           # Isolation Forest anomaly detection
│   ├── llm_sentiment.py     # LLM-powered narrative analysis
│   └── backtest.py          # Original simple backtester
├── freqtrade_bridge/        # ★ NEW — Freqtrade-inspired modules
│   ├── strategy.py          # BaseStrategy + BTCPulseStrategy
│   ├── backtester.py        # Walk-forward backtester (ROI/trailing SL)
│   ├── hyperopt.py          # Optuna-based parameter optimization
│   ├── risk_manager.py      # Position sizing (Kelly/ATR/fixed)
│   └── data_provider.py     # Multi-timeframe data access layer
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── utils/
│   ├── config.py            # YAML config loader
│   ├── logging.py           # Loguru setup
│   └── retry.py             # Exponential backoff retry
├── run_ingest.py            # Daily data collection pipeline
├── run_backtest.py          # Original backtest runner
├── run_strategy_backtest.py # ★ NEW — Freqtrade-style backtest + hyperopt
├── run_dashboard.py         # Launch Streamlit dashboard
├── config.yaml.example      # Configuration template
└── requirements.txt         # Python dependencies
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/smartcoreissolutions/BTC-Pulse.git
cd BTC-Pulse
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config.yaml.example config.yaml
# Edit config.yaml:
#   All core data is FREE — no paid API keys needed!
#   Optional: set ai_provider to "deepseek" or "ollama" for LLM sentiment
#   Optional: set ai_api_key for DeepSeek (or leave blank for Ollama local)
```

### 3. Ingest Data

```bash
python run_ingest.py
```

This fetches historical + latest data from Binance, Fear & Greed Index, and derivatives sources. Sets up the SQLite database automatically.

### 4. Run the Dashboard

```bash
python run_dashboard.py
# Opens at http://localhost:8501
```

### 5. Run Backtests

**Original backtest (simple event-driven):**
```bash
python run_backtest.py
```

**Freqtrade-style backtest (walk-forward, ROI, trailing SL):**
```bash
python run_strategy_backtest.py
python run_strategy_backtest.py --days 365   # Last year only
```

**Hyperparameter optimization:**
```bash
python run_strategy_backtest.py --hyperopt --trials 100 --objective sharpe
python run_strategy_backtest.py --hyperopt --trials 50 --objective profit_factor
```

---

## 🧠 Strategy Deep Dive

### Entry Signals (5 types, all require confluence)

| # | Signal | Conditions | Tag |
|---|--------|-----------|-----|
| 1 | **Fear Reversal** | FNG < 20 + RSI < 35 + price below Bollinger lower + MACD hist rising | `fear_oversold_reversal` |
| 2 | **Squeeze Breakout** | Bollinger squeeze (bottom 10% width) + price above mid + ADX > 20 + volume surge | `squeeze_breakout` |
| 3 | **EMA Cross** | EMA30 crosses above EMA90 + volume > 1.3x avg + RSI in healthy range | `ema_cross_bullish` |
| 4 | **Funding Squeeze** | Negative funding rate + RSI oversold + StochRSI low + OBV above average | `negative_funding_squeeze` |
| 5 | **VWAP Breakout** | Price crosses above monthly VWAP + volume confirmation + trending ADX | `vwap_breakout` |

### Exit Signals (5 types)

| # | Signal | Conditions | Tag |
|---|--------|-----------|-----|
| 1 | **Greed Overbought** | FNG > 85 + RSI > 75 + price above Bollinger upper | `greed_overbought` |
| 2 | **Bearish EMA Cross** | EMA30 crosses below EMA90 | `ema_cross_bearish` |
| 3 | **MACD Divergence** | MACD crosses below signal line + RSI > 60 | `macd_bearish_cross` |
| 4 | **OI Divergence** | Price up but open interest dropping (bearish divergence) | `oi_divergence` |
| 5 | **Leverage Overweight** | Cumulative funding > 2% + RSI elevated + weak volume | `leverage_overweight` |

### Risk Management

| Method | Description |
|--------|-------------|
| **Fixed Fractional** | Risk 2% of capital per trade |
| **Kelly Criterion** | Optimal sizing based on win rate and payoff ratio (half-Kelly for safety) |
| **ATR-Based** | Volatility-adaptive sizing — smaller positions in volatile markets |
| **Drawdown Circuit Breaker** | Halt trading at -15% drawdown from peak |
| **Dynamic Stoploss** | ATR-based stops that tighten as profit grows |
| **Trailing Stop** | Lock in gains once 5% profit reached (trail at 2.5%) |

---

## 📈 What Freqtrade Concepts Were Integrated

BTC-Pulse doesn't *run* Freqtrade — it *learns from* it. Here's what we borrowed and adapted:

| Freqtrade Concept | BTC-Pulse Implementation | Why |
|--------------------|--------------------------|----|
| `IStrategy` interface | `BaseStrategy` + `BTCPulseStrategy` | Clean, testable strategy pattern instead of scattered logic |
| `populate_indicators()` | Vectorized indicator pipeline (RSI, BB, MACD, ADX, StochRSI, OBV) | Freqtrade's indicator patterns are battle-tested on crypto |
| `populate_entry/exit_trend()` | Multi-condition entry/exit with tags | Know exactly *why* each trade triggered |
| `minimal_roi` table | Time-decay profit targets | Auto-exit when profit target met for trade duration |
| Trailing stoploss | Configurable trailing with offset | Lock in gains without premature exits |
| `custom_stoploss()` callback | ATR-based dynamic stops | Adapt to current volatility instead of fixed % |
| Hyperopt | Optuna Bayesian optimization | Find optimal stoploss, ROI, thresholds automatically |
| DataProvider | Centralized data access + multi-timeframe | Weekly data as informative context for daily signals |
| Backtesting engine | Walk-forward with per-trade metrics | Sharpe, Sortino, Calmar, max drawdown, entry tag stats |
| FreqAI concepts | Self-adaptive retraining pattern (roadmap) | Retrain models on recent data to adapt to regime changes |

---

## 🔮 Improvement Roadmap

### High-Priority Enhancements

1. **Live Signal Alerts (Telegram/Discord)**
   Push notifications when the composite signal flips. The infrastructure exists — just needs a notification dispatcher.

2. **FreqAI-Style Self-Adaptive Retraining**
   Automatically retrain XGBoost on a rolling window during live operation. Freqtrade's FreqAI does this on a background thread — same pattern works here.

3. **Multi-Asset Support**
   Extend beyond BTC to ETH, SOL, and correlated assets. Use cross-asset correlation as an additional signal (divergence = opportunity).

4. **On-Chain Data Integration**
   Add Glassnode/CryptoQuant metrics: exchange inflows/outflows, MVRV ratio, NUPL, whale wallet movements. These are the signals institutions actually use.

5. **Reinforcement Learning Agent**
   Replace rule-based entry/exit with an RL agent trained on the backtest environment. Freqtrade supports this via FreqAI+Stable-Baselines — same approach applies.

### Medium-Priority

6. **Paper Trading Mode**
   Forward-test strategies in real-time without capital at risk. Log every decision for review.

7. **Walk-Forward Cross-Validation**
   Split historical data into train/validate windows to detect overfitting before live deployment.

8. **Orderbook Imbalance Signals**
   The DataProvider already supports orderbook fetching. Use bid/ask imbalance as a short-term signal.

9. **Ensemble Model Voting**
   Run multiple strategy variants simultaneously, weight their signals, and only trade when consensus agrees.

10. **Volatility Regime Classification**
    Detect whether the market is in trending/ranging/crisis mode and switch strategy parameters accordingly (Freqtrade's `informative_pairs` + regime-specific ROI tables).

### Nice-to-Have

11. **Grafana/Prometheus Monitoring** — Real-time system health metrics
12. **Docker Compose Deployment** — One-command setup with all services
13. **REST API** — Expose signals via API for integration with other tools
14. **Backtesting Report Export** — HTML reports with trade-by-trade analysis
15. **Community Strategy Repository** — Share and import strategies (like Freqtrade's strategy repo)

---

## ⚙️ Configuration Reference

```yaml
# config.yaml
api_keys:
  # All core data uses FREE public APIs — no keys needed!
  binance_public: ""     # No key required
  okx_public: ""         # No key required
  bybit_public: ""       # No key required

  # AI Sentiment — pick one:
  ai_provider: "deepseek"   # "deepseek" | "ollama" | "openai"
  ai_api_key: ""             # DeepSeek key, or blank for Ollama
  ai_base_url: "https://api.deepseek.com"  # or http://localhost:11434/v1

data:
  db_path: "data/btc_pulse.db"
  symbol: "BTC/USDT"
  exchange: "binance"

features:
  ma_periods: [7, 14, 30, 90, 200]
  fng_extreme_low: 20
  fng_extreme_high: 80

signals:
  fng_buy_threshold: 15
  fng_sell_threshold: 85
  funding_sell_threshold: 0.02

model:
  xgboost:
    n_estimators: 200
    max_depth: 5
    learning_rate: 0.05
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-signal`)
3. Test with the backtest engine before submitting
4. Open a PR with backtest results showing improvement

---

## ⚠️ Disclaimer

BTC-Pulse is for **research and educational purposes only**. It is not financial advice. Cryptocurrency trading carries significant risk. Past performance of any signal system does not guarantee future results. Always do your own research.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Freqtrade](https://github.com/freqtrade/freqtrade) — Strategy interface, backtesting patterns, and hyperopt concepts
- [ccxt](https://github.com/ccxt/ccxt) — Unified crypto exchange API
- [Optuna](https://github.com/optuna/optuna) — Bayesian hyperparameter optimization
- [Alternative.me](https://alternative.me/crypto/fear-and-greed-index/) — Fear & Greed Index
- [Binance](https://www.binance.com/) — Market and derivatives data

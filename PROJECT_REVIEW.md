# BTC-Pulse — Project Summary & Review

> Reviewed: 2026-05-03

---

## What Is BTC-Pulse?

BTC-Pulse is a **Bitcoin monthly trend prediction system** that combines multiple data sources with machine learning to generate trading signals. It's designed as a self-contained research/analysis tool that runs locally, collects free public data, and outputs a daily "traffic light" rating (bullish / neutral / bearish).

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION                            │
│  collectors/market.py     → Binance OHLCV (daily, via CCXT)      │
│  collectors/sentiment.py  → Fear & Greed Index (Alternative.me)  │
│  collectors/derivatives.py→ Funding Rate + OI (Binance/OKX/Bybit)│
└───────────────────────────────┬──────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                         STORAGE                                   │
│  database/init_db.py → SQLite (WAL mode)                         │
│  Tables: market_price, derivatives, sentiment, ai_sentiment,     │
│          features, predictions                                    │
└───────────────────────────────┬──────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                            │
│  features/engine.py                                              │
│  - Moving averages (7/14/30/90/200-day)                          │
│  - VWAP deviation                                                │
│  - Volatility (30d annualized)                                   │
│  - FNG rolling averages & extreme flags                          │
│  - Cumulative funding rate (30d)                                 │
│  - OI-price divergence                                           │
│  - Forward 30-day return labels (for supervised training)        │
└───────────────────────────────┬──────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                      MODELS / SIGNALS                             │
│  models/signals.py       → Rule-based: 3 modules (Contrarian,    │
│                            Leverage Purge, Institutional VWAP)    │
│  models/xgboost_model.py → XGBoost classifier (bearish/neutral/  │
│                            bullish) on features                   │
│  models/anomaly.py       → Isolation Forest black-swan detector  │
│  models/llm_sentiment.py → LLM-powered text sentiment            │
│                            (DeepSeek / Ollama / OpenAI)           │
│  models/backtest.py      → Simple event-driven backtest engine   │
└───────────────────────────────┬──────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                  FREQTRADE BRIDGE (v2 layer)                     │
│  freqtrade_bridge/strategy.py       → IStrategy-style base class │
│  freqtrade_bridge/backtester.py     → Walk-forward backtest with │
│                                       ROI table, trailing stop    │
│  freqtrade_bridge/hyperopt.py       → Optuna-based param tuning  │
│  freqtrade_bridge/risk_manager.py   → Kelly / ATR position sizing│
│  freqtrade_bridge/data_provider.py  → Multi-timeframe data layer │
└───────────────────────────────┬──────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                       PRESENTATION                               │
│  dashboard/app.py → Streamlit dashboard                          │
│    - Traffic-light trend rating                                  │
│    - Candlestick + MA + VWAP chart                               │
│    - Risk radar (polar plot)                                     │
│    - Historical similarity finder                                │
│    - Prediction history table                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Entry Points (How to Run It)

| Script | Purpose |
|--------|---------|
| `run_ingest.py` | Daily pipeline: collect data → compute features → generate signals → export CSV snapshots |
| `run_backtest.py` | Train XGBoost + anomaly detector, run simple rule-based backtest |
| `run_strategy_backtest.py` | Run the Freqtrade-style backtest (optional `--hyperopt`) |
| `run_dashboard.py` | Launch Streamlit dashboard on port 8501 |

---

## Key Design Decisions

1. **All free data** — no paid APIs required. Binance public OHLCV, Alternative.me FNG, and Binance public funding rate endpoints.
2. **SQLite as the single data store** — simple, no infra overhead, WAL mode for concurrency.
3. **Incremental collection** — each collector checks the last stored timestamp and only fetches new data.
4. **Dual signal systems** — original rule-based modules + a newer Freqtrade-inspired strategy layer that adds RSI, Bollinger, MACD, ADX, etc.
5. **Multi-provider LLM** — sentiment analysis supports DeepSeek, Ollama (free local), or OpenAI via the same OpenAI SDK interface.
6. **Forward-looking labels** — the feature engine computes 30-day forward returns for supervised learning, which is standard for this kind of research.

---

## Things That Are Strange / Noteworthy / Could Be Improved

### 🔴 Concerning Issues

| # | Issue | Details |
|---|-------|---------|
| 1 | **Typo in class name** | `FreqtadeBacktester` (missing 'r') in `freqtrade_bridge/backtester.py` — should be `FreqtradeBacktester` |
| 2 | **SQL injection risk** | `get_last_timestamp()` uses f-string interpolation for the table name: `f"SELECT MAX(timestamp) FROM {table}"`. Not user-facing, but still risky practice. |
| 3 | **`pct_change(fill_method=None)`** | In `features/engine.py`, passing `fill_method=None` to `pct_change` — this parameter was deprecated in pandas 2.1 and removed in 2.2. Will crash on newer pandas. |
| 4 | **No `__main__.py` or proper CLI** | No unified entry point. Each script does `sys.path.insert(0, ...)` to fix imports — fragile; should use a proper package with `setup.py`/`pyproject.toml`. |
| 5 | **XGBoost overfitting warning threshold is backwards** | The code warns if accuracy > 0.85, but for a 3-class problem with class imbalance that's a reasonable bar. The real red flag would be if train accuracy >> test accuracy, which isn't checked. |
| 6 | **`inserted` counter in `MarketCollector.store()` counts attempts, not successes** | The counter increments even if the INSERT was ignored by `INSERT OR IGNORE`. |

### 🟡 Design Oddities

| # | Issue | Details |
|---|-------|---------|
| 7 | **Two completely independent backtesting systems** | `models/backtest.py` (simple) and `freqtrade_bridge/backtester.py` (advanced) coexist with no shared interface. The simple one is never called by the advanced one. |
| 8 | **Freqtrade bridge doesn't actually connect to Freqtrade** | Despite the name, it's a reimplementation of Freqtrade patterns, not a bridge/adapter to actual Freqtrade. Could confuse users expecting actual Freqtrade integration. |
| 9 | **`RiskManager` is never used anywhere** | It's defined but no runner script or strategy instantiates it. Orphaned code. |
| 10 | **`DataProvider` duplicates what collectors already do** | The data provider has its own `_fetch_live_ohlcv` and DB loading logic that overlaps with `MarketCollector`. Two sources of truth. |
| 11 | **XGBoost + rule-based signals don't feed into each other** | The XGBoost model predicts trend separately; the rule-based `SignalGenerator` produces its own composite. They're never combined into a final consensus signal. |
| 12 | **`vectorbt` in requirements but never imported** | Listed in `requirements.txt` but not used anywhere in the codebase. The backtest uses a manual loop instead. |
| 13 | **LLM sentiment has no data source** | `LLMSentiment.analyze()` requires text input, but nothing in the pipeline fetches news/tweets to feed it. It's only usable if you manually pass text. |
| 14 | **Dashboard imports `AnomalyDetector` and `XGBoostCombiner` but doesn't use them** | They're imported at the top of `dashboard/app.py` but never called in the dashboard logic. |
| 15 | **`ai_sentiment_score` is a feature column but never populated** | `table_features` has an `ai_sentiment_score` column, but `FeatureEngine` never reads from `table_ai_sentiment` to populate it. Dead column. |
| 16 | **Hardcoded `2020-01-01` start date** | All collectors default to 2020-01-01 on first run. Not configurable. |

### 🟢 Minor / Style

| # | Issue | Details |
|---|-------|---------|
| 17 | **No tests whatsoever** | No `tests/` directory, no pytest, no CI. |
| 18 | **No `pyproject.toml` or `setup.py`** | Can't `pip install -e .` for development. |
| 19 | **`.gitignore` has duplicate entries** | `data/`, `logs/`, `config.yaml`, `*.db` are each listed twice. |
| 20 | **No type hints on some key return values** | `collect()` methods return `pd.DataFrame` (typed) but `run()` methods return ambiguous `int` or `dict`. |
| 21 | **Emoji in code comments** | Not harmful but the `[UP]`/`[DOWN]`/`[--]` placeholders in `signals.py` suggest emoji was intended but not rendered. |
| 22 | **`config.yaml.example` leaks architecture assumptions** | The example mentions "Freqtrade-inspired" in the requirements section — could confuse users into thinking Freqtrade needs to be installed. |
| 23 | **No graceful handling if Binance is geo-blocked** | Users in restricted regions will get silent failures. No fallback exchange logic in `MarketCollector` (only derivatives has multi-exchange). |

---

## Verdict

This is a **well-structured research prototype** with clear separation of concerns (collect → store → featurize → model → present). The code is readable and well-commented. However, it has the hallmarks of a project that grew organically:

- The Freqtrade bridge is a significant addition (~40% of the codebase) that partially duplicates existing functionality without being fully integrated.
- Several components (risk manager, vectorbt, LLM sentiment) are scaffolded but not wired into the main pipeline.
- No testing or packaging infrastructure.

**If this were to go to production**, the priority would be: unify the two backtest systems, wire the XGBoost prediction into the composite signal, add proper packaging (`pyproject.toml`), and write at least integration tests for the ingestion pipeline.

---

*Generated by code review on 2026-05-03.*

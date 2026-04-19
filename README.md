# BTC-Pulse 1.0

**多维数据驱动月度趋势预测系统 / Multi-Dimensional Data-Driven Monthly Trend Prediction System**

BTC-Pulse 1.0 is a multi-dimensional, data-driven Bitcoin monthly trend prediction system. It integrates market data, derivatives, macro sentiment, and on-chain indicators through a local SQLite data hub. The system covers three core analytical modules — Contrarian Sentiment, Leverage Purge, and Institutional Benchmark — enhanced by XGBoost and LLM-based AI signal filtering, all presented through a Streamlit dashboard.

## Architecture

```
BTC-Pulse/
├── database/          # SQLite schema & DB management
├── collectors/        # Data ingestion from APIs
├── features/          # Feature engineering & preprocessing
├── models/            # XGBoost, anomaly detection, NLP sentiment
├── dashboard/         # Streamlit visualization
├── utils/             # Shared helpers (logging, config, retry)
├── data/              # SQLite DB files & CSV snapshots
├── logs/              # Execution logs
├── config.yaml        # API keys & settings
├── requirements.txt   # Python dependencies
├── run_ingest.py      # Daily ingestion entry point
├── run_backtest.py    # Offline backtest entry point
└── run_dashboard.py   # Streamlit dashboard launcher
```

## Core Data Sources

| Dimension | API Source | Key Data | Cost |
|-----------|-----------|----------|------|
| Market | Binance (CCXT) | Daily OHLCV, Volume | Free |
| Derivatives | Coinglass API | OI, Funding Rate, Liquidations | Basic tier |
| Sentiment | Alternative.me | Fear & Greed Index | Free |
| On-chain | CryptoQuant | Exchange Netflow | Partial free |

## Three Core Modules

1. **Contrarian Sentiment** — Analyzes FNG extremes (<20 / >80) and their 30-day forward returns
2. **Leverage Purge** — Detects overcrowded leverage via cumulative funding rate + OI-price divergence
3. **Institutional Benchmark** — Uses monthly VWAP as bull/bear boundary with breakout-retest logic

## AI Enhancement Layer

- **XGBoost Signal Combiner** — Learns optimal feature weights automatically
- **LLM Sentiment Engine** — Scores analyst text for narrative-driven signals
- **Anomaly Detection** — Isolation Forest for black swan early warning

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp config.yaml.example config.yaml
# Edit config.yaml with your keys

# 3. Initialize database
python -m database.init_db

# 4. Run first data ingestion
python run_ingest.py

# 5. Run backtest
python run_backtest.py

# 6. Launch dashboard
streamlit run run_dashboard.py
```

## Roadmap

- **Phase 1**: Data infrastructure — SQLite hub + multi-source ingestion
- **Phase 2**: Correlation research — Heatmaps + feature importance
- **Phase 3**: Signal generator — Rule-based + ML-enhanced signals + backtesting

## Warnings

- **Overfitting**: Crypto history is short. Perfect backtest = something is wrong.
- **GIGO**: Data gaps in funding rate / OI will invalidate predictions.
- **Black Box**: AI is a *filter*, not the sole decision-maker.

## License

MIT

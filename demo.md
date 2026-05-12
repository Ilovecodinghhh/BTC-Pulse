# BTC-Pulse — Demo Walkthrough

A step-by-step demo showing how to set up, run, and interpret BTC-Pulse end-to-end.

**Repository:** [github.com/Ilovecodinghhh/BTC-Pulse](https://github.com/Ilovecodinghhh/BTC-Pulse)

---

## Demo Scenario

You're a trader who wants to:

1. Install BTC-Pulse on a laptop
2. Ingest historical Bitcoin data back to 2020
3. View the current composite signal in the dashboard
4. Run a Freqtrade-style backtest over the last 365 days
5. Optimize strategy parameters automatically
6. Inspect the entry/exit tags to understand each trade

Estimated total time: **~30 minutes** (most of which is the initial data ingest).

---

## Prerequisites

- Python 3.10 or newer
- Git
- ~500 MB free disk space (data + dependencies)
- Internet connection (for data ingest)
- **No paid API keys required** — all core data sources are free

Optional:

- DeepSeek API key (~$0.001/call) **or** Ollama installed locally — for LLM-powered news sentiment

---

## Step 1: Clone & Install

```bash
git clone https://github.com/Ilovecodinghhh/BTC-Pulse.git
cd BTC-Pulse

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
# or, editable install:
# pip install -e .
```

**Expected output:** Dependencies install cleanly. You should see `xgboost`, `ccxt`, `streamlit`, `optuna`, `loguru`, `pandas`, etc.

---

## Step 2: Configure

```bash
cp config.yaml.example config.yaml
```

Open `config.yaml` and review:

```yaml
api_keys:
  # All core data uses FREE public APIs — no keys needed!
  ai_provider: "deepseek"          # "deepseek" | "ollama" | "openai"
  ai_api_key: ""                   # DeepSeek key, or blank for Ollama
  ai_base_url: "https://api.deepseek.com"

data:
  db_path: "data/btc_pulse.db"
  symbol: "BTC/USDT"
  exchange: "binance"              # Falls back to OKX/Bybit/Kraken
  start_date: "2020-01-01"

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

**Demo choice:** Leave `ai_api_key` blank to skip LLM sentiment, or point `ai_base_url` at `http://localhost:11434/v1` for Ollama.

---

## Step 3: Run Initial Data Ingest

```bash
python run_ingest.py
```

**What happens behind the scenes:**

1. **Bitcoinity CSV** — Downloads daily BTC price & volume back to 2010 (or your configured `start_date`).
2. **Exchange OHLCV via CCXT** — Pulls daily candles from Binance. If Binance is geo-blocked, automatically falls back to OKX → Bybit → Kraken.
3. **Fear & Greed Index** — Fetches historical and current sentiment from Alternative.me.
4. **Derivatives data** — Funding rates and open interest from Binance + OKX + Bybit (averaged).
5. **News headlines** — Pulls RSS feeds from CoinDesk and CoinTelegraph; if an LLM provider is configured, runs narrative sentiment scoring.
6. **Database** — Initializes SQLite schema at `data/btc_pulse.db` and writes everything in.

**Expected console output (abridged):**

```
[INFO] Initializing SQLite at data/btc_pulse.db
[INFO] Fetching Bitcoinity historical CSV...
[INFO] Loaded 5,978 daily rows (2010-07-17 → 2026-05-11)
[INFO] Fetching OHLCV from binance for BTC/USDT...
[INFO] Loaded 2,323 daily candles
[INFO] Fetching Fear & Greed Index...
[INFO] Loaded 2,510 FNG records
[INFO] Fetching funding rates (binance, okx, bybit)...
[INFO] Averaged funding across 3 exchanges
[INFO] Fetching news headlines (CoinDesk, CoinTelegraph)...
[INFO] LLM sentiment disabled (ai_api_key blank)
[INFO] Ingest complete in 87.3s
```

---

## Step 4: Launch the Dashboard

```bash
python run_dashboard.py
```

Your browser opens at **http://localhost:8501**.

### What You'll See

The Streamlit dashboard has several panels:

#### 4.1 Composite Signal Panel (top)

- **Today's Rating:** e.g. `🟢 BULLISH (confidence 0.72)`
- **Component breakdown:**
  - Rules-based score: 0.65 (60% weight)
  - XGBoost ML prediction: 0.82 (40% weight)
  - **Blended composite: 0.72**
- **Active entry tags:** `squeeze_breakout`, `ema_cross_bullish`
- **Active exit tags:** *(none)*

#### 4.2 Candlestick Chart

- Daily BTC/USDT candles
- Overlay: MA7, MA30, MA90, MA200
- Overlay: Monthly VWAP line (institutional benchmark)
- Volume histogram below

#### 4.3 Risk Radar

A radar chart showing five normalized risk dimensions:

- **Leverage** (cumulative funding rate)
- **Sentiment** (Fear & Greed extremes)
- **Volatility** (ATR / Bollinger width)
- **VWAP deviation** (distance from monthly VWAP)
- **MA deviation** (distance from 200-day MA)

Each axis is colored: green (calm), yellow (caution), red (elevated risk).

#### 4.4 Anomaly Detection Panel

- **Isolation Forest status:** `Normal` | `Anomaly detected`
- Shows recent anomaly scores; spikes indicate potential regime change.

#### 4.5 Historical Similarity Analysis

- Searches the database for past months whose feature vectors most closely match the current state.
- Example: "Today most resembles 2019-04 (similarity 0.89) and 2023-01 (similarity 0.84)."
- Useful for narrative context: "What happened next in those analogs?"

#### 4.6 Prediction History Log

A scrollable table of recent daily predictions, composite scores, active tags, and the realized BTC return over the following N days.

---

## Step 5: Run a Backtest

### 5.1 Original Backtest (Simple Event-Driven)

```bash
python run_backtest.py
```

This runs the basic event-driven backtester and trains the XGBoost model on historical data. Quick and useful for first-pass validation.

### 5.2 Freqtrade-Style Backtest (Recommended)

```bash
python run_strategy_backtest.py --days 365
```

**Walks forward day-by-day** over the last 365 days, simulating the `BTCPulseStrategy` with:

- ROI table (time-decay profit targets)
- Trailing stoploss (activates at +5% profit, trails at 2.5%)
- ATR-based dynamic stoploss
- Position sizing via RiskManager (default: half-Kelly)
- Drawdown circuit breaker at -15%

**Expected output (abridged):**

```
==================== BTC-Pulse Strategy Backtest ====================
Period:               2025-05-12 → 2026-05-11 (365 days)
Initial capital:      $10,000
Final capital:        $13,847
Total return:         +38.47%
Sharpe ratio:         1.82
Sortino ratio:        2.64
Calmar ratio:         3.11
Max drawdown:         -12.4%
Total trades:         27
Win rate:             63.0%
Profit factor:        2.41
Avg trade duration:   8.2 days

---------- Entry Tag Performance ----------
squeeze_breakout            8 trades | win 75% | avg +6.2%
ema_cross_bullish           7 trades | win 71% | avg +4.8%
fear_oversold_reversal      6 trades | win 67% | avg +5.1%
vwap_breakout               4 trades | win 50% | avg +2.3%
negative_funding_squeeze    2 trades | win 50% | avg +1.7%

---------- Exit Tag Performance ----------
roi_target_hit             14 trades | avg +5.4%
trailing_stop              7 trades  | avg +3.1%
greed_overbought           3 trades  | avg +7.2%
ema_cross_bearish          2 trades  | avg -1.8%
dynamic_stoploss           1 trade   | avg -3.2%
```

Notice how every trade is tagged — you can see exactly which signals are performing and which need refinement.

---

## Step 6: Hyperparameter Optimization

```bash
python run_strategy_backtest.py --hyperopt --trials 100 --objective sharpe
```

**Optuna runs a Bayesian search** over the strategy's tunable parameters (stoploss, ROI targets, FNG thresholds, etc.), optimizing for Sharpe ratio.

**Expected output (abridged):**

```
[INFO] Starting Optuna study: btc_pulse_hyperopt
[INFO] Trials: 100 | Objective: sharpe
[Trial   1/100] Sharpe: 1.45  |  params: stoploss=-0.08, roi_0=0.05, fng_buy=18
[Trial   2/100] Sharpe: 1.67  |  params: stoploss=-0.06, roi_0=0.04, fng_buy=15
[Trial   3/100] Sharpe: 1.23  |  params: stoploss=-0.12, roi_0=0.08, fng_buy=22
...
[Trial 100/100] Sharpe: 2.14  |  params: stoploss=-0.07, roi_0=0.045, fng_buy=14

==================== Best Trial ====================
Sharpe ratio: 2.14
Parameters:
  stoploss:       -0.07
  roi_0:           0.045
  roi_30:          0.025
  roi_60:          0.010
  trailing_offset: 0.025
  fng_buy:         14
  fng_sell:        86
====================================================
```

You can also optimize for other objectives:

```bash
python run_strategy_backtest.py --hyperopt --trials 50 --objective profit_factor
python run_strategy_backtest.py --hyperopt --trials 50 --objective calmar
```

---

## Step 7: Run the Test Suite

```bash
pip install pytest
pytest tests/ -v
```

**Expected output:**

```
tests/test_core.py::test_feature_engine_computes_indicators PASSED
tests/test_core.py::test_contrarian_signal_fires_on_extreme_fear PASSED
tests/test_core.py::test_composite_signal_blends_rules_and_ml PASSED
tests/test_core.py::test_database_initialization_creates_tables PASSED
tests/test_core.py::test_backtest_generates_signals PASSED
========== 5 passed in 3.42s ==========
```

---

## Step 8: Reading the Signals — A Worked Example

Suppose the dashboard shows:

```
🟢 BULLISH (confidence 0.74)
  Rules score: 0.68
  ML score:    0.83
  Composite:   0.74

Active entry tags:
  • fear_oversold_reversal
  • squeeze_breakout

Risk radar:
  Leverage:        🟢 low
  Sentiment:       🔴 extreme fear (FNG=12)
  Volatility:      🟡 elevated
  VWAP deviation:  🟢 near VWAP
  MA deviation:    🟡 -8% below MA200

Anomaly status: Normal
```

**Interpretation:**

1. **Sentiment is at extreme fear (FNG=12)** — historically a strong contrarian buy signal.
2. **Two entry signals fire simultaneously** — `fear_oversold_reversal` (extreme fear + oversold RSI + below BB lower + rising MACD) and `squeeze_breakout` (Bollinger squeeze releasing upward).
3. **Risk radar** shows leverage is low (no overcrowded longs to flush) and the price is near VWAP (not stretched).
4. **ML model agrees** at 0.83 confidence.
5. **Composite of 0.74** is a strong bullish reading.

A trader using BTC-Pulse as a research tool might:

- Allocate per the RiskManager's Kelly suggestion
- Set a stoploss at ATR-based dynamic level
- Plan to trail once +5% profit is reached
- Watch for `greed_overbought` or `leverage_overweight` exit tags

---

## Step 9: Going Further

Now that the basic demo works, you can:

- **Customize signals** by editing `freqtrade_bridge/strategy.py` and adding new entry/exit conditions with tags.
- **Add new data sources** by creating a collector in `collectors/` (e.g. Glassnode, CryptoQuant).
- **Train a custom XGBoost model** with `run_backtest.py` after adding new features in `features/engine.py`.
- **Schedule daily ingest** via cron:
  ```cron
  0 1 * * * cd /path/to/BTC-Pulse && /path/to/venv/bin/python run_ingest.py
  ```
- **Run hyperopt overnight** to discover optimal parameters for current market regime.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `binance` returns geo-block error | System auto-falls back to OKX/Bybit/Kraken; Bitcoinity data always works |
| Dashboard shows "No data" | Run `python run_ingest.py` first |
| ML predictions missing from composite | Train the model: `python run_backtest.py` |
| LLM sentiment disabled | Either blank `ai_api_key` (skip), set DeepSeek key, or point at Ollama (`http://localhost:11434/v1`) |
| Hyperopt very slow | Lower `--trials` or use `--objective profit_factor` (faster than `sharpe`) |
| `pytest` not found | `pip install pytest` |

---

## Disclaimer

This demo and BTC-Pulse itself are for **research and educational purposes only**. Nothing here is financial advice. Cryptocurrency trading carries significant risk and you can lose your entire capital. Past performance does not guarantee future results. Do your own research.

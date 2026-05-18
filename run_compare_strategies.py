#!/usr/bin/env python3
"""
Strategy Comparison — visualize equity curves for all BTC-Pulse approaches.

Compares three strategies:
1. Rules-Only (original BacktestEngine: FNG + funding rate signals)
2. Freqtrade-Style (BTCPulseStrategy: technical + sentiment + derivatives)
3. Composite (Rules 60% + XGBoost 40% blended signal)

Outputs:
- Equity curve comparison plot (PNG)
- Metrics summary table
- Monthly returns heatmaps

Usage:
    python run_compare_strategies.py [--days 730] [--output plots/comparison.png]
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from database.init_db import init_database, get_connection
from models.backtest import BacktestEngine
from models.xgboost_model import XGBoostCombiner
from freqtrade_bridge.strategy import BTCPulseStrategy
from freqtrade_bridge.backtester import FreqtradeBacktester
from utils.logging import setup_logger
from utils.statistics import (
    daily_returns_from_equity,
    annualized_sharpe,
    max_drawdown as compute_max_drawdown,
    random_entry_baseline,
    strategy_vs_random_pvalue,
)

logger = setup_logger("compare_strategies")

# Fee per side (0.1% = Binance spot taker). Round-trip = 0.2%.
FEE_PCT = 0.001


# ═══════════════════════════════════════════════════════════════
# Strategy 1: Rules-Only (original BTC-Pulse)
# ═══════════════════════════════════════════════════════════════

def run_rules_backtest(days: int = None) -> dict:
    """
    Run the original rule-based backtest.

    Fixed vs original:
    - Entry on next candle's open (not signal candle's close)
    - Fees deducted on entry and exit (0.1% per side)
    - Sharpe from daily equity curve (not per-trade returns)
    """
    engine = BacktestEngine(fee_pct=FEE_PCT)
    df = engine.load_data()
    df = engine.generate_signals(df)

    if df.empty:
        return {"equity": [], "dates": [], "metrics": {}, "name": "Rules-Only"}

    df = df.dropna(subset=["close"]).reset_index(drop=True)

    if days and not df.empty:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    capital = 10000.0
    equity = [capital]
    dates = [df.iloc[0]["timestamp"]]
    position = 0
    entry_price = 0.0
    entry_idx = 0
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]

        if position == 1:
            pnl = (row["close"] - entry_price) / entry_price
            days_held = i - entry_idx

            exit_reason = None
            if pnl <= engine.stoploss:
                exit_reason = "stoploss"
            elif row["sell_signal"] == 1:
                exit_reason = "signal"
            elif days_held >= engine.profit_exit_days and pnl > 0:
                exit_reason = "profit_time_exit"
            elif days_held >= engine.max_hold_days:
                exit_reason = "max_hold_exit"

            if exit_reason:
                # Apply exit fee
                effective_exit = row["close"] * (1 - FEE_PCT)
                pnl = (effective_exit - entry_price) / entry_price
                capital *= (1 + pnl)
                position = 0
                trades.append(pnl)

        # Entry on next candle's open after signal fires
        elif (i > 0 and df.iloc[i - 1]["buy_signal"] == 1
              and position == 0):
            position = 1
            entry_price = row["open"] * (1 + FEE_PCT)  # Entry fee
            entry_idx = i

        equity.append(capital)
        dates.append(row["timestamp"])

    # Sharpe from daily equity curve (correct method)
    daily_rets = daily_returns_from_equity(equity)
    sharpe = annualized_sharpe(daily_rets)

    wins = [r for r in trades if r > 0]
    losses = [r for r in trades if r <= 0]

    metrics = {
        "total_trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0,
        "total_return": capital / 10000 - 1,
        "sharpe": sharpe,
        "max_drawdown": compute_max_drawdown(equity),
        "best_trade": max(trades) if trades else 0,
        "worst_trade": min(trades) if trades else 0,
    }

    return {"equity": equity, "dates": dates, "metrics": metrics, "name": "Rules-Only (FNG+Funding)"}


# ═══════════════════════════════════════════════════════════════
# Strategy 2: Freqtrade-Style
# ═══════════════════════════════════════════════════════════════

def run_freqtrade_backtest(days: int = None) -> dict:
    """Run the Freqtrade-style backtest and return equity curve + metrics."""
    strategy = BTCPulseStrategy()
    backtester = FreqtradeBacktester(strategy, initial_capital=10000.0)
    result = backtester.run(days=days)

    if not result.equity_curve:
        return {"equity": [], "dates": [], "metrics": {}, "name": "Freqtrade-Style"}

    # Get dates from strategy data
    df = strategy.load_market_data(days=days)
    dates = list(df["timestamp"])
    # Equity curve has len(df) + 1 entries (initial + per-candle)
    # Align: trim equity to match dates if needed
    equity = result.equity_curve
    if len(equity) > len(dates):
        dates = [dates[0]] + dates  # Prepend start date for initial capital
    elif len(dates) > len(equity):
        dates = dates[:len(equity)]

    metrics = {
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "total_return": result.total_return,
        "sharpe": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "best_trade": result.best_trade,
        "worst_trade": result.worst_trade,
    }

    return {"equity": equity, "dates": dates, "metrics": metrics, "name": "Freqtrade-Style (Technical)"}


# ═══════════════════════════════════════════════════════════════
# Strategy 3: Composite (Rules 60% + XGBoost 40%)
# ═══════════════════════════════════════════════════════════════

def run_composite_backtest(days: int = None) -> dict:
    """
    Composite strategy: blend rule-based signal with XGBoost prediction.
    Entry when composite_score > 0.3, exit when < -0.3.
    The composite score = 0.6 * rules_score + 0.4 * ml_score.

    IMPORTANT: Uses walk-forward (expanding window) XGBoost training to
    avoid data leakage. The model is retrained periodically and only
    predicts on out-of-sample data it has never seen.
    """
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """SELECT f.*, m.close, m.open, m.high, m.low, m.volume
               FROM table_features f
               JOIN table_market_price m ON f.timestamp = m.timestamp
               ORDER BY f.timestamp""",
            conn, parse_dates=["timestamp"],
        )
    finally:
        conn.close()

    if df.empty or len(df) < 60:
        return {"equity": [], "dates": [], "metrics": {}, "name": "Composite (Rules+XGBoost)"}

    if days and not df.empty:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    # --- Rule-based scoring (from SignalGenerator logic) ---
    df["rules_score"] = 0.0

    # Contrarian sentiment component
    fng = df["fng_value"].fillna(50)
    df["rules_score"] += (50 - fng) / 100  # FNG=0→+0.5, FNG=50→0, FNG=100→-0.5

    # Extreme fear boost
    df.loc[fng < 15, "rules_score"] += 0.3
    # Extreme greed penalty
    df.loc[fng > 85, "rules_score"] -= 0.3

    # Leverage / funding component
    cum_funding = df.get("cumulative_funding_30d", pd.Series(0.0, index=df.index)).fillna(0)
    df.loc[cum_funding > 0.015, "rules_score"] -= 0.3
    df.loc[cum_funding < -0.005, "rules_score"] += 0.2

    # VWAP component
    vwap_dev = df.get("vwap_deviation", pd.Series(0.0, index=df.index)).fillna(0)
    df["rules_score"] += (vwap_dev / 20).clip(-0.3, 0.3)

    # Clip total rules score
    df["rules_score"] = df["rules_score"].clip(-1.0, 1.0)

    # --- Walk-forward XGBoost scoring (no data leakage) ---
    # Only predict on rows the model has NOT been trained on.
    # Retrain every `retrain_interval` rows using an expanding window.
    # Purge gap of 30 rows between train and predict to avoid label leakage
    # (since forward_trend uses a 30-day lookahead).
    df["ml_score"] = 0.0  # Default neutral (used for early rows with no model)

    PURGE_GAP = 30          # Must match the forward label horizon (30 days)
    MIN_TRAIN_ROWS = 200    # Minimum rows before first model
    RETRAIN_INTERVAL = 60   # Retrain every 60 rows (≈2 months)

    try:
        from xgboost import XGBClassifier

        xgb_ref = XGBoostCombiner()
        trained_features = [c for c in xgb_ref.FEATURE_COLS if c in df.columns]

        if not trained_features:
            raise ValueError("No usable features found in data")

        # Prepare feature matrix and labels for walk-forward
        X_all = df[trained_features].apply(pd.to_numeric, errors="coerce").fillna(0)
        label_col = "forward_trend"
        label_map = {"bearish": 0, "neutral": 1, "bullish": 2}

        if label_col in df.columns:
            y_all = df[label_col].map(label_map)
        else:
            y_all = pd.Series(np.nan, index=df.index)

        last_train_end = 0
        model = None

        for i in range(len(df)):
            # Can we train/retrain?
            # Training uses rows [0 .. i - PURGE_GAP] (purge avoids label overlap)
            train_end = i - PURGE_GAP

            if train_end >= MIN_TRAIN_ROWS and (model is None or (i - last_train_end) >= RETRAIN_INTERVAL):
                # Expanding window train
                X_train = X_all.iloc[:train_end]
                y_train = y_all.iloc[:train_end]

                # Drop rows without valid labels
                valid_mask = y_train.notna()
                X_train = X_train[valid_mask]
                y_train = y_train[valid_mask]

                if len(X_train) >= MIN_TRAIN_ROWS:
                    model = XGBClassifier(
                        n_estimators=150,
                        max_depth=4,
                        learning_rate=0.05,
                        objective="multi:softprob",
                        num_class=3,
                        eval_metric="mlogloss",
                        use_label_encoder=False,
                        random_state=42,
                        reg_alpha=0.1,      # L1 regularization
                        reg_lambda=1.0,     # L2 regularization
                        min_child_weight=5, # Reduce overfitting
                    )
                    model.fit(X_train, y_train, verbose=False)
                    last_train_end = i
                    logger.debug(f"Walk-forward retrain at row {i} (train size: {len(X_train)})")

            # Predict with current model (out-of-sample only)
            if model is not None:
                x_row = X_all.iloc[[i]]
                proba = model.predict_proba(x_row)[0]
                # ML score: bullish_prob - bearish_prob → [-1, +1]
                df.iloc[i, df.columns.get_loc("ml_score")] = float(proba[2] - proba[0])

        logger.info("Walk-forward XGBoost predictions complete (no data leakage)")
    except Exception as e:
        logger.warning(f"XGBoost unavailable, using rules-only: {e}")

    # --- Composite signal ---
    df["composite_score"] = 0.6 * df["rules_score"] + 0.4 * df["ml_score"]

    # --- Backtest the composite signal ---
    # Fixed: entry on next candle's open, fees on both sides
    capital = 10000.0
    equity = [capital]
    dates = [df.iloc[0]["timestamp"]]
    position = 0
    entry_price = 0.0
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]

        if position == 1:
            score = row["composite_score"]
            if score < -0.3:
                # Exit: apply fee
                effective_exit = row["close"] * (1 - FEE_PCT)
                pnl_pct = (effective_exit - entry_price) / entry_price
                capital *= (1 + pnl_pct)
                position = 0
                trades.append(pnl_pct)

        elif position == 0 and i > 0:
            prev_score = df.iloc[i - 1]["composite_score"]
            if prev_score > 0.3:
                # Entry on this candle's open (signal was previous candle)
                entry_price = row["open"] * (1 + FEE_PCT)
                position = 1

        equity.append(capital)
        dates.append(row["timestamp"])

    # Sharpe from daily equity curve (correct method)
    daily_rets = daily_returns_from_equity(equity)
    sharpe = annualized_sharpe(daily_rets)

    wins = [r for r in trades if r > 0]
    losses = [r for r in trades if r <= 0]

    metrics = {
        "total_trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0,
        "total_return": capital / 10000 - 1,
        "sharpe": sharpe,
        "max_drawdown": compute_max_drawdown(equity),
        "best_trade": max(trades) if trades else 0,
        "worst_trade": min(trades) if trades else 0,
    }

    return {"equity": equity, "dates": dates, "metrics": metrics, "name": "Composite (60% Rules + 40% XGBoost)"}


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

# NOTE: _compute_max_drawdown is now imported from utils.statistics
# as compute_max_drawdown. The series version below is only used
# for the drawdown plot panel.


def _compute_drawdown_series(equity: list) -> list:
    """Compute drawdown at each point."""
    peak = equity[0]
    drawdowns = []
    for val in equity:
        peak = max(peak, val)
        drawdowns.append((val - peak) / peak)
    return drawdowns


# ═══════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════

def plot_comparison(results: list, output_path: str):
    """
    Generate a multi-panel comparison plot:
    - Top: Equity curves (all strategies on one axis)
    - Middle: Drawdown comparison
    - Bottom: Metrics comparison table
    """
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 1, height_ratios=[3, 1.5, 1.5], hspace=0.3)

    colors = ["#2196F3", "#FF9800", "#4CAF50"]  # Blue, Orange, Green
    styles = ["-", "--", "-."]

    # ─── Panel 1: Equity Curves ──────────────────────────────
    ax1 = fig.add_subplot(gs[0])

    for idx, res in enumerate(results):
        if not res["equity"]:
            continue
        dates = res["dates"]
        equity = res["equity"]

        # Normalize to starting capital for fair comparison
        normalized = [e / equity[0] * 100 for e in equity]  # Index base 100

        # Use integer x-axis if dates aren't aligned
        ax1.plot(
            dates[:len(normalized)],
            normalized,
            label=f'{res["name"]} ({res["metrics"].get("total_return", 0):.1%})',
            color=colors[idx % len(colors)],
            linestyle=styles[idx % len(styles)],
            linewidth=2,
            alpha=0.9,
        )

    ax1.axhline(y=100, color="gray", linestyle=":", alpha=0.5, label="Break-even")
    ax1.set_title("BTC-Pulse Strategy Comparison — Equity Curves", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value (indexed to 100)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # ─── Panel 2: Drawdown Comparison ────────────────────────
    ax2 = fig.add_subplot(gs[1])

    for idx, res in enumerate(results):
        if not res["equity"]:
            continue
        dd = _compute_drawdown_series(res["equity"])
        dates = res["dates"][:len(dd)]
        ax2.fill_between(
            dates,
            [d * 100 for d in dd],
            0,
            alpha=0.3,
            color=colors[idx % len(colors)],
            label=f'{res["name"]} (max: {res["metrics"].get("max_drawdown", 0):.1%})',
        )
        ax2.plot(dates, [d * 100 for d in dd], color=colors[idx % len(colors)], linewidth=0.8)

    ax2.set_title("Drawdown Comparison", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # ─── Panel 3: Metrics Table ──────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")

    headers = ["Strategy", "Trades", "Win Rate", "Total Return", "Sharpe", "Max DD", "Best Trade", "Worst Trade"]
    table_data = []
    for res in results:
        m = res["metrics"]
        if not m:
            continue
        table_data.append([
            res["name"],
            str(m.get("total_trades", 0)),
            f'{m.get("win_rate", 0):.1%}',
            f'{m.get("total_return", 0):.2%}',
            f'{m.get("sharpe", 0):.2f}',
            f'{m.get("max_drawdown", 0):.1%}',
            f'{m.get("best_trade", 0):.2%}',
            f'{m.get("worst_trade", 0):.2%}',
        ])

    if table_data:
        table = ax3.table(
            cellText=table_data,
            colLabels=headers,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.6)

        # Style header
        for j in range(len(headers)):
            table[0, j].set_facecolor("#37474F")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Color-code total return column (col 3)
        for i in range(len(table_data)):
            ret = results[i]["metrics"].get("total_return", 0)
            if ret > 0:
                table[i + 1, 3].set_facecolor("#E8F5E9")
            else:
                table[i + 1, 3].set_facecolor("#FFEBEE")

    ax3.set_title("Performance Metrics", fontsize=12, fontweight="bold", pad=20)

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Comparison plot saved to: {output}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BTC-Pulse Strategy Comparison")
    parser.add_argument("--days", type=int, default=None, help="Limit to N days of data")
    parser.add_argument("--output", default="plots/strategy_comparison.png",
                        help="Output path for comparison plot")
    args = parser.parse_args()

    init_database()

    print("=" * 60)
    print("BTC-Pulse Strategy Comparison")
    print("=" * 60)

    results = []

    # Strategy 1: Rules-Only
    print("\n[1/3] Running Rules-Only backtest...")
    try:
        r1 = run_rules_backtest(days=args.days)
        results.append(r1)
        m = r1["metrics"]
        print(f"      → {m.get('total_trades', 0)} trades, "
              f"{m.get('total_return', 0):.2%} return, "
              f"Sharpe {m.get('sharpe', 0):.2f}")
    except Exception as e:
        logger.error(f"Rules backtest failed: {e}")
        results.append({"equity": [], "dates": [], "metrics": {}, "name": "Rules-Only"})

    # Strategy 2: Freqtrade-Style
    print("\n[2/3] Running Freqtrade-Style backtest...")
    try:
        r2 = run_freqtrade_backtest(days=args.days)
        results.append(r2)
        m = r2["metrics"]
        print(f"      → {m.get('total_trades', 0)} trades, "
              f"{m.get('total_return', 0):.2%} return, "
              f"Sharpe {m.get('sharpe', 0):.2f}")
    except Exception as e:
        logger.error(f"Freqtrade backtest failed: {e}")
        results.append({"equity": [], "dates": [], "metrics": {}, "name": "Freqtrade-Style"})

    # Strategy 3: Composite
    print("\n[3/3] Running Composite (Rules + XGBoost) backtest...")
    try:
        r3 = run_composite_backtest(days=args.days)
        results.append(r3)
        m = r3["metrics"]
        print(f"      → {m.get('total_trades', 0)} trades, "
              f"{m.get('total_return', 0):.2%} return, "
              f"Sharpe {m.get('sharpe', 0):.2f}")
    except Exception as e:
        logger.error(f"Composite backtest failed: {e}")
        results.append({"equity": [], "dates": [], "metrics": {}, "name": "Composite"})

    # Generate plot
    valid_results = [r for r in results if r["equity"]]
    if valid_results:
        print(f"\nGenerating comparison plot → {args.output}")
        plot_comparison(results, args.output)
        print(f"✅ Plot saved to: {args.output}")
    else:
        print("\n⚠️ No strategy produced results. Run `python run_ingest.py` first to populate data.")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Strategy':<35} {'Return':>10} {'Sharpe':>8} {'Win%':>7} {'Trades':>7}")
    print("-" * 60)
    for r in results:
        m = r["metrics"]
        if m:
            print(f"{r['name']:<35} {m.get('total_return', 0):>9.2%} "
                  f"{m.get('sharpe', 0):>8.2f} "
                  f"{m.get('win_rate', 0):>6.1%} "
                  f"{m.get('total_trades', 0):>7}")
    print("=" * 60)

    # ── Random Baseline Test ──────────────────────────────────
    # Answer: "Does any strategy beat a monkey picking random entries?"
    print("\n── Random-Entry Baseline (10,000 simulations) ──")
    try:
        # Load price series for baseline
        conn = get_connection()
        try:
            prices_df = pd.read_sql_query(
                "SELECT close FROM table_market_price ORDER BY timestamp",
                conn,
            )
        finally:
            conn.close()

        if not prices_df.empty:
            prices = prices_df["close"].values
            for r in results:
                m = r["metrics"]
                if not m or m.get("total_trades", 0) == 0:
                    continue
                baseline = random_entry_baseline(
                    prices=prices,
                    n_trades=m["total_trades"],
                    avg_hold_days=30,  # Conservative estimate
                    n_simulations=10_000,
                    fee_pct=2 * FEE_PCT,
                )
                pval = strategy_vs_random_pvalue(
                    m["total_return"], baseline["distribution"]
                )
                sig = "✅ significant" if pval < 0.05 else "❌ NOT significant"
                print(f"  {r['name']:<35} p-value: {pval:.4f}  ({sig})")
    except Exception as e:
        logger.warning(f"Random baseline failed: {e}")


if __name__ == "__main__":
    main()

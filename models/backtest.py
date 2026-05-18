"""
Offline Backtest Engine — tests signal combinations on historical data.
Fully offline: reads from SQLite, no network requests.

NOTE: This is the original simple backtester. For advanced walk-forward
backtesting with ROI tables, trailing stops, and per-trade tagging,
see freqtrade_bridge/backtester.py (FreqtradeBacktester).

Fixes applied (v2.1):
- Entries execute on the NEXT candle's open (not signal candle's close)
- Transaction fees deducted on both entry and exit (default 0.1%/side)
- Sharpe ratio computed from daily equity curve (not per-trade returns)
- Bootstrap confidence intervals on key metrics
"""

import pandas as pd
import numpy as np
from loguru import logger

from database.init_db import get_connection
from utils.config import load_config
from utils.statistics import (
    daily_returns_from_equity,
    annualized_sharpe,
    max_drawdown as compute_max_drawdown,
    bootstrap_ci,
)

# Default fee: 0.1% per side (Binance spot taker).
DEFAULT_FEE_PCT = 0.001


class BacktestEngine:
    """
    Backtests trading signals on historical data.
    Supports both rule-based and ML-generated signals.
    """

    # Risk management defaults (can be overridden via config)
    DEFAULT_STOPLOSS = -0.15       # -15% hard stop
    DEFAULT_MAX_HOLD_DAYS = 180    # Force-exit after 180 days
    DEFAULT_PROFIT_EXIT_DAYS = 90  # Take-profit exit if profitable after 90 days

    def __init__(self, fee_pct: float = DEFAULT_FEE_PCT):
        cfg = load_config()
        sig_cfg = cfg.get("signals", {})
        self.fng_buy = sig_cfg.get("fng_buy_threshold", 25)
        self.fng_sell = sig_cfg.get("fng_sell_threshold", 70)
        self.funding_sell = sig_cfg.get("funding_sell_threshold", 0.005)
        self.stoploss = sig_cfg.get("stoploss", self.DEFAULT_STOPLOSS)
        self.max_hold_days = sig_cfg.get("max_hold_days", self.DEFAULT_MAX_HOLD_DAYS)
        self.profit_exit_days = sig_cfg.get("profit_exit_days", self.DEFAULT_PROFIT_EXIT_DAYS)
        self.fee_pct = fee_pct

    def load_data(self) -> pd.DataFrame:
        """Load all features + price data for backtesting."""
        conn = get_connection()
        try:
            # Join features with price data
            df = pd.read_sql_query(
                """SELECT f.*, m.close, m.open, m.high, m.low, m.volume
                   FROM table_features f
                   JOIN table_market_price m ON f.timestamp = m.timestamp
                   ORDER BY f.timestamp""",
                conn, parse_dates=["timestamp"],
            )
        finally:
            conn.close()

        logger.info(f"Loaded {len(df)} rows for backtesting")
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the rule-based contrarian signal logic:
        - Buy:  FNG < 25 AND funding_rate < 0
                (fear + shorts dominating → contrarian long entry)
        - Sell: FNG > 70 AND cumulative_funding_30d > 0.5%
                (greed + leveraged longs → contrarian exit)

        Thresholds were calibrated so both buy and sell signals fire
        frequently enough across market regimes (2021–2026 validation).
        """
        df = df.copy()
        df["buy_signal"] = 0
        df["sell_signal"] = 0

        # Buy conditions: fear regime + negative funding
        fng_fear = df["fng_value"] < self.fng_buy
        funding_neg = df["funding_rate"] < 0

        # Sell conditions: greed regime + leveraged longs
        fng_greed = df["fng_value"] > self.fng_sell
        funding_high = df["cumulative_funding_30d"] > self.funding_sell

        df.loc[fng_fear & funding_neg, "buy_signal"] = 1
        df.loc[fng_greed & funding_high, "sell_signal"] = 1

        buy_count = df["buy_signal"].sum()
        sell_count = df["sell_signal"].sum()
        logger.info(f"Generated {buy_count} buy signals, {sell_count} sell signals")

        return df

    def run_backtest(self, df: pd.DataFrame) -> dict:
        """
        Event-driven backtest with risk management.

        Entry price: next candle's open after signal (avoids look-ahead).
        Fees: deducted on both entry and exit (default 0.1% per side).
        Sharpe: computed from daily equity curve (includes flat days).

        Returns performance metrics with bootstrap confidence intervals.
        """
        if df.empty:
            return {"error": "No data for backtesting"}

        df = df.dropna(subset=["close"]).reset_index(drop=True)

        initial_capital = 10000.0
        capital = initial_capital
        position = 0  # 0 = flat, 1 = long
        entry_price = 0.0
        entry_idx = 0
        total_fees = 0.0
        trades = []

        # Daily equity curve (one value per row, starting with initial)
        equity_curve = [capital]

        for i in range(len(df)):
            row = df.iloc[i]

            if position == 1:
                # Evaluate PnL against the fee-adjusted entry price
                current_pnl = (row["close"] - entry_price) / entry_price
                days_held = i - entry_idx

                exit_reason = None

                # Check stoploss
                if current_pnl <= self.stoploss:
                    exit_reason = "stoploss"

                # Check signal exit
                elif row["sell_signal"] == 1:
                    exit_reason = "signal"

                # Profit-taking time exit (profitable after N days)
                elif days_held >= self.profit_exit_days and current_pnl > 0:
                    exit_reason = "profit_time_exit"

                # Force exit (max hold exceeded)
                elif days_held >= self.max_hold_days:
                    exit_reason = "max_hold_exit"

                if exit_reason:
                    # Apply exit fee
                    effective_exit = row["close"] * (1 - self.fee_pct)
                    total_fees += row["close"] * self.fee_pct

                    pnl = (effective_exit - entry_price) / entry_price
                    capital *= (1 + pnl)
                    position = 0

                    trades.append({
                        "type": "sell",
                        "date": row["timestamp"],
                        "price": effective_exit,
                        "return": pnl,
                        "exit_reason": exit_reason,
                        "days_held": days_held,
                    })

            # Entry: signal on previous candle, execute on THIS candle's open
            elif (i > 0 and df.iloc[i - 1]["buy_signal"] == 1
                  and position == 0):
                position = 1
                # Enter on this candle's open + entry fee
                raw_price = row["open"]
                entry_price = raw_price * (1 + self.fee_pct)
                total_fees += raw_price * self.fee_pct
                entry_idx = i

                trades.append({
                    "type": "buy",
                    "date": row["timestamp"],
                    "price": entry_price,
                })

            equity_curve.append(capital)

        # ── Calculate metrics ─────────────────────────────────
        completed = [t for t in trades if t["type"] == "sell"]

        if not completed:
            return {
                "total_trades": 0,
                "equity_curve": equity_curve,
                "message": "No completed trades in backtest period",
            }

        returns = [t["return"] for t in completed]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        total_return = capital / initial_capital - 1

        # Sharpe from daily equity curve (correct methodology)
        daily_rets = daily_returns_from_equity(equity_curve)
        sharpe = annualized_sharpe(daily_rets)

        # Max drawdown from equity curve
        mdd = compute_max_drawdown(equity_curve)

        # Bootstrap CIs
        returns_arr = np.array(returns)
        win_rate_ci = bootstrap_ci(
            (returns_arr > 0).astype(float), stat_fn=np.mean
        )
        avg_return_ci = bootstrap_ci(returns_arr, stat_fn=np.mean)

        result = {
            "total_trades": len(completed),
            "win_rate": round(len(wins) / len(completed), 4) if completed else 0,
            "win_rate_ci_95": [round(win_rate_ci["ci_lower"], 4),
                               round(win_rate_ci["ci_upper"], 4)],
            "total_return": round(total_return, 4),
            "avg_trade_return": round(float(np.mean(returns)), 4),
            "avg_return_ci_95": [round(avg_return_ci["ci_lower"], 4),
                                 round(avg_return_ci["ci_upper"], 4)],
            "best_trade": round(max(returns), 4) if returns else 0,
            "worst_trade": round(min(returns), 4) if returns else 0,
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(mdd, 4),
            "total_fees_paid": round(total_fees, 2),
            "equity_curve": equity_curve,
            "trades": trades,
        }

        # Exit reason breakdown
        exit_reasons = {}
        for t in completed:
            reason = t.get("exit_reason", "unknown")
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        result["exit_reasons"] = exit_reasons

        logger.info(
            f"Backtest complete: {result['total_trades']} trades, "
            f"{result['win_rate']:.0%} win rate "
            f"(95% CI: [{win_rate_ci['ci_lower']:.0%}, {win_rate_ci['ci_upper']:.0%}]), "
            f"{result['total_return']:.2%} total return, "
            f"Sharpe: {result['sharpe_ratio']:.2f}, "
            f"fees: ${result['total_fees_paid']:.2f}"
        )

        return result

    def contrarian_fng_study(self, df: pd.DataFrame) -> dict:
        """
        Research: FNG extreme values vs 30-day forward returns.
        Validates the contrarian sentiment hypothesis.
        """
        df = df.dropna(subset=["fng_value", "forward_return_30d"]).copy()

        extreme_low = df[df["fng_value"] < 20]
        extreme_high = df[df["fng_value"] > 80]
        normal = df[(df["fng_value"] >= 20) & (df["fng_value"] <= 80)]

        result = {
            "extreme_fear": {
                "count": len(extreme_low),
                "avg_30d_return": round(extreme_low["forward_return_30d"].mean(), 4)
                if not extreme_low.empty else None,
                "median_30d_return": round(extreme_low["forward_return_30d"].median(), 4)
                if not extreme_low.empty else None,
            },
            "extreme_greed": {
                "count": len(extreme_high),
                "avg_30d_return": round(extreme_high["forward_return_30d"].mean(), 4)
                if not extreme_high.empty else None,
                "median_30d_return": round(extreme_high["forward_return_30d"].median(), 4)
                if not extreme_high.empty else None,
            },
            "normal": {
                "count": len(normal),
                "avg_30d_return": round(normal["forward_return_30d"].mean(), 4)
                if not normal.empty else None,
            },
        }

        logger.info(f"FNG Study — Fear avg return: {result['extreme_fear']['avg_30d_return']}, "
                     f"Greed avg return: {result['extreme_greed']['avg_30d_return']}")

        return result

    def run(self) -> dict:
        """Full backtest pipeline."""
        df = self.load_data()
        df = self.generate_signals(df)
        backtest_result = self.run_backtest(df)
        fng_study = self.contrarian_fng_study(df)

        return {
            "backtest": backtest_result,
            "fng_study": fng_study,
        }

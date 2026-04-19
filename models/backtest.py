"""
Offline Backtest Engine — tests signal combinations on historical data.
Fully offline: reads from SQLite, no network requests.
Uses vectorbt for efficient backtesting.
"""

import pandas as pd
import numpy as np
from loguru import logger

from database.init_db import get_connection
from utils.config import load_config


class BacktestEngine:
    """
    Backtests trading signals on historical data.
    Supports both rule-based and ML-generated signals.
    """

    def __init__(self):
        cfg = load_config()
        sig_cfg = cfg.get("signals", {})
        self.fng_buy = sig_cfg.get("fng_buy_threshold", 15)
        self.fng_sell = sig_cfg.get("fng_sell_threshold", 85)
        self.funding_sell = sig_cfg.get("funding_sell_threshold", 0.02)

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
        Apply the rule-based signal logic:
        - Buy: FNG < 15 AND funding_rate < 0
        - Sell: FNG > 85 AND cumulative_funding_30d > 2%
        """
        df = df.copy()
        df["buy_signal"] = 0
        df["sell_signal"] = 0

        # Buy conditions
        fng_fear = df["fng_value"] < self.fng_buy
        funding_neg = df["funding_rate"] < 0

        # Sell conditions
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
        Simple event-driven backtest.
        Buy on buy_signal, sell on sell_signal.
        Returns performance metrics.
        """
        if df.empty:
            return {"error": "No data for backtesting"}

        df = df.dropna(subset=["close"]).reset_index(drop=True)

        # Position tracking
        position = 0  # 0 = flat, 1 = long
        entry_price = 0
        trades = []

        for i, row in df.iterrows():
            if row["buy_signal"] == 1 and position == 0:
                position = 1
                entry_price = row["close"]
                trades.append({
                    "type": "buy",
                    "date": row["timestamp"],
                    "price": entry_price,
                })

            elif row["sell_signal"] == 1 and position == 1:
                position = 0
                exit_price = row["close"]
                pnl = (exit_price - entry_price) / entry_price
                trades.append({
                    "type": "sell",
                    "date": row["timestamp"],
                    "price": exit_price,
                    "return": pnl,
                })

        # Calculate metrics
        completed = [t for t in trades if t["type"] == "sell"]

        if not completed:
            return {
                "total_trades": 0,
                "message": "No completed trades in backtest period",
            }

        returns = [t["return"] for t in completed]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        total_return = np.prod([1 + r for r in returns]) - 1
        avg_return = np.mean(returns)
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(12)) if np.std(returns) > 0 else 0

        result = {
            "total_trades": len(completed),
            "win_rate": round(len(wins) / len(completed), 4) if completed else 0,
            "total_return": round(total_return, 4),
            "avg_trade_return": round(avg_return, 4),
            "best_trade": round(max(returns), 4) if returns else 0,
            "worst_trade": round(min(returns), 4) if returns else 0,
            "sharpe_ratio": round(sharpe, 4),
            "trades": trades,
        }

        logger.info(
            f"Backtest complete: {result['total_trades']} trades, "
            f"{result['win_rate']:.0%} win rate, "
            f"{result['total_return']:.2%} total return, "
            f"Sharpe: {result['sharpe_ratio']:.2f}"
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

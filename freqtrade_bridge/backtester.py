"""
Freqtrade-Style Backtester for BTC-Pulse.

Borrows from Freqtrade's backtesting engine:
- Walk-forward validation (no future data leakage)
- ROI table and trailing stoploss simulation
- Per-trade tagging (entry/exit reasons)
- Detailed trade log with timestamps
- Performance metrics: Sharpe, Sortino, Calmar, max drawdown
- Monthly/yearly breakdown

Major upgrade over BTC-Pulse's original simple backtest.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from loguru import logger
from typing import Optional

from freqtrade_bridge.strategy import BaseStrategy
from freqtrade_bridge.risk_manager import RiskManager


@dataclass
class Trade:
    """Represents a single backtest trade."""
    entry_date: pd.Timestamp
    entry_price: float
    entry_tag: str = ""
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_tag: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    duration_days: int = 0
    peak_profit: float = 0.0
    max_drawdown: float = 0.0

    def to_dict(self) -> dict:
        return {
            "entry_date": self.entry_date,
            "entry_price": self.entry_price,
            "entry_tag": self.entry_tag,
            "exit_date": self.exit_date,
            "exit_price": self.exit_price,
            "exit_tag": self.exit_tag,
            "pnl_pct": round(self.pnl_pct, 4),
            "duration_days": self.duration_days,
            "peak_profit": round(self.peak_profit, 4),
        }


@dataclass
class BacktestResult:
    """Full backtest results with Freqtrade-style metrics."""
    trades: list = field(default_factory=list)
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration_days: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_profit: float = 0.0
    monthly_returns: dict = field(default_factory=dict)
    entry_tag_stats: dict = field(default_factory=dict)
    exit_tag_stats: dict = field(default_factory=dict)
    equity_curve: list = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"═══ Backtest Results ═══\n"
            f"Total trades:    {self.total_trades}\n"
            f"Win rate:        {self.win_rate:.1%}\n"
            f"Total return:    {self.total_return:.2%}\n"
            f"CAGR:            {self.cagr:.2%}\n"
            f"Sharpe ratio:    {self.sharpe_ratio:.2f}\n"
            f"Sortino ratio:   {self.sortino_ratio:.2f}\n"
            f"Max drawdown:    {self.max_drawdown:.2%}\n"
            f"Calmar ratio:    {self.calmar_ratio:.2f}\n"
            f"Profit factor:   {self.profit_factor:.2f}\n"
            f"Best trade:      {self.best_trade:.2%}\n"
            f"Worst trade:     {self.worst_trade:.2%}\n"
            f"Avg duration:    {self.avg_trade_duration_days:.1f} days\n"
        )


class FreqtradeBacktester:
    """
    Walk-forward backtester inspired by Freqtrade.

    Key features over the original BTC-Pulse backtest:
    - ROI table enforcement (auto-sell at profit targets)
    - Trailing stoploss simulation
    - Dynamic stoploss via strategy callback
    - Trade tagging (why did we enter/exit?)
    - Equity curve and drawdown tracking
    - No look-ahead bias
    """

    def __init__(self, strategy: BaseStrategy, initial_capital: float = 10000.0,
                 risk_manager: Optional[RiskManager] = None):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.risk_manager = risk_manager or RiskManager(
            total_capital=initial_capital, position_mode="atr"
        )

    def _check_roi(self, trade: Trade, current_idx: int, entry_idx: int,
                   current_price: float) -> bool:
        """Check if ROI target is met (Freqtrade feature)."""
        duration_minutes = (current_idx - entry_idx) * 1440  # 1d candles
        current_profit = (current_price - trade.entry_price) / trade.entry_price

        for minutes_str, roi in sorted(self.strategy.minimal_roi.items(),
                                        key=lambda x: int(x[0])):
            if duration_minutes >= int(minutes_str):
                if current_profit >= roi:
                    return True
        return False

    def _check_stoploss(self, trade: Trade, current_price: float,
                        atr_pct: float = 3.0) -> tuple[bool, float]:
        """
        Check stoploss conditions including trailing stop.
        Returns (triggered, stoploss_price).
        """
        current_profit = (current_price - trade.entry_price) / trade.entry_price

        # Dynamic stoploss from strategy
        dynamic_sl = self.strategy.custom_stoploss(
            current_profit=current_profit,
            current_time=0,
            entry_price=trade.entry_price,
            current_price=current_price,
            atr_pct=atr_pct,
        )

        stoploss = max(dynamic_sl, self.strategy.stoploss)  # Use tighter of the two

        # Trailing stop logic
        if self.strategy.trailing_stop:
            if self.strategy.trailing_only_offset_is_reached:
                if current_profit >= self.strategy.trailing_stop_positive_offset:
                    trailing_sl = current_profit - self.strategy.trailing_stop_positive
                    if trailing_sl > stoploss:
                        stoploss = -abs(current_profit - trailing_sl)
            else:
                if current_profit > self.strategy.trailing_stop_positive:
                    stoploss = -(current_profit - self.strategy.trailing_stop_positive)

        # Check if stoploss hit
        if current_profit <= stoploss:
            return True, trade.entry_price * (1 + stoploss)

        return False, 0.0

    def run(self, days: Optional[int] = None) -> BacktestResult:
        """
        Run the full backtest.

        Walk-forward: iterate candle by candle, no future peeking.
        """
        logger.info("Running Freqtrade-style backtest...")

        # Get strategy data with signals
        df = self.strategy.run_pipeline(days=days)
        if df.empty or len(df) < 30:
            logger.warning("Insufficient data for backtest")
            return BacktestResult()

        result = BacktestResult()
        capital = self.initial_capital
        equity_curve = [capital]
        open_trade: Optional[Trade] = None
        peak_equity = capital

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            if open_trade is not None:
                # ── Manage open trade ─────────────────────────
                current_price = row["close"]
                current_profit = (current_price - open_trade.entry_price) / open_trade.entry_price
                open_trade.peak_profit = max(open_trade.peak_profit, current_profit)

                exit_reason = ""

                # Check stoploss
                atr_pct = row.get("atr_pct", 3.0)
                if pd.isna(atr_pct):
                    atr_pct = 3.0
                sl_hit, sl_price = self._check_stoploss(open_trade, current_price, atr_pct)
                if sl_hit:
                    exit_reason = "stoploss"
                    current_price = sl_price

                # Check ROI
                elif self._check_roi(open_trade, i, entry_idx, current_price):
                    exit_reason = "roi"

                # Check strategy exit signal
                elif row.get("exit_long", 0) == 1:
                    exit_reason = row.get("exit_tag", "signal_exit")
                    # Confirm exit via strategy callback
                    if not self.strategy.confirm_trade_exit(df, i, current_profit):
                        exit_reason = ""

                if exit_reason:
                    # Close trade
                    pnl_pct = (current_price - open_trade.entry_price) / open_trade.entry_price
                    open_trade.exit_date = row["timestamp"]
                    open_trade.exit_price = current_price
                    open_trade.exit_tag = exit_reason
                    open_trade.pnl_pct = pnl_pct
                    open_trade.duration_days = (
                        row["timestamp"] - open_trade.entry_date
                    ).days

                    capital *= (1 + pnl_pct)
                    result.trades.append(open_trade)
                    open_trade = None

            else:
                # ── Look for entry ────────────────────────────
                if prev_row.get("enter_long", 0) == 1:
                    # Entry on this candle's open (next candle after signal)
                    if self.strategy.confirm_trade_entry(df, i):
                        open_trade = Trade(
                            entry_date=row["timestamp"],
                            entry_price=row["open"],
                            entry_tag=prev_row.get("enter_tag", ""),
                        )
                        entry_idx = i

            equity_curve.append(capital)

            # Track drawdown
            peak_equity = max(peak_equity, capital)
            dd = (capital - peak_equity) / peak_equity
            if dd < result.max_drawdown:
                result.max_drawdown = dd

        # ── Compute final metrics ─────────────────────────────
        result.equity_curve = equity_curve

        completed = [t for t in result.trades if t.exit_date is not None]
        result.total_trades = len(completed)

        if not completed:
            logger.info("No completed trades")
            return result

        returns = [t.pnl_pct for t in completed]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = len(wins) / len(completed) if completed else 0
        result.total_return = capital / self.initial_capital - 1
        result.best_trade = max(returns)
        result.worst_trade = min(returns)
        result.avg_profit = np.mean(returns)
        result.avg_trade_duration_days = np.mean([t.duration_days for t in completed])

        # Sharpe (annualized, daily returns approximation)
        if np.std(returns) > 0:
            result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365 / max(result.avg_trade_duration_days, 1))

        # Sortino (only downside deviation)
        downside = [r for r in returns if r < 0]
        if downside:
            downside_std = np.std(downside)
            if downside_std > 0:
                result.sortino_ratio = np.mean(returns) / downside_std * np.sqrt(365 / max(result.avg_trade_duration_days, 1))

        # Calmar
        if result.max_drawdown < 0:
            # Estimate CAGR
            total_days = (completed[-1].exit_date - completed[0].entry_date).days
            if total_days > 0:
                result.cagr = (1 + result.total_return) ** (365 / total_days) - 1
                result.calmar_ratio = result.cagr / abs(result.max_drawdown)

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1e-10
        result.profit_factor = gross_profit / gross_loss

        # Entry/exit tag stats
        for trade in completed:
            tag = trade.entry_tag or "unknown"
            if tag not in result.entry_tag_stats:
                result.entry_tag_stats[tag] = {"count": 0, "avg_pnl": 0, "returns": []}
            result.entry_tag_stats[tag]["count"] += 1
            result.entry_tag_stats[tag]["returns"].append(trade.pnl_pct)

        for tag, stats in result.entry_tag_stats.items():
            stats["avg_pnl"] = round(np.mean(stats["returns"]), 4)
            stats["win_rate"] = round(len([r for r in stats["returns"] if r > 0]) / len(stats["returns"]), 2)
            del stats["returns"]

        logger.info(f"\n{result.summary()}")
        return result

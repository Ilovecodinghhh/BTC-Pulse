"""
Freqtrade-Style Backtester for BTC-Pulse.

Borrows from Freqtrade's backtesting engine:
- Walk-forward validation (no future data leakage)
- ROI table and trailing stoploss simulation
- Per-trade tagging (entry/exit reasons)
- Detailed trade log with timestamps
- Transaction cost modeling (fees + slippage)
- Position sizing via RiskManager (Kelly / ATR / fixed)
- Performance metrics from daily equity curve (Sharpe, Sortino, Calmar)
- Bootstrap confidence intervals on all key metrics
- Deflated Sharpe Ratio for multiple-testing adjustment
- Random-entry baseline for hypothesis testing
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from loguru import logger
from typing import Optional

from freqtrade_bridge.strategy import BaseStrategy
from freqtrade_bridge.risk_manager import RiskManager
from utils.statistics import (
    daily_returns_from_equity,
    annualized_sharpe,
    annualized_sortino,
    max_drawdown,
    calmar_ratio,
    bootstrap_ci,
    deflated_sharpe_ratio,
    random_entry_baseline,
    strategy_vs_random_pvalue,
)


# ── Default cost model ───────────────────────────────────────
# Binance spot taker fee = 0.1% per side → 0.2% round trip.
# We model 0.1% per trade (entry or exit), so a full round trip = 0.2%.
DEFAULT_FEE_PCT = 0.001  # 0.1% per trade (entry or exit)


@dataclass
class Trade:
    """Represents a single backtest trade."""
    entry_date: pd.Timestamp
    entry_price: float
    entry_tag: str = ""
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_tag: str = ""
    pnl_pct: float = 0.0
    duration_days: int = 0
    peak_profit: float = 0.0
    position_size_usd: float = 0.0  # Actual dollars allocated

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
            "position_size_usd": round(self.position_size_usd, 2),
        }


@dataclass
class BacktestResult:
    """
    Full backtest results with corrected metrics.

    All risk-adjusted ratios (Sharpe, Sortino, Calmar) are computed from
    the daily equity curve — not from per-trade returns. This is the
    standard methodology because it correctly accounts for flat periods.
    """
    trades: list = field(default_factory=list)
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration_days: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_profit: float = 0.0
    total_fees_paid: float = 0.0
    entry_tag_stats: dict = field(default_factory=dict)
    equity_curve: list = field(default_factory=list)

    # Statistical validation (new)
    sharpe_ci: dict = field(default_factory=dict)
    win_rate_ci: dict = field(default_factory=dict)
    avg_profit_ci: dict = field(default_factory=dict)
    deflated_sharpe: float = 0.0
    random_baseline: dict = field(default_factory=dict)
    random_pvalue: float = 1.0

    def summary(self) -> str:
        lines = [
            f"═══ Backtest Results ═══",
            f"Total trades:    {self.total_trades}",
            f"Win rate:        {self.win_rate:.1%}"
            + (f"  (95% CI: [{self.win_rate_ci.get('ci_lower', 0):.1%}, "
               f"{self.win_rate_ci.get('ci_upper', 0):.1%}])"
               if self.win_rate_ci else ""),
            f"Total return:    {self.total_return:.2%}",
            f"CAGR:            {self.cagr:.2%}",
            f"Total fees paid: ${self.total_fees_paid:.2f}",
            f"",
            f"── Risk Metrics (from daily equity curve) ──",
            f"Sharpe ratio:    {self.sharpe_ratio:.2f}"
            + (f"  (95% CI: [{self.sharpe_ci.get('ci_lower', 0):.2f}, "
               f"{self.sharpe_ci.get('ci_upper', 0):.2f}])"
               if self.sharpe_ci else ""),
            f"Sortino ratio:   {self.sortino_ratio:.2f}",
            f"Max drawdown:    {self.max_drawdown:.2%}",
            f"Calmar ratio:    {self.calmar_ratio:.2f}",
            f"Profit factor:   {self.profit_factor:.2f}",
            f"Best trade:      {self.best_trade:.2%}",
            f"Worst trade:     {self.worst_trade:.2%}",
            f"Avg duration:    {self.avg_trade_duration_days:.1f} days",
        ]

        if self.deflated_sharpe > 0:
            lines.append(f"")
            lines.append(f"── Statistical Validation ──")
            lines.append(f"Deflated Sharpe: {self.deflated_sharpe:.3f}"
                         f"  ({'PASS' if self.deflated_sharpe > 0.95 else 'FAIL'}"
                         f" at 95% confidence)")
            lines.append(f"vs Random p-val: {self.random_pvalue:.4f}"
                         f"  ({'significant' if self.random_pvalue < 0.05 else 'NOT significant'}"
                         f" at 5%)")

        return "\n".join(lines)


class FreqtradeBacktester:
    """
    Walk-forward backtester inspired by Freqtrade.

    Key features:
    - ROI table enforcement (auto-sell at profit targets)
    - Trailing stoploss simulation
    - Dynamic stoploss via strategy callback
    - Transaction cost modeling (entry fee + exit fee)
    - Position sizing via RiskManager (Kelly / ATR / fixed fractional)
    - Risk metrics from daily equity curve (not per-trade)
    - Bootstrap CIs on Sharpe, win rate, avg profit
    - Deflated Sharpe Ratio adjustment for hyperopt trials
    - Random-entry baseline for statistical significance
    - No look-ahead bias: entries on next candle's open
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 10000.0,
        fee_pct: float = DEFAULT_FEE_PCT,
        risk_manager: Optional[RiskManager] = None,
        n_hyperopt_trials: int = 1,
    ):
        """
        Args:
            strategy:           Strategy instance to backtest.
            initial_capital:    Starting capital in USD.
            fee_pct:            Fee per trade side (0.001 = 0.1%).
                                Round-trip cost is 2 × fee_pct.
            risk_manager:       Position sizer. If None, uses ATR-based
                                sizing with 2% risk per trade.
            n_hyperopt_trials:  Number of strategy variants tried during
                                hyperopt. Used for Deflated Sharpe Ratio.
                                Set to 1 if no hyperopt was run.
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct
        self.risk_manager = risk_manager or RiskManager(
            total_capital=initial_capital, position_mode="atr",
        )
        self.n_hyperopt_trials = n_hyperopt_trials

    def _apply_entry_fee(self, price: float) -> float:
        """Effective entry price after paying the fee (slippage up)."""
        return price * (1 + self.fee_pct)

    def _apply_exit_fee(self, price: float) -> float:
        """Effective exit price after paying the fee (slippage down)."""
        return price * (1 - self.fee_pct)

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

        stoploss = max(dynamic_sl, self.strategy.stoploss)

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

        if current_profit <= stoploss:
            return True, trade.entry_price * (1 + stoploss)

        return False, 0.0

    def run(self, days: Optional[int] = None) -> BacktestResult:
        """
        Run the full backtest.

        Walk-forward: iterates candle by candle. Entries execute on the
        next candle's open (not the signal candle's close). Transaction
        fees are deducted on both entry and exit. Position sizing is
        delegated to the RiskManager.
        """
        logger.info("Running Freqtrade-style backtest...")
        logger.info(f"  Fee per side: {self.fee_pct:.2%} "
                     f"(round-trip: {2 * self.fee_pct:.2%})")

        # Get strategy data with signals
        df = self.strategy.run_pipeline(days=days)
        if df.empty or len(df) < 30:
            logger.warning("Insufficient data for backtest")
            return BacktestResult()

        result = BacktestResult()
        capital = self.initial_capital
        total_fees = 0.0

        # Build a daily equity curve (one value per candle)
        equity_curve = [capital]
        open_trade: Optional[Trade] = None
        entry_idx: int = 0

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            if open_trade is not None:
                # ── Manage open trade ─────────────────────────
                current_price = row["close"]
                current_profit = (
                    (current_price - open_trade.entry_price)
                    / open_trade.entry_price
                )
                open_trade.peak_profit = max(
                    open_trade.peak_profit, current_profit
                )

                exit_reason = ""

                # Check stoploss
                atr_pct = row.get("atr_pct", 3.0)
                if pd.isna(atr_pct):
                    atr_pct = 3.0
                sl_hit, sl_price = self._check_stoploss(
                    open_trade, current_price, atr_pct
                )
                if sl_hit:
                    exit_reason = "stoploss"
                    current_price = sl_price

                # Check ROI
                elif self._check_roi(open_trade, i, entry_idx, current_price):
                    exit_reason = "roi"

                # Check strategy exit signal
                elif row.get("exit_long", 0) == 1:
                    exit_reason = row.get("exit_tag", "signal_exit")
                    if not self.strategy.confirm_trade_exit(
                        df, i, current_profit
                    ):
                        exit_reason = ""

                if exit_reason:
                    # Apply exit fee
                    effective_exit = self._apply_exit_fee(current_price)
                    exit_fee = current_price * self.fee_pct
                    total_fees += exit_fee * (
                        open_trade.position_size_usd
                        / open_trade.entry_price
                    )

                    pnl_pct = (
                        (effective_exit - open_trade.entry_price)
                        / open_trade.entry_price
                    )

                    open_trade.exit_date = row["timestamp"]
                    open_trade.exit_price = effective_exit
                    open_trade.exit_tag = exit_reason
                    open_trade.pnl_pct = pnl_pct
                    open_trade.duration_days = (
                        row["timestamp"] - open_trade.entry_date
                    ).days

                    # Update capital proportional to position size
                    capital += open_trade.position_size_usd * pnl_pct

                    result.trades.append(open_trade)
                    self.risk_manager.register_trade_result(pnl_pct)
                    open_trade = None

            else:
                # ── Look for entry ────────────────────────────
                if prev_row.get("enter_long", 0) == 1:
                    if self.strategy.confirm_trade_entry(df, i):
                        raw_entry_price = row["open"]
                        effective_entry = self._apply_entry_fee(
                            raw_entry_price
                        )

                        # Position sizing via RiskManager
                        atr = row.get("atr", None)
                        if pd.isna(atr) if atr is not None else True:
                            atr = raw_entry_price * 0.03
                        sl_price = raw_entry_price * (1 + self.strategy.stoploss)

                        sizing = self.risk_manager.calculate_position_size(
                            entry_price=raw_entry_price,
                            stoploss_price=sl_price,
                            atr=atr,
                        )
                        position_usd = sizing["size_usd"]
                        if position_usd <= 0:
                            # RiskManager blocked (circuit breaker or zero)
                            equity_curve.append(capital)
                            continue

                        # Cap at available capital
                        position_usd = min(position_usd, capital * 0.95)

                        # Entry fee
                        entry_fee = raw_entry_price * self.fee_pct
                        total_fees += entry_fee * (
                            position_usd / raw_entry_price
                        )

                        open_trade = Trade(
                            entry_date=row["timestamp"],
                            entry_price=effective_entry,
                            entry_tag=prev_row.get("enter_tag", ""),
                            position_size_usd=position_usd,
                        )
                        entry_idx = i
                        self.risk_manager.register_trade_entry()

            equity_curve.append(capital)

        # ── Compute final metrics from daily equity curve ─────
        result.equity_curve = equity_curve
        result.total_fees_paid = round(total_fees, 2)

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
        result.win_rate = len(wins) / len(completed)
        result.total_return = capital / self.initial_capital - 1
        result.best_trade = max(returns)
        result.worst_trade = min(returns)
        result.avg_profit = float(np.mean(returns))
        result.avg_trade_duration_days = float(
            np.mean([t.duration_days for t in completed])
        )

        # ── Risk metrics from daily equity curve (correct method) ──
        daily_rets = daily_returns_from_equity(equity_curve)
        result.sharpe_ratio = annualized_sharpe(daily_rets)
        result.sortino_ratio = annualized_sortino(daily_rets)
        result.max_drawdown = max_drawdown(equity_curve)
        result.calmar_ratio = calmar_ratio(equity_curve)

        # CAGR
        total_days = (completed[-1].exit_date - completed[0].entry_date).days
        if total_days > 0:
            result.cagr = (
                (1 + result.total_return) ** (365 / total_days) - 1
            )

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1e-10
        result.profit_factor = gross_profit / gross_loss

        # ── Entry tag stats ───────────────────────────────────
        for trade in completed:
            tag = trade.entry_tag or "unknown"
            if tag not in result.entry_tag_stats:
                result.entry_tag_stats[tag] = {
                    "count": 0, "returns": [],
                }
            result.entry_tag_stats[tag]["count"] += 1
            result.entry_tag_stats[tag]["returns"].append(trade.pnl_pct)

        for tag, stats in result.entry_tag_stats.items():
            tag_returns = stats["returns"]
            stats["avg_pnl"] = round(float(np.mean(tag_returns)), 4)
            stats["win_rate"] = round(
                len([r for r in tag_returns if r > 0]) / len(tag_returns), 2
            )
            stats["n_trades"] = len(tag_returns)
            del stats["returns"]

        # ── Statistical validation ────────────────────────────
        returns_arr = np.array(returns)

        # Bootstrap CI on win rate
        result.win_rate_ci = bootstrap_ci(
            (returns_arr > 0).astype(float), stat_fn=np.mean
        )

        # Bootstrap CI on average profit
        result.avg_profit_ci = bootstrap_ci(returns_arr, stat_fn=np.mean)

        # Bootstrap CI on Sharpe (resample daily returns)
        if len(daily_rets) >= 30:
            def _sharpe_fn(x):
                s = np.std(x, ddof=1)
                return np.mean(x) / s * np.sqrt(365) if s > 1e-12 else 0.0

            result.sharpe_ci = bootstrap_ci(daily_rets, stat_fn=_sharpe_fn)

        # Deflated Sharpe Ratio
        if len(daily_rets) >= 30:
            skew = float(pd.Series(daily_rets).skew())
            kurt = float(pd.Series(daily_rets).kurtosis() + 3)
            result.deflated_sharpe = deflated_sharpe_ratio(
                observed_sharpe=result.sharpe_ratio,
                n_trials=max(self.n_hyperopt_trials, 1),
                n_returns=len(daily_rets),
                skewness=skew,
                kurtosis=kurt,
            )

        # Random-entry baseline
        prices = df["close"].values
        if len(prices) > 10 and result.total_trades > 0:
            baseline = random_entry_baseline(
                prices=prices,
                n_trades=result.total_trades,
                avg_hold_days=max(int(result.avg_trade_duration_days), 1),
                n_simulations=10_000,
                fee_pct=2 * self.fee_pct,  # Round-trip fee
            )
            result.random_baseline = baseline
            result.random_pvalue = strategy_vs_random_pvalue(
                result.total_return, baseline["distribution"]
            )

        logger.info(f"\n{result.summary()}")
        return result

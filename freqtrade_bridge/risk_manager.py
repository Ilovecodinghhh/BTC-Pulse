"""
Freqtrade-Style Position Sizing & Risk Management for BTC-Pulse.

Inspired by Freqtrade's:
- Stake amount calculation
- Max open trades limiting
- Risk-per-trade models (fixed, Kelly Criterion, ATR-based)
- Drawdown-based position reduction
"""

import numpy as np
from loguru import logger
from typing import Optional


class RiskManager:
    """
    Position sizing and risk management — inspired by Freqtrade's
    stake_amount and max_open_trades logic, extended with:
    - Kelly Criterion sizing
    - ATR-based position sizing (volatility-adjusted)
    - Drawdown circuit breaker
    - Correlation-aware exposure limits
    """

    def __init__(
        self,
        total_capital: float = 10000.0,
        max_risk_per_trade: float = 0.02,  # 2% of capital per trade
        max_open_trades: int = 3,
        max_drawdown_halt: float = -0.15,  # Stop trading at -15% DD
        position_mode: str = "atr",  # 'fixed', 'kelly', 'atr'
    ):
        self.total_capital = total_capital
        self.current_capital = total_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_open_trades = max_open_trades
        self.max_drawdown_halt = max_drawdown_halt
        self.position_mode = position_mode

        self.peak_capital = total_capital
        self.open_positions = 0
        self.trade_history = []

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak."""
        if self.peak_capital == 0:
            return 0
        return (self.current_capital - self.peak_capital) / self.peak_capital

    @property
    def can_trade(self) -> bool:
        """Check if trading is allowed (circuit breaker)."""
        if self.current_drawdown <= self.max_drawdown_halt:
            logger.warning(f"Circuit breaker: drawdown {self.current_drawdown:.1%} "
                          f"exceeds limit {self.max_drawdown_halt:.1%}")
            return False
        if self.open_positions >= self.max_open_trades:
            return False
        return True

    def calculate_position_size(
        self,
        entry_price: float,
        stoploss_price: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> dict:
        """
        Calculate position size based on selected mode.

        Returns:
            dict with 'size_usd', 'size_btc', 'risk_usd', 'method'
        """
        if not self.can_trade:
            return {"size_usd": 0, "size_btc": 0, "risk_usd": 0,
                    "method": "blocked", "reason": "circuit_breaker"}

        risk_distance = abs(entry_price - stoploss_price) / entry_price

        if self.position_mode == "fixed":
            size_usd = self._fixed_size(risk_distance)
        elif self.position_mode == "kelly":
            size_usd = self._kelly_size(risk_distance, win_rate, avg_win, avg_loss)
        elif self.position_mode == "atr":
            size_usd = self._atr_size(entry_price, atr)
        else:
            size_usd = self._fixed_size(risk_distance)

        # Cap at available capital
        size_usd = min(size_usd, self.current_capital * 0.95)

        return {
            "size_usd": round(size_usd, 2),
            "size_btc": round(size_usd / entry_price, 6),
            "risk_usd": round(size_usd * risk_distance, 2),
            "risk_pct": round(size_usd * risk_distance / self.current_capital * 100, 2),
            "method": self.position_mode,
        }

    def _fixed_size(self, risk_distance: float) -> float:
        """Fixed fractional: risk X% of capital per trade."""
        risk_amount = self.current_capital * self.max_risk_per_trade
        return risk_amount / risk_distance if risk_distance > 0 else 0

    def _kelly_size(self, risk_distance: float,
                    win_rate: Optional[float],
                    avg_win: Optional[float],
                    avg_loss: Optional[float]) -> float:
        """
        Kelly Criterion sizing — optimal growth rate.
        f* = (bp - q) / b
        where b = avg_win/avg_loss, p = win_rate, q = 1-p

        Uses half-Kelly for safety (full Kelly is too aggressive).
        """
        if not all([win_rate, avg_win, avg_loss]):
            return self._fixed_size(risk_distance)

        if avg_loss == 0:
            return self._fixed_size(risk_distance)

        b = abs(avg_win / avg_loss)
        p = win_rate
        q = 1 - p

        kelly_fraction = (b * p - q) / b

        if kelly_fraction <= 0:
            # Kelly says don't bet (negative expectancy)
            logger.warning("Kelly says skip: negative edge")
            return 0

        # Half-Kelly for safety
        half_kelly = kelly_fraction / 2

        # Cap at 25% of capital
        fraction = min(half_kelly, 0.25)

        return self.current_capital * fraction

    def _atr_size(self, entry_price: float, atr: Optional[float]) -> float:
        """
        ATR-based sizing — risk a fixed amount per ATR unit.
        Position = (risk_capital) / (N * ATR)
        where N = ATR multiplier for stop distance.
        """
        if not atr or atr == 0:
            return self._fixed_size(0.05)  # Default 5% risk distance

        risk_amount = self.current_capital * self.max_risk_per_trade
        atr_multiplier = 2.0  # Stop at 2x ATR

        # Position size = risk / (atr * multiplier)
        position_value = risk_amount / (atr * atr_multiplier / entry_price)

        return position_value

    def register_trade_result(self, pnl_pct: float):
        """Update state after a trade completes."""
        self.current_capital *= (1 + pnl_pct)
        self.peak_capital = max(self.peak_capital, self.current_capital)
        self.trade_history.append(pnl_pct)
        self.open_positions = max(0, self.open_positions - 1)

    def register_trade_entry(self):
        """Track new position opening."""
        self.open_positions += 1

    def get_stats(self) -> dict:
        """Current risk manager state."""
        return {
            "capital": round(self.current_capital, 2),
            "peak": round(self.peak_capital, 2),
            "drawdown": f"{self.current_drawdown:.2%}",
            "open_positions": self.open_positions,
            "can_trade": self.can_trade,
            "total_trades": len(self.trade_history),
            "mode": self.position_mode,
        }

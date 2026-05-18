"""
Tests for utils/statistics.py — the statistical foundation.
Run with: pytest tests/test_statistics.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


class TestDailyReturns:
    def test_simple_equity(self):
        equity = [100, 110, 105, 120]
        rets = daily_returns_from_equity(equity)
        assert len(rets) == 3
        assert pytest.approx(rets[0], rel=1e-6) == 0.10
        assert pytest.approx(rets[1], rel=1e-6) == -0.04545454545

    def test_empty(self):
        assert len(daily_returns_from_equity([])) == 0
        assert len(daily_returns_from_equity([100])) == 0

    def test_flat_equity(self):
        """Flat equity → all-zero returns."""
        rets = daily_returns_from_equity([100, 100, 100, 100])
        assert np.all(rets == 0)


class TestSharpe:
    def test_positive_returns(self):
        """Steadily rising equity → positive Sharpe."""
        equity = [100 + i * 0.5 for i in range(365)]
        daily_rets = daily_returns_from_equity(equity)
        sharpe = annualized_sharpe(daily_rets)
        assert sharpe > 0

    def test_zero_returns(self):
        """Flat equity → Sharpe = 0."""
        rets = np.zeros(100)
        assert annualized_sharpe(rets) == 0.0

    def test_too_few_points(self):
        assert annualized_sharpe(np.array([0.01])) == 0.0
        assert annualized_sharpe(np.array([])) == 0.0

    def test_known_value(self):
        """Hand-calculated: mean=0.001, std=0.01, annualized sqrt(365)."""
        rng = np.random.default_rng(42)
        rets = rng.normal(0.001, 0.01, size=365)
        sharpe = annualized_sharpe(rets)
        # Should be roughly 0.001 / 0.01 * sqrt(365) ≈ 1.91
        assert 1.0 < sharpe < 3.0


class TestSortino:
    def test_only_positive_returns(self):
        """No downside → Sortino should be 0 (no downside std)."""
        rets = np.array([0.01, 0.02, 0.005, 0.015])
        assert annualized_sortino(rets) == 0.0

    def test_mixed_returns(self):
        rng = np.random.default_rng(42)
        rets = rng.normal(0.001, 0.01, size=365)
        sortino = annualized_sortino(rets)
        sharpe = annualized_sharpe(rets)
        # Sortino ≥ Sharpe when returns have positive mean
        assert sortino >= sharpe * 0.8  # Approximately


class TestMaxDrawdown:
    def test_simple_drawdown(self):
        """Peak at 120, trough at 90 → DD = (90-120)/120 = -25%."""
        equity = [100, 110, 120, 100, 90, 95]
        mdd = max_drawdown(equity)
        assert pytest.approx(mdd, rel=1e-6) == -0.25

    def test_monotonic_increase(self):
        equity = [100, 110, 120, 130]
        assert max_drawdown(equity) == 0.0

    def test_empty(self):
        assert max_drawdown([]) == 0.0
        assert max_drawdown([100]) == 0.0


class TestCalmar:
    def test_positive_calmar(self):
        # Total return: 50%, max DD: 25%, over 1 year
        equity = [100, 110, 120, 100, 90, 110, 130, 150]
        # This won't be exact but should be positive
        c = calmar_ratio(equity)
        assert c > 0

    def test_no_drawdown(self):
        equity = [100, 110, 120, 130]
        assert calmar_ratio(equity) == 0.0  # MDD=0 → division guarded


class TestBootstrap:
    def test_basic_ci(self):
        rng = np.random.default_rng(42)
        values = rng.normal(0.05, 0.1, size=100)
        result = bootstrap_ci(values)
        assert "estimate" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] < result["estimate"] < result["ci_upper"]

    def test_single_value(self):
        result = bootstrap_ci(np.array([0.5]))
        assert result["estimate"] == 0.5
        assert result["std_error"] == 0.0

    def test_ci_width_decreases_with_n(self):
        """Larger sample → narrower CI."""
        rng = np.random.default_rng(42)
        small = bootstrap_ci(rng.normal(0, 1, size=20), seed=42)
        large = bootstrap_ci(rng.normal(0, 1, size=500), seed=42)
        small_width = small["ci_upper"] - small["ci_lower"]
        large_width = large["ci_upper"] - large["ci_lower"]
        assert large_width < small_width


class TestDeflatedSharpe:
    def test_single_trial(self):
        """With 1 trial, DSR should be relatively high for decent Sharpe."""
        dsr = deflated_sharpe_ratio(
            observed_sharpe=1.5, n_trials=1, n_returns=365
        )
        assert dsr > 0.5

    def test_many_trials_reduces_dsr(self):
        """More trials → lower DSR (harder to pass)."""
        dsr_1 = deflated_sharpe_ratio(1.5, n_trials=1, n_returns=365)
        dsr_100 = deflated_sharpe_ratio(1.5, n_trials=100, n_returns=365)
        assert dsr_100 < dsr_1

    def test_edge_cases(self):
        assert deflated_sharpe_ratio(1.0, n_trials=0, n_returns=365) == 0.0
        assert deflated_sharpe_ratio(1.0, n_trials=1, n_returns=1) == 0.0


class TestRandomBaseline:
    def test_basic_output(self):
        prices = np.linspace(100, 150, 365)  # Trending up
        result = random_entry_baseline(
            prices, n_trades=10, avg_hold_days=20, n_simulations=100
        )
        assert "mean_return" in result
        assert "distribution" in result
        assert len(result["distribution"]) == 100

    def test_pvalue_perfect_strategy(self):
        """A massive return should have a low p-value."""
        dist = np.random.normal(0.1, 0.05, size=1000)
        pval = strategy_vs_random_pvalue(10.0, dist)  # 1000% return
        assert pval < 0.01

    def test_pvalue_zero_strategy(self):
        """A zero-return strategy should have a high p-value."""
        dist = np.random.normal(0.1, 0.05, size=1000)
        pval = strategy_vs_random_pvalue(0.0, dist)
        assert pval > 0.5

"""
Statistical utilities for rigorous backtest evaluation.

Provides:
- Correct Sharpe / Sortino / Calmar from daily equity curves
- Bootstrap confidence intervals on any metric
- Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
- Random-entry baseline generator for hypothesis testing

All functions are stateless and operate on numpy arrays.
"""

import numpy as np


# ── Daily-Return-Based Risk Metrics ──────────────────────────

def daily_returns_from_equity(equity_curve: list | np.ndarray) -> np.ndarray:
    """
    Convert an equity curve into a daily return series.

    Args:
        equity_curve: Sequence of portfolio values, one per day.
                      First element is the initial capital.

    Returns:
        Array of daily returns (length = len(equity_curve) - 1).
        Includes zero-return days when the strategy is flat.
    """
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) < 2:
        return np.array([])
    returns = np.diff(eq) / eq[:-1]
    return returns


def annualized_sharpe(daily_rets: np.ndarray,
                      risk_free_daily: float = 0.0,
                      trading_days: int = 365) -> float:
    """
    Annualized Sharpe ratio from daily returns.

    Uses sqrt(trading_days) annualization (365 for crypto, 252 for
    equities). Includes flat days in the denominator — this is correct
    because capital is at risk even when no trade is open (opportunity
    cost).

    Args:
        daily_rets:      Array of daily portfolio returns.
        risk_free_daily: Daily risk-free rate (default 0).
        trading_days:    Days per year for annualization.

    Returns:
        Annualized Sharpe ratio. Returns 0.0 if std is zero.
    """
    if len(daily_rets) < 2:
        return 0.0
    excess = daily_rets - risk_free_daily
    std = np.std(excess, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(trading_days))


def annualized_sortino(daily_rets: np.ndarray,
                       risk_free_daily: float = 0.0,
                       trading_days: int = 365) -> float:
    """
    Annualized Sortino ratio — penalizes only downside volatility.

    Args:
        daily_rets:      Array of daily portfolio returns.
        risk_free_daily: Daily risk-free rate.
        trading_days:    Days per year for annualization.

    Returns:
        Annualized Sortino ratio. Returns 0.0 if downside std is zero.
    """
    if len(daily_rets) < 2:
        return 0.0
    excess = daily_rets - risk_free_daily
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = np.std(downside, ddof=1)
    if downside_std < 1e-12:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(trading_days))


def max_drawdown(equity_curve: list | np.ndarray) -> float:
    """
    Maximum drawdown as a negative fraction (e.g. -0.15 = -15%).

    Args:
        equity_curve: Sequence of portfolio values.

    Returns:
        Maximum drawdown (negative float). Returns 0.0 if curve is
        monotonically increasing.
    """
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    drawdowns = (eq - peak) / peak
    return float(np.min(drawdowns))


def calmar_ratio(equity_curve: list | np.ndarray,
                 trading_days: int = 365) -> float:
    """
    Calmar ratio = CAGR / |max drawdown|.

    Args:
        equity_curve: Sequence of portfolio values.
        trading_days: Days per year.

    Returns:
        Calmar ratio. Returns 0.0 if max drawdown is zero.
    """
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) < 2 or eq[0] <= 0:
        return 0.0

    total_return = eq[-1] / eq[0] - 1
    n_days = len(eq) - 1
    if n_days <= 0:
        return 0.0
    cagr = (1 + total_return) ** (trading_days / n_days) - 1

    mdd = abs(max_drawdown(eq))
    if mdd < 1e-12:
        return 0.0
    return float(cagr / mdd)


# ── Bootstrap Confidence Intervals ───────────────────────────

def bootstrap_ci(
    values: np.ndarray,
    stat_fn=np.mean,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Non-parametric bootstrap confidence interval for any statistic.

    Resamples `values` with replacement `n_boot` times, computes
    `stat_fn` on each resample, and returns the percentile interval.

    Args:
        values:     1-D array of observations (e.g. per-trade returns).
        stat_fn:    Callable that takes an array → scalar.
        n_boot:     Number of bootstrap resamples.
        confidence: Confidence level (0.95 = 95% CI).
        seed:       Random seed for reproducibility.

    Returns:
        dict with 'estimate', 'ci_lower', 'ci_upper', 'std_error'.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        point = float(stat_fn(values)) if len(values) == 1 else 0.0
        return {"estimate": point, "ci_lower": point,
                "ci_upper": point, "std_error": 0.0}

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_stats[i] = stat_fn(sample)

    alpha = 1 - confidence
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return {
        "estimate": float(stat_fn(values)),
        "ci_lower": lower,
        "ci_upper": upper,
        "std_error": float(np.std(boot_stats, ddof=1)),
    }


# ── Deflated Sharpe Ratio ────────────────────────────────────

def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_returns: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Adjusts the observed Sharpe ratio for the number of strategy
    configurations tried (the "trials tax"). Returns the probability
    that the observed Sharpe exceeds the expected maximum Sharpe under
    the null hypothesis of zero skill.

    A DSR > 0.95 suggests the Sharpe is unlikely due to luck alone.

    Args:
        observed_sharpe: The Sharpe ratio of the best strategy.
        n_trials:        Total number of strategies / hyperopt trials tried.
        n_returns:       Number of return observations used.
        skewness:        Skewness of the return series (0 = normal).
        kurtosis:        Kurtosis of the return series (3 = normal).

    Returns:
        Probability in [0, 1]. Higher = more likely to be genuine skill.
    """
    from scipy.stats import norm

    if n_trials < 1 or n_returns < 2:
        return 0.0

    # Expected maximum Sharpe under null (Euler-Mascheroni approximation)
    euler_mascheroni = 0.5772156649
    e_max_sharpe = (
        norm.ppf(1 - 1 / n_trials)
        * (1 - euler_mascheroni)
        + euler_mascheroni * norm.ppf(1 - 1 / (n_trials * np.e))
    )

    # Variance of Sharpe estimator (Lo, 2002) with skew/kurtosis correction
    sr_var = (
        1
        + 0.5 * observed_sharpe ** 2
        - skewness * observed_sharpe
        + ((kurtosis - 3) / 4) * observed_sharpe ** 2
    ) / (n_returns - 1)

    if sr_var <= 0:
        return 0.0

    sr_std = np.sqrt(sr_var)

    # Probability that observed Sharpe exceeds E[max] under null
    z = (observed_sharpe - e_max_sharpe) / sr_std
    return float(norm.cdf(z))


# ── Random Entry Baseline ────────────────────────────────────

def random_entry_baseline(
    prices: np.ndarray,
    n_trades: int,
    avg_hold_days: int,
    n_simulations: int = 10_000,
    fee_pct: float = 0.001,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo random-entry baseline.

    Generates `n_simulations` random strategies, each entering
    `n_trades` times at random dates with the specified average holding
    period, and computes the distribution of total returns. This
    answers: "How often would a monkey with the same trade frequency
    beat my strategy?"

    Args:
        prices:          Array of daily close prices.
        n_trades:        Number of trades per simulation.
        avg_hold_days:   Average holding period in days.
        n_simulations:   Number of random strategies to simulate.
        fee_pct:         Round-trip transaction cost as a fraction.
        seed:            Random seed.

    Returns:
        dict with 'mean_return', 'median_return', 'std_return',
        'percentile_95', 'percentile_5', and the full 'distribution'.
    """
    rng = np.random.default_rng(seed)
    prices = np.asarray(prices, dtype=float)
    n = len(prices)

    if n < avg_hold_days + 1 or n_trades < 1:
        return {"mean_return": 0.0, "median_return": 0.0,
                "std_return": 0.0, "percentile_5": 0.0,
                "percentile_95": 0.0, "distribution": []}

    max_entry = n - avg_hold_days - 1
    if max_entry < 1:
        max_entry = 1

    sim_returns = np.empty(n_simulations)

    for sim in range(n_simulations):
        capital = 1.0
        entries = rng.integers(0, max_entry, size=n_trades)
        hold_days = rng.poisson(avg_hold_days, size=n_trades).clip(1, n - 1)

        for entry, hold in zip(entries, hold_days):
            exit_idx = min(entry + hold, n - 1)
            trade_return = (prices[exit_idx] / prices[entry]) - 1
            trade_return -= fee_pct  # Deduct transaction cost
            capital *= (1 + trade_return)

        sim_returns[sim] = capital - 1

    return {
        "mean_return": float(np.mean(sim_returns)),
        "median_return": float(np.median(sim_returns)),
        "std_return": float(np.std(sim_returns)),
        "percentile_5": float(np.percentile(sim_returns, 5)),
        "percentile_95": float(np.percentile(sim_returns, 95)),
        "distribution": sim_returns.tolist(),
    }


def strategy_vs_random_pvalue(
    strategy_return: float,
    random_distribution: list | np.ndarray,
) -> float:
    """
    Empirical p-value: fraction of random strategies that beat ours.

    Args:
        strategy_return:     Total return of the actual strategy.
        random_distribution: Array of total returns from random baseline.

    Returns:
        p-value in [0, 1]. Lower = strategy is more likely to have
        genuine edge. Convention: p < 0.05 is "significant".
    """
    dist = np.asarray(random_distribution, dtype=float)
    if len(dist) == 0:
        return 1.0
    return float(np.mean(dist >= strategy_return))

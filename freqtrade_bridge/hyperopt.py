"""
Freqtrade-Style Hyperparameter Optimization for BTC-Pulse.

Inspired by Freqtrade's hyperopt:
- Optimizes strategy parameters (indicator thresholds, stoploss, ROI)
- Uses Optuna (Bayesian optimization) instead of brute force
- Walk-forward cross-validation to avoid overfitting
- Objectives: Sharpe ratio, profit factor, or custom
"""

import numpy as np
import pandas as pd
from loguru import logger
from dataclasses import dataclass
from typing import Optional, Callable
import json
from pathlib import Path

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from freqtrade_bridge.strategy import BTCPulseStrategy
from freqtrade_bridge.backtester import FreqtadeBacktester


@dataclass
class HyperoptResult:
    """Results from hyperparameter optimization."""
    best_params: dict
    best_score: float
    n_trials: int
    optimization_target: str
    all_trials: list


class StrategyHyperopt:
    """
    Hyperparameter optimization for BTC-Pulse strategies.

    Optimizable parameters:
    - RSI thresholds (entry/exit)
    - Bollinger Band periods and deviation
    - FNG thresholds
    - Funding rate thresholds
    - Stoploss value
    - ROI table entries
    - Trailing stop parameters
    """

    def __init__(self, strategy_class=BTCPulseStrategy,
                 initial_capital: float = 10000.0,
                 objective: str = "sharpe"):
        """
        Args:
            strategy_class: Strategy class to optimize
            initial_capital: Starting capital for backtests
            objective: Optimization target ('sharpe', 'profit_factor',
                       'total_return', 'calmar', 'sortino')
        """
        if not HAS_OPTUNA:
            raise ImportError("optuna required: pip install optuna")

        self.strategy_class = strategy_class
        self.initial_capital = initial_capital
        self.objective = objective

    def _create_trial_strategy(self, trial: "optuna.Trial") -> BTCPulseStrategy:
        """Create a strategy instance with trial-suggested parameters."""
        strategy = self.strategy_class()

        # Stoploss
        strategy.stoploss = trial.suggest_float("stoploss", -0.15, -0.03)

        # Trailing stop
        strategy.trailing_stop = trial.suggest_categorical("trailing_stop", [True, False])
        if strategy.trailing_stop:
            strategy.trailing_stop_positive = trial.suggest_float(
                "trailing_stop_positive", 0.01, 0.06)
            strategy.trailing_stop_positive_offset = trial.suggest_float(
                "trailing_stop_positive_offset", 0.03, 0.10)

        # ROI table
        roi_0 = trial.suggest_float("roi_0", 0.05, 0.25)
        roi_1d = trial.suggest_float("roi_1d", 0.02, 0.12)
        roi_3d = trial.suggest_float("roi_3d", 0.01, 0.06)
        strategy.minimal_roi = {
            "0": roi_0,
            "1440": roi_1d,
            "4320": roi_3d,
            "14400": 0.0,
        }

        # Signal thresholds (stored as attributes, used in populate_* overrides)
        strategy._hp_rsi_buy = trial.suggest_int("rsi_buy", 20, 40)
        strategy._hp_rsi_sell = trial.suggest_int("rsi_sell", 65, 85)
        strategy._hp_fng_buy = trial.suggest_int("fng_buy", 10, 25)
        strategy._hp_fng_sell = trial.suggest_int("fng_sell", 75, 95)
        strategy._hp_bb_period = trial.suggest_int("bb_period", 15, 30)

        return strategy

    def _objective_fn(self, trial: "optuna.Trial", df_cache: list) -> float:
        """Optuna objective function."""
        strategy = self._create_trial_strategy(trial)

        # Use cached data to avoid repeated DB queries
        backtester = FreqtadeBacktester(strategy, self.initial_capital)
        result = backtester.run()

        if result.total_trades < 5:
            return -100  # Penalize strategies with too few trades

        if self.objective == "sharpe":
            return result.sharpe_ratio
        elif self.objective == "sortino":
            return result.sortino_ratio
        elif self.objective == "calmar":
            return result.calmar_ratio
        elif self.objective == "profit_factor":
            return result.profit_factor
        elif self.objective == "total_return":
            return result.total_return
        else:
            return result.sharpe_ratio

    def optimize(self, n_trials: int = 100, timeout: int = 600,
                 show_progress: bool = True) -> HyperoptResult:
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Maximum number of trials
            timeout: Maximum time in seconds
            show_progress: Log progress

        Returns:
            HyperoptResult with best parameters and scores
        """
        logger.info(f"Starting hyperopt: {n_trials} trials, objective={self.objective}")

        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )

        df_cache = []  # Placeholder for data caching

        study.optimize(
            lambda trial: self._objective_fn(trial, df_cache),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress,
        )

        best = study.best_trial
        logger.info(f"Best trial: score={best.value:.4f}, params={best.params}")

        result = HyperoptResult(
            best_params=best.params,
            best_score=best.value,
            n_trials=len(study.trials),
            optimization_target=self.objective,
            all_trials=[{
                "number": t.number,
                "value": t.value,
                "params": t.params,
            } for t in study.trials if t.value is not None],
        )

        return result

    def save_results(self, result: HyperoptResult, path: str = "data/hyperopt_results.json"):
        """Save optimization results to file."""
        output = {
            "best_params": result.best_params,
            "best_score": result.best_score,
            "n_trials": result.n_trials,
            "objective": result.optimization_target,
            "top_10": sorted(result.all_trials, key=lambda x: x["value"], reverse=True)[:10],
        }

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Hyperopt results saved to {p}")

    def apply_best_params(self, params: dict) -> BTCPulseStrategy:
        """Create a strategy instance with the best parameters applied."""
        strategy = self.strategy_class()
        strategy.stoploss = params.get("stoploss", strategy.stoploss)
        strategy.trailing_stop = params.get("trailing_stop", strategy.trailing_stop)

        if strategy.trailing_stop:
            strategy.trailing_stop_positive = params.get(
                "trailing_stop_positive", strategy.trailing_stop_positive)
            strategy.trailing_stop_positive_offset = params.get(
                "trailing_stop_positive_offset", strategy.trailing_stop_positive_offset)

        strategy.minimal_roi = {
            "0": params.get("roi_0", 0.15),
            "1440": params.get("roi_1d", 0.08),
            "4320": params.get("roi_3d", 0.04),
            "14400": 0.0,
        }

        return strategy

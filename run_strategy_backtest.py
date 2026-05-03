#!/usr/bin/env python3
"""
Run the Freqtrade-style backtest on BTC-Pulse data.

Usage:
    python run_strategy_backtest.py [--days 365] [--hyperopt] [--trials 50]
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freqtrade_bridge.strategy import BTCPulseStrategy
from freqtrade_bridge.backtester import FreqtradeBacktester
from utils.logging import setup_logger

logger = setup_logger("strategy_backtest")


def main():
    parser = argparse.ArgumentParser(description="BTC-Pulse Strategy Backtest (Freqtrade-style)")
    parser.add_argument("--days", type=int, default=None, help="Days of data to backtest")
    parser.add_argument("--hyperopt", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--trials", type=int, default=50, help="Number of hyperopt trials")
    parser.add_argument("--objective", default="sharpe",
                        choices=["sharpe", "sortino", "calmar", "profit_factor", "total_return"],
                        help="Optimization objective")
    args = parser.parse_args()

    if args.hyperopt:
        from freqtrade_bridge.hyperopt import StrategyHyperopt

        logger.info(f"Running hyperopt: {args.trials} trials, objective={args.objective}")
        optimizer = StrategyHyperopt(objective=args.objective)
        result = optimizer.optimize(n_trials=args.trials)
        optimizer.save_results(result)

        logger.info(f"\nBest parameters (score={result.best_score:.4f}):")
        for k, v in result.best_params.items():
            logger.info(f"  {k}: {v}")

        # Run final backtest with best params
        logger.info("\nRunning backtest with best parameters...")
        strategy = optimizer.apply_best_params(result.best_params)
        backtester = FreqtradeBacktester(strategy)
        bt_result = backtester.run(days=args.days)
        print(bt_result.summary())

    else:
        logger.info("Running BTC-Pulse Strategy Backtest (Freqtrade-style)")
        strategy = BTCPulseStrategy()
        backtester = FreqtradeBacktester(strategy)
        result = backtester.run(days=args.days)
        print(result.summary())

        # Print entry tag breakdown
        if result.entry_tag_stats:
            print("\n── Entry Signal Breakdown ──")
            for tag, stats in sorted(result.entry_tag_stats.items(),
                                      key=lambda x: x[1]["count"], reverse=True):
                print(f"  {tag}: {stats['count']} trades, "
                      f"win={stats['win_rate']:.0%}, avg_pnl={stats['avg_pnl']:.2%}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
BTC-Pulse Offline Backtest Runner
Tests signal combinations on historical data.
Fully offline — reads from SQLite only.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from database.init_db import init_database
from models.backtest import BacktestEngine
from models.xgboost_model import XGBoostCombiner
from models.anomaly import AnomalyDetector
from utils.logging import setup_logger

logger = setup_logger("backtest")


def main():
    logger.info("=" * 60)
    logger.info("BTC-Pulse Backtest Engine Started")
    logger.info("=" * 60)

    init_database()

    # Run rule-based backtest
    logger.info("── Rule-Based Backtest ──")
    try:
        engine = BacktestEngine()
        results = engine.run()

        bt = results.get("backtest", {})
        if "error" not in bt:
            print("\n[BACKTEST RESULTS]")
            print(f"  Total Trades:    {bt.get('total_trades', 0)}")
            print(f"  Win Rate:        {bt.get('win_rate', 0):.1%}")
            print(f"  Total Return:    {bt.get('total_return', 0):.2%}")
            print(f"  Avg Trade:       {bt.get('avg_trade_return', 0):.2%}")
            print(f"  Best Trade:      {bt.get('best_trade', 0):.2%}")
            print(f"  Worst Trade:     {bt.get('worst_trade', 0):.2%}")
            print(f"  Sharpe Ratio:    {bt.get('sharpe_ratio', 0):.2f}")
        else:
            print(f"  WARNING: {bt['error']}")

        fng = results.get("fng_study", {})
        if fng:
            print("\n[FNG CONTRARIAN STUDY]")
            ef = fng.get("extreme_fear", {})
            eg = fng.get("extreme_greed", {})
            print(f"  Extreme Fear (FNG<20):  {ef.get('count', 0)} days, "
                  f"avg 30d return: {ef.get('avg_30d_return', 'N/A')}")
            print(f"  Extreme Greed (FNG>80): {eg.get('count', 0)} days, "
                  f"avg 30d return: {eg.get('avg_30d_return', 'N/A')}")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")

    # Train XGBoost model
    logger.info("\n── XGBoost Model Training ──")
    try:
        xgb = XGBoostCombiner()
        train_result = xgb.train()
        print("\n[XGBOOST MODEL]")
        print(f"  Accuracy:        {train_result.get('accuracy', 'N/A')}")
        print(f"  Train Samples:   {train_result.get('train_samples', 0)}")
        print(f"  Test Samples:    {train_result.get('test_samples', 0)}")
        print("  Feature Importance:")
        for feat, imp in list(train_result.get("feature_importance", {}).items())[:5]:
            print(f"    {feat}: {imp:.4f}")

        # Run prediction
        pred = xgb.predict()
        if "error" not in pred:
            print(f"\n  Prediction: {pred.get('predicted_trend', 'N/A')} "
                  f"(confidence: {pred.get('confidence', 0):.1%})")
            print(f"  P(trend up): {pred.get('trend_up_probability', 0):.1%}")
    except Exception as e:
        logger.error(f"XGBoost training/prediction failed: {e}")

    # Train anomaly detector
    logger.info("\n── Anomaly Detection ──")
    try:
        anomaly = AnomalyDetector()
        train_result = anomaly.train()
        if "error" not in train_result:
            detection = anomaly.detect()
            print(f"\n[ANOMALY CHECK]: {detection.get('alert', 'N/A')}")
            print(f"  Score: {detection.get('anomaly_score', 'N/A')}")
        else:
            print(f"  WARNING: {train_result['error']}")
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")

    logger.info("=" * 60)
    logger.info("BTC-Pulse Backtest Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

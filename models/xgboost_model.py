"""
XGBoost Signal Combiner — learns optimal feature weights automatically.
Replaces manual weight assignment with data-driven importance.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from database.init_db import get_connection
from utils.config import load_config, get_project_root


class XGBoostCombiner:
    """
    Trains XGBoost on historical features to predict monthly trend.
    Used as an AI filter on top of rule-based signals.
    """

    FEATURE_COLS = [
        "ma30_deviation", "price_change_7d", "price_change_30d",
        "volatility_30d", "volume_change", "vwap_deviation",
        "fng_value", "fng_ma7", "fng_extreme_low", "fng_extreme_high",
        "funding_rate", "cumulative_funding_30d", "oi_change_rate",
        "oi_price_divergence",
    ]

    LABEL_COL = "forward_trend"

    def __init__(self):
        cfg = load_config()
        model_cfg = cfg.get("model", {}).get("xgboost", {})

        self.n_estimators = model_cfg.get("n_estimators", 200)
        self.max_depth = model_cfg.get("max_depth", 5)
        self.learning_rate = model_cfg.get("learning_rate", 0.05)
        self.test_size = 1 - model_cfg.get("train_test_split", 0.8)

        self.model_path = get_project_root() / "data" / "xgboost_model.joblib"
        self.model = None

    def _load_training_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load features and labels from database."""
        conn = get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM table_features WHERE forward_trend IS NOT NULL "
                "ORDER BY timestamp", conn,
            )
        finally:
            conn.close()

        if df.empty:
            raise ValueError("No training data available. Run feature engineering first.")

        # Select available feature columns
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        X = df[available].copy()
        y = df[self.LABEL_COL].copy()

        # Drop rows with NaN in features
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]

        logger.info(f"Training data: {len(X)} samples, {len(available)} features")
        return X, y

    def train(self) -> dict:
        """Train XGBoost model with time-series aware splitting."""
        X, y = self._load_training_data()

        if len(X) < 100:
            logger.warning(f"Only {len(X)} samples — model may overfit!")

        # Encode labels
        label_map = {"bearish": 0, "neutral": 1, "bullish": 2}
        y_encoded = y.map(label_map)

        # Time-series split (no future leakage)
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y_encoded.iloc[:split_idx], y_encoded.iloc[split_idx:]

        # Train
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Feature importance
        importance = dict(zip(X.columns, self.model.feature_importances_))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        # Save model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")

        result = {
            "accuracy": round(accuracy, 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_importance": {k: round(v, 4) for k, v in importance.items()},
        }

        logger.info(f"Training complete — accuracy: {accuracy:.2%}")
        if accuracy > 0.85:
            logger.warning("⚠️ Suspiciously high accuracy — possible overfitting!")

        return result

    def predict(self, features: pd.DataFrame | None = None) -> dict:
        """
        Predict trend probability for the latest data point.
        Returns probability distribution over [bearish, neutral, bullish].
        """
        if self.model is None:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info("Loaded saved XGBoost model")
            else:
                return {"error": "No trained model. Run train() first."}

        if features is None:
            conn = get_connection()
            try:
                df = pd.read_sql_query(
                    "SELECT * FROM table_features ORDER BY timestamp DESC LIMIT 1",
                    conn,
                )
            finally:
                conn.close()

            if df.empty:
                return {"error": "No features available for prediction."}

            available = [c for c in self.FEATURE_COLS if c in df.columns]
            features = df[available]

        # Handle NaN
        features = features.fillna(0)

        proba = self.model.predict_proba(features)[0]
        labels = ["bearish", "neutral", "bullish"]

        return {
            "probabilities": dict(zip(labels, [round(p, 4) for p in proba])),
            "predicted_trend": labels[int(np.argmax(proba))],
            "confidence": round(float(np.max(proba)), 4),
            "trend_up_probability": round(float(proba[2]), 4),
        }

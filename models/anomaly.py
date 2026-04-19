"""
Anomaly Detection — Isolation Forest for black swan early warning.
Identifies abnormal price/volume/OI behavior patterns.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from loguru import logger
from sklearn.ensemble import IsolationForest

from database.init_db import get_connection
from utils.config import load_config, get_project_root


class AnomalyDetector:
    """Detects anomalous market conditions using Isolation Forest."""

    ANOMALY_FEATURES = [
        "price_change_1d", "price_change_7d", "volatility_30d",
        "volume_change", "funding_rate", "oi_change_rate",
    ]

    def __init__(self):
        cfg = load_config()
        self.contamination = cfg.get("model", {}).get("anomaly", {}).get("contamination", 0.05)
        self.model_path = get_project_root() / "data" / "anomaly_model.joblib"
        self.model = None

    def train(self) -> dict:
        """Train Isolation Forest on historical features."""
        conn = get_connection()
        try:
            df = pd.read_sql_query("SELECT * FROM table_features ORDER BY timestamp", conn)
        finally:
            conn.close()

        available = [c for c in self.ANOMALY_FEATURES if c in df.columns]
        X = df[available].apply(pd.to_numeric, errors="coerce")
        # Drop columns >90% NaN, then fill remaining
        null_pct = X.isna().mean()
        X = X.drop(columns=null_pct[null_pct > 0.9].index.tolist())
        X = X.fillna(0)
        # Drop rows that were all-NaN originally
        X = X[(X != 0).any(axis=1)]

        if len(X) < 50:
            return {"error": f"Not enough data ({len(X)} rows). Need at least 50."}

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=200,
        )
        self.model.fit(X)

        # Score training data
        scores = self.model.decision_function(X)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._trained_features = list(X.columns)
        joblib.dump({"model": self.model, "features": self._trained_features}, self.model_path)

        return {
            "trained_on": len(X),
            "features": available,
            "anomaly_threshold": round(float(np.percentile(scores, self.contamination * 100)), 4),
        }

    def detect(self) -> dict:
        """Check if the latest data point is anomalous."""
        if self.model is None:
            if self.model_path.exists():
                saved = joblib.load(self.model_path)
                if isinstance(saved, dict):
                    self.model = saved["model"]
                    self._trained_features = saved.get("features", self.ANOMALY_FEATURES)
                else:
                    self.model = saved
                    self._trained_features = self.ANOMALY_FEATURES
            else:
                return {"error": "No trained model. Run train() first."}

        trained_feats = getattr(self, "_trained_features", self.ANOMALY_FEATURES)

        conn = get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM table_features ORDER BY timestamp DESC LIMIT 1", conn,
            )
        finally:
            conn.close()

        if df.empty:
            return {"error": "No features available."}

        # Use same features as training
        for col in trained_feats:
            if col not in df.columns:
                df[col] = 0
        X = df[trained_feats].apply(pd.to_numeric, errors="coerce").fillna(0)

        prediction = self.model.predict(X)[0]
        score = self.model.decision_function(X)[0]

        is_anomaly = prediction == -1

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(float(score), 4),
            "alert": "⚠️ ANOMALY DETECTED — unusual market behavior" if is_anomaly else "✅ Normal",
            "timestamp": df["timestamp"].iloc[0] if "timestamp" in df.columns else None,
        }

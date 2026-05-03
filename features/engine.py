"""
Feature engineering pipeline for BTC-Pulse.
Computes all technical, sentiment, and derivative features from raw data.
Stores results in table_features for model training and inference.
"""

import pandas as pd
import numpy as np
from loguru import logger

from database.init_db import get_connection
from utils.config import load_config


class FeatureEngine:
    """Computes multi-dimensional features from raw data tables."""

    def __init__(self):
        cfg = load_config()
        feat_cfg = cfg.get("features", {})
        self.ma_periods = feat_cfg.get("ma_periods", [7, 14, 30, 90, 200])
        self.fng_low = feat_cfg.get("fng_extreme_low", 20)
        self.fng_high = feat_cfg.get("fng_extreme_high", 80)
        self.funding_days = feat_cfg.get("funding_cumulative_days", 30)

    def _load_raw_data(self) -> dict[str, pd.DataFrame]:
        """Load all raw data from SQLite into DataFrames."""
        conn = get_connection()
        try:
            market = pd.read_sql_query(
                "SELECT timestamp, open, high, low, close, volume, quote_volume "
                "FROM table_market_price ORDER BY timestamp",
                conn, parse_dates=["timestamp"],
            )
            sentiment = pd.read_sql_query(
                "SELECT timestamp, fng_value, fng_classification "
                "FROM table_sentiment ORDER BY timestamp",
                conn, parse_dates=["timestamp"],
            )
            derivatives = pd.read_sql_query(
                "SELECT timestamp, funding_rate, open_interest "
                "FROM table_derivatives ORDER BY timestamp",
                conn, parse_dates=["timestamp"],
            )
        finally:
            conn.close()

        return {"market": market, "sentiment": sentiment, "derivatives": derivatives}

    def _compute_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute moving averages, returns, volatility, VWAP."""
        # Moving averages
        for period in self.ma_periods:
            df[f"ma{period}"] = df["close"].rolling(window=period).mean()

        # MA30 deviation (percentage)
        df["ma30_deviation"] = (df["close"] - df["ma30"]) / df["ma30"] * 100

        # Price changes
        df["price_change_1d"] = df["close"].pct_change(1)
        df["price_change_7d"] = df["close"].pct_change(7)
        df["price_change_30d"] = df["close"].pct_change(30)

        # 30-day rolling volatility (annualized)
        df["volatility_30d"] = df["price_change_1d"].rolling(30).std() * np.sqrt(365)

        # Volume features
        df["volume_ma7"] = df["volume"].rolling(7).mean()
        df["volume_change"] = df["volume"].pct_change(1)

        # Monthly VWAP (rolling 30-day)
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["tp_vol"] = df["typical_price"] * df["volume"]
        df["monthly_vwap"] = df["tp_vol"].rolling(30).sum() / df["volume"].rolling(30).sum()
        df["vwap_deviation"] = (df["close"] - df["monthly_vwap"]) / df["monthly_vwap"] * 100

        # Clean up temp columns
        df = df.drop(columns=["typical_price", "tp_vol"], errors="ignore")

        return df

    def _compute_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute FNG-based features."""
        if "fng_value" not in df.columns or df["fng_value"].isna().all():
            return df

        df["fng_ma7"] = df["fng_value"].rolling(7).mean()
        df["fng_extreme_low"] = (df["fng_value"] < self.fng_low).astype(int)
        df["fng_extreme_high"] = (df["fng_value"] > self.fng_high).astype(int)

        return df

    def _compute_derivatives_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute funding rate and OI features."""
        if "funding_rate" not in df.columns:
            return df

        # Cumulative funding over N days
        df["cumulative_funding_30d"] = df["funding_rate"].rolling(self.funding_days).sum()

        # OI change rate
        if "open_interest" in df.columns:
            # Don't forward-fill NaN values - compute percent change as-is
            df["oi_change_rate"] = df["open_interest"].pct_change(1)

            # OI-Price divergence: price up but OI down (bearish signal)
            price_up = df["price_change_7d"] > 0
            oi_down = df["open_interest"].pct_change(7) < -0.05
            df["oi_price_divergence"] = (price_up & oi_down).astype(int)

        return df

    def _compute_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute forward-looking labels for model training."""
        # Forward 30-day return
        df["forward_return_30d"] = df["close"].shift(-30) / df["close"] - 1

        # Trend classification
        df["forward_trend"] = "neutral"
        df.loc[df["forward_return_30d"] > 0.05, "forward_trend"] = "bullish"
        df.loc[df["forward_return_30d"] < -0.05, "forward_trend"] = "bearish"

        return df

    def _merge_ai_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge the latest AI sentiment score from table_ai_sentiment."""
        conn = get_connection()
        try:
            ai_sent = pd.read_sql_query(
                """SELECT DATE(timestamp) as timestamp, AVG(sentiment_score) as ai_sentiment_score
                   FROM table_ai_sentiment
                   GROUP BY DATE(timestamp)
                   ORDER BY timestamp""",
                conn,
            )
        finally:
            conn.close()

        if ai_sent.empty:
            return df

        # Convert to date string for merging
        ai_sent["timestamp"] = pd.to_datetime(ai_sent["timestamp"])
        df = df.merge(ai_sent, on="timestamp", how="left")
        return df

    def compute_all(self) -> pd.DataFrame:
        """Run full feature engineering pipeline."""
        logger.info("Loading raw data...")
        raw = self._load_raw_data()

        market = raw["market"].copy()
        if market.empty:
            logger.warning("No market data available for feature computation")
            return pd.DataFrame()

        # Compute price features
        logger.info("Computing price features...")
        market = self._compute_price_features(market)

        # Merge sentiment
        sentiment = raw["sentiment"]
        if not sentiment.empty:
            market = market.merge(sentiment, on="timestamp", how="left")

        # Merge derivatives
        derivatives = raw["derivatives"]
        if not derivatives.empty:
            market = market.merge(derivatives, on="timestamp", how="left")

        # Compute derived features
        logger.info("Computing sentiment features...")
        market = self._compute_sentiment_features(market)

        logger.info("Computing derivatives features...")
        market = self._compute_derivatives_features(market)

        logger.info("Computing forward labels...")
        market = self._compute_labels(market)

        # Merge AI sentiment scores (from LLM analysis)
        logger.info("Merging AI sentiment scores...")
        market = self._merge_ai_sentiment(market)

        # Select feature columns
        feature_cols = [
            "timestamp", "ma7", "ma14", "ma30", "ma90", "ma200",
            "ma30_deviation", "price_change_1d", "price_change_7d",
            "price_change_30d", "volatility_30d", "volume_ma7", "volume_change",
            "monthly_vwap", "vwap_deviation",
            "fng_value", "fng_ma7", "fng_extreme_low", "fng_extreme_high",
            "funding_rate", "cumulative_funding_30d", "oi_change_rate",
            "oi_price_divergence", "ai_sentiment_score",
            "forward_return_30d", "forward_trend",
        ]

        # Only keep columns that exist
        available = [c for c in feature_cols if c in market.columns]
        features = market[available].copy()

        logger.info(f"Computed {len(features)} feature rows with {len(available)} columns")
        return features

    def store(self, df: pd.DataFrame) -> int:
        """Store computed features into table_features."""
        if df.empty:
            return 0

        conn = get_connection()
        inserted = 0
        try:
            # Format timestamp as string for storage
            df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d")

            for _, row in df.iterrows():
                vals = {k: (None if pd.isna(v) else v) for k, v in row.items()}
                cols = ", ".join(vals.keys())
                placeholders = ", ".join(["?"] * len(vals))
                try:
                    conn.execute(
                        f"INSERT OR REPLACE INTO table_features ({cols}) VALUES ({placeholders})",
                        list(vals.values()),
                    )
                    inserted += 1
                except Exception as e:
                    logger.warning(f"Feature store error for {vals.get('timestamp')}: {e}")

            conn.commit()
            logger.info(f"Stored {inserted} feature rows")
        finally:
            conn.close()

        return inserted

    def run(self) -> int:
        """Full compute-and-store pipeline."""
        df = self.compute_all()
        return self.store(df)

"""
Signal Generator — combines rule-based signals with ML predictions.
Three core modules: Contrarian Sentiment, Leverage Purge, Institutional Benchmark.
"""

import pandas as pd
import numpy as np
from loguru import logger

from database.init_db import get_connection
from utils.config import load_config


class SignalGenerator:
    """
    Generates trading signals from computed features.
    Each module produces a score from -1 (bearish) to +1 (bullish).
    """

    def __init__(self):
        cfg = load_config()
        sig_cfg = cfg.get("signals", {})
        feat_cfg = cfg.get("features", {})

        self.fng_buy = sig_cfg.get("fng_buy_threshold", 15)
        self.fng_sell = sig_cfg.get("fng_sell_threshold", 85)
        self.funding_sell = sig_cfg.get("funding_sell_threshold", 0.02)
        self.fng_low = feat_cfg.get("fng_extreme_low", 20)
        self.fng_high = feat_cfg.get("fng_extreme_high", 80)

    def load_features(self, days: int = 90) -> pd.DataFrame:
        """Load recent features from database."""
        conn = get_connection()
        try:
            df = pd.read_sql_query(
                f"""SELECT * FROM table_features 
                    ORDER BY timestamp DESC LIMIT {days}""",
                conn,
            )
        finally:
            conn.close()

        if df.empty:
            return df

        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def contrarian_sentiment(self, df: pd.DataFrame) -> dict:
        """
        Module 1: Contrarian Sentiment
        - FNG < 15 + negative funding → bullish signal
        - FNG > 85 + high cumulative funding → bearish signal
        """
        if df.empty or "fng_value" not in df.columns:
            return {"score": 0, "signal": "neutral", "detail": "No sentiment data"}

        latest = df.iloc[-1]
        fng = latest.get("fng_value")

        if pd.isna(fng):
            return {"score": 0, "signal": "neutral", "detail": "FNG unavailable"}

        fng = int(fng)
        score = 0
        detail_parts = [f"FNG={fng}"]

        # Extreme fear → contrarian bullish
        if fng < self.fng_buy:
            score = 0.8
            # Check if funding is also negative (stronger signal)
            funding = latest.get("funding_rate")
            if not pd.isna(funding) and funding < 0:
                score = 1.0
                detail_parts.append(f"funding={funding:.6f} (negative=bullish)")
        # Extreme greed → contrarian bearish
        elif fng > self.fng_sell:
            score = -0.8
            cum_funding = latest.get("cumulative_funding_30d")
            if not pd.isna(cum_funding) and cum_funding > self.funding_sell:
                score = -1.0
                detail_parts.append(f"cum_funding_30d={cum_funding:.4f} (overheated)")
        else:
            # Moderate range
            score = (50 - fng) / 100  # Linear scale: FNG=0→+0.5, FNG=50→0, FNG=100→-0.5

        # Check for consecutive extreme days
        recent_fng = df["fng_value"].tail(3)
        if (recent_fng < self.fng_low).all():
            score = min(score + 0.2, 1.0)
            detail_parts.append("3-day extreme fear streak")
        elif (recent_fng > self.fng_high).all():
            score = max(score - 0.2, -1.0)
            detail_parts.append("3-day extreme greed streak")

        signal = "bullish" if score > 0.3 else ("bearish" if score < -0.3 else "neutral")

        return {"score": round(score, 3), "signal": signal, "detail": " | ".join(detail_parts)}

    def leverage_purge(self, df: pd.DataFrame) -> dict:
        """
        Module 2: Leverage Purge
        - Cumulative funding > 1.5% + price stagnation → bearish (overweight longs)
        - OI-Price divergence → trend exhaustion warning
        """
        if df.empty:
            return {"score": 0, "signal": "neutral", "detail": "No data"}

        latest = df.iloc[-1]
        score = 0
        detail_parts = []

        cum_funding = latest.get("cumulative_funding_30d")
        price_change = latest.get("price_change_30d")

        if not pd.isna(cum_funding):
            detail_parts.append(f"cum_funding_30d={cum_funding:.4f}")

            # High cumulative funding + price stagnation = "heavy car"
            if cum_funding > 0.015:  # >1.5%
                if not pd.isna(price_change) and abs(price_change) < 0.05:
                    score = -0.8
                    detail_parts.append("overweight longs + stagnant price")
                else:
                    score = -0.5
                    detail_parts.append("high leverage cost")
            elif cum_funding < -0.005:
                score = 0.5
                detail_parts.append("shorts paying → potential squeeze")

        # OI-Price divergence
        oi_div = latest.get("oi_price_divergence")
        if not pd.isna(oi_div) and oi_div == 1:
            score = max(score - 0.3, -1.0)
            detail_parts.append("OI-price divergence detected")

        if not detail_parts:
            detail_parts.append("Insufficient derivatives data")

        signal = "bullish" if score > 0.3 else ("bearish" if score < -0.3 else "neutral")

        return {"score": round(score, 3), "signal": signal, "detail": " | ".join(detail_parts)}

    def institutional_benchmark(self, df: pd.DataFrame) -> dict:
        """
        Module 3: Institutional Benchmark (VWAP)
        - Price above monthly VWAP with volume → bullish continuation
        - Price below VWAP → bearish
        """
        if df.empty or "monthly_vwap" not in df.columns:
            return {"score": 0, "signal": "neutral", "detail": "No VWAP data"}

        latest = df.iloc[-1]
        score = 0
        detail_parts = []

        vwap_dev = latest.get("vwap_deviation")

        if pd.isna(vwap_dev):
            return {"score": 0, "signal": "neutral", "detail": "VWAP unavailable"}

        detail_parts.append(f"VWAP_deviation={vwap_dev:.2f}%")

        if vwap_dev > 2:
            # Price well above VWAP
            vol_change = latest.get("volume_change")
            if not pd.isna(vol_change) and vol_change > 0:
                score = 0.7
                detail_parts.append("above VWAP + volume expansion")
            else:
                score = 0.4
                detail_parts.append("above VWAP but weak volume")
        elif vwap_dev < -2:
            score = -0.6
            detail_parts.append("below VWAP")
        else:
            score = vwap_dev / 10  # Small linear score near VWAP
            detail_parts.append("near VWAP (neutral zone)")

        # Check breakout-retest pattern
        if len(df) >= 3:
            recent_dev = df["vwap_deviation"].tail(3)
            if not recent_dev.isna().any():
                # Was below, now above = breakout
                if recent_dev.iloc[0] < 0 and recent_dev.iloc[-1] > 0:
                    score = min(score + 0.3, 1.0)
                    detail_parts.append("VWAP breakout detected")

        signal = "bullish" if score > 0.3 else ("bearish" if score < -0.3 else "neutral")

        return {"score": round(score, 3), "signal": signal, "detail": " | ".join(detail_parts)}

    def composite_signal(self, df: pd.DataFrame) -> dict:
        """
        Combine all three modules into a composite trend rating.
        Equal weights by default (AI model can override).
        """
        cs = self.contrarian_sentiment(df)
        lp = self.leverage_purge(df)
        ib = self.institutional_benchmark(df)

        # Equal-weighted composite
        composite = (cs["score"] + lp["score"] + ib["score"]) / 3

        # Trend rating
        if composite > 0.3:
            rating = "BULLISH"
            emoji = "🟢"
        elif composite < -0.3:
            rating = "BEARISH"
            emoji = "🔴"
        else:
            rating = "NEUTRAL"
            emoji = "🟡"

        return {
            "composite_score": round(composite, 3),
            "rating": rating,
            "emoji": emoji,
            "modules": {
                "contrarian_sentiment": cs,
                "leverage_purge": lp,
                "institutional_benchmark": ib,
            },
        }

    def store_prediction(self, result: dict) -> None:
        """Store prediction to database."""
        from datetime import datetime, timezone

        conn = get_connection()
        try:
            modules = result.get("modules", {})
            conn.execute(
                """INSERT INTO table_predictions 
                   (timestamp, model_version, trend_probability_up, trend_rating,
                    contrarian_score, leverage_score, institutional_score,
                    composite_score, signal)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "rules_v1",
                    max(0, (result["composite_score"] + 1) / 2),  # Map to 0-1
                    result["rating"],
                    modules.get("contrarian_sentiment", {}).get("score", 0),
                    modules.get("leverage_purge", {}).get("score", 0),
                    modules.get("institutional_benchmark", {}).get("score", 0),
                    result["composite_score"],
                    result["rating"],
                ),
            )
            conn.commit()
            logger.info(f"Stored prediction: {result['rating']} ({result['composite_score']})")
        finally:
            conn.close()

    def run(self) -> dict:
        """Generate and store composite signal (rule-based + ML consensus)."""
        df = self.load_features()
        result = self.composite_signal(df)

        # Attempt to blend with XGBoost prediction for consensus
        try:
            from models.xgboost_model import XGBoostCombiner
            xgb = XGBoostCombiner()
            ml_pred = xgb.predict()
            if "error" not in ml_pred:
                # Blend: 60% rule-based, 40% ML
                ml_score = (ml_pred["trend_up_probability"] - 0.5) * 2  # Map 0-1 → -1 to +1
                blended = result["composite_score"] * 0.6 + ml_score * 0.4
                result["ml_prediction"] = ml_pred
                result["blended_score"] = round(blended, 3)

                # Override rating if blended score gives different signal
                if blended > 0.3:
                    result["rating"] = "BULLISH"
                    result["emoji"] = "🟢"
                elif blended < -0.3:
                    result["rating"] = "BEARISH"
                    result["emoji"] = "🔴"
                else:
                    result["rating"] = "NEUTRAL"
                    result["emoji"] = "🟡"

                result["composite_score"] = blended
                logger.info(f"Blended signal (rules+ML): {result['rating']} ({blended:.3f})")
        except Exception as e:
            logger.debug(f"ML blend unavailable (falling back to rules-only): {e}")

        self.store_prediction(result)
        return result

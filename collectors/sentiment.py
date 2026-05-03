"""
Sentiment collector — fetches Fear & Greed Index from Alternative.me.
Free API, daily updates.
"""

import requests
import pandas as pd
from datetime import datetime, timezone
from loguru import logger

from database.init_db import get_connection, get_last_timestamp
from utils.retry import safe_api_call


class SentimentCollector:
    """Collects Bitcoin Fear & Greed Index from Alternative.me."""

    FNG_URL = "https://api.alternative.me/fng/"

    @safe_api_call
    def _fetch_fng(self, limit: int = 0) -> list[dict]:
        """
        Fetch FNG data. limit=0 means all available history.
        Returns list of {value, value_classification, timestamp}.
        """
        params = {"limit": limit, "format": "json"}
        resp = requests.get(self.FNG_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])

    def collect(self) -> pd.DataFrame:
        """Fetch new FNG data since last stored record."""
        last_ts = get_last_timestamp("table_sentiment")

        if last_ts:
            # Fetch recent data (last 30 days should cover any gaps)
            raw = self._fetch_fng(limit=30)
            logger.info(f"Incremental FNG fetch, last stored: {last_ts}")
        else:
            # First run: fetch all history
            raw = self._fetch_fng(limit=0)
            logger.info(f"Initial FNG fetch: {len(raw)} records")

        if not raw:
            logger.info("No FNG data returned")
            return pd.DataFrame()

        records = []
        for item in raw:
            ts = datetime.fromtimestamp(int(item["timestamp"]), tz=timezone.utc)
            records.append({
                "timestamp": ts.strftime("%Y-%m-%d"),
                "fng_value": int(item["value"]),
                "fng_classification": item.get("value_classification", ""),
            })

        df = pd.DataFrame(records)

        # Filter out already-stored dates
        if last_ts:
            df = df[df["timestamp"] > last_ts]

        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        return df

    def store(self, df: pd.DataFrame) -> int:
        """Store FNG data into SQLite."""
        if df.empty:
            return 0

        conn = get_connection()
        inserted = 0
        try:
            for _, row in df.iterrows():
                try:
                    cursor = conn.execute(
                        """INSERT OR IGNORE INTO table_sentiment
                           (timestamp, fng_value, fng_classification)
                           VALUES (?, ?, ?)""",
                        (row["timestamp"], row["fng_value"], row["fng_classification"]),
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                except Exception as e:
                    logger.warning(f"Skip {row['timestamp']}: {e}")
            conn.commit()
            logger.info(f"Stored {inserted} FNG records")
        finally:
            conn.close()

        return inserted

    def run(self) -> int:
        """Full collect-and-store pipeline."""
        df = self.collect()
        return self.store(df)

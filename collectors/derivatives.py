"""
Derivatives collector — fetches funding rate, OI, and liquidation data.
Uses Coinglass API (requires API key for full access) with Binance public fallback.
"""

import requests
import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
from loguru import logger

from database.init_db import get_connection, get_last_timestamp
from utils.config import get_api_key, load_config
from utils.retry import safe_api_call


class DerivativesCollector:
    """Collects derivatives data: funding rate, OI, liquidations."""

    COINGLASS_BASE = "https://open-api.coinglass.com/public/v2"

    def __init__(self):
        self.coinglass_key = get_api_key("coinglass")
        cfg = load_config()
        exchange_id = cfg.get("data", {}).get("exchange", "binance")
        self.exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})

    @safe_api_call
    def _fetch_funding_from_binance(self, since_ms: int) -> list[dict]:
        """Fallback: fetch funding rate history from Binance futures API."""
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        all_data = []
        start = since_ms

        while True:
            params = {
                "symbol": "BTCUSDT",
                "startTime": start,
                "limit": 1000,
            }
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            batch = resp.json()

            if not batch:
                break

            all_data.extend(batch)
            start = batch[-1]["fundingTime"] + 1

            if len(batch) < 1000:
                break

        return all_data

    @safe_api_call
    def _fetch_oi_from_binance(self) -> dict | None:
        """Fetch current open interest from Binance futures."""
        url = "https://fapi.binance.com/fapi/v1/openInterest"
        resp = requests.get(url, params={"symbol": "BTCUSDT"}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    @safe_api_call
    def _fetch_coinglass_funding(self) -> list[dict]:
        """Fetch funding rate from Coinglass (if key available)."""
        if not self.coinglass_key:
            return []

        headers = {"coinglassSecret": self.coinglass_key}
        url = f"{self.COINGLASS_BASE}/funding"
        params = {"symbol": "BTC", "time_type": "all"}
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])

    def collect(self) -> pd.DataFrame:
        """Fetch derivatives data. Uses Binance public API as primary source."""
        last_ts = get_last_timestamp("table_derivatives")

        if last_ts:
            since_dt = datetime.fromisoformat(last_ts) + timedelta(hours=8)
            logger.info(f"Incremental derivatives fetch from {since_dt}")
        else:
            since_dt = datetime(2020, 1, 1, tzinfo=timezone.utc)
            logger.info("Initial derivatives fetch from 2020-01-01")

        since_ms = int(since_dt.timestamp() * 1000)

        # Fetch funding rate history from Binance
        raw_funding = self._fetch_funding_from_binance(since_ms)

        if not raw_funding:
            logger.info("No new funding rate data")
            return pd.DataFrame()

        # Group by date and take the daily average funding rate
        records = []
        for item in raw_funding:
            ts = datetime.fromtimestamp(item["fundingTime"] / 1000, tz=timezone.utc)
            records.append({
                "timestamp": ts.strftime("%Y-%m-%d"),
                "funding_rate": float(item["fundingRate"]),
                "funding_time": ts,
            })

        df = pd.DataFrame(records)

        # Aggregate to daily: mean funding rate
        daily = df.groupby("timestamp").agg(
            funding_rate=("funding_rate", "mean"),
        ).reset_index()

        # Try to get current OI
        try:
            oi_data = self._fetch_oi_from_binance()
            if oi_data:
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                oi_val = float(oi_data.get("openInterest", 0))
                # Add OI to the most recent row or create a row
                if not daily.empty:
                    daily.loc[daily["timestamp"] == daily["timestamp"].max(), "open_interest"] = oi_val
        except Exception as e:
            logger.warning(f"Could not fetch OI: {e}")

        if "open_interest" not in daily.columns:
            daily["open_interest"] = None

        daily["long_liquidations"] = None
        daily["short_liquidations"] = None
        daily["long_short_ratio"] = None

        # Filter out already stored
        if last_ts:
            daily = daily[daily["timestamp"] > last_ts]

        daily = daily.drop_duplicates(subset=["timestamp"], keep="last")
        return daily

    def store(self, df: pd.DataFrame) -> int:
        """Store derivatives data into SQLite."""
        if df.empty:
            return 0

        conn = get_connection()
        inserted = 0
        try:
            for _, row in df.iterrows():
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO table_derivatives
                           (timestamp, funding_rate, open_interest,
                            long_liquidations, short_liquidations, long_short_ratio)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (row["timestamp"], row.get("funding_rate"),
                         row.get("open_interest"), row.get("long_liquidations"),
                         row.get("short_liquidations"), row.get("long_short_ratio")),
                    )
                    inserted += 1
                except Exception as e:
                    logger.warning(f"Skip {row['timestamp']}: {e}")
            conn.commit()
            logger.info(f"Stored {inserted} derivatives records")
        finally:
            conn.close()

        return inserted

    def run(self) -> int:
        """Full collect-and-store pipeline."""
        df = self.collect()
        return self.store(df)

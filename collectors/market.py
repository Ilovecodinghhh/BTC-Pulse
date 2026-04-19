"""
Market price collector — fetches daily OHLCV from Binance via CCXT.
Supports incremental updates (only fetches missing data).
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
from loguru import logger

from database.init_db import get_connection, get_last_timestamp
from utils.config import load_config
from utils.retry import safe_api_call


class MarketCollector:
    """Collects BTC/USDT daily OHLCV data from Binance."""

    def __init__(self):
        cfg = load_config()
        exchange_id = cfg.get("data", {}).get("exchange", "binance")
        self.symbol = cfg.get("data", {}).get("symbol", "BTC/USDT")
        self.exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self.timeframe = "1d"

    @safe_api_call
    def _fetch_ohlcv(self, since_ms: int, limit: int = 1000) -> list:
        """Fetch OHLCV data from exchange."""
        return self.exchange.fetch_ohlcv(
            self.symbol,
            timeframe=self.timeframe,
            since=since_ms,
            limit=limit,
        )

    def collect(self) -> pd.DataFrame:
        """
        Fetch new OHLCV data since last stored record.
        Returns DataFrame of newly fetched rows.
        """
        last_ts = get_last_timestamp("table_market_price")

        if last_ts:
            since_dt = datetime.fromisoformat(last_ts) + timedelta(days=1)
            logger.info(f"Incremental fetch from {since_dt.date()}")
        else:
            # First run: fetch from 2020-01-01
            since_dt = datetime(2020, 1, 1, tzinfo=timezone.utc)
            logger.info("Initial fetch from 2020-01-01")

        since_ms = int(since_dt.timestamp() * 1000)
        all_data = []

        while True:
            batch = self._fetch_ohlcv(since_ms)
            if not batch:
                break

            all_data.extend(batch)
            logger.debug(f"Fetched {len(batch)} candles")

            # Move to next batch
            since_ms = batch[-1][0] + 1

            # If we got less than limit, we've reached the end
            if len(batch) < 1000:
                break

        if not all_data:
            logger.info("No new market data to fetch")
            return pd.DataFrame()

        df = pd.DataFrame(all_data, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.strftime("%Y-%m-%d")
        df["quote_volume"] = df["volume"] * df["close"]  # Approximate
        df = df.drop(columns=["timestamp_ms"])

        # Remove today's incomplete candle
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        df = df[df["timestamp"] < today]

        # Deduplicate
        df = df.drop_duplicates(subset=["timestamp"], keep="last")

        return df

    def store(self, df: pd.DataFrame) -> int:
        """Store fetched data into SQLite. Returns number of rows inserted."""
        if df.empty:
            return 0

        conn = get_connection()
        inserted = 0
        try:
            for _, row in df.iterrows():
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO table_market_price 
                           (timestamp, open, high, low, close, volume, quote_volume)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (row["timestamp"], row["open"], row["high"], row["low"],
                         row["close"], row["volume"], row["quote_volume"]),
                    )
                    inserted += 1
                except Exception as e:
                    logger.warning(f"Skip duplicate or error for {row['timestamp']}: {e}")
            conn.commit()
            logger.info(f"Stored {inserted} market price records")
        finally:
            conn.close()

        return inserted

    def run(self) -> int:
        """Full collect-and-store pipeline."""
        df = self.collect()
        return self.store(df)

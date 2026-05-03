"""
Market price collector — fetches BTC price and volume data.

Primary source: Bitcoinity (free CSV, monthly data back to 2010)
Secondary source: Exchange OHLCV via CCXT (daily granularity, fills recent gaps)

The Bitcoinity data provides long-history coverage. CCXT daily data is
overlaid on top for finer granularity (daily candles with OHLC).
"""

import ccxt
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from io import StringIO
from loguru import logger

from database.init_db import get_connection, get_last_timestamp
from utils.config import load_config
from utils.retry import safe_api_call


BITCOINITY_URL = (
    "https://data.bitcoinity.org/export_data.csv"
    "?currency=USD&data_type=price_volume&t=lb&timespan=all&vu=curr"
)


class MarketCollector:
    """
    Collects BTC/USD(T) price and volume data.

    Strategy:
      1. Fetch Bitcoinity CSV (monthly price+volume, 2010–present) — always
         attempted first for maximum historical coverage.
      2. Fetch daily OHLCV from exchange (Binance/OKX/Bybit/Kraken) for
         granular recent data.
      3. Daily exchange data takes precedence over Bitcoinity for any
         overlapping dates (higher resolution).
    """

    FALLBACK_EXCHANGES = ["binance", "okx", "bybit", "kraken"]

    def __init__(self):
        cfg = load_config()
        exchange_id = cfg.get("data", {}).get("exchange", "binance")
        self.symbol = cfg.get("data", {}).get("symbol", "BTC/USDT")
        self.timeframe = "1d"
        self.start_date = cfg.get("data", {}).get("start_date", "2020-01-01")

        self._exchange = None
        self._preferred_exchange = exchange_id

    @property
    def exchange(self):
        """Lazy-init exchange (avoids network call if only Bitcoinity is needed)."""
        if self._exchange is None:
            self._exchange = self._init_exchange(self._preferred_exchange)
        return self._exchange

    def _init_exchange(self, preferred: str):
        """Initialize exchange with fallback if primary is unavailable."""
        exchanges_to_try = [preferred] + [e for e in self.FALLBACK_EXCHANGES if e != preferred]

        for ex_id in exchanges_to_try:
            try:
                ex_class = getattr(ccxt, ex_id, None)
                if ex_class is None:
                    continue
                ex = ex_class({"enableRateLimit": True})
                ex.load_markets()
                logger.info(f"Using exchange: {ex_id}")
                return ex
            except Exception as e:
                logger.warning(f"Exchange {ex_id} unavailable: {e}")

        logger.error("All exchange fallbacks failed — using preferred without validation")
        return getattr(ccxt, preferred)({"enableRateLimit": True})

    # ── Bitcoinity (primary: long history) ─────────────────────────

    def collect_bitcoinity(self) -> pd.DataFrame:
        """
        Fetch Bitcoinity CSV — monthly price+volume since 2010.
        Returns DataFrame with columns: timestamp, open, high, low, close, volume, quote_volume
        (open/high/low are set equal to close since Bitcoinity only provides avg price)
        """
        try:
            logger.info("Fetching Bitcoinity historical data...")
            resp = requests.get(BITCOINITY_URL, timeout=30, headers={
                "User-Agent": "BTC-Pulse/2.0 (market-collector)"
            })
            resp.raise_for_status()

            df = pd.read_csv(StringIO(resp.text))

            if df.empty:
                logger.warning("Bitcoinity returned empty data")
                return pd.DataFrame()

            # Columns: Time, price, volume
            df = df.rename(columns={
                "Time": "timestamp_raw",
                "price": "close",
                "volume": "quote_volume",
            })

            df["timestamp"] = pd.to_datetime(df["timestamp_raw"]).dt.strftime("%Y-%m-%d")

            # Bitcoinity provides avg price, not OHLC — use price for all
            df["open"] = df["close"]
            df["high"] = df["close"]
            df["low"] = df["close"]

            # Estimate BTC volume from quote volume / price
            df["volume"] = df["quote_volume"] / df["close"].replace(0, float("nan"))

            df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]]
            df = df.dropna(subset=["close"])
            df = df.drop_duplicates(subset=["timestamp"], keep="last")

            logger.info(f"Bitcoinity: {len(df)} monthly records ({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]})")
            return df

        except Exception as e:
            logger.warning(f"Bitcoinity fetch failed: {e}")
            return pd.DataFrame()

    # ── Exchange OHLCV (secondary: daily granularity) ──────────────

    @safe_api_call
    def _fetch_ohlcv(self, since_ms: int, limit: int = 1000) -> list:
        """Fetch OHLCV data from exchange."""
        return self.exchange.fetch_ohlcv(
            self.symbol,
            timeframe=self.timeframe,
            since=since_ms,
            limit=limit,
        )

    def collect_exchange(self) -> pd.DataFrame:
        """
        Fetch new daily OHLCV data from exchange since last stored record.
        Returns DataFrame of newly fetched rows.
        """
        last_ts = get_last_timestamp("table_market_price")

        if last_ts:
            since_dt = datetime.fromisoformat(last_ts) + timedelta(days=1)
            logger.info(f"Exchange incremental fetch from {since_dt.date()}")
        else:
            since_dt = datetime.fromisoformat(self.start_date).replace(tzinfo=timezone.utc)
            logger.info(f"Exchange initial fetch from {self.start_date}")

        since_ms = int(since_dt.timestamp() * 1000)
        all_data = []

        while True:
            batch = self._fetch_ohlcv(since_ms)
            if not batch:
                break

            all_data.extend(batch)
            logger.debug(f"Fetched {len(batch)} candles")

            since_ms = batch[-1][0] + 1

            if len(batch) < 1000:
                break

        if not all_data:
            logger.info("No new exchange data to fetch")
            return pd.DataFrame()

        df = pd.DataFrame(all_data, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.strftime("%Y-%m-%d")
        df["quote_volume"] = df["volume"] * df["close"]
        df = df.drop(columns=["timestamp_ms"])

        # Remove today's incomplete candle
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        df = df[df["timestamp"] < today]

        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        return df

    # ── Combined collect ───────────────────────────────────────────

    def collect(self) -> pd.DataFrame:
        """
        Collect from both sources and merge.
        Daily exchange data overwrites Bitcoinity for overlapping dates.
        """
        bitcoinity_df = self.collect_bitcoinity()
        exchange_df = self.collect_exchange()

        if bitcoinity_df.empty and exchange_df.empty:
            logger.info("No new market data from any source")
            return pd.DataFrame()

        if bitcoinity_df.empty:
            return exchange_df
        if exchange_df.empty:
            return bitcoinity_df

        # Exchange data takes precedence (daily OHLC > monthly avg)
        # Mark source for merge priority
        bitcoinity_df["_source"] = "bitcoinity"
        exchange_df["_source"] = "exchange"

        combined = pd.concat([bitcoinity_df, exchange_df], ignore_index=True)

        # Keep exchange rows where dates overlap (they have real OHLC)
        combined = combined.sort_values(
            ["timestamp", "_source"],
            ascending=[True, False],  # 'exchange' > 'bitcoinity' alphabetically reversed
        )
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = combined.drop(columns=["_source"])
        combined = combined.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Combined: {len(combined)} records "
                    f"({combined['timestamp'].iloc[0]} → {combined['timestamp'].iloc[-1]})")
        return combined

    # ── Store ──────────────────────────────────────────────────────

    def store(self, df: pd.DataFrame) -> int:
        """Store fetched data into SQLite. Returns number of rows inserted."""
        if df.empty:
            return 0

        conn = get_connection()
        inserted = 0
        try:
            for _, row in df.iterrows():
                try:
                    # Use INSERT OR REPLACE so exchange data can overwrite Bitcoinity
                    cursor = conn.execute(
                        """INSERT OR REPLACE INTO table_market_price
                           (timestamp, open, high, low, close, volume, quote_volume)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (row["timestamp"], row["open"], row["high"], row["low"],
                         row["close"], row["volume"], row["quote_volume"]),
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                except Exception as e:
                    logger.warning(f"Store error for {row['timestamp']}: {e}")
            conn.commit()
            logger.info(f"Stored {inserted} market price records")
        finally:
            conn.close()

        return inserted

    def run(self) -> int:
        """Full collect-and-store pipeline."""
        df = self.collect()
        return self.store(df)

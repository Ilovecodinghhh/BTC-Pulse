"""
Freqtrade-Style Data Provider for BTC-Pulse.

Inspired by Freqtrade's DataProvider:
- Multi-timeframe data merging (informative pairs)
- Automatic data staleness detection
- Rate-limited exchange access
- Orderbook data (when available)
- Ticker data caching
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from loguru import logger
from typing import Optional

from database.init_db import get_connection, get_last_timestamp
from utils.config import load_config
from utils.retry import safe_api_call


class DataProvider:
    """
    Centralized data access layer — inspired by Freqtrade's DataProvider.

    Replaces scattered data loading across BTC-Pulse modules with a
    single source of truth. Supports multi-timeframe analysis.
    """

    def __init__(self):
        cfg = load_config()
        exchange_id = cfg.get("data", {}).get("exchange", "binance")
        self.symbol = cfg.get("data", {}).get("symbol", "BTC/USDT")
        self.exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self._cache = {}
        self._cache_ts = {}

    def ohlcv(self, timeframe: str = "1d", days: Optional[int] = None,
              use_cache: bool = True) -> pd.DataFrame:
        """
        Get OHLCV data — from DB for daily, live-fetched for others.

        Multi-timeframe support (Freqtrade's informative_pairs concept):
        - '1d' → from SQLite (primary data)
        - '4h', '1h', '1w' → aggregated or fetched live
        """
        cache_key = f"ohlcv_{timeframe}_{days}"
        if use_cache and cache_key in self._cache:
            age = (datetime.now() - self._cache_ts[cache_key]).seconds
            if age < 300:  # 5 min cache
                return self._cache[cache_key]

        if timeframe == "1d":
            df = self._load_daily_from_db(days)
        elif timeframe == "1w":
            daily = self._load_daily_from_db(days)
            df = self._resample_to_weekly(daily)
        else:
            df = self._fetch_live_ohlcv(timeframe, days)

        self._cache[cache_key] = df
        self._cache_ts[cache_key] = datetime.now()
        return df

    def _load_daily_from_db(self, days: Optional[int] = None) -> pd.DataFrame:
        """Load daily OHLCV from BTC-Pulse database."""
        conn = get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT timestamp, open, high, low, close, volume, quote_volume "
                "FROM table_market_price ORDER BY timestamp",
                conn, parse_dates=["timestamp"],
            )
        finally:
            conn.close()

        if days and not df.empty:
            cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
            df = df[df["timestamp"] >= cutoff]

        return df.reset_index(drop=True)

    def _resample_to_weekly(self, daily: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily data to weekly candles."""
        if daily.empty:
            return daily

        df = daily.set_index("timestamp")
        weekly = df.resample("W").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "quote_volume": "sum",
        }).dropna()

        return weekly.reset_index()

    @safe_api_call
    def _fetch_live_ohlcv(self, timeframe: str, days: Optional[int] = None) -> pd.DataFrame:
        """Fetch live OHLCV from exchange for non-daily timeframes."""
        since = None
        if days:
            since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        data = self.exchange.fetch_ohlcv(
            self.symbol, timeframe=timeframe, since=since, limit=1000)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
        df["quote_volume"] = df["volume"] * df["close"]
        return df.drop(columns=["timestamp_ms"])

    def features(self, days: Optional[int] = None) -> pd.DataFrame:
        """Load computed features from BTC-Pulse feature store."""
        conn = get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM table_features ORDER BY timestamp",
                conn, parse_dates=["timestamp"],
            )
        finally:
            conn.close()

        if days and not df.empty:
            cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
            df = df[df["timestamp"] >= cutoff]

        return df.reset_index(drop=True)

    def sentiment(self, days: Optional[int] = None) -> pd.DataFrame:
        """Load sentiment data."""
        conn = get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM table_sentiment ORDER BY timestamp",
                conn, parse_dates=["timestamp"],
            )
        finally:
            conn.close()

        if days and not df.empty:
            cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
            df = df[df["timestamp"] >= cutoff]

        return df.reset_index(drop=True)

    def derivatives(self, days: Optional[int] = None) -> pd.DataFrame:
        """Load derivatives data."""
        conn = get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM table_derivatives ORDER BY timestamp",
                conn, parse_dates=["timestamp"],
            )
        finally:
            conn.close()

        if days and not df.empty:
            cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
            df = df[df["timestamp"] >= cutoff]

        return df.reset_index(drop=True)

    @safe_api_call
    def ticker(self) -> dict:
        """Get current ticker (last price, bid, ask, volume)."""
        return self.exchange.fetch_ticker(self.symbol)

    @safe_api_call
    def orderbook(self, depth: int = 20) -> dict:
        """
        Get current orderbook — Freqtrade feature for custom pricing.
        Useful for detecting support/resistance walls.
        """
        book = self.exchange.fetch_order_book(self.symbol, limit=depth)

        # Compute bid/ask imbalance (simple version)
        bid_volume = sum(b[1] for b in book["bids"][:depth])
        ask_volume = sum(a[1] for a in book["asks"][:depth])
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-10)

        return {
            "bids": book["bids"][:depth],
            "asks": book["asks"][:depth],
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "imbalance": round(imbalance, 4),
            "spread_pct": round(
                (book["asks"][0][0] - book["bids"][0][0]) / book["bids"][0][0] * 100, 4
            ) if book["bids"] and book["asks"] else 0,
        }

    def data_staleness_check(self) -> dict:
        """
        Check if data is stale — Freqtrade concept.
        Returns staleness info for each data source.
        """
        now = datetime.now(timezone.utc)
        result = {}

        for table, name in [
            ("table_market_price", "market"),
            ("table_sentiment", "sentiment"),
            ("table_derivatives", "derivatives"),
            ("table_features", "features"),
        ]:
            last_ts = get_last_timestamp(table)
            if last_ts:
                last_dt = datetime.fromisoformat(last_ts).replace(tzinfo=timezone.utc)
                age_hours = (now - last_dt).total_seconds() / 3600
                result[name] = {
                    "last_update": last_ts,
                    "age_hours": round(age_hours, 1),
                    "stale": age_hours > 36,  # > 1.5 days is stale for daily data
                }
            else:
                result[name] = {"last_update": None, "age_hours": None, "stale": True}

        return result

    def merge_timeframes(self, primary: pd.DataFrame,
                         informative: pd.DataFrame,
                         suffix: str = "_inf") -> pd.DataFrame:
        """
        Merge informative timeframe data into primary — Freqtrade pattern.
        Forward-fills to avoid look-ahead bias.
        """
        if informative.empty:
            return primary

        # Rename informative columns
        inf_renamed = informative.rename(
            columns={c: f"{c}{suffix}" for c in informative.columns if c != "timestamp"}
        )

        # Merge with forward-fill (no future data)
        merged = pd.merge_asof(
            primary.sort_values("timestamp"),
            inf_renamed.sort_values(f"timestamp"),
            on="timestamp",
            direction="backward",
        )

        return merged

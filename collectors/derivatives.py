"""
Derivatives collector — fetches funding rate, OI, and liquidation data.
Uses CCXT public endpoints (Binance, OKX, Bybit) — NO API keys required.
Multi-exchange aggregation for more robust signals.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from loguru import logger

from database.init_db import get_connection, get_last_timestamp
from utils.config import load_config
from utils.retry import safe_api_call


class DerivativesCollector:
    """
    Collects derivatives data from multiple exchanges via CCXT public APIs.
    No API keys required — uses publicly available endpoints.

    Sources:
    - Funding Rate: averaged across Binance, OKX, Bybit
    - Open Interest: from Binance futures (public)
    """

    def __init__(self):
        cfg = load_config()
        self.futures_symbol = cfg.get("data", {}).get("futures_symbol", "BTC/USDT:USDT")
        self.start_date = cfg.get("data", {}).get("start_date", "2020-01-01")

        # Initialize multiple exchanges for aggregation (all public, no keys)
        self.exchanges = []
        for ExClass in [ccxt.binance, ccxt.okx, ccxt.bybit]:
            try:
                ex = ExClass({
                    "enableRateLimit": True,
                    "options": {"defaultType": "swap"},
                })
                self.exchanges.append(ex)
            except Exception as e:
                logger.warning(f"Failed to init {ExClass.__name__}: {e}")

        # Primary exchange for historical funding
        self.primary_exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })

    @safe_api_call
    def _fetch_multi_exchange_funding(self) -> dict:
        """
        Fetch current funding rate from multiple exchanges.
        Returns average across all available sources.
        No API keys needed — CCXT public endpoints.
        """
        rates = {}
        for ex in self.exchanges:
            try:
                data = ex.fetch_funding_rate(self.futures_symbol)
                if data and "fundingRate" in data:
                    rates[ex.id] = float(data["fundingRate"])
                    logger.debug(f"  {ex.id} funding: {data['fundingRate']}")
            except Exception as e:
                logger.debug(f"  {ex.id} funding unavailable: {e}")
                continue

        if not rates:
            return {"avg_rate": None, "sources": 0, "rates": {}}

        avg_rate = np.mean(list(rates.values()))
        return {
            "avg_rate": avg_rate,
            "sources": len(rates),
            "rates": rates,
        }

    @safe_api_call
    def _fetch_funding_history(self, since_ms: int) -> list[dict]:
        """
        Fetch historical funding rates from Binance futures (public endpoint).
        No API key needed for historical funding rate data.
        """
        import requests

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
    def _fetch_open_interest(self) -> float | None:
        """
        Fetch current open interest via CCXT public API.
        Tries multiple exchanges as fallback.
        """
        for ex in self.exchanges:
            try:
                # CCXT unified method for open interest
                oi = ex.fetch_open_interest(self.futures_symbol)
                if oi and "openInterestAmount" in oi:
                    return float(oi["openInterestAmount"])
                elif oi and "openInterest" in oi:
                    return float(oi["openInterest"])
            except Exception as e:
                logger.debug(f"  {ex.id} OI unavailable: {e}")
                continue

        # Fallback: Binance REST directly
        try:
            import requests
            resp = requests.get(
                "https://fapi.binance.com/fapi/v1/openInterest",
                params={"symbol": "BTCUSDT"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("openInterest", 0))
        except Exception as e:
            logger.warning(f"OI fetch failed entirely: {e}")
            return None

    def collect(self) -> pd.DataFrame:
        """
        Fetch derivatives data using only free/public APIs.
        - Historical funding: Binance public endpoint
        - Current funding: averaged across Binance, OKX, Bybit
        - Open interest: CCXT public endpoint
        """
        last_ts = get_last_timestamp("table_derivatives")

        if last_ts:
            since_dt = datetime.fromisoformat(last_ts) + timedelta(hours=8)
            logger.info(f"Incremental derivatives fetch from {since_dt}")
        else:
            since_dt = datetime.fromisoformat(self.start_date).replace(tzinfo=timezone.utc)
            logger.info(f"Initial derivatives fetch from {self.start_date}")

        since_ms = int(since_dt.timestamp() * 1000)

        # Fetch historical funding rate (free Binance endpoint)
        logger.info("Fetching funding rate history (Binance public API)...")
        raw_funding = self._fetch_funding_history(since_ms)

        if not raw_funding:
            # Try getting at least the current rate from multi-exchange
            current = self._fetch_multi_exchange_funding()
            if current and current["avg_rate"] is not None:
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                oi = self._fetch_open_interest()
                return pd.DataFrame([{
                    "timestamp": today,
                    "funding_rate": current["avg_rate"],
                    "open_interest": oi,
                    "long_liquidations": None,
                    "short_liquidations": None,
                    "long_short_ratio": None,
                }])
            logger.info("No new funding rate data available")
            return pd.DataFrame()

        # Process historical funding into daily records
        records = []
        for item in raw_funding:
            ts = datetime.fromtimestamp(item["fundingTime"] / 1000, tz=timezone.utc)
            records.append({
                "timestamp": ts.strftime("%Y-%m-%d"),
                "funding_rate": float(item["fundingRate"]),
            })

        df = pd.DataFrame(records)

        # Aggregate to daily: mean funding rate
        daily = df.groupby("timestamp").agg(
            funding_rate=("funding_rate", "mean"),
        ).reset_index()

        # Enrich latest row with current multi-exchange average
        current = self._fetch_multi_exchange_funding()
        if current and current["avg_rate"] is not None and current["sources"] > 1:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if not daily.empty and daily["timestamp"].max() == today:
                # Replace today's single-exchange rate with multi-exchange average
                daily.loc[daily["timestamp"] == today, "funding_rate"] = current["avg_rate"]
                logger.info(f"Multi-exchange funding avg: {current['avg_rate']:.6f} "
                           f"(from {current['sources']} sources)")

        # Fetch current open interest (public)
        oi = self._fetch_open_interest()
        if oi is not None:
            daily["open_interest"] = None
            if not daily.empty:
                daily.loc[daily.index[-1], "open_interest"] = oi
                logger.info(f"Current OI: {oi:,.0f} BTC")
        else:
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
                    cursor = conn.execute(
                        """INSERT OR IGNORE INTO table_derivatives
                           (timestamp, funding_rate, open_interest,
                            long_liquidations, short_liquidations, long_short_ratio)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (row["timestamp"], row.get("funding_rate"),
                         row.get("open_interest"), row.get("long_liquidations"),
                         row.get("short_liquidations"), row.get("long_short_ratio")),
                    )
                    if cursor.rowcount > 0:
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

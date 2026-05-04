"""
Freqtrade-Style Strategy Interface for BTC-Pulse.

Borrows from Freqtrade's IStrategy pattern:
- populate_indicators(): compute all technical indicators on a DataFrame
- populate_entry_trend(): generate buy/entry signals
- populate_exit_trend(): generate sell/exit signals
- Custom stoploss and ROI table
- Multi-timeframe support

This gives BTC-Pulse a proper, testable strategy layer instead of ad-hoc
signal generation scattered across modules.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from loguru import logger
from typing import Optional

from database.init_db import get_connection


class BaseStrategy(ABC):
    """
    Abstract base strategy — mirrors Freqtrade's IStrategy interface.
    All BTC-Pulse strategies inherit from this.
    """

    INTERFACE_VERSION = 1

    # Minimal ROI table (time in minutes → minimum ROI to sell)
    # Freqtrade concept: auto-exit when profit target reached
    minimal_roi = {
        "0": 0.10,    # 10% immediately
        "1440": 0.05, # 5% after 1 day
        "4320": 0.02, # 2% after 3 days
        "10080": 0.0, # break-even after 7 days
    }

    # Stoploss
    stoploss = -0.08  # -8% hard stop

    # Trailing stoploss (Freqtrade feature)
    trailing_stop = True
    trailing_stop_positive = 0.03      # Lock in once 3% profit
    trailing_stop_positive_offset = 0.05  # Start trailing at 5%
    trailing_only_offset_is_reached = True

    # Timeframes to analyze (primary + informative)
    timeframe = "1d"
    informative_timeframes = ["1w"]

    @abstractmethod
    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all indicators to the DataFrame. Must be vectorized."""
        pass

    @abstractmethod
    def populate_entry_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set df['enter_long'] = 1 where entry conditions are met."""
        pass

    @abstractmethod
    def populate_exit_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set df['exit_long'] = 1 where exit conditions are met."""
        pass

    def custom_stoploss(self, current_profit: float, current_time: int,
                        entry_price: float, current_price: float,
                        **kwargs) -> float:
        """
        Dynamic stoploss — override for smarts. Returns negative float.
        Freqtrade concept: adjust stoploss based on profit/time/conditions.
        """
        # Default: tighten stoploss as profit grows
        if current_profit > 0.10:
            return -0.03  # Tighten to 3% when up 10%+
        elif current_profit > 0.05:
            return -0.05  # Tighten to 5% when up 5%+
        return self.stoploss

    def confirm_trade_entry(self, df: pd.DataFrame, row_idx: int) -> bool:
        """
        Optional guard — veto a trade before execution.
        Freqtrade concept: last-minute confirmation checks.
        """
        return True

    def confirm_trade_exit(self, df: pd.DataFrame, row_idx: int,
                           profit: float) -> bool:
        """
        Optional guard — veto an exit.
        Useful for: don't sell during anomaly events, etc.
        """
        return True

    def load_market_data(self, days: Optional[int] = None) -> pd.DataFrame:
        """Load OHLCV from BTC-Pulse database."""
        conn = get_connection()
        try:
            query = """
                SELECT timestamp, open, high, low, close, volume, quote_volume
                FROM table_market_price ORDER BY timestamp
            """
            df = pd.read_sql_query(query, conn, parse_dates=["timestamp"])
        finally:
            conn.close()

        if days and not df.empty:
            cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
            df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

        return df

    def run_pipeline(self, days: Optional[int] = None) -> pd.DataFrame:
        """Run the full strategy pipeline: load → indicators → signals."""
        df = self.load_market_data(days=days)
        if df.empty:
            logger.warning("No market data for strategy pipeline")
            return df

        df = self.populate_indicators(df)
        df = self.populate_entry_trend(df)
        df = self.populate_exit_trend(df)

        entries = df["enter_long"].sum() if "enter_long" in df.columns else 0
        exits = df["exit_long"].sum() if "exit_long" in df.columns else 0
        logger.info(f"Strategy pipeline: {entries} entries, {exits} exits over {len(df)} candles")

        return df


class BTCPulseStrategy(BaseStrategy):
    """
    BTC-Pulse v2 Strategy — combines Freqtrade indicator patterns with
    BTC-Pulse's existing contrarian/leverage/institutional signals.

    Indicators from Freqtrade ecosystem:
    - RSI (Relative Strength Index) with divergence detection
    - Bollinger Bands with squeeze detection
    - MACD with histogram analysis
    - ADX (trend strength)
    - Stochastic RSI

    Combined with BTC-Pulse originals:
    - Fear & Greed extremes
    - Funding rate analysis
    - VWAP deviation
    """

    # Tighter ROI for BTC
    minimal_roi = {
        "0": 0.15,
        "2880": 0.08,
        "7200": 0.04,
        "14400": 0.01,
    }

    stoploss = -0.07
    trailing_stop = True
    trailing_stop_positive = 0.025
    trailing_stop_positive_offset = 0.04

    # ── Hyperopt-tunable signal thresholds ────────────────────
    # These are the defaults; hyperopt overrides them via _hp_* attributes.
    _hp_rsi_buy: int = 35
    _hp_rsi_sell: int = 75
    _hp_fng_buy: int = 20
    _hp_fng_sell: int = 85
    _hp_bb_period: int = 20

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators — Freqtrade-style vectorized ops.
        """
        # ── RSI ──────────────────────────────────────────────
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # Stochastic RSI
        rsi_min = df["rsi"].rolling(14).min()
        rsi_max = df["rsi"].rolling(14).max()
        df["stoch_rsi"] = (df["rsi"] - rsi_min) / (rsi_max - rsi_min + 1e-10)
        df["stoch_rsi_k"] = df["stoch_rsi"].rolling(3).mean()
        df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(3).mean()

        # ── Bollinger Bands ──────────────────────────────────
        bb_period = getattr(self, "_hp_bb_period", 20)
        df["bb_mid"] = df["close"].rolling(bb_period).mean()
        bb_std = df["close"].rolling(bb_period).std()
        df["bb_upper"] = df["bb_mid"] + 2 * bb_std
        df["bb_lower"] = df["bb_mid"] - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

        # Bollinger Band squeeze (low volatility → imminent breakout)
        bb_width_percentile = df["bb_width"].rolling(120).rank(pct=True)
        df["bb_squeeze"] = (bb_width_percentile < 0.1).astype(int)

        # ── MACD ─────────────────────────────────────────────
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # MACD histogram rising (momentum building)
        df["macd_hist_rising"] = (df["macd_hist"] > df["macd_hist"].shift(1)).astype(int)

        # ── ADX (Average Directional Index) ──────────────────
        high_diff = df["high"].diff()
        low_diff = -df["low"].diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)

        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr14 = tr.ewm(span=14, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr14
        minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr14

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        df["adx"] = dx.ewm(span=14, adjust=False).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di

        # ── ATR (for position sizing / stoploss) ─────────────
        df["atr"] = atr14
        df["atr_pct"] = df["atr"] / df["close"] * 100

        # ── Moving Averages (already in BTC-Pulse, enhanced) ─
        for period in [7, 14, 30, 50, 90, 200]:
            df[f"sma{period}"] = df["close"].rolling(period).mean()
            df[f"ema{period}"] = df["close"].ewm(span=period, adjust=False).mean()

        # EMA cross signals
        df["ema_cross_bull"] = (
            (df["ema30"] > df["ema90"]) & (df["ema30"].shift(1) <= df["ema90"].shift(1))
        ).astype(int)
        df["ema_cross_bear"] = (
            (df["ema30"] < df["ema90"]) & (df["ema30"].shift(1) >= df["ema90"].shift(1))
        ).astype(int)

        # ── Volume analysis ──────────────────────────────────
        df["volume_sma20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma20"]
        # High volume candle (>2x average)
        df["high_volume"] = (df["volume_ratio"] > 2.0).astype(int)

        # On-Balance Volume (OBV) — Freqtrade common indicator
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
        df["obv_sma20"] = df["obv"].rolling(20).mean()

        # ── Merge BTC-Pulse features from database ───────────
        conn = get_connection()
        try:
            features = pd.read_sql_query(
                """SELECT timestamp, fng_value, funding_rate, cumulative_funding_30d,
                          oi_change_rate, oi_price_divergence, monthly_vwap, vwap_deviation
                   FROM table_features ORDER BY timestamp""",
                conn, parse_dates=["timestamp"],
            )
        finally:
            conn.close()

        if not features.empty:
            df = df.merge(features, on="timestamp", how="left")

        return df

    def populate_entry_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Multi-condition entry — inspired by Freqtrade's vectorized approach.
        Requires confluence of multiple signals (reduces false entries).

        All indicator thresholds read from _hp_* attributes so that
        hyperopt can tune them (falls back to class defaults).
        """
        df["enter_long"] = 0
        df["enter_tag"] = ""

        # Read tunable thresholds
        rsi_buy = getattr(self, "_hp_rsi_buy", 35)
        fng_buy = getattr(self, "_hp_fng_buy", 20)

        # ── Entry 1: Contrarian Extreme Fear + Technical Oversold ────
        fear_entry = (
            (df.get("fng_value", pd.Series(dtype=float)) < fng_buy) &
            (df["rsi"] < rsi_buy) &
            (df["close"] < df["bb_lower"]) &
            (df["macd_hist_rising"] == 1)
        )
        df.loc[fear_entry, "enter_long"] = 1
        df.loc[fear_entry, "enter_tag"] = "fear_oversold_reversal"

        # ── Entry 2: Bollinger Squeeze Breakout ──────────────────────
        squeeze_entry = (
            (df["bb_squeeze"] == 1) &
            (df["close"] > df["bb_mid"]) &
            (df["adx"] > 20) &
            (df["plus_di"] > df["minus_di"]) &
            (df["volume_ratio"] > 1.5)
        )
        df.loc[squeeze_entry, "enter_long"] = 1
        df.loc[squeeze_entry & (df["enter_tag"] == ""), "enter_tag"] = "squeeze_breakout"

        # ── Entry 3: EMA Cross + Volume Confirmation ─────────────────
        ema_entry = (
            (df["ema_cross_bull"] == 1) &
            (df["volume_ratio"] > 1.3) &
            (df["rsi"] > 40) &
            (df["rsi"] < 70)
        )
        df.loc[ema_entry, "enter_long"] = 1
        df.loc[ema_entry & (df["enter_tag"] == ""), "enter_tag"] = "ema_cross_bullish"

        # ── Entry 4: Funding Rate Squeeze (BTC-Pulse original) ───────
        if "funding_rate" in df.columns:
            funding_entry = (
                (df["funding_rate"] < -0.001) &
                (df["rsi"] < rsi_buy + 5) &
                (df["stoch_rsi_k"] < 0.2) &
                (df["obv"] > df["obv_sma20"])
            )
            df.loc[funding_entry, "enter_long"] = 1
            df.loc[funding_entry & (df["enter_tag"] == ""), "enter_tag"] = "negative_funding_squeeze"

        # ── Entry 5: VWAP Breakout (BTC-Pulse original) ──────────────
        if "vwap_deviation" in df.columns:
            vwap_entry = (
                (df["vwap_deviation"] > 0) &
                (df["vwap_deviation"].shift(1) < 0) &  # Just crossed above
                (df["volume_ratio"] > 1.2) &
                (df["adx"] > 15)
            )
            df.loc[vwap_entry, "enter_long"] = 1
            df.loc[vwap_entry & (df["enter_tag"] == ""), "enter_tag"] = "vwap_breakout"

        return df

    def populate_exit_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Multi-condition exit — Freqtrade-style.

        Thresholds read from _hp_* attributes (hyperopt-tunable).
        """
        df["exit_long"] = 0
        df["exit_tag"] = ""

        # Read tunable thresholds
        rsi_sell = getattr(self, "_hp_rsi_sell", 75)
        fng_sell = getattr(self, "_hp_fng_sell", 85)

        # ── Exit 1: Extreme Greed + Overbought ──────────────────────
        greed_exit = (
            (df.get("fng_value", pd.Series(dtype=float)) > fng_sell) &
            (df["rsi"] > rsi_sell) &
            (df["close"] > df["bb_upper"])
        )
        df.loc[greed_exit, "exit_long"] = 1
        df.loc[greed_exit, "exit_tag"] = "greed_overbought"

        # ── Exit 2: Bearish EMA Cross ────────────────────────────────
        df.loc[df["ema_cross_bear"] == 1, "exit_long"] = 1
        df.loc[(df["ema_cross_bear"] == 1) & (df["exit_tag"] == ""), "exit_tag"] = "ema_cross_bearish"

        # ── Exit 3: MACD Bearish Divergence ──────────────────────────
        macd_exit = (
            (df["macd"] < df["macd_signal"]) &
            (df["macd"].shift(1) >= df["macd_signal"].shift(1)) &
            (df["rsi"] > 60)
        )
        df.loc[macd_exit, "exit_long"] = 1
        df.loc[macd_exit & (df["exit_tag"] == ""), "exit_tag"] = "macd_bearish_cross"

        # ── Exit 4: OI-Price Divergence (BTC-Pulse original) ─────────
        if "oi_price_divergence" in df.columns:
            oi_exit = (
                (df["oi_price_divergence"] == 1) &
                (df["rsi"] > 55)
            )
            df.loc[oi_exit, "exit_long"] = 1
            df.loc[oi_exit & (df["exit_tag"] == ""), "exit_tag"] = "oi_divergence"

        # ── Exit 5: Leverage Overweight ──────────────────────────────
        if "cumulative_funding_30d" in df.columns:
            leverage_exit = (
                (df["cumulative_funding_30d"] > 0.02) &
                (df["rsi"] > rsi_sell - 10) &
                (df["volume_ratio"] < 0.8)
            )
            df.loc[leverage_exit, "exit_long"] = 1
            df.loc[leverage_exit & (df["exit_tag"] == ""), "exit_tag"] = "leverage_overweight"

        return df

    def custom_stoploss(self, current_profit: float, current_time: int,
                        entry_price: float, current_price: float,
                        **kwargs) -> float:
        """ATR-based dynamic stoploss — adapts to volatility."""
        atr_pct = kwargs.get("atr_pct", 3.0)

        # Use ATR as stoploss distance, with profit-dependent tightening
        if current_profit > 0.15:
            return -0.02
        elif current_profit > 0.08:
            return -(atr_pct / 100)  # 1x ATR
        elif current_profit > 0.03:
            return -(atr_pct * 1.5 / 100)  # 1.5x ATR
        return -(atr_pct * 2 / 100)  # 2x ATR (widest at start)

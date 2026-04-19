"""
SQLite schema definitions and database initialization for BTC-Pulse.
Three core tables + AI sentiment table + feature store.
"""

import sqlite3
from pathlib import Path
from loguru import logger
from utils.config import get_db_path


SCHEMA_SQL = """
-- Market price data (daily OHLCV from Binance)
CREATE TABLE IF NOT EXISTS table_market_price (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL UNIQUE,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    quote_volume REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_market_timestamp ON table_market_price(timestamp);

-- Derivatives data (funding rate, OI, liquidations)
CREATE TABLE IF NOT EXISTS table_derivatives (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL UNIQUE,
    funding_rate REAL,
    open_interest REAL,
    long_liquidations REAL,
    short_liquidations REAL,
    long_short_ratio REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_deriv_timestamp ON table_derivatives(timestamp);

-- Sentiment data (Fear & Greed Index)
CREATE TABLE IF NOT EXISTS table_sentiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL UNIQUE,
    fng_value INTEGER NOT NULL,
    fng_classification TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sentiment_timestamp ON table_sentiment(timestamp);

-- AI-generated sentiment scores from LLM analysis
CREATE TABLE IF NOT EXISTS table_ai_sentiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    source TEXT,
    text_snippet TEXT,
    sentiment_score REAL,
    narrative_tags TEXT,
    model_used TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_ai_sentiment_timestamp ON table_ai_sentiment(timestamp);

-- Feature store for model training/inference
CREATE TABLE IF NOT EXISTS table_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL UNIQUE,
    -- Price features
    ma7 REAL, ma14 REAL, ma30 REAL, ma90 REAL, ma200 REAL,
    ma30_deviation REAL,
    price_change_1d REAL,
    price_change_7d REAL,
    price_change_30d REAL,
    volatility_30d REAL,
    -- Volume features
    volume_ma7 REAL,
    volume_change REAL,
    -- VWAP features
    monthly_vwap REAL,
    vwap_deviation REAL,
    -- Sentiment features
    fng_value INTEGER,
    fng_ma7 REAL,
    fng_extreme_low INTEGER DEFAULT 0,
    fng_extreme_high INTEGER DEFAULT 0,
    -- Derivatives features
    funding_rate REAL,
    cumulative_funding_30d REAL,
    oi_change_rate REAL,
    oi_price_divergence INTEGER DEFAULT 0,
    -- AI features
    ai_sentiment_score REAL,
    -- Labels (for training)
    forward_return_30d REAL,
    forward_trend TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_features_timestamp ON table_features(timestamp);

-- Model predictions log
CREATE TABLE IF NOT EXISTS table_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model_version TEXT,
    trend_probability_up REAL,
    trend_rating TEXT,
    contrarian_score REAL,
    leverage_score REAL,
    institutional_score REAL,
    composite_score REAL,
    signal TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON table_predictions(timestamp);
"""


def init_database(db_path: str | Path | None = None) -> str:
    """Initialize the SQLite database with all required tables."""
    if db_path is None:
        db_path = get_db_path()

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        logger.info(f"Database initialized at {db_path}")
    finally:
        conn.close()

    return str(db_path)


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode for better concurrency."""
    if db_path is None:
        db_path = get_db_path()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def get_last_timestamp(table: str, db_path: str | Path | None = None) -> str | None:
    """Get the most recent timestamp from a table for incremental updates."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(f"SELECT MAX(timestamp) FROM {table}")
        row = cursor.fetchone()
        return row[0] if row and row[0] else None
    finally:
        conn.close()


if __name__ == "__main__":
    path = init_database()
    print(f"Database created at: {path}")

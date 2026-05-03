#!/usr/bin/env python3
"""
BTC-Pulse Daily Ingestion Script
Fetches data from all sources and stores in SQLite.
Run daily at 08:05 via cron/scheduler.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from database.init_db import init_database
from collectors.market import MarketCollector
from collectors.sentiment import SentimentCollector
from collectors.derivatives import DerivativesCollector
from collectors.news import NewsCollector
from features.engine import FeatureEngine
from models.signals import SignalGenerator
from models.llm_sentiment import LLMSentiment
from utils.logging import setup_logger
from utils.config import get_snapshot_dir

logger = setup_logger("ingest")


def export_daily_snapshot():
    """Export current database state to CSV as backup."""
    import pandas as pd
    from database.init_db import get_connection

    snap_dir = get_snapshot_dir()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    conn = get_connection()
    try:
        for table in ["table_market_price", "table_sentiment", "table_derivatives"]:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            if not df.empty:
                path = snap_dir / f"{table}_{today}.csv"
                df.to_csv(path, index=False)
                logger.info(f"Snapshot: {path} ({len(df)} rows)")
    finally:
        conn.close()


def main():
    logger.info("=" * 60)
    logger.info("BTC-Pulse Daily Ingestion Started")
    logger.info("=" * 60)

    # Step 0: Ensure database exists
    db_path = init_database()
    logger.info(f"Database: {db_path}")

    # Step 1: Collect market data
    logger.info("── Collecting market data (Binance) ──")
    try:
        market = MarketCollector()
        n = market.run()
        logger.info(f"Market: {n} new records")
    except Exception as e:
        logger.error(f"Market collection failed: {e}")

    # Step 2: Collect sentiment data
    logger.info("── Collecting sentiment data (FNG) ──")
    try:
        sentiment = SentimentCollector()
        n = sentiment.run()
        logger.info(f"Sentiment: {n} new records")
    except Exception as e:
        logger.error(f"Sentiment collection failed: {e}")

    # Step 3: Collect derivatives data
    logger.info("── Collecting derivatives data ──")
    try:
        derivatives = DerivativesCollector()
        n = derivatives.run()
        logger.info(f"Derivatives: {n} new records")
    except Exception as e:
        logger.error(f"Derivatives collection failed: {e}")

    # Step 4: Collect news and run LLM sentiment
    logger.info("── Collecting news for LLM sentiment ──")
    try:
        news = NewsCollector()
        text = news.collect_as_text()
        if text:
            llm = LLMSentiment()
            result = llm.analyze(text, source="rss_headlines")
            logger.info(f"LLM sentiment: {result.get('sentiment_score', 'N/A')}")
        else:
            logger.info("No BTC news found for sentiment analysis")
    except Exception as e:
        logger.error(f"News/LLM sentiment failed: {e}")

    # Step 5: Compute features
    logger.info("── Computing features ──")
    try:
        engine = FeatureEngine()
        n = engine.run()
        logger.info(f"Features: {n} rows computed")
    except Exception as e:
        logger.error(f"Feature computation failed: {e}")

    # Step 6: Generate signals
    logger.info("── Generating signals ──")
    try:
        signals = SignalGenerator()
        result = signals.run()
        logger.info(f"Signal: {result['emoji']} {result['rating']} (score: {result['composite_score']})")
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")

    # Step 7: Export daily snapshot
    logger.info("── Exporting snapshot ──")
    try:
        export_daily_snapshot()
    except Exception as e:
        logger.error(f"Snapshot export failed: {e}")

    logger.info("=" * 60)
    logger.info("BTC-Pulse Daily Ingestion Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

"""
Basic tests for BTC-Pulse core modules.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestFeatureEngine:
    """Test feature engineering computations."""

    def test_price_features_basic(self):
        """Test that price features are computed on simple OHLCV data."""
        from features.engine import FeatureEngine
        engine = FeatureEngine()

        # Create minimal test data (35 days to cover 30-day MA)
        dates = pd.date_range("2024-01-01", periods=35, freq="D")
        prices = np.linspace(40000, 45000, 35)
        df = pd.DataFrame({
            "timestamp": dates,
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(1000, 5000, 35).astype(float),
            "quote_volume": np.random.randint(40000000, 200000000, 35).astype(float),
        })

        result = engine._compute_price_features(df)

        assert "ma30" in result.columns
        assert "volatility_30d" in result.columns
        assert "monthly_vwap" in result.columns
        assert len(result) == 35

    def test_labels_computation(self):
        """Test forward label generation."""
        from features.engine import FeatureEngine
        engine = FeatureEngine()

        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        prices = np.concatenate([
            np.linspace(40000, 50000, 30),  # +25% in first 30 days
            np.linspace(50000, 45000, 30),  # -10% in next 30 days
        ])
        df = pd.DataFrame({"timestamp": dates, "close": prices})

        result = engine._compute_labels(df)

        assert "forward_return_30d" in result.columns
        assert "forward_trend" in result.columns
        # First few rows should be bullish (price goes up 25% over next 30 days)
        assert result.iloc[0]["forward_trend"] == "bullish"


class TestSignalGenerator:
    """Test signal generation logic."""

    def test_contrarian_sentiment_extreme_fear(self):
        """Test that extreme fear produces bullish signal."""
        from models.signals import SignalGenerator
        gen = SignalGenerator()

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "fng_value": [10, 12, 8, 11, 10],  # Extreme fear
            "funding_rate": [-0.001, -0.002, -0.001, -0.0015, -0.001],
            "cumulative_funding_30d": [-0.01, -0.01, -0.01, -0.01, -0.01],
        })

        result = gen.contrarian_sentiment(df)
        assert result["score"] > 0, "Extreme fear should produce bullish score"
        assert result["signal"] == "bullish"

    def test_contrarian_sentiment_extreme_greed(self):
        """Test that extreme greed produces bearish signal."""
        from models.signals import SignalGenerator
        gen = SignalGenerator()

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "fng_value": [90, 88, 92, 91, 90],  # Extreme greed
            "funding_rate": [0.01, 0.01, 0.01, 0.01, 0.01],
            "cumulative_funding_30d": [0.03, 0.03, 0.03, 0.03, 0.03],
        })

        result = gen.contrarian_sentiment(df)
        assert result["score"] < 0, "Extreme greed should produce bearish score"
        assert result["signal"] == "bearish"

    def test_composite_signal_returns_all_modules(self):
        """Test that composite signal includes all three modules."""
        from models.signals import SignalGenerator
        gen = SignalGenerator()

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "fng_value": [50, 50, 50, 50, 50],
            "funding_rate": [0.0001] * 5,
            "cumulative_funding_30d": [0.005] * 5,
            "monthly_vwap": [42000] * 5,
            "vwap_deviation": [1.0] * 5,
            "volume_change": [0.1] * 5,
            "price_change_30d": [0.03] * 5,
            "oi_price_divergence": [0] * 5,
        })

        result = gen.composite_signal(df)
        assert "composite_score" in result
        assert "rating" in result
        assert "modules" in result
        assert "contrarian_sentiment" in result["modules"]
        assert "leverage_purge" in result["modules"]
        assert "institutional_benchmark" in result["modules"]


class TestDatabaseInit:
    """Test database initialization."""

    def test_init_creates_db(self, tmp_path):
        """Test that init_database creates the SQLite file."""
        from database.init_db import init_database
        db_file = tmp_path / "test.db"
        init_database(str(db_file))
        assert db_file.exists()

    def test_allowed_tables_validation(self):
        """Test that get_last_timestamp rejects invalid table names."""
        from database.init_db import get_last_timestamp
        with pytest.raises(ValueError, match="Invalid table name"):
            get_last_timestamp("DROP TABLE users; --")


class TestBacktestEngine:
    """Test backtest logic."""

    def test_generate_signals(self):
        """Test signal generation for backtest."""
        from models.backtest import BacktestEngine
        engine = BacktestEngine()

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
            "close": [40000 + i * 100 for i in range(10)],
            "fng_value": [10, 50, 50, 50, 50, 50, 90, 50, 50, 50],
            "funding_rate": [-0.01, 0, 0, 0, 0, 0, 0.01, 0, 0, 0],
            "cumulative_funding_30d": [0, 0, 0, 0, 0, 0, 0.03, 0, 0, 0],
        })

        result = engine.generate_signals(df)
        assert result["buy_signal"].sum() >= 1  # FNG=10 + neg funding
        assert result["sell_signal"].sum() >= 1  # FNG=90 + high cum funding

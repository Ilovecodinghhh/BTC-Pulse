"""
BTC-Pulse Streamlit Dashboard
Real-time trend rating, risk radar, and historical similarity analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.init_db import get_connection, init_database
from models.signals import SignalGenerator
from models.xgboost_model import XGBoostCombiner
from models.anomaly import AnomalyDetector
from utils.config import load_config


# Page config
st.set_page_config(
    page_title="BTC-Pulse 1.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 BTC-Pulse 1.0")
st.caption("Multi-Dimensional Bitcoin Monthly Trend Prediction System")


# ─── Helpers ───────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_market_data():
    conn = get_connection()
    try:
        return pd.read_sql_query(
            "SELECT * FROM table_market_price ORDER BY timestamp",
            conn, parse_dates=["timestamp"],
        )
    finally:
        conn.close()


@st.cache_data(ttl=300)
def load_features():
    conn = get_connection()
    try:
        return pd.read_sql_query(
            "SELECT * FROM table_features ORDER BY timestamp",
            conn, parse_dates=["timestamp"],
        )
    finally:
        conn.close()


@st.cache_data(ttl=300)
def load_sentiment():
    conn = get_connection()
    try:
        return pd.read_sql_query(
            "SELECT * FROM table_sentiment ORDER BY timestamp",
            conn, parse_dates=["timestamp"],
        )
    finally:
        conn.close()


@st.cache_data(ttl=300)
def load_predictions():
    conn = get_connection()
    try:
        return pd.read_sql_query(
            "SELECT * FROM table_predictions ORDER BY timestamp DESC LIMIT 30",
            conn, parse_dates=["timestamp"],
        )
    finally:
        conn.close()


# ─── Sidebar ───────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.header("📅 Date Range")
    days_back = st.slider("Days to display", 30, 1825, 365)

    st.divider()
    st.header("ℹ️ About")
    st.markdown("""
    **BTC-Pulse 1.0** integrates:
    - 📈 Price & Volume (Binance)
    - 📊 Derivatives (Funding/OI)
    - 😱 Fear & Greed Index
    - 🤖 XGBoost + LLM AI Layer
    """)


# ─── Main Content ──────────────────────────────────────────

# Load data
market = load_market_data()
features = load_features()
sentiment = load_sentiment()
predictions = load_predictions()

if market.empty:
    st.warning("⚠️ No data found. Run `python run_ingest.py` first to populate the database.")
    st.stop()

# Filter by date range
cutoff = datetime.now() - timedelta(days=days_back)
market = market[market["timestamp"] >= cutoff]
features = features[features["timestamp"] >= cutoff] if not features.empty else features
sentiment = sentiment[sentiment["timestamp"] >= cutoff] if not sentiment.empty else sentiment


# ─── Section 1: Trend Traffic Light ───────────────────────

st.header("🚦 Monthly Trend Rating")

col1, col2, col3, col4 = st.columns(4)

# Generate current signal
signal_gen = SignalGenerator()
sig_features = signal_gen.load_features()

if not sig_features.empty:
    result = signal_gen.composite_signal(sig_features)
    modules = result["modules"]

    with col1:
        emoji = result["emoji"]
        st.metric(
            "Composite Rating",
            f"{emoji} {result['rating']}",
            f"Score: {result['composite_score']:.3f}",
        )

    with col2:
        cs = modules["contrarian_sentiment"]
        st.metric("Contrarian Sentiment", cs["signal"].upper(), f"{cs['score']:.3f}")

    with col3:
        lp = modules["leverage_purge"]
        st.metric("Leverage Purge", lp["signal"].upper(), f"{lp['score']:.3f}")

    with col4:
        ib = modules["institutional_benchmark"]
        st.metric("Institutional (VWAP)", ib["signal"].upper(), f"{ib['score']:.3f}")

    # Module details
    with st.expander("📋 Signal Details"):
        for name, mod in modules.items():
            st.write(f"**{name.replace('_', ' ').title()}**: {mod['detail']}")
else:
    st.info("Run feature engineering to see trend ratings.")


# ─── Section 2: Price Chart with Indicators ───────────────

st.header("📈 Price & Indicators")

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.5, 0.25, 0.25],
    subplot_titles=("BTC/USDT Price", "Volume", "Fear & Greed Index"),
)

# Price + MAs
fig.add_trace(
    go.Candlestick(
        x=market["timestamp"],
        open=market["open"], high=market["high"],
        low=market["low"], close=market["close"],
        name="BTC/USDT",
    ),
    row=1, col=1,
)

if not features.empty:
    for ma_col, color in [("ma30", "orange"), ("ma90", "blue"), ("ma200", "purple")]:
        if ma_col in features.columns:
            fig.add_trace(
                go.Scatter(
                    x=features["timestamp"], y=features[ma_col],
                    name=ma_col.upper(), line=dict(color=color, width=1),
                ),
                row=1, col=1,
            )

    # VWAP
    if "monthly_vwap" in features.columns:
        fig.add_trace(
            go.Scatter(
                x=features["timestamp"], y=features["monthly_vwap"],
                name="Monthly VWAP", line=dict(color="cyan", width=1, dash="dash"),
            ),
            row=1, col=1,
        )

# Volume
fig.add_trace(
    go.Bar(x=market["timestamp"], y=market["volume"], name="Volume", marker_color="gray"),
    row=2, col=1,
)

# Fear & Greed
if not sentiment.empty:
    colors = sentiment["fng_value"].apply(
        lambda x: "red" if x < 25 else ("orange" if x < 45 else ("yellow" if x < 55 else ("lightgreen" if x < 75 else "green")))
    )
    fig.add_trace(
        go.Bar(x=sentiment["timestamp"], y=sentiment["fng_value"], name="FNG", marker_color=colors),
        row=3, col=1,
    )
    fig.add_hline(y=20, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="green", row=3, col=1)

fig.update_layout(
    height=800,
    template="plotly_dark",
    showlegend=True,
    xaxis_rangeslider_visible=False,
)

st.plotly_chart(fig, use_container_width=True)


# ─── Section 3: Risk Radar ────────────────────────────────

st.header("🎯 Risk Radar")

if not features.empty:
    latest = features.iloc[-1]

    radar_categories = []
    radar_values = []

    # Leverage Crowding (funding rate normalized)
    cum_fund = latest.get("cumulative_funding_30d")
    if not pd.isna(cum_fund):
        leverage_risk = min(abs(cum_fund) / 0.03 * 100, 100)
        radar_categories.append("Leverage Crowding")
        radar_values.append(leverage_risk)

    # Sentiment Extreme
    fng = latest.get("fng_value")
    if not pd.isna(fng):
        sent_risk = abs(fng - 50) / 50 * 100
        radar_categories.append("Sentiment Extreme")
        radar_values.append(sent_risk)

    # Volatility
    vol = latest.get("volatility_30d")
    if not pd.isna(vol):
        vol_risk = min(vol / 1.5 * 100, 100)
        radar_categories.append("Volatility")
        radar_values.append(vol_risk)

    # VWAP Deviation
    vwap_dev = latest.get("vwap_deviation")
    if not pd.isna(vwap_dev):
        vwap_risk = min(abs(vwap_dev) / 10 * 100, 100)
        radar_categories.append("VWAP Deviation")
        radar_values.append(vwap_risk)

    # Price vs MA
    ma_dev = latest.get("ma30_deviation")
    if not pd.isna(ma_dev):
        ma_risk = min(abs(ma_dev) / 20 * 100, 100)
        radar_categories.append("MA30 Deviation")
        radar_values.append(ma_risk)

    if radar_categories:
        radar_fig = go.Figure(data=go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=radar_categories + [radar_categories[0]],
            fill="toself",
            fillcolor="rgba(255, 100, 100, 0.3)",
            line_color="red",
        ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            template="plotly_dark",
            height=400,
            title="Current Risk Levels (0-100)",
        )
        st.plotly_chart(radar_fig, use_container_width=True)


# ─── Section 4: Historical Similarity ─────────────────────

st.header("🔍 Historical Similarity")
st.caption("Finds past months with the most similar data distribution to now")

if not features.empty and len(features) > 60:
    # Current 30-day feature vector
    recent = features.tail(30)
    current_profile = {}

    for col in ["ma30_deviation", "volatility_30d", "fng_value", "cumulative_funding_30d"]:
        if col in recent.columns:
            vals = recent[col].dropna()
            if not vals.empty:
                current_profile[col] = vals.mean()

    if current_profile:
        # Compare with rolling 30-day windows in history
        similarities = []
        for i in range(30, len(features) - 30, 30):  # Monthly windows
            window = features.iloc[i:i + 30]
            window_profile = {}
            for col in current_profile:
                if col in window.columns:
                    vals = window[col].dropna()
                    if not vals.empty:
                        window_profile[col] = vals.mean()

            # Cosine-like similarity
            if window_profile:
                common = set(current_profile.keys()) & set(window_profile.keys())
                if common:
                    a = np.array([current_profile[c] for c in common])
                    b = np.array([window_profile[c] for c in common])
                    # Normalize and compute distance
                    norm_a = a / (np.abs(a).max() + 1e-10)
                    norm_b = b / (np.abs(b).max() + 1e-10)
                    dist = np.sqrt(np.sum((norm_a - norm_b) ** 2))
                    similarity = max(0, 1 - dist / 2)

                    mid_date = window["timestamp"].iloc[15] if len(window) > 15 else window["timestamp"].iloc[0]
                    similarities.append({
                        "date": mid_date,
                        "similarity": round(similarity, 3),
                    })

        if similarities:
            sim_df = pd.DataFrame(similarities).sort_values("similarity", ascending=False).head(5)
            st.dataframe(sim_df, use_container_width=True)

            best = sim_df.iloc[0]
            st.info(f"📌 Current market conditions most resemble **{best['date'].strftime('%B %Y')}** "
                    f"(similarity: {best['similarity']:.1%})")
else:
    st.info("Need at least 60 days of data for similarity analysis.")


# ─── Section 5: Prediction History ────────────────────────

st.header("📜 Recent Predictions")

if not predictions.empty:
    pred_display = predictions[["timestamp", "trend_rating", "composite_score",
                                "contrarian_score", "leverage_score", "institutional_score"]].copy()
    pred_display.columns = ["Date", "Rating", "Composite", "Sentiment", "Leverage", "VWAP"]
    st.dataframe(pred_display, use_container_width=True)
else:
    st.info("No predictions yet. Run `python run_ingest.py` followed by signal generation.")


# ─── Footer ───────────────────────────────────────────────

st.divider()
st.caption(
    "⚠️ BTC-Pulse is for research and educational purposes only. "
    "Not financial advice. AI predictions have inherent limitations."
)

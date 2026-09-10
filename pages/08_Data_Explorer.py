"""Data Explorer: Quality Diagnostics, Missing Bar Detection, and Hygiene Audits."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quant_engine.config.markets import load_market_config
from quant_engine.data.downloader import DataDownloader
from quant_engine.data.cache import DataCache
from quant_engine.ui_helpers import (
    apply_terminal_theme,
    render_terminal_header,
    get_terminal_plotly_layout,
    render_disclaimer,
)

st.set_page_config(page_title="Data Explorer | Systematic Engine", page_icon="🔍", layout="wide")
apply_terminal_theme()

force_demo = st.session_state.get("force_demo", False)

render_terminal_header(
    title="MARKET DATA EXPLORER & HYGIENE AUDITOR",
    subtitle="Inspect Pricing Health Scores, Missing Calendars, Anomalies, and Cache State",
    is_demo=force_demo,
)

col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
with col_d1:
    market_code = st.selectbox("Select Market", ["MY", "US", "JP", "EU"], index=0)
    market_cfg = load_market_config(market_code)
with col_d2:
    symbol = st.selectbox("Select Asset", [u.symbol for u in market_cfg.universe], index=0)
with col_d3:
    st.write("")
    st.write("")
    clear_cache = st.button("🧹 Clear Parquet Cache")

if clear_cache:
    cache = DataCache()
    deleted = cache.clear()
    st.success(f"Cleared {deleted} cached parquet files.")

downloader = DataDownloader(force_demo=force_demo)
df, health = downloader.load_price_data(symbol, "2018-01-01", "2025-12-31")

if df.empty:
    st.error("No data available.")
    st.stop()

# Health Scorecard
st.markdown('<div class="section-header">DATA HEALTH AUDIT REPORT</div>', unsafe_allow_html=True)
h1, h2, h3, h4, h5 = st.columns(5)
h1.metric("Data Health Score", f"{health.data_health_score:.1f} / 100")
h2.metric("Total Trading Bars", health.total_bars)
h3.metric("Duplicate Dates", health.duplicate_dates_count)
h4.metric("Zero/Negative Prices", health.zero_negative_prices_count)
h5.metric("Stale Price Bars", health.stale_bars_count)

if health.warnings:
    st.warning(f"⚠️ **DATA HYGIENE NOTICES**: {'; '.join(health.warnings)}")
else:
    st.success("✅ **CLEAN DATA AUDIT**: Zero duplicates, zero negative prices, monotonic chronological order confirmed.")

# Candlestick & Volume Chart
st.markdown('<div class="section-header">HISTORICAL OHLCV CHART</div>', unsafe_allow_html=True)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.06)

fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC",
        increasing_line_color="#00D4B2",
        decreasing_line_color="#FF5630",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume",
        marker_color="#2684FF",
        opacity=0.6,
    ),
    row=2,
    col=1,
)

fig.update_layout(get_terminal_plotly_layout(height=520))
fig.update_xaxes(rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# Raw Data Table
st.markdown('<div class="section-header">RAW PRICE SAMPLE (FIRST & LAST 5 BARS)</div>', unsafe_allow_html=True)
sample_df = pd.concat([df.head(5), df.tail(5)])
st.dataframe(sample_df, use_container_width=True)

render_disclaimer()

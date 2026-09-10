"""Strategy Builder: Signal Formulation and Indicator Inspection."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

from quant_engine.config.markets import load_market_config
from quant_engine.config.strategies import StrategyConfig, StrategyType
from quant_engine.data.downloader import DataDownloader
from quant_engine.signals.momentum import MovingAverageMomentum, TimeSeriesMomentum
from quant_engine.signals.mean_reversion import MeanReversion, BollingerReversion
from quant_engine.signals.breakout import DonchianBreakout, ATRBreakout
from quant_engine.features.technical import sma, ema, bollinger_bands
from quant_engine.ui_helpers import (
    apply_terminal_theme,
    render_terminal_header,
    get_terminal_plotly_layout,
    render_disclaimer,
)

st.set_page_config(page_title="Strategy Builder | Systematic Engine", page_icon="⚙️", layout="wide")
apply_terminal_theme()

force_demo = st.session_state.get("force_demo", False)

render_terminal_header(
    title="STRATEGY BUILDER & SIGNAL INSPECTOR",
    subtitle="Deconstruct Technical Indicators, Mathematical Channels, and Signal Triggers",
    is_demo=force_demo,
)

col_s1, col_s2, col_s3 = st.columns([1, 1, 1])
with col_s1:
    market_code = st.selectbox("Market Selection", ["MY", "US", "JP", "EU"], index=0)
    market_cfg = load_market_config(market_code)
with col_s2:
    symbol = st.selectbox("Underlying Asset", [u.symbol for u in market_cfg.universe], index=0)
with col_s3:
    strat_choice = st.selectbox(
        "Strategy Architecture",
        [
            "Dual Moving Average Crossover",
            "Time-Series (Absolute) Momentum",
            "Z-Score Mean Reversion",
            "Bollinger Band Mean Reversion",
            "Donchian Channel Breakout",
            "ATR Volatility Breakout",
        ],
    )

# Load data
downloader = DataDownloader(force_demo=force_demo)
df, health = downloader.load_price_data(symbol, "2021-01-01", "2024-12-31")

if df.empty:
    st.error("Data unavailable.")
    st.stop()

# Build Strategy
if strat_choice == "Dual Moving Average Crossover":
    c1, c2 = st.columns(2)
    fast_p = c1.slider("Fast Period", 5, 50, 20)
    slow_p = c2.slider("Slow Period", 20, 200, 50)
    strat = MovingAverageMomentum(StrategyConfig(parameters={"fast_period": fast_p, "slow_period": slow_p}, long_only=True))
    overlay_fast = sma(df["Close"], fast_p)
    overlay_slow = sma(df["Close"], slow_p)
    overlay_names = [f"Fast SMA ({fast_p})", f"Slow SMA ({slow_p})"]
    overlays = [overlay_fast, overlay_slow]
elif strat_choice == "Bollinger Band Mean Reversion":
    c1, c2 = st.columns(2)
    bb_p = c1.slider("Bollinger Period", 10, 50, 20)
    bb_std = c2.slider("Band StDev", 1.0, 3.0, 2.0, 0.1)
    strat = BollingerReversion(StrategyConfig(parameters={"period": bb_p, "num_std": bb_std}, long_only=True))
    mid, up, low = bollinger_bands(df["Close"], period=bb_p, num_std=bb_std)
    overlay_names = ["BB Mid", "BB Upper", "BB Lower"]
    overlays = [mid, up, low]
elif strat_choice == "Donchian Channel Breakout":
    c1, c2 = st.columns(2)
    lb = c1.slider("Entry Lookback", 20, 100, 50)
    ex_lb = c2.slider("Exit Lookback", 10, 50, 20)
    strat = DonchianBreakout(StrategyConfig(parameters={"lookback": lb, "exit_lookback": ex_lb}, long_only=True))
    up = df["High"].rolling(lb).max().shift(1)
    low = df["Low"].rolling(ex_lb).min().shift(1)
    overlay_names = [f"Donchian High ({lb}d)", f"Donchian Low ({ex_lb}d)"]
    overlays = [up, low]
else:
    lb = st.slider("Momentum Lookback (Days)", 20, 250, 120)
    strat = TimeSeriesMomentum(StrategyConfig(parameters={"lookback_days": lb}, long_only=True))
    overlays = []
    overlay_names = []

signals = strat.generate_signals(df)

# Multi-panel Price & Signal Plot
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.7, 0.3],
    subplot_titles=[f"{symbol} Price Action & Indicators", "Strategy Active Signal Target Weight [-1.0, +1.0]"],
)

# Price candlesticks
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
        increasing_line_color="#00D4B2",
        decreasing_line_color="#FF5630",
    ),
    row=1,
    col=1,
)

colors = ["#FFAB00", "#00B8D9", "#6554C0"]
for idx, (ov, name) in enumerate(zip(overlays, overlay_names)):
    fig.add_trace(
        go.Scatter(x=ov.index, y=ov, name=name, line=dict(color=colors[idx % len(colors)], width=1.4)),
        row=1,
        col=1,
    )

# Signal line
fig.add_trace(
    go.Scatter(
        x=signals.index,
        y=signals,
        name="Signal",
        line=dict(color="#36B37E", width=2.0),
        fill="tozeroy",
        fillcolor="rgba(54, 179, 126, 0.2)",
    ),
    row=2,
    col=1,
)

fig.update_layout(get_terminal_plotly_layout(height=600))
fig.update_xaxes(rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# Signal Statistics
col_stat1, col_stat2, col_stat3 = st.columns(3)
long_pct = (signals > 0).mean() * 100.0
flat_pct = (signals == 0).mean() * 100.0
short_pct = (signals < 0).mean() * 100.0

col_stat1.metric("Long Allocation (% Days)", f"{long_pct:.1f}%")
col_stat2.metric("Cash / Flat (% Days)", f"{flat_pct:.1f}%")
col_stat3.metric("Short Allocation (% Days)", f"{short_pct:.1f}%")

st.markdown('<div class="section-header">SIGNAL AUDIT LEDGER (LAST 10 BARS)</div>', unsafe_allow_html=True)
recent_df = pd.DataFrame(
    {
        "Close": df["Close"].iloc[-10:],
        "Signal": signals.iloc[-10:],
        "Action": signals.diff().iloc[-10:].apply(
            lambda x: "BUY / LONG" if x > 0 else ("SELL / EXIT" if x < 0 else "HOLD")
        ),
    }
)
st.dataframe(recent_df, use_container_width=True)

render_disclaimer()

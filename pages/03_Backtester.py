"""Backtester Deep-Dive: Monthly Returns Heatmap, Return Distribution, and Full Trade Ledger."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from quant_engine.config.markets import load_market_config
from quant_engine.config.strategies import StrategyConfig
from quant_engine.data.downloader import DataDownloader
from quant_engine.signals.momentum import TimeSeriesMomentum
from quant_engine.backtest.engine import BacktestEngine
from quant_engine.analytics.statistics import calculate_return_statistics
from quant_engine.ui_helpers import (
    apply_terminal_theme,
    render_terminal_header,
    get_terminal_plotly_layout,
    render_disclaimer,
)

st.set_page_config(page_title="Backtester Audit | Systematic Engine", page_icon="📑", layout="wide")
apply_terminal_theme()

force_demo = st.session_state.get("force_demo", False)

render_terminal_header(
    title="BACKTEST EXECUTION AUDIT",
    subtitle="Granular Trade Ledger, Monthly Returns Matrix, and Tail Risk Distribution",
    is_demo=force_demo,
)

# Load existing results from session or generate fresh baseline
res_df = st.session_state.get("last_res_df")
ledger = st.session_state.get("last_ledger")
summary = st.session_state.get("last_summary")
symbol = st.session_state.get("last_symbol", "^KLSE")

if res_df is None or ledger is None:
    st.info("Running baseline simulation for demonstration (Run in Research Lab to customize).")
    downloader = DataDownloader(force_demo=force_demo)
    df, _ = downloader.load_price_data("^KLSE", "2019-01-01", "2024-12-31")
    strat = TimeSeriesMomentum(StrategyConfig(parameters={"lookback_days": 120, "holding_days": 20}))
    sig = strat.generate_signals(df)
    engine = BacktestEngine()
    res_df, ledger, summary = engine.run(df, sig, symbol="^KLSE")

# Top summary KPIs
k1, k2, k3, k4, k5 = st.columns(5)
stats = summary["trade_stats"]
k1.metric("Total Executed Trades", stats["total_trades"])
k2.metric("Trade Win Rate", f"{stats['win_rate']*100:.1f}%")
k3.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
k4.metric("Average Holding Bars", f"{stats['avg_holding_bars']:.1f} bars")
k5.metric("Net Expectancy", f"${stats['expectancy']:.2f}")

# 1. Monthly Returns Heatmap
st.markdown('<div class="section-header">MONTHLY RETURNS MATRIX (%)</div>', unsafe_allow_html=True)
daily_rets = res_df["Net_Return"].copy()
monthly_rets = daily_rets.resample("M").apply(lambda r: (1.0 + r).prod() - 1.0)

monthly_matrix_data = []
for dt, val in monthly_rets.items():
    monthly_matrix_data.append({"Year": dt.year, "Month": dt.strftime("%b"), "Return": val * 100.0})

if monthly_matrix_data:
    month_df = pd.DataFrame(monthly_matrix_data)
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    heatmap_df = month_df.pivot(index="Year", columns="Month", values="Return")
    # Reindex columns to calendar order
    available_months = [m for m in month_order if m in heatmap_df.columns]
    heatmap_df = heatmap_df[available_months]

    fig_heat = px.imshow(
        heatmap_df,
        text_auto=".2f",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0.0,
        aspect="auto",
    )
    fig_heat.update_layout(get_terminal_plotly_layout(height=340, title="MONTH-BY-MONTH RETURN PERFORMANCE (%)"))
    st.plotly_chart(fig_heat, use_container_width=True)

# 2. Return Distribution & Tail Risk
st.markdown('<div class="section-header">RETURN DISTRIBUTION & TAIL RISK</div>', unsafe_allow_html=True)
stat_metrics = calculate_return_statistics(res_df["Net_Return"])

col_hist, col_stats = st.columns([2, 1])

with col_hist:
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=res_df["Net_Return"] * 100.0,
            nbinsx=60,
            marker_color="#00D4B2",
            opacity=0.75,
            name="Daily Returns (%)",
        )
    )
    fig_hist.add_vline(
        x=stat_metrics["var_95_daily"] * 100.0,
        line_dash="dash",
        line_color="#FF5630",
        annotation_text="VaR 95%",
    )
    fig_hist.add_vline(
        x=stat_metrics["cvar_95_daily"] * 100.0,
        line_dash="dot",
        line_color="#FFAB00",
        annotation_text="CVaR (Expected Shortfall)",
    )
    fig_hist.update_layout(get_terminal_plotly_layout(height=340, title="DAILY RETURN HISTOGRAM & VALUE-AT-RISK (%)"))
    st.plotly_chart(fig_hist, use_container_width=True)

with col_stats:
    st.markdown("**Distribution Statistics**")
    stats_table = pd.DataFrame(
        [
            {"Statistic": "Skewness", "Value": f"{stat_metrics['skewness']:.3f}"},
            {"Statistic": "Excess Kurtosis", "Value": f"{stat_metrics['excess_kurtosis']:.3f}"},
            {"Statistic": "Historical VaR (95%)", "Value": f"{stat_metrics['var_95_daily']*100:.2f}%"},
            {"Statistic": "Historical VaR (99%)", "Value": f"{stat_metrics['var_99_daily']*100:.2f}%"},
            {"Statistic": "Conditional VaR (95%)", "Value": f"{stat_metrics['cvar_95_daily']*100:.2f}%"},
            {"Statistic": "Best Single Day", "Value": f"+{stat_metrics['best_day']*100:.2f}%"},
            {"Statistic": "Worst Single Day", "Value": f"{stat_metrics['worst_day']*100:.2f}%"},
        ]
    )
    st.dataframe(stats_table, use_container_width=True, hide_index=True)

# 3. Trade Ledger
st.markdown('<div class="section-header">AUDIT TRADE LEDGER</div>', unsafe_allow_html=True)
trade_df = ledger.to_dataframe()
if not trade_df.empty:
    st.dataframe(trade_df, use_container_width=True)
else:
    st.info("No completed round-trip trades recorded in this backtest.")

render_disclaimer()

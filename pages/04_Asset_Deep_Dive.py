"""Systematic Research Engine - Asset Deep Dive.

Detailed asset-level quantitative research inspection.
Displays precomputed daily equity curves, drawdown regimes, and walk-forward statistics.
"""

import sys
from pathlib import Path

# Add src to sys.path
_SRC_PATH = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from quant_engine.research.store import ResearchStore
from quant_engine.ui_helpers import apply_terminal_theme, render_terminal_header, render_disclaimer

st.set_page_config(page_title="Asset Deep Dive", page_icon="🔍", layout="wide")
apply_terminal_theme()

render_terminal_header(
    title="ASSET QUANTITATIVE DEEP DIVE",
    subtitle="Inspect historical daily equity curves, drawdown profiles, and regime robustness for any asset",
)

store = ResearchStore()
markets = store.get_distinct_markets()

if not markets:
    st.warning("No research data found in database. Run `python scripts/run_research.py` to populate.")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    selected_m = st.selectbox("Market", markets, index=0 if "MY" not in markets else markets.index("MY"))

available_symbols = store.get_distinct_symbols(market=selected_m)
with col2:
    selected_sym = st.selectbox("Asset / Instrument", available_symbols, index=0)

strat_records = store.get_strategy_results(market=selected_m, symbol=selected_sym)
if strat_records.empty:
    st.info(f"No completed experiments for {selected_sym} in market {selected_m}.")
    st.stop()

with col3:
    variant_names = list(strat_records["strategy_name"].unique())
    selected_strat_name = st.selectbox("Evaluated Strategy", variant_names, index=0)

# Retrieve selected experiment record
exp_record = strat_records[strat_records["strategy_name"] == selected_strat_name].iloc[0]
exp_id = exp_record["experiment_id"]

# Key Metric Cards
st.markdown('<div class="section-header">PRECOMPUTED PERFORMANCE & VALIDATION METRICS</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Realized Net Sharpe", f"{exp_record['sharpe_ratio']:.2f}", f"Gross {exp_record['total_gross_return']*100:.1f}%")
k2.metric("OOS Sharpe", f"{exp_record['oos_sharpe']:.2f}", f"WFE {exp_record['walk_forward_efficiency']:.2f}")
k3.metric("Net CAGR", f"{exp_record['cagr']*100:.1f}%", f"Vol {exp_record['annualized_volatility']*100:.1f}%")
k4.metric("Maximum Drawdown", f"{exp_record['max_drawdown']*100:.1f}%", f"Calmar {exp_record['calmar_ratio']:.2f}")
k5.metric("Friction Survival", f"{exp_record['friction_survival_ratio']*100:.1f}%", f"${exp_record['total_fees_paid']:,.0f} Fees")
k6.metric("Overfitting Risk", exp_record["overfitting_risk_level"], f"Score {exp_record['overfitting_score']:.0f}/100")

# Daily Equity Curve from SQLite
curve_df = store.get_equity_curve(exp_id)

if not curve_df.empty:
    st.markdown('<div class="section-header">HISTORICAL NAV & EXECUTION FRICTION PROFILE</div>', unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve_df.index,
            y=curve_df["NAV"],
            mode="lines",
            name="Net Realized NAV (After All Costs)",
            line=dict(color="#00e5ff", width=2),
        )
    )
    if "Gross_NAV" in curve_df.columns:
        fig.add_trace(
            go.Scatter(
                x=curve_df.index,
                y=curve_df["Gross_NAV"],
                mode="lines",
                name="Frictionless Gross NAV",
                line=dict(color="#94a3b8", width=1.5, dash="dot"),
            )
        )
    if "Benchmark_NAV" in curve_df.columns:
        fig.add_trace(
            go.Scatter(
                x=curve_df.index,
                y=curve_df["Benchmark_NAV"],
                mode="lines",
                name="Market Benchmark NAV",
                line=dict(color="#64748b", width=1, dash="dash"),
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Growth of $1.00",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Underwater Drawdown Profile
    st.markdown('<div class="section-header">UNDERWATER DRAWDOWN PROFILE</div>', unsafe_allow_html=True)
    
    dd_fig = go.Figure()
    dd_fig.add_trace(
        go.Scatter(
            x=curve_df.index,
            y=curve_df["Drawdown"] * 100.0,
            fill="tozeroy",
            mode="lines",
            name="Drawdown (%)",
            line=dict(color="#ef4444", width=1),
            fillcolor="rgba(239, 68, 68, 0.2)",
        )
    )
    dd_fig.update_layout(
        template="plotly_dark",
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="Drawdown %",
        showlegend=False,
    )
    st.plotly_chart(dd_fig, use_container_width=True)
else:
    st.info("Daily equity curve data not found for this experiment.")

# Regime Analysis Breakdown
st.markdown('<div class="section-header">MARKET REGIME PERFORMANCE DECOMPOSITION</div>', unsafe_allow_html=True)

r_cols = st.columns(4)
r_cols[0].metric("Bull Market Sharpe", f"{exp_record['bull_sharpe']:.2f}")
r_cols[1].metric("Bear Market Sharpe", f"{exp_record['bear_sharpe']:.2f}")
r_cols[2].metric("High-Volatility Sharpe", f"{exp_record['high_vol_sharpe']:.2f}")
r_cols[3].metric("Low-Volatility Sharpe", f"{exp_record['low_vol_sharpe']:.2f}")

# Breakeven & Trade Statistics
st.markdown('<div class="section-header">TRADE EXECUTION & CAPACITY ATTRIBUTION</div>', unsafe_allow_html=True)

t1, t2, t3, t4, t5 = st.columns(5)
t1.metric("Total Trades", f"{exp_record['total_trades']}")
t2.metric("Win Rate", f"{exp_record['win_rate']*100:.1f}%")
t3.metric("Profit Factor", f"{exp_record['profit_factor']:.2f}")
t4.metric("Annualized Turnover", f"{exp_record['annualized_turnover']:.1f}x")
t5.metric("Breakeven Cost", f"{exp_record['breakeven_cost_bps']:.1f} bps", "Max Friction Allowed")

render_disclaimer()

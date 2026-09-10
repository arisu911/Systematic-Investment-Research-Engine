"""Systematic Research Engine - Strategy Explorer.

Precomputed systematic strategy deep-dive across multiple geographic markets.
Evaluates strategy universality, friction decay, and parameter stability without client-side computation.
"""

import sys
from pathlib import Path

# Add src to sys.path
_SRC_PATH = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

import streamlit as st
import pandas as pd
import plotly.express as px
from quant_engine.research.store import ResearchStore
from quant_engine.ui_helpers import apply_terminal_theme, render_terminal_header, render_disclaimer

st.set_page_config(page_title="Strategy Explorer", page_icon="📈", layout="wide")
apply_terminal_theme()

render_terminal_header(
    title="STRATEGY RESEARCH EXPLORER",
    subtitle="Inspect systematic strategy behavior, cross-market persistence, and friction tolerance",
)

store = ResearchStore()
all_results = store.get_strategy_results()

if all_results.empty:
    st.warning("No strategy results found in database. Run `python scripts/run_research.py` to populate.")
    st.stop()

strat_types = sorted(list(all_results["strategy_type"].unique()))

col1, col2 = st.columns([2, 2])
with col1:
    selected_type = st.selectbox("Select Strategy Family", strat_types, index=0)

available_variants = sorted(list(all_results[all_results["strategy_type"] == selected_type]["strategy_name"].unique()))
with col2:
    selected_variant = st.selectbox("Select Specific Parameter Variant", ["ALL VARIANTS"] + available_variants, index=0)

# Filter results
if selected_variant == "ALL VARIANTS":
    strat_df = all_results[all_results["strategy_type"] == selected_type].copy()
else:
    strat_df = all_results[all_results["strategy_name"] == selected_variant].copy()

# Strategy Headline Metrics
st.markdown('<div class="section-header">CROSS-MARKET STRATEGY BENCHMARK</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Evaluations", f"{len(strat_df)} Runs", f"{len(strat_df['market'].unique())} Markets")
c2.metric("Median Net Sharpe", f"{strat_df['sharpe_ratio'].median():.2f}")
c3.metric("Median OOS Sharpe", f"{strat_df['oos_sharpe'].median():.2f}")
avg_survival = strat_df["friction_survival_ratio"].mean() * 100.0
c4.metric("Alpha Friction Survival", f"{avg_survival:.1f}%", "After Real Costs")
robust_pct = (len(strat_df[strat_df["walk_forward_efficiency"] >= 0.60]) / len(strat_df)) * 100.0 if len(strat_df) > 0 else 0.0
c5.metric("OOS Robust Rate", f"{robust_pct:.1f}%", "WFE >= 0.60")

# Cross-Market Comparison Chart
st.markdown('<div class="section-header">CROSS-MARKET SHARPE COMPARISON</div>', unsafe_allow_html=True)

fig = px.box(
    strat_df,
    x="market",
    y="sharpe_ratio",
    color="market",
    points="all",
    hover_data=["symbol", "strategy_name", "oos_sharpe", "friction_survival_ratio"],
    labels={"market": "Market Universe", "sharpe_ratio": "Realized Net Sharpe"},
    template="plotly_dark",
    title=f"Net Sharpe Distribution Across Markets for {selected_type}",
)
fig.add_hline(y=0.0, line_dash="dash", line_color="#ef4444")
fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Friction Reality Breakdown: Gross vs Net Return
st.markdown('<div class="section-header">EXECUTION FRICTION ATTRITION (GROSS VS NET)</div>', unsafe_allow_html=True)

f_fig = px.bar(
    strat_df,
    x="symbol",
    y=["total_gross_return", "total_net_return"],
    barmode="group",
    color_discrete_map={"total_gross_return": "#38bdf8", "total_net_return": "#22c55e"},
    labels={"value": "Total Return", "variable": "Return Metric", "symbol": "Asset"},
    template="plotly_dark",
    title="Gross vs Realized Net Return (Impact of Statutory Stamp Duty, Brokerage & Slippage)",
)
f_fig.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(f_fig, use_container_width=True)

# Precomputed Detailed Results Table
st.markdown('<div class="section-header">DETAILED EXPERIMENT RECORDS</div>', unsafe_allow_html=True)

disp_cols = [
    "market", "symbol", "strategy_name", "sharpe_ratio", "oos_sharpe",
    "walk_forward_efficiency", "cagr", "max_drawdown", "annualized_turnover",
    "friction_survival_ratio", "breakeven_cost_bps", "overfitting_risk_level"
]
formatted_df = strat_df[disp_cols].sort_values("sharpe_ratio", ascending=False).copy()
formatted_df.rename(
    columns={
        "market": "Market",
        "symbol": "Asset",
        "strategy_name": "Variant",
        "sharpe_ratio": "IS Sharpe",
        "oos_sharpe": "OOS Sharpe",
        "walk_forward_efficiency": "WFE",
        "cagr": "Net CAGR",
        "max_drawdown": "Max DD",
        "annualized_turnover": "Turnover",
        "friction_survival_ratio": "Survival",
        "breakeven_cost_bps": "Breakeven Cost",
        "overfitting_risk_level": "Risk Level",
    },
    inplace=True,
)
st.dataframe(
    formatted_df.style.format({
        "IS Sharpe": "{:.2f}",
        "OOS Sharpe": "{:.2f}",
        "WFE": "{:.2f}",
        "Net CAGR": "{:.1%}",
        "Max DD": "{:.1%}",
        "Turnover": "{:.1f}x",
        "Survival": "{:.1%}",
        "Breakeven Cost": "{:.1f} bps",
    }),
    use_container_width=True,
    hide_index=True,
)

render_disclaimer()

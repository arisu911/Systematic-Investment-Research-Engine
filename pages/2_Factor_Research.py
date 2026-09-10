"""Factor Research: Cross-Market Correlation, Rolling Betas & Lead-Lag Analysis.

Displays:
1. Full 25x25 cross-market Pearson correlation heatmap with regional group markers (MY, US, JP, Macro).
2. Dynamic rolling CAPM beta against selected regional benchmarks (^KLSE, ^GSPC, ^N225).
3. Cross-market lead-lag cross-correlation matrix (shifts from -5d to +5d).
4. Multi-horizon momentum scorecard (1M, 3M, 6M, 12M, composite).
"""

import sys
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data.aligner import load_universe_registry
from data.loader import get_cached_universe_prices, get_all_universe_tickers, compute_lookback_dates, LOOKBACK_HORIZONS
from research.signals import SignalEngine
from research.utils import inject_metric_css, render_data_freshness_badge, render_backfill_warning_badge

try:
    st.set_page_config(page_title="Factor Research", page_icon="🌐", layout="wide")
except Exception:
    pass

inject_metric_css()

# Initialize Session State Defaults
if "lookback_horizon" not in st.session_state:
    st.session_state["lookback_horizon"] = "5Y"
if "start_date" not in st.session_state or "end_date" not in st.session_state:
    s_date, e_date = compute_lookback_dates(st.session_state["lookback_horizon"])
    st.session_state["start_date"] = s_date
    st.session_state["end_date"] = e_date

st.markdown(
    """
    <div style="background-color: #131722; border: 1px solid #2a2e39; border-left: 5px solid #3b82f6;
                padding: 14px 18px; border-radius: 4px; margin-bottom: 20px;">
        <span style="font-size: 11px; font-weight: 800; color: #3b82f6; letter-spacing: 1.5px; text-transform: uppercase;">
            CROSS-MARKET FACTOR DYNAMICS • STATISTICAL COVARIANCE
        </span>
        <h2 style="margin: 4px 0 0 0; color: #ffffff; font-size: 22px; font-weight: 700;">
            Global 25x25 Factor Correlation & Lead-Lag Matrix
        </h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render Data Freshness & Proxy Backfill Warning Badges
render_data_freshness_badge()
render_backfill_warning_badge()

# Load full 25-instrument universe
base_curr = st.session_state.get("selected_currency", st.session_state.get("base_currency", "USD"))
is_hedged = st.session_state.get("hedged_toggle", False)

with st.expander("⚙️ Fine-Tune Historical Horizon & Base Currency", expanded=False):
    f_c1, f_c2 = st.columns(2)
    curr_options = ["USD", "MYR", "JPY", "EUR", "GBP", "LOCAL"]
    new_curr = f_c1.selectbox("Base Currency", curr_options, index=curr_options.index(base_curr) if base_curr in curr_options else 0, key="factor_curr")
    if new_curr != base_curr:
        st.session_state["selected_currency"] = new_curr
        st.rerun()
    cur_lb = st.session_state.get("lookback_horizon", "5Y")
    cur_lb_idx = LOOKBACK_HORIZONS.index(cur_lb) if cur_lb in LOOKBACK_HORIZONS else 3
    new_lb = f_c2.selectbox("Historical Horizon", LOOKBACK_HORIZONS, index=cur_lb_idx, key="factor_horizon")
    if new_lb != cur_lb:
        st.session_state["lookback_horizon"] = new_lb
        s_date, e_date = compute_lookback_dates(new_lb)
        st.session_state["start_date"] = s_date
        st.session_state["end_date"] = e_date
        st.rerun()

prices_df, is_demo = get_cached_universe_prices(
    start_date=st.session_state.get("start_date"),
    end_date=st.session_state.get("end_date"),
    base_currency=base_curr,
    lookback_horizon=st.session_state.get("lookback_horizon", "5Y"),
    ttl_seconds=st.session_state.get("cache_ttl_seconds", 14400),
    force_reload=st.session_state.get("force_reload", False),
)

registry = load_universe_registry()
all_tickers = list(registry.keys())
available_tickers = [t for t in all_tickers if t in prices_df.columns]

# Sort tickers systematically by region: MY -> US -> JP -> MACRO
def sort_key(t):
    meta = registry.get(t, {})
    reg = meta.get("region", "OTHER")
    order = {"MY": 1, "US": 2, "JP": 3, "GLOBAL": 4}
    return (order.get(reg, 5), t)

sorted_tickers = sorted(available_tickers, key=sort_key)
prices_sorted = prices_df[sorted_tickers]

if is_hedged:
    from data.fx_engine import FXEngine
    fx_eng = FXEngine(prices_sorted, registry=registry)
    returns_sorted = fx_eng.compute_asset_returns(target_currency=base_curr, is_hedged=True)
else:
    returns_sorted = prices_sorted.pct_change().dropna()

# 1. Full 25x25 Correlation Matrix Heatmap
st.markdown("#### 🌐 Full 25x25 Cross-Market Correlation Matrix")
st.caption(
    "Pairwise daily return correlations across Malaysia, US, Japan, and Cross-Market Macro instruments. "
    "Delineated into regional clusters."
)

corr_matrix = SignalEngine.calculate_cross_market_correlation(returns_sorted)

fig_corr = px.imshow(
    corr_matrix,
    x=sorted_tickers,
    y=sorted_tickers,
    color_continuous_scale="RdBu_r",
    zmin=-1.0,
    zmax=1.0,
    labels=dict(color="Pearson Corr"),
    aspect="auto",
)

# Regional cluster division lines
# Indices where region changes
reg_labels = [registry.get(t, {}).get("region", "") for t in sorted_tickers]
div_indices = []
for i in range(1, len(reg_labels)):
    if reg_labels[i] != reg_labels[i-1]:
        div_indices.append(i - 0.5)

for d in div_indices:
    fig_corr.add_vline(x=d, line_width=1.5, line_color="#00c805", line_dash="dash")
    fig_corr.add_hline(y=d, line_width=1.5, line_color="#00c805", line_dash="dash")

fig_corr.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#131722",
    margin=dict(l=10, r=10, t=20, b=10),
    height=620,
    coloraxis_colorbar=dict(title="Corr", thickness=15, len=0.8),
    xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
    yaxis=dict(tickfont=dict(size=10)),
)
st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False, "responsive": True})

st.markdown("---")

# 2. Rolling Beta Against Selected Benchmark
c_beta1, c_beta2 = st.columns([3, 7])

with c_beta1:
    st.markdown("#### 🎯 Rolling Beta Engine")
    bench_select = st.selectbox(
        "Target Benchmark",
        ["^KLSE", "^GSPC", "^N225", "^NDX"],
        index=1,
        help="Compute 60-day rolling CAPM beta of selected assets against this index.",
    )
    
    # Let user pick 4 assets to compare
    default_sub = [t for t in ["1155.KL", "AAPL", "NVDA", "7203.T", "GC=F"] if t in sorted_tickers][:4]
    selected_assets = st.multiselect(
        "Compare Assets",
        sorted_tickers,
        default=default_sub,
    )
    
    roll_window = st.slider("Rolling Window (Days)", 21, 126, 60)

with c_beta2:
    if selected_assets and bench_select in prices_df.columns:
        bench_ret = prices_df[bench_select].pct_change().dropna()
        sub_returns = prices_df[selected_assets].pct_change().dropna()
        rolling_betas = SignalEngine.calculate_rolling_beta(sub_returns, bench_ret, window=roll_window)

        fig_beta = go.Figure()
        colors = ["#00c805", "#3b82f6", "#ec4899", "#f59e0b", "#a855f7"]
        for i, col in enumerate(selected_assets):
            if col in rolling_betas.columns:
                fig_beta.add_trace(
                    go.Scatter(
                        x=rolling_betas.index,
                        y=rolling_betas[col],
                        name=f"{registry.get(col, {}).get('name', col)} ({col})",
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate="%{y:.2f}<extra></extra>",
                    )
                )

        fig_beta.add_hline(y=1.0, line_dash="dot", line_color="#64748b", annotation_text="Beta = 1.0")
        fig_beta.add_hline(y=0.0, line_dash="solid", line_color="#334155")

        fig_beta.update_layout(
            title=dict(text=f"60-Day Rolling Beta vs {bench_select}", font=dict(size=14, color="#e0e0e0")),
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#131722",
            margin=dict(l=10, r=10, t=40, b=10),
            height=340,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor="#2a2e39"),
            yaxis=dict(showgrid=True, gridcolor="#2a2e39"),
        )
        st.plotly_chart(fig_beta, use_container_width=True, config={"displayModeBar": False, "responsive": True})
    else:
        st.info("Select assets and ensure the benchmark is present to plot rolling beta.")

st.markdown("---")

# 3. Lead-Lag Cross Correlation & Momentum Scorecard
c_lag, c_mom = st.columns([5, 5])

with c_lag:
    st.markdown("#### ⏱️ Cross-Market Lead-Lag Matrix")
    st.caption("Investigates if macro/overseas market moves lead domestic Bursa Malaysia equities.")

    lead_pair_x = st.selectbox("Leading Candidate (t - k)", ["^GSPC", "^VIX", "BZ=F", "^N225", "USDMYR=X"], index=0)
    lead_pair_y = st.selectbox("Lagging Candidate (t)", ["^KLSE", "1155.KL", "6742.KL", "7203.T"], index=0)

    if lead_pair_x in prices_df.columns and lead_pair_y in prices_df.columns:
        rx = prices_df[lead_pair_x].pct_change().dropna()
        ry = prices_df[lead_pair_y].pct_change().dropna()
        lead_lag_s = SignalEngine.calculate_lead_lag_cross_correlation(ry, rx, max_lag=5)

        fig_lag = go.Figure(
            data=[
                go.Bar(
                    x=[f"Lag {k}d" for k in lead_lag_s.index],
                    y=lead_lag_s.values,
                    marker_color=["#00c805" if v > 0 else "#ef4444" for v in lead_lag_s.values],
                    hovertemplate="%{y:.3f}<extra></extra>",
                )
            ]
        )
        fig_lag.update_layout(
            title=dict(text=f"Cross-Correlation: {lead_pair_x} vs {lead_pair_y}", font=dict(size=12)),
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#131722",
            margin=dict(l=10, r=10, t=40, b=10),
            height=280,
            yaxis=dict(range=[-0.5, 0.5], gridcolor="#2a2e39"),
        )
        st.plotly_chart(fig_lag, use_container_width=True, config={"displayModeBar": False, "responsive": True})

with c_mom:
    st.markdown("#### ⚡ Cross-Asset Momentum Scorecard")
    mom_df = SignalEngine.calculate_momentum_scores(prices_sorted)
    mom_df["Name"] = [registry.get(i, {}).get("name", i) for i in mom_df.index]
    mom_df["Region"] = [registry.get(i, {}).get("region", "OTHER") for i in mom_df.index]

    mom_display = mom_df[["Name", "Region", "1M_Return", "3M_Return", "12M_Return", "Composite_Momentum"]].copy()
    mom_display = mom_display.sort_values("Composite_Momentum", ascending=False)
    
    st.dataframe(
        mom_display.style.format({
            "1M_Return": "{:.1%}",
            "3M_Return": "{:.1%}",
            "12M_Return": "{:.1%}",
            "Composite_Momentum": "{:.2%}",
        }),
        use_container_width=True,
        height=320,
    )

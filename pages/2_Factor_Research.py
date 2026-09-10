"""Institutional Multi-Asset Portfolio - Factor Research & Dispersion Terminal.

Evaluates cross-asset correlation heatmaps, rolling CAPM beta against domestic and global
benchmarks, rolling realized volatility, and momentum / mean-reversion factor signals.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add root directory to sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml

from data.loader import get_cached_multi_asset_data
from research.signals import SignalEngine

st.set_page_config(page_title="Factor Research", page_icon="🧬", layout="wide")

with open(_ROOT / "configs" / "universe.yaml", "r", encoding="utf-8") as f:
    u_cfg = yaml.safe_load(f)

core_symbols = []
for ac in u_cfg.get("asset_classes", []):
    for a in ac.get("assets", []):
        if a["symbol"] not in core_symbols:
            core_symbols.append(a["symbol"])

st.markdown(
    """
    <div style="background-color: #131722; border: 1px solid #2a2e39; border-left: 5px solid #38bdf8;
                padding: 14px 18px; border-radius: 4px; margin-bottom: 18px;">
        <span style="font-size: 11px; font-weight: 800; color: #38bdf8; letter-spacing: 1px; text-transform: uppercase;">
            FACTOR MODELING & CORRELATION DYNAMICS
        </span>
        <h3 style="margin: 3px 0 0 0; color: #ffffff; font-size: 20px; font-weight: 700;">
            Cross-Asset Factor Research & Dispersion Matrix
        </h3>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Controls
st.sidebar.markdown("### ⚙️ Factor Horizon Settings")
lookback_days = st.sidebar.selectbox("Estimation Window", [252, 504, 756], index=1, format_func=lambda x: f"{x} Trading Days ({x//248}Y)")
beta_window = st.sidebar.slider("Rolling Beta Window (Days)", min_value=20, max_value=120, value=60, step=10)
force_offline = st.sidebar.checkbox("Force Offline Mode", value=False)

end_dt = datetime.now()
start_dt = end_dt - timedelta(days=int(lookback_days * 1.55))

prices_df, is_demo = get_cached_multi_asset_data(
    core_symbols,
    start_date=start_dt.strftime("%Y-%m-%d"),
    end_date=end_dt.strftime("%Y-%m-%d"),
    base_currency="MYR",
    force_offline=force_offline,
)

valid_prices = prices_df.dropna()
returns_df = valid_prices.pct_change().dropna()

# Section 1: Correlation Matrix
st.markdown("### 🌐 Cross-Asset Correlation Heatmap")

corr_matrix = SignalEngine.calculate_cross_asset_correlation(returns_df)

corr_fig = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1.0,
    zmax=1.0,
    labels=dict(color="Correlation"),
    template="plotly_dark",
)
corr_fig.update_layout(
    height=480,
    margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
)
st.plotly_chart(corr_fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

# Section 2: Rolling Beta to Domestic (^KLSE) and Global (SPY)
st.markdown("### 📈 Rolling 60-Day CAPM Beta Dynamics")

b_left, b_right = st.columns(2)

with b_left:
    st.markdown("**Beta relative to Domestic Equity Benchmark (`^KLSE`)**")
    if "^KLSE" in returns_df.columns:
        domestic_betas = SignalEngine.calculate_rolling_beta(returns_df, returns_df["^KLSE"], window=beta_window)
        # Select representative assets to plot
        sample_cols = [c for c in ["1155.KL", "1023.KL", "5347.KL", "SPY", "MGS_10Y"] if c in domestic_betas.columns]
        
        beta_dom_fig = go.Figure()
        for col in sample_cols:
            beta_dom_fig.add_trace(go.Scatter(x=domestic_betas.index, y=domestic_betas[col], mode="lines", name=col))
        
        beta_dom_fig.add_hline(y=1.0, line_dash="dash", line_color="#64748b")
        beta_dom_fig.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="Beta vs KLCI",
        )
        st.plotly_chart(beta_dom_fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
    else:
        st.info("Domestic benchmark (^KLSE) not present in active selection.")

with b_right:
    st.markdown("**Beta relative to Global Equity Benchmark (`SPY`)**")
    if "SPY" in returns_df.columns:
        global_betas = SignalEngine.calculate_rolling_beta(returns_df, returns_df["SPY"], window=beta_window)
        sample_cols = [c for c in ["1155.KL", "QQQ", "EEM", "GC=F", "MGS_10Y"] if c in global_betas.columns]
        
        beta_glob_fig = go.Figure()
        for col in sample_cols:
            beta_glob_fig.add_trace(go.Scatter(x=global_betas.index, y=global_betas[col], mode="lines", name=col))
        
        beta_glob_fig.add_hline(y=1.0, line_dash="dash", line_color="#64748b")
        beta_glob_fig.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="Beta vs SPY",
        )
        st.plotly_chart(beta_glob_fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
    else:
        st.info("Global benchmark (SPY) not present in active selection.")

# Section 3: Momentum & Mean Reversion Factor Scorecard
st.markdown("### ⚡ Systematic Factor Scorecard (Momentum vs Mean Reversion)")

momentum_df = SignalEngine.calculate_momentum_scores(valid_prices)
z_scores = SignalEngine.calculate_mean_reversion_zscores(valid_prices, window=20)

factor_summary = momentum_df.copy()
factor_summary["Mean_Rev_ZScore_20d"] = z_scores

st.dataframe(
    factor_summary.style.format({
        "1M_Return": "{:+.2%}",
        "3M_Return": "{:+.2%}",
        "6M_Return": "{:+.2%}",
        "12M_Return": "{:+.2%}",
        "Composite_Momentum": "{:+.2%}",
        "Mean_Rev_ZScore_20d": "{:+.2f}",
    }).background_gradient(subset=["Composite_Momentum"], cmap="RdYlGn", vmin=-0.2, vmax=0.2),
    use_container_width=True,
)

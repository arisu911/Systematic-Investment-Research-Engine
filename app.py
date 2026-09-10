"""Institutional Multi-Asset Portfolio Allocation & Systematic Research Workstation.

Global dashboard entry point and executive workstation overview.
Adheres strictly to zero-cost, open-access financial data architecture.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add root directory to sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yaml

from data.loader import get_cached_multi_asset_data
from research.optimization import PortfolioOptimizer
from research.risk import RiskEngine
from experiments.backtester import PortfolioBacktester

# Page Configuration
st.set_page_config(
    page_title="Institutional Multi-Asset Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Universe & Risk Configurations
@st.cache_data(ttl=3600)
def load_app_configs():
    with open(_ROOT / "configs" / "universe.yaml", "r", encoding="utf-8") as f:
        u_cfg = yaml.safe_load(f)
    with open(_ROOT / "configs" / "risk_limits.yaml", "r", encoding="utf-8") as f:
        r_cfg = yaml.safe_load(f)
    return u_cfg, r_cfg


universe_cfg, risk_cfg = load_app_configs()

# Flatten default asset universe
core_symbols = []
offshore_flags = []
asset_metadata = {}

for ac in universe_cfg.get("asset_classes", []):
    domicile = ac.get("domicile", "domestic")
    for a in ac.get("assets", []):
        sym = a["symbol"]
        if sym not in core_symbols:
            core_symbols.append(sym)
            is_offshore = (domicile == "offshore") or (a.get("domicile") == "offshore")
            offshore_flags.append(is_offshore)
            asset_metadata[sym] = {
                "name": a.get("name", sym),
                "sector": a.get("sector", "General"),
                "class": ac.get("name", "Other"),
                "is_offshore": is_offshore,
                "weight_mkt": a.get("weight_mkt", 0.05),
            }

# Terminal Header Banner
st.markdown(
    """
    <div style="background-color: #131722; border: 1px solid #2a2e39; border-left: 5px solid #00c805;
                padding: 16px 20px; border-radius: 4px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 11px; font-weight: 800; color: #00c805; letter-spacing: 1.5px; text-transform: uppercase;">
                    INSTITUTIONAL QUANTITATIVE WORKSTATION • FACTSET / BLOOMBERG ERGONOMICS
                </span>
                <h2 style="margin: 4px 0 0 0; color: #ffffff; font-size: 24px; font-weight: 700; letter-spacing: -0.5px;">
                    Malaysian Multi-Asset Risk & Allocation Terminal
                </h2>
            </div>
            <div style="text-align: right;">
                <span style="background-color: #1e2433; color: #00e5ff; font-size: 11px; font-weight: 700;
                             padding: 5px 10px; border-radius: 3px; border: 1px solid #2d3748;">
                    ● ZERO-COST DATA ARCHITECTURE (NO PAID APIS)
                </span>
                <div style="color: #94a3b8; font-size: 11px; margin-top: 5px; font-family: monospace;">
                    BASE CURRENCY: MYR • STATUTORY FRICTIONS MODELED
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Institutional Controls
st.sidebar.markdown("### ⚙️ Allocation Mandates & Controls")

preset = st.sidebar.selectbox(
    "Strategy Allocation Preset",
    [
        "EPF Institutional Balanced (30% Max Offshore, 5% Cash)",
        "Maximum Sharpe Ratio (Unconstrained)",
        "Minimum Volatility (Capital Preservation)",
        "Black-Litterman Active Views (Overweight Bursa & Tech)",
        "Equal Weight Baseline (1/N)",
    ],
    index=0,
)

lookback_choice = st.sidebar.selectbox("Estimation Lookback Window", ["1 Year (252d)", "2 Years (504d)", "3 Years (756d)"], index=1)
lookback_days = 252 if "1" in lookback_choice else 756 if "3" in lookback_choice else 504

base_curr = st.sidebar.radio("Base Accounting Currency", ["MYR", "USD"], index=0, horizontal=True)
rebal_freq = st.sidebar.selectbox("Rebalancing Frequency", ["Monthly", "Quarterly", "Annual"], index=0)
force_offline = st.sidebar.checkbox("Force Offline Mode (Deterministic Demo)", value=False)

# Date calculations
end_dt = datetime.now()
start_dt = end_dt - timedelta(days=int(lookback_days * 1.55))
start_str = start_dt.strftime("%Y-%m-%d")
end_str = end_dt.strftime("%Y-%m-%d")

# Load Pricing Dataset
with st.spinner("Synchronizing multi-asset datasets via free public endpoints..."):
    prices_df, is_demo = get_cached_multi_asset_data(
        core_symbols,
        start_date=start_str,
        end_date=end_str,
        base_currency=base_curr,
        force_offline=force_offline,
    )

if prices_df.empty:
    st.error("Unable to load price series. Toggle 'Force Offline Mode' in the sidebar to use local synthetic fallback.")
    st.stop()

# Align assets present
valid_assets = [s for s in core_symbols if s in prices_df.columns]
valid_prices = prices_df[valid_assets].dropna()
returns_df = valid_prices.pct_change().dropna()
offshore_mask = [asset_metadata[s]["is_offshore"] for s in valid_assets]

# Compute Benchmark: Domestic 60/40 (^KLSE + MGS 10Y)
bench_equity = valid_prices["^KLSE"] if "^KLSE" in valid_prices.columns else valid_prices.iloc[:, 0]
bench_bond = valid_prices["MGS_10Y"] if "MGS_10Y" in valid_prices.columns else valid_prices.iloc[:, -1]
bench_60_40_series = 0.60 * (bench_equity / bench_equity.iloc[0]) + 0.40 * (bench_bond / bench_bond.iloc[0])
bench_60_40_series = bench_60_40_series * 100.0

# Initialize Quantitative Optimizer
optimizer = PortfolioOptimizer(returns_df, risk_free_rate=0.030, annual_trading_days=248)

# Compute Weights based on Preset
cash_idx = valid_assets.index("MGS_3Y") if "MGS_3Y" in valid_assets else valid_assets.index("MGS_5Y") if "MGS_5Y" in valid_assets else None

if "Equal Weight" in preset:
    opt_weights = np.ones(len(valid_assets)) / len(valid_assets)
elif "Min Volatility" in preset:
    res = optimizer.optimize_mean_variance(
        objective="min_volatility",
        max_single_asset=0.25,
        max_offshore=0.30,
        offshore_mask=offshore_mask,
    )
    opt_weights = res["weights"]
elif "Black-Litterman" in preset:
    mkt_weights = np.array([asset_metadata[s]["weight_mkt"] for s in valid_assets])
    mkt_weights = mkt_weights / np.sum(mkt_weights)
    views = {
        "1155.KL": (0.09, 0.70),  # Maybank expected return 9% (70% confidence)
        "SPY": (0.11, 0.65),      # S&P 500 expected return 11% (65% confidence)
        "MGS_10Y": (0.042, 0.85), # MGS 10Y yield 4.2% (85% confidence)
    }
    pi, post_er, post_cov = optimizer.compute_black_litterman(mkt_weights, views)
    res = optimizer.optimize_mean_variance(
        objective="max_sharpe",
        max_single_asset=0.20,
        max_offshore=0.30,
        offshore_mask=offshore_mask,
        expected_returns=post_er,
        cov_matrix=post_cov,
    )
    opt_weights = res["weights"]
elif "Unconstrained" in preset:
    res = optimizer.optimize_mean_variance(objective="max_sharpe", max_single_asset=0.40, max_offshore=1.0)
    opt_weights = res["weights"]
else:
    # EPF Institutional Balanced (Default)
    res = optimizer.optimize_mean_variance(
        objective="max_sharpe",
        max_single_asset=0.20,
        max_offshore=0.30,
        offshore_mask=offshore_mask,
        min_cash_buffer=0.05,
        cash_index=cash_idx,
    )
    opt_weights = res["weights"]

# Backtest Engine Execution
backtester = PortfolioBacktester(valid_prices, benchmark_prices=bench_60_40_series, initial_capital=100_000.0)
equity_df, kpis = backtester.run_rebalancing_backtest(opt_weights, frequency=rebal_freq)

# Top KPI Scorecards
st.markdown('<div style="margin-top: 5px; margin-bottom: 15px;">', unsafe_allow_html=True)
k1, k2, k3, k4, k5, k6, k7 = st.columns(7)

k1.metric("CAGR", f"{kpis['CAGR']*100:.2f}%", "Annualized")
k2.metric("Volatility", f"{kpis['Annualized_Volatility']*100:.2f}%", "Annualized")
k3.metric("Sharpe Ratio", f"{kpis['Sharpe_Ratio']:.2f}", f"Rf = 3.0%")
k4.metric("Sortino Ratio", f"{kpis['Sortino_Ratio']:.2f}", "Downside")
k5.metric("Max Drawdown", f"{kpis['Max_Drawdown']*100:.2f}%", "Peak-to-Trough")
k6.metric("Calmar Ratio", f"{kpis['Calmar_Ratio']:.2f}", "Return/DD")
k7.metric("Tracking Error", f"{kpis['Tracking_Error']*100:.2f}%", "vs Domestic 60/40")
st.markdown('</div>', unsafe_allow_html=True)

# Main Institutional Visual Layout: Cumulative Wealth Chart (60%) beside Allocation Donut (40%)
c_left, c_right = st.columns([6, 4])

with c_left:
    st.markdown(
        """
        <div style="font-size: 13px; font-weight: 700; color: #e0e0e0; margin-bottom: 8px; text-transform: uppercase;">
            Cumulative Wealth Trajectory (Growth of RM 100,000)
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    growth_fig = go.Figure()
    growth_fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df["NAV"] * 100_000.0,
            mode="lines",
            name="Allocated Portfolio (Net of All Frictions)",
            line=dict(color="#00c805", width=2.5),
            hovertemplate="Allocated Portfolio: <b>RM %{y:,.2f}</b><extra></extra>",
        )
    )
    growth_fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df["Benchmark_NAV"] * 100_000.0,
            mode="lines",
            name="Domestic 60/40 Benchmark (60% KLCI + 40% MGS 10Y)",
            line=dict(color="#38bdf8", width=1.5, dash="dash"),
            hovertemplate="Domestic 60/40 Benchmark: <b>RM %{y:,.2f}</b><extra></extra>",
        )
    )
    if "^KLSE" in valid_prices.columns:
        klci_norm = (valid_prices["^KLSE"] / valid_prices["^KLSE"].iloc[0]) * 100_000.0
        growth_fig.add_trace(
            go.Scatter(
                x=valid_prices.index,
                y=klci_norm,
                mode="lines",
                name="Bursa KLCI Equity Index (^KLSE)",
                line=dict(color="#64748b", width=1, dash="dot"),
                hovertemplate="Bursa KLCI: <b>RM %{y:,.2f}</b><extra></extra>",
            )
        )

    growth_fig.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(
            title="Portfolio Value (RM)",
            tickprefix="RM ",
            gridcolor="#2a2e39",
            zerolinecolor="#2a2e39",
        ),
        xaxis=dict(gridcolor="#2a2e39"),
    )
    st.plotly_chart(growth_fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

with c_right:
    offshore_total = sum(opt_weights[i] for i, off in enumerate(offshore_mask) if off)
    badge_col = "#00c805" if offshore_total <= 0.301 else "#ff1744"
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 13px; font-weight: 700; color: #e0e0e0; text-transform: uppercase;">
                Optimal Asset Allocation
            </span>
            <span style="font-size: 11px; font-weight: 700; color: {badge_col}; font-family: monospace;">
                OFFSHORE: {offshore_total*100:.1f}% / 30.0% LIMIT
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Donut Chart
    active_idx = [i for i, w in enumerate(opt_weights) if w >= 0.005]
    donut_labels = [asset_metadata[valid_assets[i]]["name"][:22] for i in active_idx]
    donut_values = [opt_weights[i] for i in active_idx]

    donut_fig = go.Figure(
        data=[
            go.Pie(
                labels=donut_labels,
                values=donut_values,
                hole=0.55,
                textinfo="label+percent",
                insidetextorientation="radial",
                hovertemplate="<b>%{label}</b><br>Allocation: <b>%{percent:.1%}</b><extra></extra>",
            )
        ]
    )
    donut_fig.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        showlegend=False,
    )
    st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

# Navigation Callout to Multi-Page Modules
st.markdown("---")
st.markdown(
    """
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
        <div style="background-color: #131722; padding: 14px; border-radius: 4px; border: 1px solid #2a2e39;">
            <div style="color: #00c805; font-weight: 700; font-size: 13px;">PAGE 1: EXECUTIVE TEARSHEET</div>
            <div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">Detailed weights ledger, monthly returns heatmap, and benchmark tracking.</div>
        </div>
        <div style="background-color: #131722; padding: 14px; border-radius: 4px; border: 1px solid #2a2e39;">
            <div style="color: #38bdf8; font-weight: 700; font-size: 13px;">PAGE 2: FACTOR RESEARCH</div>
            <div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">Cross-asset correlation matrix, rolling beta, volatility, and dispersion.</div>
        </div>
        <div style="background-color: #131722; padding: 14px; border-radius: 4px; border: 1px solid #2a2e39;">
            <div style="color: #ffab00; font-weight: 700; font-size: 13px;">PAGE 3: STRATEGY BACKTEST</div>
            <div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">Drawdown underwater analysis, turnover fees, and 4 historical crisis stress tests.</div>
        </div>
        <div style="background-color: #131722; padding: 14px; border-radius: 4px; border: 1px solid #2a2e39;">
            <div style="color: #e040fb; font-weight: 700; font-size: 13px;">PAGE 4: RISK & TAIL ENGINE</div>
            <div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">Non-Gaussian Cornish-Fisher VaR/CVaR, fat-tail skewness, and risk contribution.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

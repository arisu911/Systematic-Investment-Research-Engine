"""Executive Summary: Consolidated Multi-Asset Portfolio Analytics Workstation.

Displays:
1. Executive KPI Scorecard (CAGR, Annualized Volatility, Sharpe, Sortino, Max Drawdown, Calmar).
2. Capital Growth Curve ($100k baseline) vs Selected Global Benchmark (^KLSE, ^GSPC, ^N225).
3. Optimal Asset Allocation Donut Chart (Asset Level & Regional Cluster Level).
4. Interactive Model Selector: Max Sharpe, Min Volatility, Risk Parity, Black-Litterman.
5. Monthly Returns Performance Matrix Heatmap.
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

from data.aligner import load_universe_registry, get_tradable_tickers
from data.loader import get_cached_universe_prices
from research.optimization import PortfolioOptimizer
from experiments.backtester import PortfolioBacktester

try:
    st.set_page_config(page_title="Executive Summary", page_icon="🏛️", layout="wide")
except Exception:
    pass

st.markdown(
    """
    <div style="background-color: #131722; border: 1px solid #2a2e39; border-left: 5px solid #00c805;
                padding: 14px 18px; border-radius: 4px; margin-bottom: 20px;">
        <span style="font-size: 11px; font-weight: 800; color: #00c805; letter-spacing: 1.5px; text-transform: uppercase;">
            PORTFOLIO TEARSHEET • EXECUTIVE ANALYTICS
        </span>
        <h2 style="margin: 4px 0 0 0; color: #ffffff; font-size: 22px; font-weight: 700;">
            Multi-Market Optimal Allocation & Tearsheet
        </h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# Configuration controls
c_ctl1, c_ctl2, c_ctl3, c_ctl4 = st.columns(4)

base_curr = c_ctl1.selectbox(
    "Currency Normalization",
    ["USD", "MYR", "LOCAL"],
    index=0,
    key="exec_base_curr",
)

alloc_model = c_ctl2.selectbox(
    "Allocation Strategy",
    ["Max Sharpe (MVO)", "Minimum Volatility", "Risk Parity (ERC)", "Black-Litterman"],
    index=0,
)

benchmark_choice = c_ctl3.selectbox(
    "Benchmark Overlay",
    ["^GSPC", "^KLSE", "^N225", "^NDX"],
    index=0,
    help="Global equity benchmark for relative alpha/beta evaluation.",
)

rebal_freq = c_ctl4.selectbox(
    "Rebalancing Frequency",
    ["Monthly", "Quarterly", "Annual"],
    index=0,
)

# Load data
prices_df, is_demo = get_cached_universe_prices(
    start_date="2020-01-01",
    end_date="2024-12-31",
    base_currency=base_curr,
)

registry = load_universe_registry()
tradable_symbols = get_tradable_tickers(registry)
available_tradable = [s for s in tradable_symbols if s in prices_df.columns]

if len(available_tradable) < 3:
    st.error("Insufficient tradable asset history loaded. Please verify data stream.")
    st.stop()

# Returns
tradable_prices = prices_df[available_tradable]
returns_df = tradable_prices.pct_change().dropna()

# Benchmark series
bench_series = prices_df[benchmark_choice] if benchmark_choice in prices_df.columns else prices_df.iloc[:, 0]

# Optimize portfolio
optimizer = PortfolioOptimizer(returns_df, risk_free_rate=0.040, registry=registry, filter_tradable=False)

if alloc_model == "Max Sharpe (MVO)":
    opt_res = optimizer.optimize_mean_variance(objective="max_sharpe", max_single_asset=0.25)
elif alloc_model == "Minimum Volatility":
    opt_res = optimizer.optimize_mean_variance(objective="min_volatility", max_single_asset=0.25)
elif alloc_model == "Risk Parity (ERC)":
    opt_res = optimizer.optimize_risk_parity(mode="asset_level")
else: # Black-Litterman
    views = [
        {"asset_long": "NVDA", "asset_short": "7203.T", "relative_return": 0.04, "confidence": 0.70},
        {"asset_long": "1155.KL", "view_return": 0.08, "confidence": 0.65},
        {"asset_long": "GC=F", "view_return": 0.07, "confidence": 0.60},
    ]
    opt_res = optimizer.optimize_black_litterman(views_list=views, max_single_asset=0.25)

weights = opt_res["weights"]
assets = opt_res["assets"]

# Run backtest
backtester = PortfolioBacktester(
    prices_df=tradable_prices[assets],
    benchmark_prices=bench_series,
    annual_trading_days=252,
    initial_capital=100_000.0,
)

equity_df, summary = backtester.run_rebalancing_backtest(weights=weights, frequency=rebal_freq)

# Top KPI Scorecard
st.markdown("#### 📊 Key Performance Indicators (Friction-Adjusted)")
k1, k2, k3, k4, k5, k6 = st.columns(6)

currency_sym = "$" if base_curr == "USD" else "RM " if base_curr == "MYR" else ""
k1.metric("CAGR", f"{summary['cagr']:.2%}", f"vs {summary['benchmark_cagr']:.2%} Bench")
k2.metric("Annual Volatility", f"{summary['annualized_volatility']:.2%}")
k3.metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
k4.metric("Sortino Ratio", f"{summary['sortino_ratio']:.2f}")
k5.metric("Max Drawdown", f"{summary['max_drawdown']:.2%}", delta_color="inverse")
k6.metric("Calmar Ratio", f"{summary['calmar_ratio']:.2f}")

st.markdown("---")

# Visual Layout: Equity Curve (60%) beside Allocation Donut (40%)
c_left, c_right = st.columns([6, 4])

with c_left:
    st.markdown(f"#### 📈 Growth of {currency_sym}100,000 Portfolio vs {benchmark_choice}")

    fig_equity = go.Figure()
    fig_equity.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df["NAV"],
            name=f"Portfolio NAV ({alloc_model})",
            line=dict(color="#00c805", width=2.5),
            hovertemplate=f"{currency_sym}%{{y:,.2f}}<extra></extra>",
        )
    )
    fig_equity.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df["Benchmark_NAV"],
            name=f"Benchmark ({benchmark_choice})",
            line=dict(color="#64748b", width=1.5, dash="dot"),
            hovertemplate=f"{currency_sym}%{{y:,.2f}}<extra></extra>",
        )
    )

    fig_equity.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#131722",
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor="#2a2e39"),
        yaxis=dict(showgrid=True, gridcolor="#2a2e39"),
    )
    st.plotly_chart(fig_equity, use_container_width=True, config={"displayModeBar": False, "responsive": True})

with c_right:
    st.markdown("#### 🍩 Optimal Asset Allocation")

    w_df = pd.DataFrame({"Asset": assets, "Weight": weights})
    w_df = w_df[w_df["Weight"] >= 0.01].sort_values("Weight", ascending=False)
    w_df["Name"] = [registry.get(a, {}).get("name", a) for a in w_df["Asset"]]
    w_df["Region"] = [registry.get(a, {}).get("region", "OTHER") for a in w_df["Asset"]]

    fig_donut = px.pie(
        w_df,
        values="Weight",
        names="Name",
        hole=0.55,
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )
    fig_donut.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="%{label}: %{percent:.1%}<extra></extra>",
    )
    fig_donut.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#131722",
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
        showlegend=False,
    )
    st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False, "responsive": True})

st.markdown("---")

# Allocation Breakdown Table & Regional Exposure
c_tab1, c_tab2 = st.columns([6, 4])

with c_tab1:
    st.markdown("#### 📑 Position Weights & Region Segregation")
    w_table = []
    for a, w in zip(assets, weights):
        meta = registry.get(a, {})
        w_table.append({
            "Ticker": a,
            "Name": meta.get("name", a),
            "Region": meta.get("region", "OTHER"),
            "Role": meta.get("role", "equity"),
            "Weight": f"{w:.2%}",
            "Value Allocation": f"{currency_sym}{w * summary['ending_nav']:,.2f}",
        })
    st.dataframe(pd.DataFrame(w_table).sort_values("Weight", ascending=False), use_container_width=True, hide_index=True)

with c_tab2:
    st.markdown("#### 🗺️ Regional & Sector Concentration")
    reg_alloc = {}
    for a, w in zip(assets, weights):
        meta = registry.get(a, {})
        reg = meta.get("region", "GLOBAL")
        if meta.get("asset_class") == "commodity":
            reg = "COMMODITY"
        reg_alloc[reg] = reg_alloc.get(reg, 0.0) + w

    reg_df = pd.DataFrame([{"Bucket": k, "Weight": v} for k, v in reg_alloc.items()])
    fig_bar = px.bar(
        reg_df,
        x="Weight",
        y="Bucket",
        orientation="h",
        color="Bucket",
        color_discrete_sequence=["#00c805", "#3b82f6", "#ec4899", "#f59e0b"],
    )
    fig_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#131722",
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
        showlegend=False,
        xaxis=dict(tickformat=".0%"),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False, "responsive": True})

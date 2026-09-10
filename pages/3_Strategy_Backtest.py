"""Institutional Multi-Asset Portfolio - Backtest & Crisis Stress Replay.

Vectorized multi-asset rebalancing backtest under statutory Bursa transaction costs,
underwater drawdown analytics, and historical crisis stress test simulations.
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
import plotly.graph_objects as go
import yaml

from data.loader import get_cached_multi_asset_data
from research.optimization import PortfolioOptimizer
from experiments.backtester import PortfolioBacktester

st.set_page_config(page_title="Strategy Backtest", page_icon="📈", layout="wide")

with open(_ROOT / "configs" / "universe.yaml", "r", encoding="utf-8") as f:
    u_cfg = yaml.safe_load(f)

core_symbols = []
offshore_flags = []
for ac in u_cfg.get("asset_classes", []):
    domicile = ac.get("domicile", "domestic")
    for a in ac.get("assets", []):
        if a["symbol"] not in core_symbols:
            core_symbols.append(a["symbol"])
            offshore_flags.append(domicile == "offshore" or a.get("domicile") == "offshore")

st.markdown(
    """
    <div style="background-color: #131722; border: 1px solid #2a2e39; border-left: 5px solid #ffab00;
                padding: 14px 18px; border-radius: 4px; margin-bottom: 18px;">
        <span style="font-size: 11px; font-weight: 800; color: #ffab00; letter-spacing: 1px; text-transform: uppercase;">
            EXECUTION BACKTEST & CRISIS STRESS ENGINE
        </span>
        <h3 style="margin: 3px 0 0 0; color: #ffffff; font-size: 20px; font-weight: 700;">
            Portfolio Rebalancing Simulation & Historical Crisis Replay
        </h3>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Controls
st.sidebar.markdown("### ⚙️ Backtest Execution Parameters")
initial_capital = st.sidebar.number_input("Initial Investment Capital (RM)", min_value=10_000.0, value=100_000.0, step=10_000.0)
rebal_freq = st.sidebar.selectbox("Rebalancing Schedule", ["Monthly", "Quarterly", "Annual", "Daily"], index=0)
lookback_days = st.sidebar.selectbox("Estimation Horizon", [504, 756, 1008], index=0, format_func=lambda x: f"{x} Bars ({x//248}Y)")
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
valid_assets = valid_prices.columns.tolist()
offshore_mask = [offshore_flags[core_symbols.index(s)] for s in valid_assets]

# Benchmark: Domestic 60/40
bench_equity = valid_prices["^KLSE"] if "^KLSE" in valid_prices.columns else valid_prices.iloc[:, 0]
bench_bond = valid_prices["MGS_10Y"] if "MGS_10Y" in valid_prices.columns else valid_prices.iloc[:, -1]
bench_60_40 = 0.60 * (bench_equity / bench_equity.iloc[0]) + 0.40 * (bench_bond / bench_bond.iloc[0])
bench_60_40 = bench_60_40 * 100.0

# Optimize with EPF Mandate Constraints
optimizer = PortfolioOptimizer(returns_df, risk_free_rate=0.030, annual_trading_days=248)
opt_res = optimizer.optimize_mean_variance(
    objective="max_sharpe",
    max_single_asset=0.20,
    max_offshore=0.30,
    offshore_mask=offshore_mask,
)
weights = opt_res["weights"]

# Run Backtest
backtester = PortfolioBacktester(
    valid_prices,
    benchmark_prices=bench_60_40,
    initial_capital=initial_capital,
    brokerage_bps=10.0,
    clearing_bps=3.0,
    stamp_duty_bps=10.0,
    stamp_duty_cap=1000.0,
    slippage_bps=5.0,
)
equity_df, summary = backtester.run_rebalancing_backtest(weights, frequency=rebal_freq)

# Key Performance Scorecards
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Ending Capital", f"RM {equity_df['NAV'].iloc[-1]*initial_capital:,.2f}", f"Net Return {equity_df['NAV'].iloc[-1]*100 - 100:+.1f}%")
k2.metric("Annualized CAGR", f"{summary['CAGR']*100:.2f}%")
k3.metric("Net Sharpe Ratio", f"{summary['Sharpe_Ratio']:.2f}", "After All Frictions")
k4.metric("Max Drawdown", f"{summary['Max_Drawdown']*100:.2f}%")
k5.metric("Total Fees Incurred", f"RM {summary['Total_Fees_Paid']:,.2f}", f"{summary['Rebalance_Count']} Rebalances")

# Section 1: Growth Chart
st.markdown("### 📈 Net Equity Trajectory vs Benchmarks")

equity_fig = go.Figure()
equity_fig.add_trace(
    go.Scatter(
        x=equity_df.index,
        y=equity_df["NAV"] * initial_capital,
        mode="lines",
        name="Allocated Portfolio (Net of All Costs)",
        line=dict(color="#00c805", width=2.5),
        hovertemplate="Portfolio: <b>RM %{y:,.2f}</b><extra></extra>",
    )
)
equity_fig.add_trace(
    go.Scatter(
        x=equity_df.index,
        y=equity_df["Gross_NAV"] * initial_capital,
        mode="lines",
        name="Frictionless Gross NAV (Theoretical)",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
        hovertemplate="Gross NAV: <b>RM %{y:,.2f}</b><extra></extra>",
    )
)
equity_fig.add_trace(
    go.Scatter(
        x=equity_df.index,
        y=equity_df["Benchmark_NAV"] * initial_capital,
        mode="lines",
        name="Domestic 60/40 Benchmark (60% KLCI + 40% MGS 10Y)",
        line=dict(color="#38bdf8", width=1.5, dash="dash"),
        hovertemplate="Benchmark 60/40: <b>RM %{y:,.2f}</b><extra></extra>",
    )
)

equity_fig.update_layout(
    template="plotly_dark",
    height=400,
    margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis=dict(title="Portfolio Value (RM)", tickprefix="RM ", gridcolor="#2a2e39"),
    xaxis=dict(gridcolor="#2a2e39"),
)
st.plotly_chart(equity_fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

# Section 2: Underwater Drawdown Chart
st.markdown("### 🌊 Underwater Drawdown Profile")

dd_fig = go.Figure()
dd_fig.add_trace(
    go.Scatter(
        x=equity_df.index,
        y=equity_df["Drawdown"] * 100.0,
        fill="tozeroy",
        mode="lines",
        name="Portfolio Drawdown",
        line=dict(color="#ef4444", width=1.5),
        fillcolor="rgba(239, 68, 68, 0.2)",
        hovertemplate="Portfolio DD: <b>%{y:.2f}%</b><extra></extra>",
    )
)
dd_fig.add_trace(
    go.Scatter(
        x=equity_df.index,
        y=equity_df["Bench_Drawdown"] * 100.0,
        mode="lines",
        name="Benchmark 60/40 DD",
        line=dict(color="#64748b", width=1, dash="dash"),
        hovertemplate="Benchmark DD: <b>%{y:.2f}%</b><extra></extra>",
    )
)
dd_fig.update_layout(
    template="plotly_dark",
    height=260,
    margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis=dict(title="Drawdown (%)", ticksuffix="%", gridcolor="#2a2e39"),
    xaxis=dict(gridcolor="#2a2e39"),
)
st.plotly_chart(dd_fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

# Section 3: Historical Crisis Stress Testing
st.markdown("### 💥 Historical Crisis Stress Replay Module")
st.markdown("*Simulated impact of historical market shocks on the allocated multi-asset portfolio:*")

crisis_results = backtester.replay_historical_crisis(weights)
crisis_df = pd.DataFrame(crisis_results)

st.dataframe(
    crisis_df.style.format({
        "Estimated_Return": "{:+.2%}",
        "Estimated_Max_Drawdown": "{:.2%}",
    }).background_gradient(subset=["Estimated_Max_Drawdown"], cmap="Reds_r", vmin=-0.4, vmax=0.0),
    use_container_width=True,
    hide_index=True,
)

# Export Artifacts Button
st.markdown("---")
if st.button("💾 Persist Run Artifacts to results/runs/"):
    run_dir = backtester.save_run_artifacts(weights, summary, equity_df, run_name="streamlit_backtest")
    st.success(f"Execution artifacts saved to `{run_dir}` (summary.json, weights.csv, equity_curve.parquet).")

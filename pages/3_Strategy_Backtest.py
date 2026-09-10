"""Strategy Backtest: Multi-Market Rebalancing & Walk-Forward Validation Workstation.

Displays:
1. Portfolio Equity Curve (Net NAV vs Gross NAV vs Benchmark NAV).
2. Interactive Friction & Slippage Slider (0 - 50 bps).
3. Friction Drag & Cumulative Fee Attrition Breakdown.
4. Underwater Drawdown Profiles & Maximum Peak-to-Trough Metrics.
5. In-Sample vs. Out-of-Sample Walk-Forward Stability Table (WFE Metric).
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

from data.aligner import load_universe_registry, get_tradable_tickers
from data.loader import get_cached_universe_prices, compute_lookback_dates, LOOKBACK_HORIZONS
from data.fx_engine import get_currency_symbol, SUPPORTED_CURRENCIES
from research.strategies import StrategyDispatcher, STRATEGY_REGISTRY
from research.optimization import PortfolioOptimizer
from research.utils import inject_metric_css, format_money, render_data_freshness_badge, render_backfill_warning_badge
from experiments.backtester import PortfolioBacktester

try:
    st.set_page_config(page_title="Strategy Backtest", page_icon="📈", layout="wide")
except Exception:
    pass

# Inject global metric CSS to prevent ellipsis truncation
inject_metric_css()

# Retrieve Global Session State
capital = float(st.session_state.get("capital_amount", 100_000.0))
base_curr = st.session_state.get("selected_currency", "USD")
active_strat = st.session_state.get("selected_strategy", "max_sharpe")
bench_choice_default = st.session_state.get("benchmark_ticker", "^GSPC")
is_hedged = st.session_state.get("hedged_toggle", False)
if "lookback_horizon" not in st.session_state:
    st.session_state["lookback_horizon"] = "5Y"
if "start_date" not in st.session_state or "end_date" not in st.session_state:
    s_date, e_date = compute_lookback_dates(st.session_state["lookback_horizon"])
    st.session_state["start_date"] = s_date
    st.session_state["end_date"] = e_date

st.markdown(
    """
    <div style="background-color: #131722; border: 1px solid #2a2e39; border-left: 5px solid #ec4899;
                padding: 14px 18px; border-radius: 4px; margin-bottom: 20px;">
        <span style="font-size: 11px; font-weight: 800; color: #ec4899; letter-spacing: 1.5px; text-transform: uppercase;">
            SYSTEMATIC SIMULATION • EXECUTION FRICTION & WALK-FORWARD
        </span>
        <h2 style="margin: 4px 0 0 0; color: #ffffff; font-size: 22px; font-weight: 700;">
            Multi-Market Strategy Backtester & Walk-Forward Engine
        </h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render Data Freshness & Proxy Backfill Warning Badges
render_data_freshness_badge()
render_backfill_warning_badge()

curr_sym = get_currency_symbol(base_curr)

# Control Panel
b_c1, b_c2, b_c3, b_c4, b_c5 = st.columns(5)

curr_options = SUPPORTED_CURRENCIES + ["LOCAL"]
base_curr = b_c1.selectbox("Base Currency", curr_options, index=curr_options.index(base_curr) if base_curr in curr_options else 0, key="backtest_curr")
curr_sym = get_currency_symbol(base_curr)
currency_sym = curr_sym
strat_keys = list(STRATEGY_REGISTRY.keys())
strat_names = list(STRATEGY_REGISTRY.values())
cur_idx = strat_keys.index(active_strat) if active_strat in strat_keys else 0
strategy_type_name = b_c2.selectbox("Strategy Model", strat_names, index=cur_idx)
strategy_type = strat_keys[strat_names.index(strategy_type_name)]

cur_lb = st.session_state.get("lookback_horizon", "5Y")
cur_lb_idx = LOOKBACK_HORIZONS.index(cur_lb) if cur_lb in LOOKBACK_HORIZONS else 3
selected_lb = b_c3.selectbox("Historical Horizon", LOOKBACK_HORIZONS, index=cur_lb_idx, key="backtest_horizon")
if selected_lb != cur_lb:
    st.session_state["lookback_horizon"] = selected_lb
    s_date, e_date = compute_lookback_dates(selected_lb)
    st.session_state["start_date"] = s_date
    st.session_state["end_date"] = e_date
    st.rerun()

rebal_freq = b_c4.selectbox("Rebalancing Schedule", ["Monthly", "Quarterly", "Annual"], index=0)
benchmarks = ["^GSPC", "^KLSE", "^N225", "^NDX", "^RUT"]
cur_bench_idx = benchmarks.index(bench_choice_default) if bench_choice_default in benchmarks else 0
bench_choice = b_c5.selectbox("Benchmark", benchmarks, index=cur_bench_idx)

# Interactive Slippage & Friction Slider
friction_bps = st.slider(
    "Rebalancing Friction / Slippage (bps)",
    min_value=0,
    max_value=50,
    value=10,
    step=1,
    help="Realistic trading frictions including brokerage commissions, exchange clearing, bid-ask spread, and stamp duty per turnover event.",
)

with st.spinner("Executing strategy dispatch and portfolio rebalancing simulation..."):
    prices_df, is_demo = get_cached_universe_prices(
        start_date=st.session_state.get("start_date"),
        end_date=st.session_state.get("end_date"),
        base_currency=base_curr,
        lookback_horizon=st.session_state.get("lookback_horizon", "5Y"),
        ttl_seconds=st.session_state.get("cache_ttl_seconds", 14400),
        force_reload=st.session_state.get("force_reload", False),
    )

registry = load_universe_registry()
tradable_symbols = get_tradable_tickers(registry)
available_tradable = [s for s in tradable_symbols if s in prices_df.columns]

if len(available_tradable) < 3:
    st.error("Insufficient tradable assets for strategy backtest.")
    st.stop()

tradable_prices = prices_df[available_tradable]
returns_df = tradable_prices.pct_change().dropna()
bench_series = prices_df[bench_choice] if bench_choice in prices_df.columns else prices_df.iloc[:, 0]

# Strategy Dispatcher
strat_res = StrategyDispatcher.dispatch(
    strategy_name=strategy_type,
    returns_df=returns_df,
    lookback_days=252,
    risk_free_rate=0.040,
    max_single_asset=0.25,
    top_n=5,
    registry=registry,
    filter_tradable=False,
)

weights = strat_res["weights"]
assets = strat_res["assets"]

backtester = PortfolioBacktester(
    prices_df=tradable_prices[assets],
    benchmark_prices=bench_series,
    annual_trading_days=252,
    initial_capital=capital,
)

equity_df, summary = backtester.run_rebalancing_backtest(
    weights=weights,
    frequency=rebal_freq,
    friction_bps=float(friction_bps),
)

# Top KPI Balanced Metric Grid (2x3)
st.markdown("#### 📊 Strategy Performance & Friction Drag Matrix")
k_row1_c1, k_row1_c2, k_row1_c3 = st.columns(3)
k_row1_c1.metric(
    f"Final Net NAV ({base_curr})",
    format_money(summary["ending_nav"], base_curr),
    f"{summary['total_return']:+.1%} Total Net Return",
)
k_row1_c2.metric(
    f"Gross Ending NAV (Zero Drag)",
    format_money(summary["gross_ending_nav"], base_curr),
    f"{summary['gross_total_return']:+.1%} Gross Return",
)
k_row1_c3.metric(
    "Friction & Slippage Drag",
    format_money(summary["total_friction_drag"], base_curr),
    f"-{summary['friction_drag_bps']:.1f} bps CAGR Impact",
    delta_color="inverse",
)

k_row2_c1, k_row2_c2, k_row2_c3 = st.columns(3)
k_row2_c1.metric(
    "Net CAGR",
    f"{summary['cagr']:.2%}",
    f"vs {summary['gross_cagr']:.2%} Gross ({summary['benchmark_cagr']:.2%} Bench)",
)
k_row2_c2.metric(
    "Sharpe Ratio (Net)",
    f"{summary['sharpe_ratio']:.2f}",
    f"Sortino: {summary['sortino_ratio']:.2f} | Vol: {summary['annualized_volatility']:.2%}",
)
k_row2_c3.metric(
    "Maximum Drawdown",
    f"{summary['max_drawdown']:.2%}",
    f"Calmar: {summary['calmar_ratio']:.2f}",
    delta_color="inverse",
)

st.markdown("---")

# Chart 1: Equity Curves
st.markdown(f"#### 📈 Net Equity NAV vs Gross NAV vs Benchmark ({bench_choice})")

fig_eq = go.Figure()
fig_eq.add_trace(
    go.Scatter(
        x=equity_df.index,
        y=equity_df["NAV"],
        name=f"Net NAV ({friction_bps} bps Slippage)",
        line=dict(color="#00c805", width=2.5),
        hovertemplate=f"{curr_sym}%{{y:,.2f}}<extra></extra>",
    )
)
fig_eq.add_trace(
    go.Scatter(
        x=equity_df.index,
        y=equity_df["Gross_NAV"],
        name="Gross NAV (Zero Frictions)",
        line=dict(color="#38bdf8", width=1.8, dash="dash"),
        hovertemplate=f"{curr_sym}%{{y:,.2f}}<extra></extra>",
    )
)
fig_eq.add_trace(
    go.Scatter(
        x=equity_df.index,
        y=equity_df["Benchmark_NAV"],
        name=f"Benchmark ({bench_choice})",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
        hovertemplate=f"{curr_sym}%{{y:,.2f}}<extra></extra>",
    )
)

fig_eq.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#131722",
    margin=dict(l=10, r=10, t=10, b=10),
    height=360,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=True, gridcolor="#2a2e39"),
    yaxis=dict(showgrid=True, gridcolor="#2a2e39"),
)
st.plotly_chart(fig_eq, use_container_width=True, config={"displayModeBar": False, "responsive": True})

# Chart 2: Underwater Drawdown Profile
st.markdown("#### 🌊 Underwater Drawdown Profile")

fig_dd = go.Figure()
fig_dd.add_trace(
    go.Scatter(
        x=equity_df.index,
        y=equity_df["Drawdown"],
        name="Portfolio Drawdown",
        fill="tozeroy",
        fillcolor="rgba(239, 68, 68, 0.25)",
        line=dict(color="#ef4444", width=1.5),
        hovertemplate="%{y:.2%}<extra></extra>",
    )
)
fig_dd.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#131722",
    margin=dict(l=10, r=10, t=10, b=10),
    height=240,
    yaxis=dict(tickformat=".0%", range=[min(equity_df["Drawdown"].min() * 1.15, -0.05), 0.0], gridcolor="#2a2e39"),
    xaxis=dict(showgrid=True, gridcolor="#2a2e39"),
)
st.plotly_chart(fig_dd, use_container_width=True, config={"displayModeBar": False, "responsive": True})

st.markdown("---")

# Walk-Forward Validation & Artifact Persistence
c_wf, c_art = st.columns([6, 4])

with c_wf:
    st.markdown("#### 🔁 Rolling Walk-Forward Efficiency (WFE)")
    st.caption("Evaluates 504-day In-Sample training vs 126-day Out-of-Sample testing slices.")

    def wf_weight_func(train_ret):
        sub_opt = PortfolioOptimizer(train_ret, filter_tradable=False)
        return sub_opt.optimize_mean_variance("max_sharpe")["weights"]

    wf_slices = backtester.run_walk_forward_validation(wf_weight_func, train_window=504, test_window=126)
    if wf_slices:
        wf_df = pd.DataFrame(wf_slices)
        wf_display = pd.DataFrame({
            "Slice": wf_df["slice_id"],
            "Train End": wf_df["train_end"],
            "Test End": wf_df["test_end"],
            "IS Sharpe": wf_df["in_sample_sharpe"],
            "OOS Sharpe": wf_df["out_of_sample_sharpe"],
            "Walk-Forward Efficiency": wf_df["walk_forward_efficiency"],
        })
        st.dataframe(
            wf_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "IS Sharpe": st.column_config.NumberColumn("IS Sharpe", format="%.2f"),
                "OOS Sharpe": st.column_config.NumberColumn("OOS Sharpe", format="%.2f"),
                "Walk-Forward Efficiency": st.column_config.NumberColumn("WFE Ratio", format="%.2f"),
            },
        )
    else:
        st.info("Insufficient history for rolling 504d/126d walk-forward slices.")

with c_art:
    st.markdown("#### 💾 Run Artifacts Persistence")
    st.caption("Exports summary metrics, weight vectors, and equity curve into results/runs/.")
    if st.button("Export Run Artifacts", use_container_width=True):
        saved_dir = backtester.save_run_artifacts(weights, summary, equity_df, run_name=f"run_{strategy_type.lower().replace(' ', '_')}")
        st.success(f"Artifacts persisted to:\n`{saved_dir.name}`")
        st.json(summary)

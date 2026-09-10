"""Strategy Backtest: Multi-Market Rebalancing & Walk-Forward Validation Workstation.

Displays:
1. Portfolio Equity Curve (Net NAV vs Gross NAV vs Benchmark NAV).
2. Underwater Drawdown Profiles & Maximum Peak-to-Trough Metrics.
3. In-Sample vs. Out-of-Sample Walk-Forward Stability Table (WFE Metric).
4. Periodic Rebalancing Turnover Ledger & Friction Cost Attrition.
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
from data.loader import get_cached_universe_prices
from research.optimization import PortfolioOptimizer
from experiments.backtester import PortfolioBacktester

try:
    st.set_page_config(page_title="Strategy Backtest", page_icon="📈", layout="wide")
except Exception:
    pass

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

# Retrieve Global Session State
capital = float(st.session_state.get("capital_amount", 100_000.0))
base_curr = st.session_state.get("selected_currency", "USD")
active_strat = st.session_state.get("selected_strategy", "max_sharpe")
bench_choice_default = st.session_state.get("benchmark_ticker", "^GSPC")
is_hedged = st.session_state.get("hedged_toggle", False)

from data.fx_engine import get_currency_symbol, SUPPORTED_CURRENCIES
curr_sym = get_currency_symbol(base_curr)

# Control Panel
b_c1, b_c2, b_c3, b_c4 = st.columns(4)

curr_options = SUPPORTED_CURRENCIES + ["LOCAL"]
base_curr = b_c1.selectbox("Base Currency", curr_options, index=curr_options.index(base_curr) if base_curr in curr_options else 0, key="backtest_curr")
strat_keys = list(STRATEGY_REGISTRY.keys())
strat_names = list(STRATEGY_REGISTRY.values())
cur_idx = strat_keys.index(active_strat) if active_strat in strat_keys else 0
strategy_type_name = b_c2.selectbox("Strategy Model", strat_names, index=cur_idx)
strategy_type = strat_keys[strat_names.index(strategy_type_name)]

rebal_freq = b_c3.selectbox("Rebalancing Schedule", ["Monthly", "Quarterly", "Annual"], index=0)
benchmarks = ["^GSPC", "^KLSE", "^N225", "^NDX", "^RUT"]
cur_bench_idx = benchmarks.index(bench_choice_default) if bench_choice_default in benchmarks else 0
bench_choice = b_c4.selectbox("Benchmark", benchmarks, index=cur_bench_idx)

with st.spinner("Executing strategy dispatch and portfolio rebalancing simulation..."):
    prices_df, is_demo = get_cached_universe_prices(
        start_date="2019-01-01",
        end_date="2024-12-31",
        base_currency=base_curr,
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
from research.strategies import StrategyDispatcher
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

equity_df, summary = backtester.run_rebalancing_backtest(weights=weights, frequency=rebal_freq)

# Top KPIs
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric(f"Final NAV ({base_curr})", f"{curr_sym}{summary['ending_nav']:,.0f}", f"{summary['total_return']:+.1%}")
m2.metric("CAGR", f"{summary['cagr']:.2%}")
m3.metric("Annual Volatility", f"{summary['annualized_volatility']:.2%}")
m4.metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
m5.metric("Max Drawdown", f"{summary['max_drawdown']:.2%}", delta_color="inverse")
m6.metric("Total Frictions", f"{curr_sym}{summary['total_friction_drag']:,.2f}")

st.markdown("---")

# Chart 1: Equity Curves
st.markdown("#### 📈 Net Equity NAV vs Gross NAV vs Benchmark")

fig_eq = go.Figure()
fig_eq.add_trace(
    go.Scatter(
        x=equity_df.index,
        y=equity_df["NAV"],
        name="Net NAV (Post-Friction)",
        line=dict(color="#00c805", width=2.5),
        hovertemplate=f"{currency_sym}%{{y:,.2f}}<extra></extra>",
    )
)
fig_eq.add_trace(
    go.Scatter(
        x=equity_df.index,
        y=equity_df["Gross_NAV"],
        name="Gross NAV (Zero Frictions)",
        line=dict(color="#38bdf8", width=1.5, dash="dash"),
        hovertemplate=f"{currency_sym}%{{y:,.2f}}<extra></extra>",
    )
)
fig_eq.add_trace(
    go.Scatter(
        x=equity_df.index,
        y=equity_df["Benchmark_NAV"],
        name=f"Benchmark ({bench_choice})",
        line=dict(color="#64748b", width=1.5, dash="dot"),
        hovertemplate=f"{currency_sym}%{{y:,.2f}}<extra></extra>",
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
        wf_display = wf_df[["slice_id", "train_end", "test_end", "in_sample_sharpe", "out_of_sample_sharpe", "walk_forward_efficiency"]].copy()
        st.dataframe(
            wf_display.style.format({
                "in_sample_sharpe": "{:.2f}",
                "out_of_sample_sharpe": "{:.2f}",
                "walk_forward_efficiency": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
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

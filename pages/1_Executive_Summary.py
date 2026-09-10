"""Executive Summary: Capital Allocation Tearsheet, Donut Allocation & Order Ticket Workstation.

Displays:
1. Active Capital Allocation & Strategy Overview.
2. Executive KPI Scorecard scaled to active nominal capital.
3. Nominal Capital Growth Curve vs Global Benchmark.
4. Optimal Asset Allocation Donut Chart.
5. Actionable Order Execution Ticket with 100-share Bursa board lot sizing,
   unallocated cash remainder, and CSV export.
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
from data.fx_engine import get_currency_symbol, FXEngine, SUPPORTED_CURRENCIES
from research.strategies import StrategyDispatcher, STRATEGY_REGISTRY
from research.execution import ExecutionEngine
from research.utils import inject_metric_css, format_money
from experiments.backtester import PortfolioBacktester

try:
    st.set_page_config(page_title="Executive Summary & Orders", page_icon="🏛️", layout="wide")
except Exception:
    pass

# Inject global metric CSS to prevent ellipsis truncation
inject_metric_css()

# Retrieve Global Session State or Defaults
capital = float(st.session_state.get("capital_amount", 100_000.0))
base_curr = st.session_state.get("selected_currency", "USD")
curr_sym = get_currency_symbol(base_curr)
active_strategy = st.session_state.get("selected_strategy", "max_sharpe")
is_hedged = st.session_state.get("hedged_toggle", False)
benchmark_ticker = st.session_state.get("benchmark_ticker", "^GSPC")
risk_free_rate = float(st.session_state.get("risk_free_rate", 0.040))

st.markdown(
    f"""
    <div style="background-color: #131722; border: 1px solid #2a2e39; border-left: 5px solid #00c805;
                padding: 14px 18px; border-radius: 4px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 11px; font-weight: 800; color: #00c805; letter-spacing: 1.5px; text-transform: uppercase;">
                    PORTFOLIO ALLOCATION & ORDER DISPATCH • FACTSET / BLOOMBERG PORT
                </span>
                <h2 style="margin: 4px 0 0 0; color: #ffffff; font-size: 22px; font-weight: 700;">
                    Executive Tearsheet & Actionable Order Ticket
                </h2>
            </div>
            <div style="text-align: right; color: #848e9c; font-family: monospace; font-size: 12px;">
                CAPITAL: <span style="color:#00c805; font-weight:700;">{format_money(capital, base_curr)}</span> | BASE: <span style="color:#fff;">{base_curr}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pinned interactive control bar for on-page adjustments
with st.expander("⚙️ Fine-Tune Allocation & Strategy Controls", expanded=False):
    ctl1, ctl2, ctl3, ctl4 = st.columns(4)
    
    curr_options = SUPPORTED_CURRENCIES + ["LOCAL"]
    new_curr = ctl1.selectbox("Base Currency", curr_options, index=curr_options.index(base_curr) if base_curr in curr_options else 0)
    if new_curr != base_curr:
        st.session_state["selected_currency"] = new_curr
        st.rerun()

    new_cap = ctl2.number_input("Capital", min_value=100.0, value=capital, step=10_000.0, format="%.2f")
    if new_cap != capital:
        st.session_state["capital_amount"] = new_cap
        st.rerun()

    strat_keys = list(STRATEGY_REGISTRY.keys())
    strat_names = list(STRATEGY_REGISTRY.values())
    new_strat_name = ctl3.selectbox("Strategy Model", strat_names, index=strat_keys.index(active_strategy) if active_strategy in strat_keys else 0)
    new_strat_key = strat_keys[strat_names.index(new_strat_name)]
    if new_strat_key != active_strategy:
        st.session_state["selected_strategy"] = new_strat_key
        st.rerun()

    new_bench = ctl4.selectbox("Benchmark", ["^GSPC", "^KLSE", "^N225", "^NDX", "^RUT"], index=0)
    st.session_state["benchmark_ticker"] = new_bench

# Ingest and convert prices
with st.spinner("Triangulating multi-currency exchange rates and computing strategy allocation..."):
    prices_df, is_demo = get_cached_universe_prices(
        start_date=st.session_state.get("start_date", "2020-01-01"),
        end_date=st.session_state.get("end_date", "2024-12-31"),
        base_currency=base_curr,
    )

registry = load_universe_registry()
tradable_symbols = get_tradable_tickers(registry)
available_tradable = [s for s in tradable_symbols if s in prices_df.columns]

if len(available_tradable) < 3:
    st.error("Insufficient tradable assets for portfolio allocation.")
    st.stop()

# Returns calculation (Hedged vs Unhedged)
if is_hedged:
    fx_eng = FXEngine(prices_df, registry=registry)
    returns_df = fx_eng.compute_asset_returns(target_currency=base_curr, is_hedged=True)[available_tradable]
else:
    returns_df = prices_df[available_tradable].pct_change().dropna()

# Strategy Dispatcher execution
strat_result = StrategyDispatcher.dispatch(
    strategy_name=active_strategy,
    returns_df=returns_df,
    lookback_days=252,
    risk_free_rate=risk_free_rate,
    max_single_asset=0.25,
    top_n=5,
    registry=registry,
    filter_tradable=True,
)

weights = strat_result["weights"]
assets = strat_result["assets"]
strategy_title = strat_result.get("strategy_name", active_strategy)

# Run Portfolio Backtester to generate tearsheet curves
bench_series = prices_df[benchmark_ticker] if benchmark_ticker in prices_df.columns else prices_df.iloc[:, 0]
backtester = PortfolioBacktester(
    prices_df=prices_df[assets],
    benchmark_prices=bench_series,
    annual_trading_days=252,
    initial_capital=capital,
)
equity_df, summary = backtester.run_rebalancing_backtest(weights=weights, frequency="Monthly")

# Balanced 2x3 Grid of Executive KPI Metric Cards
st.markdown("#### 📊 Nominal Performance Scorecard")

ending_nav = summary["ending_nav"]
nominal_gain = ending_nav - capital
nominal_max_loss = summary["max_drawdown"] * capital

row1_col1, row1_col2, row1_col3 = st.columns(3)
row1_col1.metric(
    f"Ending Portfolio NAV ({base_curr})",
    format_money(ending_nav, base_curr),
    f"{format_money(nominal_gain, base_curr)} ({summary['total_return']:+.1%})",
)
row1_col2.metric(
    "Compound Annual Growth (CAGR)",
    f"{summary['cagr']:.2%}",
    f"vs {summary['benchmark_cagr']:.2%} Benchmark",
)
row1_col3.metric(
    "Annualized Volatility",
    f"{summary['annualized_volatility']:.2%}",
    f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}",
)

row2_col1, row2_col2, row2_col3 = st.columns(3)
row2_col1.metric(
    "Sharpe Ratio",
    f"{summary['sharpe_ratio']:.2f}",
    f"Sortino: {summary['sortino_ratio']:.2f}",
)
row2_col2.metric(
    "Maximum Historical Drawdown",
    f"{summary['max_drawdown']:.2%}",
    f"{format_money(nominal_max_loss, base_curr)} Peak-to-Trough",
    delta_color="inverse",
)
row2_col3.metric(
    "Calmar Ratio",
    f"{summary['calmar_ratio']:.2f}",
    f"Frictions: {format_money(summary['total_friction_drag'], base_curr)}",
)

st.markdown("---")

# Visual Layout: Equity Curve (60%) beside Allocation Donut (40%)
c_left, c_right = st.columns([6, 4])

with c_left:
    st.markdown(f"#### 📈 Growth of {format_money(capital, base_curr)} Portfolio vs {benchmark_ticker}")

    fig_equity = go.Figure()
    fig_equity.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df["NAV"],
            name=f"Portfolio ({strategy_title})",
            line=dict(color="#00c805", width=2.5),
            hovertemplate=f"{curr_sym}%{{y:,.2f}}<extra></extra>",
        )
    )
    fig_equity.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df["Benchmark_NAV"],
            name=f"Benchmark ({benchmark_ticker})",
            line=dict(color="#64748b", width=1.5, dash="dot"),
            hovertemplate=f"{curr_sym}%{{y:,.2f}}<extra></extra>",
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
    st.markdown(f"#### 🍩 Asset Allocation ({strategy_title})")

    w_df = pd.DataFrame({"Asset": assets, "Weight": weights})
    w_df = w_df[w_df["Weight"] >= 0.005].sort_values("Weight", ascending=False)
    w_df["Name"] = [registry.get(a, {}).get("name", a) for a in w_df["Asset"]]

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

# Execution Order Ticket Generator & Residual Reconciliation
st.markdown("#### 🎫 Institutional Order Execution Ticket & Board Lot Sizing")
st.caption(
    "Translates portfolio allocation weights into actionable exchange orders. Strictly enforces **100-share board lots** "
    "for Bursa Malaysia counters (`.KL`) and whole-share sizing for US and Japanese equities."
)

latest_prices_map = {a: float(prices_df[a].iloc[-1]) for a in assets if a in prices_df.columns}

order_ticket = ExecutionEngine.generate_order_ticket(
    weights=weights,
    assets=assets,
    latest_prices=latest_prices_map,
    capital=capital,
    currency_symbol=curr_sym,
    base_currency=base_curr,
    registry=registry,
)

ticket_df = order_ticket["order_ticket_df"]
unallocated = order_ticket["unallocated_cash"]
unallocated_pct = order_ticket["unallocated_cash_pct"]
allocated = order_ticket["allocated_cash"]

t_col1, t_col2, t_col3 = st.columns(3)
t_col1.metric("Target Portfolio Capital", format_money(capital, base_curr))
t_col2.metric("Allocated Trade Capital", format_money(allocated, base_curr), f"{allocated/capital:.1%} of Total")
t_col3.metric("Unallocated Cash Remainder", format_money(unallocated, base_curr), f"{unallocated_pct:.2%} Lot Rounding Buffer", delta_color="inverse")

# Strict DataFrame Column Configuration
ticket_display = pd.DataFrame({
    "Ticker": ticket_df["Ticker"],
    "Name": ticket_df["Asset Name"],
    "Exchange": ticket_df["Exchange"],
    "Target Weight": ticket_df["target_weight_num"],
    "Target Value": ticket_df["target_cash_num"],
    "Price": ticket_df["price_num"],
    "Units": ticket_df["units_num"],
    "Allocated Capital": ticket_df["allocated_cash_num"],
    "Effective Weight": ticket_df["effective_weight_num"],
})

st.dataframe(
    ticket_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Name": st.column_config.TextColumn("Asset Name", width="medium"),
        "Exchange": st.column_config.TextColumn("Exchange", width="medium"),
        "Target Weight": st.column_config.NumberColumn("Target Weight (%)", format="%.2f %%"),
        "Target Value": st.column_config.NumberColumn(f"Target Value ({base_curr})", format=f"{curr_sym}%,.2f"),
        "Price": st.column_config.NumberColumn(f"Current Price ({base_curr})", format=f"{curr_sym}%,.2f"),
        "Units": st.column_config.NumberColumn("Target Units (Board Lot Compliant)", format="%d"),
        "Allocated Capital": st.column_config.NumberColumn(f"Allocated Capital ({base_curr})", format=f"{curr_sym}%,.2f"),
        "Effective Weight": st.column_config.NumberColumn("Effective Weight (%)", format="%.2f %%"),
    },
)

# Export Order Ticket Action
st.download_button(
    label="📥 Export Order Ticket (CSV)",
    data=order_ticket["csv_string"],
    file_name="order_ticket.csv",
    mime="text/csv",
    use_container_width=True,
)

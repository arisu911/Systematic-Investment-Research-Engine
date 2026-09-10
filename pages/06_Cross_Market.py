"""Cross-Market Systematic Research & Macro Hypothesis Testing."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from quant_engine.config.markets import load_market_config
from quant_engine.config.strategies import StrategyConfig, StrategyType
from quant_engine.data.downloader import DataDownloader
from quant_engine.signals.momentum import TimeSeriesMomentum, MovingAverageMomentum
from quant_engine.features.cross_market import macro_conditioned_signal
from quant_engine.backtest.engine import BacktestEngine
from quant_engine.analytics.performance import calculate_comprehensive_performance
from quant_engine.ui_helpers import (
    apply_terminal_theme,
    render_terminal_header,
    get_terminal_plotly_layout,
    render_disclaimer,
)

st.set_page_config(page_title="Cross-Market | Systematic Engine", page_icon="🌐", layout="wide")
apply_terminal_theme()

force_demo = st.session_state.get("force_demo", False)

render_terminal_header(
    title="CROSS-MARKET RESEARCH LABORATORY",
    subtitle="Evaluate Strategy Universality Across Malaysia, United States, Japan, and Europe",
    is_demo=force_demo,
)

# Strategy selection
col_str1, col_str2 = st.columns([2, 1])
with col_str1:
    strategy_choice = st.selectbox(
        "Standardized Strategy Logic",
        ["120-Day Time-Series Momentum", "20/100 MA Trend Following"],
    )
with col_str2:
    period_selection = st.selectbox("Historical Window", ["2019 to 2024", "2021 to 2024"])

start_yr = "2019-01-01" if period_selection == "2019 to 2024" else "2021-01-01"

st.markdown(
    """
    **Research Question**: Does the identical quantitative alpha thesis generate positive risk-adjusted returns
    globally, or is it an idiosyncratic artifact of a single market?
    """
)

# Market specifications
markets_to_test = [
    {"market": "MY", "name": "Malaysia (Bursa)", "symbol": "^KLSE"},
    {"market": "US", "name": "United States", "symbol": "SPY"},
    {"market": "JP", "name": "Japan", "symbol": "^N225"},
    {"market": "EU", "name": "Europe (DAX)", "symbol": "^GDAXI"},
]

downloader = DataDownloader(force_demo=force_demo)
results_summary = []
nav_traces = {}

for m in markets_to_test:
    m_cfg = load_market_config(m["market"])
    price_df, health = downloader.load_price_data(m["symbol"], start_yr, "2024-12-31")

    if price_df.empty:
        continue

    # Formulate strategy
    if strategy_choice == "120-Day Time-Series Momentum":
        strat = TimeSeriesMomentum(StrategyConfig(parameters={"lookback_days": 120, "holding_days": 20}, long_only=True))
    else:
        strat = MovingAverageMomentum(StrategyConfig(parameters={"fast_period": 20, "slow_period": 100}, long_only=True))

    sig = strat.generate_signals(price_df)

    # Backtest with market specific cost config
    engine = BacktestEngine(cost_config=m_cfg.transaction_costs, slippage_bps=float(m_cfg.transaction_costs.default_slippage_bps))
    res_df, _, sum_dict = engine.run(price_df, sig, symbol=m["symbol"])

    perf = calculate_comprehensive_performance(
        res_df["Net_Return"], res_df["NAV"], annual_trading_days=m_cfg.annual_trading_days
    )

    results_summary.append(
        {
            "Market": m["name"],
            "Asset": m["symbol"],
            "Net CAGR (%)": round(perf.get("cagr", 0.0) * 100.0, 2),
            "Net Sharpe": perf.get("sharpe_ratio", 0.0),
            "Sortino": perf.get("sortino_ratio", 0.0),
            "Max DD (%)": round(abs(perf.get("max_drawdown", 0.0)) * 100.0, 2),
            "Annual Turnover": f"{sum_dict['annualized_turnover']:.1f}x",
            "Total Fees Paid": f"${sum_dict['total_fees_paid']:,.0f}",
        }
    )

    nav_traces[m["name"]] = res_df["NAV"]

# Cross-Market Comparative Table
st.markdown('<div class="section-header">CROSS-MARKET PERFORMANCE MATRIX</div>', unsafe_allow_html=True)
st.dataframe(pd.DataFrame(results_summary), use_container_width=True, hide_index=True)

# Comparative Normalized Overlay Chart
fig_compare = go.Figure()
colors = {"Malaysia (Bursa)": "#00D4B2", "United States": "#00B8D9", "Japan": "#FFAB00", "Europe (DAX)": "#6554C0"}

for name, s_nav in nav_traces.items():
    fig_compare.add_trace(
        go.Scatter(
            x=s_nav.index,
            y=s_nav,
            name=name,
            line=dict(color=colors.get(name, "#FFFFFF"), width=2.0),
        )
    )

fig_compare.update_layout(get_terminal_plotly_layout(height=450, title="NORMALIZED STRATEGY EQUITY ACROSS GLOBAL MARKETS (BASE = 1.0)"))
st.plotly_chart(fig_compare, use_container_width=True)

# Section: Macro-Aware Hypothesis Testing
st.markdown('<div class="section-header">MACRO-AWARE HYPOTHESIS TESTING</div>', unsafe_allow_html=True)
st.markdown(
    """
    **Hypothesis**: Does filtering Bursa Malaysia equity momentum with USD/MYR currency dynamics improve performance?
    * *Condition*: When USD/MYR trailing 20-day return is positive (MYR depreciating sharply against USD), gate equity exposure to cash.
    """
)

klci_df, _ = downloader.load_price_data("^KLSE", "2019-01-01", "2024-12-31")
usmyr_series, _ = downloader.load_macro_data("MYR=X", "2019-01-01", "2024-12-31")

if not klci_df.empty and not usmyr_series.empty:
    strat_base = TimeSeriesMomentum(StrategyConfig(parameters={"lookback_days": 120, "holding_days": 20}, long_only=True))
    raw_sig = strat_base.generate_signals(klci_df)

    fx_aligned = usmyr_series.reindex(klci_df.index).ffill()
    fx_ret = (fx_aligned / fx_aligned.shift(20)) - 1.0

    # Macro conditioned: strictly lagged by 1 bar to prevent forward leakage
    cond_sig = macro_conditioned_signal(raw_sig, fx_ret, threshold=0.01, lag_macro=1)

    m_cfg_my = load_market_config("MY")
    eng_my = BacktestEngine(cost_config=m_cfg_my.transaction_costs)

    res_unfiltered, _, _ = eng_my.run(klci_df, raw_sig, symbol="^KLSE")
    res_filtered, _, _ = eng_my.run(klci_df, cond_sig, symbol="^KLSE")

    perf_unfiltered = calculate_comprehensive_performance(res_unfiltered["Net_Return"], res_unfiltered["NAV"])
    perf_filtered = calculate_comprehensive_performance(res_filtered["Net_Return"], res_filtered["NAV"])

    h_col1, h_col2 = st.columns(2)
    h_col1.metric("Unfiltered Strategy Sharpe", f"{perf_unfiltered.get('sharpe_ratio', 0.0):.2f}", f"CAGR: {perf_unfiltered.get('cagr', 0.0)*100:.1f}%")
    h_col2.metric("USD/MYR Filtered Strategy Sharpe", f"{perf_filtered.get('sharpe_ratio', 0.0):.2f}", f"CAGR: {perf_filtered.get('cagr', 0.0)*100:.1f}%")

    fig_hyp = go.Figure()
    fig_hyp.add_trace(go.Scatter(x=res_unfiltered.index, y=res_unfiltered["NAV"], name="Unfiltered Momentum", line=dict(color="#8C9BAE", width=1.5)))
    fig_hyp.add_trace(go.Scatter(x=res_filtered.index, y=res_filtered["NAV"], name="USD/MYR Filtered Momentum", line=dict(color="#00D4B2", width=2.2)))
    fig_hyp.update_layout(get_terminal_plotly_layout(height=380, title="MACRO-GATED BURSA MALAYSIA STRATEGY EQUITY"))
    st.plotly_chart(fig_hyp, use_container_width=True)

render_disclaimer()

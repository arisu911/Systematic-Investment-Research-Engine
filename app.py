"""Global Router & Capital Allocation Terminal (`systematic-research-engine`).

Modern programmatic multi-page navigation router using st.navigation.
Hosts the global interactive Capital Allocation Controller, dynamic Multi-Currency FX Engine,
multi-model Strategy Dispatcher, and system health status.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yaml

from data.aligner import load_universe_registry, get_tradable_tickers
from data.loader import (
    get_cached_universe_prices,
    get_all_universe_tickers,
    fetch_universe_data,
    purge_cache,
    compute_lookback_dates,
    load_cache_metadata,
    LOOKBACK_HORIZONS,
    CACHE_TTL_POLICIES,
)
from data.fx_engine import SUPPORTED_CURRENCIES, get_currency_symbol, FXEngine
from research.strategies import STRATEGY_REGISTRY
from research.utils import (
    inject_metric_css,
    format_money,
    render_data_freshness_badge,
    render_backfill_warning_badge,
)
try:
    st.set_page_config(
        page_title="Multi-Market Universe Engine",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

# Inject global metric CSS to prevent ellipsis truncation across all pages
inject_metric_css()

# Initialize Global Session State Defaults
if "capital_amount" not in st.session_state:
    st.session_state["capital_amount"] = 100_000.0
if "selected_currency" not in st.session_state:
    st.session_state["selected_currency"] = "USD"
if "selected_strategy" not in st.session_state:
    st.session_state["selected_strategy"] = "max_sharpe"
if "hedged_toggle" not in st.session_state:
    st.session_state["hedged_toggle"] = False
if "risk_free_rate" not in st.session_state:
    st.session_state["risk_free_rate"] = 0.040
if "benchmark_ticker" not in st.session_state:
    st.session_state["benchmark_ticker"] = "^GSPC"
if "lookback_horizon" not in st.session_state:
    st.session_state["lookback_horizon"] = "5Y"
if "cache_ttl_label" not in st.session_state:
    st.session_state["cache_ttl_label"] = "4 Hours"
if "cache_ttl_seconds" not in st.session_state:
    st.session_state["cache_ttl_seconds"] = 14400
if "last_sync_time" not in st.session_state:
    _meta = load_cache_metadata()
    st.session_state["last_sync_time"] = _meta.get("last_refresh_timestamp", "N/A")
if "start_date" not in st.session_state or "end_date" not in st.session_state:
    s_date, e_date = compute_lookback_dates(st.session_state["lookback_horizon"])
    st.session_state["start_date"] = s_date
    st.session_state["end_date"] = e_date


def render_sidebar_controls():
    """Render global interactive Capital Allocation & FX Controller in sidebar."""
    st.sidebar.markdown("### 💼 Capital & FX Controller")

    # Base Currency Selection
    curr_idx = SUPPORTED_CURRENCIES.index(st.session_state["selected_currency"]) if st.session_state["selected_currency"] in SUPPORTED_CURRENCIES else 0
    selected_curr = st.sidebar.selectbox(
        "Reporting Base Currency",
        SUPPORTED_CURRENCIES + ["LOCAL"],
        index=curr_idx,
        help="Normalizes all foreign assets into this base denomination using dynamic FX triangulation.",
    )
    st.session_state["selected_currency"] = selected_curr
    curr_sym = get_currency_symbol(selected_curr)

    # Capital Input with Presets
    st.sidebar.markdown(f"**Portfolio Capital ({selected_curr})**")
    
    # Preset quick-fill buttons
    preset_cols = st.sidebar.columns(4)
    if preset_cols[0].button("10k", use_container_width=True):
        st.session_state["capital_amount"] = 10_000.0
    if preset_cols[1].button("50k", use_container_width=True):
        st.session_state["capital_amount"] = 50_000.0
    if preset_cols[2].button("100k", use_container_width=True):
        st.session_state["capital_amount"] = 100_000.0
    if preset_cols[3].button("500k", use_container_width=True):
        st.session_state["capital_amount"] = 500_000.0

    preset_cols2 = st.sidebar.columns(3)
    if preset_cols2[0].button("1M", use_container_width=True):
        st.session_state["capital_amount"] = 1_000_000.0
    if preset_cols2[1].button("5M", use_container_width=True):
        st.session_state["capital_amount"] = 5_000_000.0
    if preset_cols2[2].button("10M", use_container_width=True):
        st.session_state["capital_amount"] = 10_000_000.0

    capital_val = st.sidebar.number_input(
        "Nominal Cash Allocation",
        min_value=100.0,
        max_value=1_000_000_000.0,
        value=float(st.session_state["capital_amount"]),
        step=10_000.0,
        format="%.2f",
        label_visibility="collapsed",
    )
    st.session_state["capital_amount"] = capital_val

    # Strategy Selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Strategy Dispatcher")
    strat_keys = list(STRATEGY_REGISTRY.keys())
    strat_names = list(STRATEGY_REGISTRY.values())
    cur_strat_idx = strat_keys.index(st.session_state["selected_strategy"]) if st.session_state["selected_strategy"] in strat_keys else 0

    selected_strat_name = st.sidebar.selectbox(
        "Active Strategy Model",
        strat_names,
        index=cur_strat_idx,
        help="Algorithm used to compute optimal target allocation weights.",
    )
    st.session_state["selected_strategy"] = strat_keys[strat_names.index(selected_strat_name)]

    # Hedged vs Unhedged returns toggle
    is_hedged = st.sidebar.toggle(
        "Hedge Currency Exposure",
        value=st.session_state["hedged_toggle"],
        help="If enabled, isolates local asset returns, ignoring FX volatility drag.",
    )
    st.session_state["hedged_toggle"] = is_hedged

    # Global Benchmark & Risk-Free Rate
    benchmarks = ["^GSPC", "^KLSE", "^N225", "^NDX", "^RUT", "^TOPX"]
    cur_bench_idx = benchmarks.index(st.session_state["benchmark_ticker"]) if st.session_state["benchmark_ticker"] in benchmarks else 0
    st.session_state["benchmark_ticker"] = st.sidebar.selectbox("Global Benchmark", benchmarks, index=cur_bench_idx)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⏱️ Data Freshness & Lookback")
    lookback_opts = ["2Y", "3Y", "4Y", "5Y", "10Y"]
    cur_lb = st.session_state.get("lookback_horizon", "5Y")
    sb_lookback = st.sidebar.selectbox(
        "Historical Horizon",
        lookback_opts,
        index=lookback_opts.index(cur_lb) if cur_lb in lookback_opts else 3,
        key="sb_lookback_select",
    )
    if sb_lookback != cur_lb:
        st.session_state["lookback_horizon"] = sb_lookback
        s_date, e_date = compute_lookback_dates(sb_lookback)
        st.session_state["start_date"] = s_date
        st.session_state["end_date"] = e_date
        st.rerun()

    ttl_opts = {"1 Hour": 3600, "4 Hours": 14400, "Daily (24h)": 86400}
    cur_ttl = st.session_state.get("cache_ttl_label", "4 Hours")
    sb_ttl = st.sidebar.selectbox(
        "Cache Policy",
        list(ttl_opts.keys()),
        index=list(ttl_opts.keys()).index(cur_ttl) if cur_ttl in ttl_opts else 1,
        key="sb_ttl_select",
    )
    st.session_state["cache_ttl_label"] = sb_ttl
    st.session_state["cache_ttl_seconds"] = ttl_opts[sb_ttl]

    if st.sidebar.button("↻ Hard Refresh Data", use_container_width=True, help="Purge all local cache files and re-download fresh market data."):
        with st.spinner("Purging cache and fetching fresh market data..."):
            purge_cache()
            st.cache_data.clear()
            st.session_state["force_reload"] = True
            st.rerun()


def render_overview():
    """Render Terminal Status, Universe Registry, and Live FX Matrix."""
    render_sidebar_controls()

    # --- Data Refresh & Lookback Control Bar ---
    with st.container():
        c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.2, 1.0, 1.4])
        
        with c1:
            lookback_options = ["2Y", "3Y", "4Y", "5Y", "10Y"]
            current_lookback = st.session_state.get("lookback_horizon", "5Y")
            cur_lb_idx = lookback_options.index(current_lookback) if current_lookback in lookback_options else 3
            selected_lookback = c1.selectbox(
                "Historical Horizon", 
                lookback_options, 
                index=cur_lb_idx,
                key="lookback_select"
            )
            if selected_lookback != current_lookback:
                st.session_state["lookback_horizon"] = selected_lookback
                s_date, e_date = compute_lookback_dates(selected_lookback)
                st.session_state["start_date"] = s_date
                st.session_state["end_date"] = e_date
                st.rerun()

        with c2:
            ttl_options = {"1 Hour": 3600, "4 Hours": 14400, "Daily (24h)": 86400}
            current_ttl_label = st.session_state.get("cache_ttl_label", "4 Hours")
            cur_ttl_idx = list(ttl_options.keys()).index(current_ttl_label) if current_ttl_label in ttl_options else 1
            selected_ttl_label = c2.selectbox(
                "Cache Policy", 
                list(ttl_options.keys()), 
                index=cur_ttl_idx,
                key="ttl_select"
            )
            st.session_state["cache_ttl_label"] = selected_ttl_label
            st.session_state["cache_ttl_seconds"] = ttl_options[selected_ttl_label]

        with c3:
            c3.write("") # Spacer
            c3.write("")
            if c3.button("↻ Hard Refresh", type="primary", use_container_width=True, help="Purge all local cache files and re-download fresh market data."):
                with st.spinner("Purging cache and fetching fresh market data..."):
                    purge_cache()
                    st.cache_data.clear()
                    st.session_state["force_reload"] = True
                    st.rerun()

        with c4:
            c4.write("")
            c4.write("")
            meta = load_cache_metadata()
            last_sync = meta.get("last_refresh_timestamp", st.session_state.get("last_sync_time", "N/A"))
            st.session_state["last_sync_time"] = last_sync
            c4.caption(f"Last Sync:\n**{last_sync}**")

        with c5:
            active_curr = st.session_state.get("selected_currency", "USD")
            active_cap = st.session_state.get("capital_amount", 100000.0)
            c5.metric(label=f"Active Portfolio ({active_curr})", value=f"{active_cap:,.0f}")

    meta = load_cache_metadata()
    render_data_freshness_badge(meta)
    render_backfill_warning_badge(meta)

    base_curr = st.session_state["selected_currency"]
    capital = st.session_state["capital_amount"]
    curr_sym = get_currency_symbol(base_curr)
    strategy_key = st.session_state["selected_strategy"]
    strategy_name = STRATEGY_REGISTRY.get(strategy_key, strategy_key)

    registry = load_universe_registry()
    tradable_symbols = get_tradable_tickers(registry)

    # Load Universe Prices
    prices_df, is_demo = get_cached_universe_prices(
        start_date=st.session_state["start_date"],
        end_date=st.session_state["end_date"],
        base_currency=base_curr,
        lookback_horizon=st.session_state["lookback_horizon"],
        ttl_seconds=st.session_state.get("cache_ttl_seconds", 14400),
    )

    # Top Status Banner
    demo_badge = (
        "<span style='background:#f59e0b;color:#000;font-size:11px;font-weight:700;padding:2px 8px;border-radius:3px;'>OFFLINE DEMO MODE</span>"
        if is_demo
        else "<span style='background:#00c805;color:#000;font-size:11px;font-weight:700;padding:2px 8px;border-radius:3px;'>LIVE FINANCIAL STREAM</span>"
    )

    st.markdown(
        f"""
        <div style="background-color: #131722; border: 1px solid #2a2e39; border-left: 5px solid #00c805;
                    padding: 16px 20px; border-radius: 4px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 11px; font-weight: 800; color: #00c805; letter-spacing: 1.5px; text-transform: uppercase;">
                        CAPITAL ALLOCATION & NOMINAL RISK WORKSTATION • FACTSET/BLOOMBERG PORT
                    </span>
                    <h2 style="margin: 4px 0 0 0; color: #ffffff; font-size: 24px; font-weight: 700; letter-spacing: -0.5px;">
                        Global 25-Instrument Systematic Research Terminal
                    </h2>
                </div>
                <div style="text-align: right;">
                    {demo_badge}
                    <div style="color: #848e9c; font-size: 11px; font-family: monospace; margin-top: 4px;">
                        CAPITAL: {curr_sym}{capital:,.2f} • BASE: {base_curr} • STRATEGY: {strategy_name}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Real-Time Key Indicators
    st.markdown("#### 🌐 Global Macro, Volatility & FX Matrix")
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    def get_val(sym):
        return float(prices_df[sym].iloc[-1]) if sym in prices_df.columns else 0.0

    def get_chg(sym):
        if sym in prices_df.columns and len(prices_df) >= 2:
            return float((prices_df[sym].iloc[-1] / prices_df[sym].iloc[-2] - 1.0) * 100.0)
        return 0.0

    m1.metric("VIX Volatility", f"{get_val('^VIX'):.2f}", f"{get_chg('^VIX'):+.2f}%", delta_color="inverse")
    m2.metric("US 10Y Yield", f"{get_val('^TNX'):.2f}%", f"{get_chg('^TNX'):+.2f}%", delta_color="inverse")
    m3.metric("USD/MYR", f"{get_val('USDMYR=X'):.4f}", f"{get_chg('USDMYR=X'):+.2f}%")
    m4.metric("USD/JPY", f"{get_val('JPY=X'):.2f}", f"{get_chg('JPY=X'):+.2f}%")
    m5.metric("Gold (GC=F)", f"{curr_sym}{get_val('GC=F'):,.1f}", f"{get_chg('GC=F'):+.2f}%")
    m6.metric("Brent Crude (BZ=F)", f"{curr_sym}{get_val('BZ=F'):.2f}", f"{get_chg('BZ=F'):+.2f}%")

    st.markdown("---")

    # 25-Instrument Registry & Valuation Table
    st.markdown(f"#### 📋 25-Instrument Universe Registry & Nominal Sizing (Base: {base_curr})")
    
    reg_records = []
    for ticker, meta in registry.items():
        has_data = ticker in prices_df.columns
        last_price = prices_df[ticker].iloc[-1] if has_data else 0.0
        tot_bars = len(prices_df[ticker].dropna()) if has_data else 0
        is_tradable = ticker in tradable_symbols
        
        reg_records.append({
            "Ticker": ticker,
            "Name": meta.get("name"),
            "Region": meta.get("region"),
            "Local Currency": meta.get("local_currency"),
            "Role": meta.get("role"),
            "Asset Class": meta.get("asset_class"),
            "Tradable": "✅ Yes" if is_tradable else "ℹ️ Overlay/Macro",
            f"Price ({base_curr})": f"{curr_sym}{last_price:,.2f}",
            "Coverage Bars": tot_bars,
            "Health": "🟢 Online" if tot_bars > 50 else "🔴 Unavailable",
        })

    st.dataframe(pd.DataFrame(reg_records), use_container_width=True, hide_index=True, height=360)

    # Analytical Subsystems Grid
    st.markdown("#### 🧭 Analytical Subsystem Workstations")
    nav1, nav2, nav3, nav4 = st.columns(4)
    with nav1:
        st.markdown(
            f"""
            <div style="background:#1a1c24;padding:16px;border-radius:6px;border-top:3px solid #00c805;">
                <h4 style="margin:0 0 8px 0;color:#fff;">1. Executive Summary</h4>
                <p style="color:#848e9c;font-size:12px;margin:0;">
                    Tearsheet KPIs, donut allocation, and lot-sized <b>Order Execution Ticket</b> for {curr_sym}{capital:,.0f}.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with nav2:
        st.markdown(
            """
            <div style="background:#1a1c24;padding:16px;border-radius:6px;border-top:3px solid #3b82f6;">
                <h4 style="margin:0 0 8px 0;color:#fff;">2. Factor Research</h4>
                <p style="color:#848e9c;font-size:12px;margin:0;">
                    25x25 correlation matrix, rolling betas vs global benchmarks, and lead-lag analysis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with nav3:
        st.markdown(
            """
            <div style="background:#1a1c24;padding:16px;border-radius:6px;border-top:3px solid #ec4899;">
                <h4 style="margin:0 0 8px 0;color:#fff;">3. Strategy Backtest</h4>
                <p style="color:#848e9c;font-size:12px;margin:0;">
                    Multi-model backtests (MVO, Risk Parity, Momentum, 1/N), walk-forward validation, and drawdowns.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with nav4:
        st.markdown(
            f"""
            <div style="background:#1a1c24;padding:16px;border-radius:6px;border-top:3px solid #f59e0b;">
                <h4 style="margin:0 0 8px 0;color:#fff;">4. Risk Engine</h4>
                <p style="color:#848e9c;font-size:12px;margin:0;">
                    <b>Nominal Cash-at-Risk</b> (1D, 5D, 21D), Component VaR cash attribution, and macro shock simulator.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Programmatic Navigation Configuration
pg = st.navigation(
    {
        "System": [
            st.Page(render_overview, title="Capital Overview", icon=":material/dashboard:", default=True)
        ],
        "Quantitative Analytics": [
            st.Page("pages/1_Executive_Summary.py", title="Executive Summary & Orders", icon=":material/analytics:"),
            st.Page("pages/2_Factor_Research.py", title="Factor Research", icon=":material/hub:"),
            st.Page("pages/3_Strategy_Backtest.py", title="Strategy Backtest", icon=":material/timeline:"),
            st.Page("pages/4_Risk_Engine.py", title="Nominal Risk Engine", icon=":material/shield:"),
        ],
    }
)

pg.run()

"""Global Router & System Overview Terminal (`systematic-research-engine`).

Modern programmatic multi-page navigation router using st.navigation.
Displays system health, universe registry audit across 25 instruments,
and configuration cache controls.
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
from data.loader import get_cached_universe_prices, get_all_universe_tickers

# Global Terminal Page Configuration
try:
    st.set_page_config(
        page_title="Multi-Market Universe Engine",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass


def render_overview():
    """Render Terminal Status and Universe Health Check."""
    registry = load_universe_registry()
    tradable_symbols = get_tradable_tickers(registry)

    # Sidebar global parameters
    st.sidebar.markdown("### ⚙️ Engine Environment")
    base_curr = st.sidebar.selectbox(
        "Base Currency Alignment",
        ["USD", "MYR", "LOCAL"],
        index=0,
        help="Normalizes multi-market assets into a common denomination.",
    )
    st.session_state["base_currency"] = base_curr

    col_s1, col_s2 = st.sidebar.columns(2)
    start_d = col_s1.date_input("Start Date", datetime(2021, 1, 1))
    end_d = col_s2.date_input("End Date", datetime(2024, 12, 31))

    st.session_state["start_date"] = str(start_d)
    st.session_state["end_date"] = str(end_d)

    if st.sidebar.button("🧹 Invalidate Data Cache", use_container_width=True):
        st.cache_data.clear()
        st.sidebar.success("Parquet cache invalidated!")

    # Load Universe
    prices_df, is_demo = get_cached_universe_prices(
        start_date=str(start_d),
        end_date=str(end_d),
        base_currency=base_curr,
    )

    # Header Banner
    demo_badge = (
        "<span style='background:#f59e0b;color:#000;font-size:11px;font-weight:700;padding:2px 8px;border-radius:3px;'>OFFLINE DEMO MODE</span>"
        if is_demo
        else "<span style='background:#00c805;color:#000;font-size:11px;font-weight:700;padding:2px 8px;border-radius:3px;'>LIVE YFINANCE STREAM</span>"
    )

    st.markdown(
        f"""
        <div style="background-color: #131722; border: 1px solid #2a2e39; border-left: 5px solid #00c805;
                    padding: 16px 20px; border-radius: 4px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 11px; font-weight: 800; color: #00c805; letter-spacing: 1.5px; text-transform: uppercase;">
                        INSTITUTIONAL QUANTITATIVE WORKSTATION • MULTI-MARKET UNIVERSE ENGINE
                    </span>
                    <h2 style="margin: 4px 0 0 0; color: #ffffff; font-size: 24px; font-weight: 700; letter-spacing: -0.5px;">
                        Global 25-Instrument Systematic Research Terminal
                    </h2>
                </div>
                <div style="text-align: right;">
                    {demo_badge}
                    <div style="color: #848e9c; font-size: 11px; font-family: monospace; margin-top: 4px;">
                        BASE: {base_curr} • 25 ASSETS MONITORED
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Key Macro Gauges
    st.markdown("#### 🌐 Real-Time Macro & Volatility Indicators")
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    def get_latest_val(sym):
        if sym in prices_df.columns:
            return float(prices_df[sym].iloc[-1])
        return 0.0

    def get_latest_pct(sym):
        if sym in prices_df.columns and len(prices_df) >= 2:
            return float((prices_df[sym].iloc[-1] / prices_df[sym].iloc[-2] - 1.0) * 100.0)
        return 0.0

    vix_val = get_latest_val("^VIX")
    vix_chg = get_latest_pct("^VIX")
    m1.metric("VIX Volatility", f"{vix_val:.2f}", f"{vix_chg:+.2f}%", delta_color="inverse")

    tnx_val = get_latest_val("^TNX")
    tnx_chg = get_latest_pct("^TNX")
    m2.metric("US 10Y Yield", f"{tnx_val:.2f}%", f"{tnx_chg:+.2f}%", delta_color="inverse")

    myr_val = get_latest_val("USDMYR=X")
    myr_chg = get_latest_pct("USDMYR=X")
    m3.metric("USD/MYR", f"{myr_val:.4f}", f"{myr_chg:+.2f}%")

    jpy_val = get_latest_val("JPY=X")
    jpy_chg = get_latest_pct("JPY=X")
    m4.metric("USD/JPY", f"{jpy_val:.2f}", f"{jpy_chg:+.2f}%")

    gold_val = get_latest_val("GC=F")
    gold_chg = get_latest_pct("GC=F")
    m5.metric("Gold (GC=F)", f"${gold_val:,.1f}", f"{gold_chg:+.2f}%")

    brent_val = get_latest_val("BZ=F")
    brent_chg = get_latest_pct("BZ=F")
    m6.metric("Brent Crude (BZ=F)", f"${brent_val:.2f}", f"{brent_chg:+.2f}%")

    st.markdown("---")

    # 25-Instrument Registry Audit
    st.markdown("#### 📋 25-Instrument Universe Registry & Data Health")
    c_reg1, c_reg2, c_reg3, c_reg4 = st.columns(4)
    c_reg1.metric("Total Instruments", "25", "4 Regional Buckets")
    c_reg2.metric("Tradable Universe", f"{len(tradable_symbols)}", "Equities, ETFs, Commodities")
    c_reg3.metric("Benchmark Overlays", "6", "KLCI, S&P, NDX, RUT, N225, TOPX")
    c_reg4.metric("Macro / Rates / Vol", "5", "VIX, TNX, DXY, FX")

    reg_records = []
    for ticker, meta in registry.items():
        has_data = ticker in prices_df.columns
        last_price = prices_df[ticker].iloc[-1] if has_data else 0.0
        tot_bars = len(prices_df[ticker].dropna()) if has_data else 0
        reg_records.append({
            "Ticker": ticker,
            "Name": meta.get("name"),
            "Region": meta.get("region"),
            "Local Currency": meta.get("local_currency"),
            "Role": meta.get("role"),
            "Asset Class": meta.get("asset_class"),
            "Tradable": "✅ Yes" if ticker in tradable_symbols else "ℹ️ Benchmark/Macro",
            f"Latest Price ({base_curr})": f"{last_price:,.2f}",
            "Coverage Bars": tot_bars,
            "Health": "🟢 Healthy" if tot_bars > 50 else "🔴 Missing",
        })

    reg_df = pd.DataFrame(reg_records)
    st.dataframe(
        reg_df,
        use_container_width=True,
        hide_index=True,
        height=380,
    )

    # Quick Navigation Summary
    st.markdown("#### 🧭 Analytical Subsystem Workstations")
    nav1, nav2, nav3, nav4 = st.columns(4)
    with nav1:
        st.markdown(
            """
            <div style="background:#1a1c24;padding:16px;border-radius:6px;border-top:3px solid #00c805;">
                <h4 style="margin:0 0 8px 0;color:#fff;">1. Executive Summary</h4>
                <p style="color:#848e9c;font-size:12px;margin:0;">
                    Consolidated KPI tearsheet, Sharpe, Sortino, Calmar, and asset allocation donut chart.
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
                    25x25 correlation matrix heatmap, rolling betas against KLCI, S&P 500, Nikkei, and lead-lag analysis.
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
                    In-sample vs out-of-sample walk-forward validation, rebalancing friction, and underwater drawdowns.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with nav4:
        st.markdown(
            """
            <div style="background:#1a1c24;padding:16px;border-radius:6px;border-top:3px solid #f59e0b;">
                <h4 style="margin:0 0 8px 0;color:#fff;">4. Risk Engine</h4>
                <p style="color:#848e9c;font-size:12px;margin:0;">
                    Non-Gaussian Cornish-Fisher mVaR/mCVaR, fat-tail QQ plots, and macro shock simulations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Configure Navigation Pages
pg = st.navigation(
    {
        "Overview": [
            st.Page(render_overview, title="Terminal Status", icon=":material/speed:", default=True)
        ],
        "Quantitative Analytics": [
            st.Page("pages/1_Executive_Summary.py", title="Executive Summary", icon=":material/analytics:"),
            st.Page("pages/2_Factor_Research.py", title="Factor Research", icon=":material/hub:"),
            st.Page("pages/3_Strategy_Backtest.py", title="Strategy Backtest", icon=":material/timeline:"),
            st.Page("pages/4_Risk_Engine.py", title="Risk Engine", icon=":material/shield:"),
        ],
    }
)

pg.run()

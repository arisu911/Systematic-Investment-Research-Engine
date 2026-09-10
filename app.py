"""Systematic Research Engine - Terminal Home & Executive Summary.

Automated Cross-Market Quantitative Research Exploration Terminal.
Streamlit serves exclusively as an exploration interface over precomputed research results.
"""

import sys
from pathlib import Path
from datetime import datetime

# Ensure src/ directory is on sys.path
_SRC_PATH = str(Path(__file__).resolve().parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

import streamlit as st
import pandas as pd
from quant_engine.research.store import ResearchStore
from quant_engine.ui_helpers import apply_terminal_theme, render_terminal_header, render_disclaimer

st.set_page_config(
    page_title="Systematic Research Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_terminal_theme()


@st.cache_data(ttl=60)
def load_overview_data():
    """Query persistent research database for terminal headline statistics."""
    store = ResearchStore()
    latest_run = store.get_latest_run()
    results_df = store.get_strategy_results()
    findings = store.get_all_findings()
    data_status_df = store.get_data_status()
    return latest_run, results_df, findings, data_status_df


latest_run, results_df, findings, data_status_df = load_overview_data()

# Global Terminal Header
is_demo = False
if not data_status_df.empty:
    is_demo = bool(data_status_df["is_demo"].any())

render_terminal_header(
    title="QUANTITATIVE RESEARCH TERMINAL",
    subtitle="Cross-Market Systematic Quantitative Research Platform • Exploration Engine",
    is_demo=is_demo,
)

# Sidebar System Metadata
st.sidebar.markdown("### 🏛️ Research Engine State")
if latest_run:
    st.sidebar.info(
        f"**Run ID:** `{latest_run['run_id']}`\n\n"
        f"**Completed:** {latest_run['timestamp'][:19]}\n\n"
        f"**Protocol:** v{latest_run['protocol_version']}\n\n"
        f"**Engine:** v{latest_run['engine_version']}\n\n"
        f"**Status:** `{latest_run['status']}`"
    )
else:
    st.sidebar.warning("No precomputed research runs found in database. Execute `python scripts/run_research.py`.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Architecture Mode:**  
    `PRECOMPUTED EXPLORATION`  
    *Zero client-side computation.*  
    *All metrics persisted in SQLite.*
    """
)

# Top Metrics Row
if not results_df.empty and latest_run:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Precomputed Experiments", f"{len(results_df)}", f"{latest_run.get('total_markets', 4)} Markets")
    
    robust_count = len(results_df[results_df["walk_forward_efficiency"] >= 0.60])
    robust_pct = (robust_count / len(results_df)) * 100.0 if len(results_df) > 0 else 0.0
    col2.metric("OOS Robust Strategies", f"{robust_count}", f"{robust_pct:.1f}% Pass Rate")

    avg_sharpe = results_df["sharpe_ratio"].median()
    col3.metric("Median Net Sharpe", f"{avg_sharpe:.2f}", "Post-Friction")

    health_avg = data_status_df["health_score"].mean() if not data_status_df.empty else 100.0
    col4.metric("Data Quality Score", f"{health_avg:.1f}%", f"{len(data_status_df)} Assets Validated")

    protocol_ver = latest_run.get("protocol_version", "1.0.0")
    col5.metric("Research Protocol", f"v{protocol_ver}", "Deterministic & Locked")
else:
    st.warning("Database empty. Please run `python scripts/run_research.py --force-demo` in terminal to populate.")

# Section: Empirical Research Findings Ticker
st.markdown('<div class="section-header">NOTABLE EMPIRICAL DISCOVERIES</div>', unsafe_allow_html=True)
st.markdown("*Key empirical insights distilled autonomously from precomputed research records:*")

if findings:
    cols = st.columns(min(len(findings), 3))
    for i, finding in enumerate(findings[:3]):
        with cols[i % len(cols)]:
            severity = finding.get("severity", "INFO")
            badge_color = (
                "#00e676" if severity == "SUCCESS"
                else "#ffab00" if severity == "WARNING"
                else "#ff1744" if severity == "CAUTION"
                else "#00e5ff"
            )
            st.markdown(
                f"""
                <div style="background-color: #121926; border: 1px solid #1e293b; border-left: 4px solid {badge_color};
                            padding: 14px; border-radius: 6px; margin-bottom: 12px; height: 175px;">
                    <div style="font-size: 11px; color: {badge_color}; font-weight: 700; text-transform: uppercase;">
                        {finding.get('category', 'Finding')}
                    </div>
                    <div style="font-size: 14px; font-weight: 600; color: #f8fafc; margin: 4px 0 8px 0;">
                        {finding.get('headline', '')}
                    </div>
                    <div style="font-size: 12px; color: #94a3b8; line-height: 1.4;">
                        {finding.get('narrative', '')[:140]}...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.info("No empirical findings cataloged yet.")

# Section: Top Out-of-Sample Strategies League Table
st.markdown('<div class="section-header">TOP GENERALIZING STRATEGIES (OUT-OF-SAMPLE LEAGUE TABLE)</div>', unsafe_allow_html=True)

if not results_df.empty:
    # Top 10 sorted by OOS Sharpe & Walk-Forward Efficiency
    top_df = results_df.sort_values(["oos_sharpe", "walk_forward_efficiency"], ascending=[False, False]).head(8)
    
    display_cols = [
        "market", "symbol", "strategy_name", "sharpe_ratio", "oos_sharpe",
        "walk_forward_efficiency", "cagr", "max_drawdown", "annualized_turnover", "friction_survival_ratio"
    ]
    formatted_df = top_df[display_cols].copy()
    formatted_df.rename(
        columns={
            "market": "Market",
            "symbol": "Asset",
            "strategy_name": "Strategy",
            "sharpe_ratio": "IS Sharpe",
            "oos_sharpe": "OOS Sharpe",
            "walk_forward_efficiency": "WFE Score",
            "cagr": "Net CAGR",
            "max_drawdown": "Max DD",
            "annualized_turnover": "Turnover",
            "friction_survival_ratio": "Friction Survival",
        },
        inplace=True,
    )
    st.dataframe(
        formatted_df.style.format({
            "IS Sharpe": "{:.2f}",
            "OOS Sharpe": "{:.2f}",
            "WFE Score": "{:.2f}",
            "Net CAGR": "{:.1%}",
            "Max DD": "{:.1%}",
            "Turnover": "{:.1f}x",
            "Friction Survival": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("🔍 Explore individual strategies in depth via the **Strategy Explorer** and **Asset Deep Dive** pages.")

# Section: System Architecture & Workflow
st.markdown('<div class="section-header">AUTOMATED PIPELINE ARCHITECTURE</div>', unsafe_allow_html=True)

st.code(
    """
    1. HEADLESS INGESTION        python scripts/update_data.py
       Acquires market data (Bursa Malaysia, US, Japan, Europe, FX, Commodities) and asserts bar integrity.

    2. AUTOMATED RESEARCH SWEEP  python scripts/run_research.py
       Executes locked research protocol: Strategy generation, execution accounting with statutory fees,
       walk-forward train/test rolling splits, parameter sensitivity, and regime breakdowns.

    3. PERSISTENT STORAGE        results/database/research.db
       Atomic SQLite database housing experiment metadata, metrics, equity curves, and findings.

    4. EXPLORATION TERMINAL      streamlit run app.py
       Instantaneous query and comparison terminal. Zero client-side computation overhead.
    """,
    language="text",
)

render_disclaimer()

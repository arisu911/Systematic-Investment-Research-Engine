"""Systematic Research Engine - Main Terminal Dashboard.

A modular, extensible, reproducible cross-market systematic quantitative research framework.
"""

import sys
from pathlib import Path

# Ensure src/ directory is on sys.path
_SRC_PATH = str(Path(__file__).resolve().parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

import streamlit as st
import pandas as pd
from quant_engine.config.settings import get_settings
from quant_engine.config.markets import load_market_config, load_global_config
from quant_engine.registry.experiments import ExperimentRegistry
from quant_engine.ui_helpers import apply_terminal_theme, render_terminal_header, render_disclaimer

st.set_page_config(
    page_title="Systematic Research Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_terminal_theme()

# Global Sidebar Controls
st.sidebar.markdown("### ⚙️ Engine Control")
force_demo = st.sidebar.checkbox("Force Demo / Offline Mode", value=False, help="Use deterministic synthetic/cached data for guaranteed reliability.")
st.session_state["force_demo"] = force_demo

render_terminal_header(
    title="SYSTEMATIC RESEARCH ENGINE",
    subtitle="Cross-Market Quantitative Research Laboratory & Statistical Validation Terminal",
    is_demo=force_demo,
)

# Executive Overview Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("First-Class Universes", "4 Markets", "MY, US, JP, EU")
col2.metric("Core Strategies", "5 Families", "Momentum, Rev, Breakout, Factor")
col3.metric("Validation Modes", "Walk-Forward & MC", "IS vs OOS Decoupled")

registry = ExperimentRegistry()
all_exps = registry.list_all()
col4.metric("Logged Experiments", f"{len(all_exps)} Runs", "Reproducible Ledger")

st.markdown('<div class="section-header">CORE RESEARCH CAPABILITIES</div>', unsafe_allow_html=True)

st.markdown(
    """
    This laboratory is designed to systematically investigate empirical market anomalies and answer core quant research questions:
    * **Cross-Market Universality**: Does momentum or mean reversion work in Bursa Malaysia, and does the exact same strategy logic translate to the US, Japan, or Europe?
    * **Execution Friction Reality**: Does a strategy survive real-world transaction costs (brokerage, clearing, stamp duties) and market impact slippage?
    * **Look-Ahead Prevention**: Strict chronological bar separation ensures signals at bar $t$ are only executed on bar $t+1$ with explicit logging.
    * **Walk-Forward Validation**: Anchored or rolling walk-forward train/test splits ensure out-of-sample statistical integrity.
    * **Overfitting Auditing**: Real-time heuristic and statistical diagnostics flag parameter fragility and performance decay.
    """
)

# Market Universes Grid
st.markdown('<div class="section-header">SUPPORTED MARKET UNIVERSES</div>', unsafe_allow_html=True)

markets_info = [
    {
        "Market": "🇲🇾 Malaysia (Bursa)",
        "Benchmark": "^KLSE (FBM KLCI)",
        "Equities": "Maybank (1155.KL), Public Bank (1295.KL), Tenaga (5347.KL), CIMB (1023.KL), YTL Power",
        "Cost Structure": "10 bps Brokerage + 3 bps Clearing + 10 bps Stamp Duty (Cap RM 1000) + Min RM 8",
        "FX / Macro": "USD/MYR (MYR=X), Brent Crude (BZ=F), Gold (GC=F)",
    },
    {
        "Market": "🇺🇸 United States",
        "Benchmark": "SPY (S&P 500)",
        "Equities": "QQQ, IWM, Apple (AAPL), Microsoft (MSFT), Nvidia (NVDA), JPMorgan (JPM)",
        "Cost Structure": "2 bps Brokerage + 0.5 bps Regulatory + Min $1.00",
        "FX / Macro": "CBOE VIX (^VIX), 10Y Yield (^TNX), US Dollar Index",
    },
    {
        "Market": "🇯🇵 Japan",
        "Benchmark": "^N225 (Nikkei 225)",
        "Equities": "Toyota (7203.T), Sony (6758.T), SoftBank (9984.T), MUFG (8306.T)",
        "Cost Structure": "5 bps Brokerage + 1 bp Clearing + Min ¥50",
        "FX / Macro": "USD/JPY (JPY=X), MSCI Japan (EWJ)",
    },
    {
        "Market": "🇪🇺 Europe",
        "Benchmark": "^GDAXI (DAX)",
        "Equities": "FTSE 100 (^FTSE), Euro Stoxx 50 (^STOXX50E), SAP (SAP.DE), LVMH (MC.PA)",
        "Cost Structure": "6 bps Brokerage + 1.5 bps Clearing + Min €3.00",
        "FX / Macro": "EUR/USD (EURUSD=X)",
    },
]

st.dataframe(pd.DataFrame(markets_info), use_container_width=True, hide_index=True)

st.markdown('<div class="section-header">SYSTEM ARCHITECTURE PIPELINE</div>', unsafe_allow_html=True)

st.code(
    """
    DATA PROVIDERS (Yahoo, FRED, CSV, Demo Fallback)
          │
          ▼
    DATA CLEANING & VALIDATION (Calendar Alignment, Monotonic Checks, No-Leakage Asserts)
          │
          ▼
    FEATURE & SIGNAL ENGINE (SMA/EMA, Momentum, Volatility, Z-Scores, Donchian Channels)
          │
          ▼
    EXECUTION TIMELINE (Bar t Close Signal -> Bar t+1 Open/Close Fill with 1-Bar Lag)
          │
          ▼
    TRANSACTION COST & SLIPPAGE (Commissions, Stamp Duty, Bid-Ask Spread, Market Impact)
          │
          ▼
    PORTFOLIO CONSTRAINTS & SIZING (Equal Weight, Volatility Targeting, Leverage Limits)
          │
          ▼
    ANALYTICS & VALIDATION (CAGR, Sharpe, Drawdowns, Walk-Forward OOS, Monte Carlo, Regimes)
          │
          ▼
    PERSISTENT EXPERIMENT REGISTRY (Reproducible Research Artifacts)
    """,
    language="text",
)

render_disclaimer()

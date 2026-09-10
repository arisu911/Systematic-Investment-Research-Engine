"""Systematic Research Engine - Market Explorer.

Precomputed quantitative research terminal filtered by regional financial markets.
Zero on-the-fly backtesting; all metrics are loaded instantaneously from SQLite.
"""

import sys
from pathlib import Path

# Add src to sys.path
_SRC_PATH = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

import streamlit as st
import pandas as pd
import plotly.express as px
from quant_engine.research.store import ResearchStore
from quant_engine.config.markets import load_market_config
from quant_engine.ui_helpers import apply_terminal_theme, render_terminal_header, render_disclaimer

st.set_page_config(page_title="Market Explorer", page_icon="🌐", layout="wide")
apply_terminal_theme()

render_terminal_header(
    title="MARKET RESEARCH EXPLORER",
    subtitle="Precomputed systematic performance, friction survival, and alpha across regional financial markets",
)

store = ResearchStore()
markets = store.get_distinct_markets()

if not markets:
    st.warning("No market research data found in database. Run `python scripts/run_research.py` to populate.")
    st.stop()

# Friendly market display names
market_labels = {
    "MY": "🇲🇾 Malaysia (Bursa Malaysia)",
    "US": "🇺🇸 United States (S&P / Nasdaq)",
    "JP": "🇯🇵 Japan (Tokyo Stock Exchange)",
    "EU": "🇪🇺 Europe (DAX / STOXX)",
}

col1, col2 = st.columns([2, 2])
with col1:
    selected_m_code = st.selectbox(
        "Select Market Universe",
        markets,
        format_func=lambda x: market_labels.get(x, x),
        index=0 if "MY" not in markets else markets.index("MY"),
    )

with col2:
    strat_types = ["ALL"] + sorted(list(store.get_strategy_results(market=selected_m_code)["strategy_type"].unique()))
    selected_type = st.selectbox("Filter Strategy Family", strat_types, index=0)

# Load precomputed data
results_df = store.get_strategy_results(market=selected_m_code, strategy_type=selected_type)

# Market Institutional Configuration Card
m_cfg = load_market_config(selected_m_code)
c_box = st.container()
with c_box:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Benchmark", m_cfg.benchmark_symbol, f"{m_cfg.trading_days_per_year} Trading Days/Yr")
    c2.metric("Currency", m_cfg.currency, "Local Settlement")
    c3.metric(
        "Brokerage & Clearing",
        f"{m_cfg.costs.brokerage_bps + m_cfg.costs.clearing_fee_bps:.1f} bps",
        f"Min {m_cfg.costs.minimum_commission:.1f} {m_cfg.currency}",
    )
    stamp_info = f"{m_cfg.costs.stamp_duty_bps:.0f} bps" if m_cfg.costs.stamp_duty_bps > 0 else "0.0 bps (None)"
    if m_cfg.costs.stamp_duty_cap:
        stamp_info += f" (Cap {m_cfg.costs.stamp_duty_cap:.0f})"
    c4.metric("Stamp Duty", stamp_info, "Statutory Friction")

st.markdown('<div class="section-header">MARKET RESEARCH SUMMARY</div>', unsafe_allow_html=True)

if not results_df.empty:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Evaluated Strategies", f"{len(results_df)}", f"{len(results_df['symbol'].unique())} Assets")
    
    avg_is_sharpe = results_df["sharpe_ratio"].mean()
    m2.metric("Mean In-Sample Sharpe", f"{avg_is_sharpe:.2f}")

    avg_oos_sharpe = results_df["oos_sharpe"].mean()
    m3.metric("Mean OOS Sharpe", f"{avg_oos_sharpe:.2f}")

    robust_pct = (len(results_df[results_df["walk_forward_efficiency"] >= 0.60]) / len(results_df)) * 100.0
    m4.metric("OOS Robust Rate", f"{robust_pct:.1f}%", "WFE >= 0.60")

    best_strat = results_df.sort_values("sharpe_ratio", ascending=False).iloc[0]
    m5.metric("Top Strategy Alpha", f"{best_strat['sharpe_ratio']:.2f} SR", best_strat['symbol'])

    # Scatter Chart: IS vs OOS Generalization
    st.markdown('<div class="section-header">GENERALIZATION MAPPING: IN-SAMPLE VS OUT-OF-SAMPLE SHARPE</div>', unsafe_allow_html=True)
    
    fig = px.scatter(
        results_df,
        x="sharpe_ratio",
        y="oos_sharpe",
        color="strategy_type",
        hover_data=["symbol", "strategy_name", "walk_forward_efficiency", "friction_survival_ratio"],
        labels={
            "sharpe_ratio": "In-Sample Sharpe (Realized)",
            "oos_sharpe": "Out-of-Sample Sharpe (Walk-Forward)",
            "strategy_type": "Strategy Family",
        },
        template="plotly_dark",
        title=f"Statistical Persistence in {selected_m_code} (Diagonal = Perfect Walk-Forward Generalization)",
    )
    # Add 45-degree guideline
    max_val = max(results_df["sharpe_ratio"].max(), results_df["oos_sharpe"].max(), 1.0)
    min_val = min(results_df["sharpe_ratio"].min(), results_df["oos_sharpe"].min(), -1.0)
    fig.add_shape(
        type="line",
        x0=min_val, y0=min_val, x1=max_val, y1=max_val,
        line=dict(color="#64748b", dash="dash", width=1),
    )
    fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # League Table
    st.markdown('<div class="section-header">PRECOMPUTED STRATEGY PERFORMANCE LEAGUE TABLE</div>', unsafe_allow_html=True)
    
    disp_cols = [
        "symbol", "strategy_name", "strategy_type", "sharpe_ratio", "oos_sharpe",
        "walk_forward_efficiency", "cagr", "max_drawdown", "annualized_turnover",
        "friction_survival_ratio", "total_trades", "overfitting_risk_level"
    ]
    formatted_df = results_df[disp_cols].copy()
    formatted_df.rename(
        columns={
            "symbol": "Asset",
            "strategy_name": "Strategy Name",
            "strategy_type": "Family",
            "sharpe_ratio": "IS Sharpe",
            "oos_sharpe": "OOS Sharpe",
            "walk_forward_efficiency": "WFE Score",
            "cagr": "Net CAGR",
            "max_drawdown": "Max DD",
            "annualized_turnover": "Turnover",
            "friction_survival_ratio": "Friction Survival",
            "total_trades": "Trades",
            "overfitting_risk_level": "Risk Audit",
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
else:
    st.info("No strategies match the selected criteria.")

render_disclaimer()

"""Systematic Research Engine - Empirical Research Findings.

Displays structured discoveries, cross-market anomalies, execution realities,
and statistical warnings distilled directly from the persistent research database.
"""

import sys
from pathlib import Path

# Add src to sys.path
_SRC_PATH = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

import streamlit as st
import pandas as pd
import json
from quant_engine.research.store import ResearchStore
from quant_engine.ui_helpers import apply_terminal_theme, render_terminal_header, render_disclaimer

st.set_page_config(page_title="Research Findings", page_icon="🔬", layout="wide")
apply_terminal_theme()

render_terminal_header(
    title="EMPIRICAL RESEARCH FINDINGS",
    subtitle="Objective quantitative discoveries & statistical audits derived from precomputed research records",
)

store = ResearchStore()
findings = store.get_all_findings()
results_df = store.get_strategy_results()

if not findings:
    st.warning("No empirical findings found in database. Run `python scripts/run_research.py` to populate.")
    st.stop()

# Controls Row
col1, col2 = st.columns([2, 2])
with col1:
    categories = ["ALL"] + sorted(list(set(f["category"] for f in findings)))
    selected_cat = st.selectbox("Filter Finding Category", categories, index=0)

filtered_findings = findings if selected_cat == "ALL" else [f for f in findings if f["category"] == selected_cat]

st.markdown('<div class="section-header">SYNTHESIZED QUANTITATIVE DISCOVERIES</div>', unsafe_allow_html=True)

for item in filtered_findings:
    sev = item.get("severity", "INFO")
    badge_color = (
        "#00e676" if sev == "SUCCESS"
        else "#ffab00" if sev == "WARNING"
        else "#ff1744" if sev == "CAUTION"
        else "#00e5ff"
    )
    
    with st.container():
        st.markdown(
            f"""
            <div style="background-color: #121926; border: 1px solid #1e293b; border-left: 5px solid {badge_color};
                        padding: 16px; border-radius: 6px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 11px; font-weight: 700; color: {badge_color}; text-transform: uppercase;">
                        {item.get('category')} • [{sev}]
                    </span>
                    <span style="font-size: 11px; color: #64748b; font-family: monospace;">
                        {item.get('finding_id')}
                    </span>
                </div>
                <h4 style="margin: 0 0 8px 0; color: #f8fafc; font-size: 16px;">
                    {item.get('headline')}
                </h4>
                <p style="margin: 0 0 12px 0; color: #cbd5e1; font-size: 13px; line-height: 1.5;">
                    {item.get('narrative')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_raw = item.get("supporting_metric", "{}")
        if isinstance(metric_raw, str):
            try:
                metrics_dict = json.loads(metric_raw)
            except Exception:
                metrics_dict = {}
        else:
            metrics_dict = metric_raw or {}

        if metrics_dict:
            with st.expander(f"📊 Supporting Evidence Data ({item.get('finding_id')})", expanded=False):
                m_cols = st.columns(min(len(metrics_dict), 4))
                for idx, (k, v) in enumerate(metrics_dict.items()):
                    with m_cols[idx % len(m_cols)]:
                        val_str = f"{v:.2f}" if isinstance(v, float) else str(v)
                        st.metric(k.replace("_", " ").title(), val_str)

st.markdown('<div class="section-header">EMPIRICAL FRAGILITY & OVERFITTING AUDIT</div>', unsafe_allow_html=True)
st.markdown("*Systematic inspection of strategy parameter sets displaying severe performance decay or high risk:*")

if not results_df.empty:
    fragile_df = results_df[results_df["overfitting_risk_level"].isin(["HIGH", "MODERATE"])].copy()
    if not fragile_df.empty:
        fragile_cols = [
            "market", "symbol", "strategy_name", "sharpe_ratio", "oos_sharpe",
            "walk_forward_efficiency", "annualized_turnover", "friction_survival_ratio", "overfitting_risk_level"
        ]
        disp_df = fragile_df[fragile_cols].sort_values("sharpe_ratio", ascending=False)
        disp_df.rename(
            columns={
                "market": "Market",
                "symbol": "Asset",
                "strategy_name": "Strategy",
                "sharpe_ratio": "In-Sample Sharpe",
                "oos_sharpe": "Out-of-Sample Sharpe",
                "walk_forward_efficiency": "WFE Score",
                "annualized_turnover": "Annual Turnover",
                "friction_survival_ratio": "Friction Survival",
                "overfitting_risk_level": "Risk Audit",
            },
            inplace=True,
        )
        st.dataframe(
            disp_df.style.format({
                "In-Sample Sharpe": "{:.2f}",
                "Out-of-Sample Sharpe": "{:.2f}",
                "WFE Score": "{:.2f}",
                "Annual Turnover": "{:.1f}x",
                "Friction Survival": "{:.1%}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No high-risk overfit strategies detected in current research dataset.")

render_disclaimer()

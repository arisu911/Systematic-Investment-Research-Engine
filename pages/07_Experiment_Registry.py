"""Experiment Registry: Search, Compare, and Audit Historical Research Runs."""

import sys
from pathlib import Path

# Ensure src/ directory is on sys.path
_SRC_PATH = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

import streamlit as st
import pandas as pd
import json

from quant_engine.registry.experiments import ExperimentRegistry
from quant_engine.ui_helpers import (
    apply_terminal_theme,
    render_terminal_header,
    render_disclaimer,
)

st.set_page_config(page_title="Experiment Registry | Systematic Engine", page_icon="🗄️", layout="wide")
apply_terminal_theme()

force_demo = st.session_state.get("force_demo", False)

render_terminal_header(
    title="EMPIRICAL EXPERIMENT REGISTRY",
    subtitle="Audit Trail, Reproducibility Ledger, and Parameter Comparison",
    is_demo=force_demo,
)

registry = ExperimentRegistry()
experiments = registry.list_all()

if not experiments:
    st.info("No experiments currently logged. Run a backtest in the Research Lab and click 'Save Experiment to Registry'.")
    render_disclaimer()
    st.stop()

# Table Overview
reg_df = registry.to_dataframe()

st.markdown('<div class="section-header">LOGGED EXPERIMENT LEDGER</div>', unsafe_allow_html=True)

# Filter Controls
f1, f2 = st.columns([1, 1])
with f1:
    market_filter = st.multiselect("Filter by Market", options=list(reg_df["Market"].unique()), default=list(reg_df["Market"].unique()))
with f2:
    min_sharpe = st.slider("Minimum Sharpe Ratio", -2.0, 3.0, -1.0, 0.1)

filtered_df = reg_df[(reg_df["Market"].isin(market_filter)) & (reg_df["Sharpe"] >= min_sharpe)]
st.dataframe(filtered_df, use_container_width=True, hide_index=True)

# Download CSV
csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Export Registry to CSV", data=csv_data, file_name="systematic_research_experiments.csv", mime="text/csv")

# Detail Inspector & Comparison
st.markdown('<div class="section-header">INSPECT EXPERIMENT ARTIFACTS</div>', unsafe_allow_html=True)
selected_id = st.selectbox("Select Experiment ID to Inspect", [e.experiment_id for e in experiments])

exp_obj = registry.get(selected_id)
if exp_obj:
    col_e1, col_e2 = st.columns([1, 1])
    with col_e1:
        st.markdown(f"**Experiment**: `{exp_obj.experiment_id}` ({exp_obj.strategy_name})")
        st.write(f"**Timestamp**: {exp_obj.timestamp}")
        st.write(f"**Market & Asset**: {exp_obj.market} | {exp_obj.symbol}")
        st.write(f"**Period**: {exp_obj.start_date} to {exp_obj.end_date}")
        st.write(f"**Overfitting Risk**: `{exp_obj.overfitting_risk_level}`")
        if exp_obj.notes:
            st.info(f"**Notes**: {exp_obj.notes}")

    with col_e2:
        st.markdown("**Parameters & Assumptions JSON**")
        st.json(
            {
                "parameters": exp_obj.parameters,
                "transaction_costs": exp_obj.transaction_costs,
                "slippage_bps": exp_obj.slippage_bps,
                "metrics": exp_obj.metrics,
            }
        )

    if st.button("🗑️ Delete This Experiment"):
        if registry.delete(selected_id):
            st.success(f"Deleted {selected_id}. Refreshing...")
            st.rerun()

render_disclaimer()

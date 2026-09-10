"""Systematic Research Engine - Walk-Forward Archive.

Inspects precomputed walk-forward optimization and out-of-sample statistical persistence.
Examines rolling train/test slices, Walk-Forward Efficiency (WFE), and parameter stability.
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
import plotly.graph_objects as go
from quant_engine.research.store import ResearchStore
from quant_engine.ui_helpers import apply_terminal_theme, render_terminal_header, render_disclaimer

st.set_page_config(page_title="Walk-Forward Archive", page_icon="⏳", layout="wide")
apply_terminal_theme()

render_terminal_header(
    title="WALK-FORWARD VALIDATION ARCHIVE",
    subtitle="Audit chronological out-of-sample persistence, slice-by-slice degradation, and Walk-Forward Efficiency",
)

store = ResearchStore()
markets = store.get_distinct_markets()

if not markets:
    st.warning("No research data found in database. Run `python scripts/run_research.py` to populate.")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    selected_m = st.selectbox("Market", markets, index=0 if "MY" not in markets else markets.index("MY"))

available_symbols = store.get_distinct_symbols(market=selected_m)
with col2:
    selected_sym = st.selectbox("Asset", available_symbols, index=0)

strat_records = store.get_strategy_results(market=selected_m, symbol=selected_sym)
if strat_records.empty:
    st.info(f"No records found for {selected_sym}.")
    st.stop()

with col3:
    variants = list(strat_records["strategy_name"].unique())
    selected_strat_name = st.selectbox("Strategy Variant", variants, index=0)

exp_record = strat_records[strat_records["strategy_name"] == selected_strat_name].iloc[0]
exp_id = exp_record["experiment_id"]

st.markdown('<div class="section-header">WALK-FORWARD EFFICIENCY AUDIT</div>', unsafe_allow_html=True)

w1, w2, w3, w4 = st.columns(4)
wfe = exp_record["walk_forward_efficiency"]
w1.metric("Walk-Forward Efficiency (WFE)", f"{wfe:.2f}", "Target >= 0.60" if wfe >= 0.60 else "Decay < 0.60")
w2.metric("In-Sample Sharpe", f"{exp_record['sharpe_ratio']:.2f}")
w3.metric("Out-of-Sample Sharpe", f"{exp_record['oos_sharpe']:.2f}")
w4.metric("OOS Max Drawdown", f"{exp_record['oos_max_drawdown']*100:.1f}%")

# Load Slices from Database
slices_df = store.get_walk_forward_slices(exp_id)

if not slices_df.empty:
    st.markdown('<div class="section-header">SLICE-BY-SLICE TRAIN / TEST STABILITY</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f"Slice {s}" for s in slices_df["slice_id"]],
            y=slices_df["in_sample_sharpe"],
            name="In-Sample Sharpe (Training)",
            marker_color="#38bdf8",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[f"Slice {s}" for s in slices_df["slice_id"]],
            y=slices_df["out_of_sample_sharpe"],
            name="Out-of-Sample Sharpe (Testing)",
            marker_color="#22c55e",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis_title="Sharpe Ratio",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed Table of Slices
    disp_slices = slices_df[[
        "slice_id", "train_start", "train_end", "test_start", "test_end",
        "in_sample_sharpe", "out_of_sample_sharpe", "in_sample_cagr", "out_of_sample_cagr", "walk_forward_efficiency"
    ]].copy()
    disp_slices.rename(
        columns={
            "slice_id": "Slice",
            "train_start": "Train Start",
            "train_end": "Train End",
            "test_start": "Test Start",
            "test_end": "Test End",
            "in_sample_sharpe": "IS Sharpe",
            "out_of_sample_sharpe": "OOS Sharpe",
            "in_sample_cagr": "IS CAGR",
            "out_of_sample_cagr": "OOS CAGR",
            "walk_forward_efficiency": "WFE Ratio",
        },
        inplace=True,
    )
    st.dataframe(
        disp_slices.style.format({
            "IS Sharpe": "{:.2f}",
            "OOS Sharpe": "{:.2f}",
            "IS CAGR": "{:.1%}",
            "OOS CAGR": "{:.1%}",
            "WFE Ratio": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No rolling walk-forward slices logged for this specific combination.")

render_disclaimer()

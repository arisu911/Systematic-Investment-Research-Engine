"""Walk-Forward Validation: Rolling & Expanding Window In-Sample vs Out-of-Sample Testing."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from quant_engine.config.markets import load_market_config
from quant_engine.config.strategies import StrategyConfig, StrategyType
from quant_engine.data.downloader import DataDownloader
from quant_engine.signals.momentum import MovingAverageMomentum, TimeSeriesMomentum
from quant_engine.signals.mean_reversion import MeanReversion
from quant_engine.signals.breakout import DonchianBreakout
from quant_engine.validation.walk_forward import WalkForwardEngine
from quant_engine.ui_helpers import (
    apply_terminal_theme,
    render_terminal_header,
    get_terminal_plotly_layout,
    render_disclaimer,
)

st.set_page_config(page_title="Walk-Forward | Systematic Engine", page_icon="🔄", layout="wide")
apply_terminal_theme()

force_demo = st.session_state.get("force_demo", False)

render_terminal_header(
    title="WALK-FORWARD VALIDATION LABORATORY",
    subtitle="Rigorous Anchored & Rolling Out-of-Sample Generalization Testing",
    is_demo=force_demo,
)

# Configuration controls
c1, c2, c3, c4 = st.columns(4)
with c1:
    market_code = st.selectbox("Market Selection", ["MY", "US", "JP", "EU"], index=0)
    market_cfg = load_market_config(market_code)
with c2:
    symbol = st.selectbox("Underlying Security", [u.symbol for u in market_cfg.universe], index=0)
with c3:
    strat_choice = st.selectbox(
        "Strategy Family",
        ["Time-Series Momentum", "Moving Average Crossover", "Donchian Breakout", "Z-Score Mean Reversion"],
    )
with c4:
    wf_mode = st.selectbox("Window Method", ["Rolling Windows", "Expanding Anchored Windows"])

col_w1, col_w2, col_w3 = st.columns(3)
train_bars = col_w1.slider("Training Window (Bars)", 250, 750, 500, 50, help="In-sample training window (~2 years = 500 bars)")
test_bars = col_w2.slider("Testing Window (Bars)", 60, 250, 125, 25, help="Out-of-sample forward test window (~6 months = 125 bars)")
step_bars = col_w3.slider("Step Size (Bars)", 60, 250, 125, 25, help="Forward shift between slices")

run_wf = st.button("🚀 Execute Walk-Forward Analysis", type="primary")

downloader = DataDownloader(force_demo=force_demo)
df, health = downloader.load_price_data(symbol, "2018-01-01", "2025-12-31")

if df.empty or len(df) < (train_bars + test_bars):
    st.error(f"Insufficient history for {symbol}. Required: {train_bars + test_bars} bars, Available: {len(df)} bars.")
    st.stop()

# Map strategy class
if strat_choice == "Time-Series Momentum":
    strat_cls = TimeSeriesMomentum
    strat_cfg = StrategyConfig(parameters={"lookback_days": 120, "holding_days": 20})
elif strat_choice == "Moving Average Crossover":
    strat_cls = MovingAverageMomentum
    strat_cfg = StrategyConfig(parameters={"fast_period": 20, "slow_period": 100})
elif strat_choice == "Donchian Breakout":
    strat_cls = DonchianBreakout
    strat_cfg = StrategyConfig(parameters={"lookback": 50, "exit_lookback": 25})
else:
    strat_cls = MeanReversion
    strat_cfg = StrategyConfig(parameters={"window": 20, "entry_z": 2.0, "exit_z": 0.5})

wf_engine = WalkForwardEngine(
    train_window_bars=train_bars,
    test_window_bars=test_bars,
    step_bars=step_bars,
    expanding=(wf_mode == "Expanding Anchored Windows"),
)

wf_result, oos_equity = wf_engine.run(df, strat_cls, strat_cfg)

# Summary KPIs
k1, k2, k3, k4, k5 = st.columns(5)
wfe = wf_result.overall_wfe
k1.metric("Walk-Forward Efficiency (WFE)", f"{wfe:.2f}", "OOS / IS Sharpe")
k2.metric("Mean In-Sample Sharpe", f"{wf_result.mean_is_sharpe:.2f}")
k3.metric("Out-of-Sample Sharpe", f"{wf_result.oos_sharpe:.2f}")
k4.metric("Out-of-Sample CAGR", f"{wf_result.oos_cagr*100.0:.2f}%")
k5.metric("OOS Max Drawdown", f"{wf_result.oos_max_drawdown*100.0:.2f}%")

if wfe >= 0.70:
    st.success("✅ **EXCELLENT OUT-OF-SAMPLE STABILITY**: Strategy preserves over 70% of its in-sample Sharpe ratio out-of-sample.")
elif wfe >= 0.40:
    st.info("ℹ️ **MODERATE GENERALIZATION**: Strategy exhibits normal degradation out-of-sample.")
else:
    st.warning("⚠️ **OVERFITTING DETECTED**: Substantial performance collapse out-of-sample (WFE < 0.40).")

# Continuous OOS Equity Curve
st.markdown('<div class="section-header">COMBINED OUT-OF-SAMPLE EQUITY TRAJECTORY</div>', unsafe_allow_html=True)
if not oos_equity.empty:
    fig_oos = go.Figure()
    fig_oos.add_trace(
        go.Scatter(
            x=oos_equity.index,
            y=oos_equity,
            name="Continuous OOS Equity",
            line=dict(color="#00D4B2", width=2.2),
        )
    )
    # Benchmark overlay for test period
    bench_sub = df["Close"].loc[oos_equity.index]
    bench_nav = bench_sub / bench_sub.iloc[0]
    fig_oos.add_trace(
        go.Scatter(
            x=bench_nav.index,
            y=bench_nav,
            name="Buy & Hold Benchmark",
            line=dict(color="#8C9BAE", width=1.4, dash="dot"),
        )
    )
    fig_oos.update_layout(get_terminal_plotly_layout(height=420, title="STRICT OUT-OF-SAMPLE TEST EQUITY (SPLICED)"))
    st.plotly_chart(fig_oos, use_container_width=True)

# Slice by slice inspection table
st.markdown('<div class="section-header">WALK-FORWARD SLICE-BY-SLICE REPORT</div>', unsafe_allow_html=True)
slices_data = []
for s in wf_result.slices:
    slices_data.append(
        {
            "Slice #": s.slice_id,
            "Train Period": f"{s.train_start} to {s.train_end}",
            "Test Period": f"{s.test_start} to {s.test_end}",
            "In-Sample Sharpe": s.in_sample_sharpe,
            "Out-of-Sample Sharpe": s.out_of_sample_sharpe,
            "In-Sample CAGR (%)": round(s.in_sample_cagr * 100.0, 2),
            "Out-of-Sample CAGR (%)": round(s.out_of_sample_cagr * 100.0, 2),
            "Efficiency (WFE)": s.walk_forward_efficiency,
        }
    )
st.dataframe(pd.DataFrame(slices_data), use_container_width=True, hide_index=True)

render_disclaimer()

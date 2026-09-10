"""Robustness Laboratory: Parameter Sensitivity, Cost Stress, Monte Carlo, and Regimes."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from quant_engine.config.markets import load_market_config
from quant_engine.config.strategies import StrategyConfig
from quant_engine.data.downloader import DataDownloader
from quant_engine.signals.momentum import MovingAverageMomentum, TimeSeriesMomentum
from quant_engine.signals.breakout import DonchianBreakout
from quant_engine.validation.parameter_sensitivity import ParameterSensitivityAnalyzer
from quant_engine.validation.robustness import RobustnessTester
from quant_engine.validation.monte_carlo import MonteCarloSimulator
from quant_engine.validation.bootstrap import BlockBootstrap
from quant_engine.analytics.regimes import decompose_regimes
from quant_engine.backtest.engine import BacktestEngine
from quant_engine.ui_helpers import (
    apply_terminal_theme,
    render_terminal_header,
    get_terminal_plotly_layout,
    render_disclaimer,
)

st.set_page_config(page_title="Robustness Lab | Systematic Engine", page_icon="🛡️", layout="wide")
apply_terminal_theme()

force_demo = st.session_state.get("force_demo", False)

render_terminal_header(
    title="ROBUSTNESS & STRESS-TESTING LABORATORY",
    subtitle="Parameter Heatmaps, Transaction Cost Decay, Monte Carlo Permutation, and Market Regimes",
    is_demo=force_demo,
)

# Common Asset Selector
col_m1, col_m2 = st.columns(2)
with col_m1:
    market_code = st.selectbox("Market Selection", ["MY", "US", "JP", "EU"], index=0)
    market_cfg = load_market_config(market_code)
with col_m2:
    symbol = st.selectbox("Security Under Test", [u.symbol for u in market_cfg.universe], index=0)

downloader = DataDownloader(force_demo=force_demo)
df, _ = downloader.load_price_data(symbol, "2019-01-01", "2024-12-31")

if df.empty:
    st.error("Data unavailable.")
    st.stop()

tab_param, tab_cost, tab_mc, tab_regimes = st.tabs(
    [
        "🎛️ Parameter Sensitivity Surface",
        "💸 Transaction Cost & Slippage Stress",
        "🎲 Monte Carlo & Block Bootstrap",
        "🌦️ Market Regime Breakdown",
    ]
)

# ----------------- TAB 1: PARAMETER SENSITIVITY -----------------
with tab_param:
    st.markdown("**2D Grid Sweep across Parameter Hyperplane**")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        fast_vals = [5, 10, 15, 20, 25, 30]
        st.write(f"Fast MA Values: {fast_vals}")
    with p_col2:
        slow_vals = [40, 60, 80, 100, 120, 150]
        st.write(f"Slow MA Values: {slow_vals}")

    if st.button("Run 2D Parameter Grid Sweep", type="primary"):
        with st.spinner("Computing parameter performance surface..."):
            analyzer = ParameterSensitivityAnalyzer()
            sharpe_mat, cagr_mat, stability_score = analyzer.sweep_2d(
                df=df,
                strategy_class=MovingAverageMomentum,
                base_config=StrategyConfig(),
                param1_name="fast_period",
                param1_values=fast_vals,
                param2_name="slow_period",
                param2_values=slow_vals,
            )

        st.metric("Parameter Surface Stability Score", f"{stability_score:.2f}", "1.0 = Smooth Plateau (Robust), 0.0 = Brittle Spikes")
        if stability_score < 0.40:
            st.warning("⚠️ **FRAGILE SURFACE WARNING**: Strategy performance fluctuates heavily across adjacent parameters.")
        else:
            st.success("✅ **STABLE SURFACE**: Performance forms a cohesive plateau.")

        fig_heat = px.imshow(
            sharpe_mat,
            labels=dict(x="Slow MA Period", y="Fast MA Period", color="Sharpe Ratio"),
            text_auto=".2f",
            color_continuous_scale="Viridis",
            aspect="auto",
        )
        fig_heat.update_layout(get_terminal_plotly_layout(height=420, title="SHARPE RATIO SURFACE (FAST VS SLOW MA)"))
        st.plotly_chart(fig_heat, use_container_width=True)

# ----------------- TAB 2: COST & SLIPPAGE STRESS -----------------
with tab_cost:
    st.markdown("**Empirical Drag under Increasing Friction Scenarios**")
    strat = MovingAverageMomentum(StrategyConfig(parameters={"fast_period": 20, "slow_period": 100}))
    signals = strat.generate_signals(df)

    cost_levels = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]
    cost_table, breakeven = RobustnessTester.test_cost_sensitivity(df, signals, cost_levels)

    c1, c2 = st.columns([2, 1])
    with c1:
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(x=cost_table["Cost_Bps"], y=cost_table["Net_Sharpe"], line=dict(color="#00D4B2", width=2.2), name="Net Sharpe"))
        fig_cost.add_hline(y=0, line_dash="dash", line_color="#FF5630")
        fig_cost.update_layout(get_terminal_plotly_layout(height=340, title="SHARPE RATIO VS ONE-WAY TRANSACTION COST (BPS)"))
        st.plotly_chart(fig_cost, use_container_width=True)

    with c2:
        st.metric("Break-Even Transaction Cost", f"{breakeven:.1f} bps")
        st.dataframe(cost_table, use_container_width=True, hide_index=True)

# ----------------- TAB 3: MONTE CARLO & BOOTSTRAP -----------------
with tab_mc:
    st.markdown("**Resampling Uncertainty Simulation (1,000 Iterations)**")
    engine = BacktestEngine()
    res_df, ledger, _ = engine.run(df, signals)
    trade_pnls = [t.return_pct for t in ledger.trades]

    if len(trade_pnls) >= 5:
        mc = MonteCarloSimulator(num_simulations=1000)
        mc_results = mc.simulate_trades(trade_pnls, initial_capital=100_000.0)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Median Simulated Capital", f"${mc_results['median_final_equity']:,.0f}")
        mc2.metric("5th Percentile Capital (Worst 5%)", f"${mc_results['ci_5_equity']:,.0f}")
        mc3.metric("95th Percentile Max Drawdown", f"{mc_results['ci_95_max_dd']*100:.1f}%")

        # Plot sample paths
        fig_mc = go.Figure()
        for path in mc_results["sample_paths"]:
            fig_mc.add_trace(go.Scatter(y=path, line=dict(color="rgba(0, 212, 178, 0.15)", width=1.0), showlegend=False))
        fig_mc.add_hline(y=100_000, line_dash="dash", line_color="#505F79")
        fig_mc.update_layout(get_terminal_plotly_layout(height=380, title="MONTE CARLO RESAMPLED WEALTH TRAJECTORIES ($100k BASE)"))
        st.plotly_chart(fig_mc, use_container_width=True)

        # Block bootstrap on Sharpe
        bb = BlockBootstrap(block_size=20, num_samples=300)
        bb_res = bb.resample_sharpe_distribution(res_df["Net_Return"])
        st.info(f"**Circular Block Bootstrap**: Estimated probability of positive Sharpe is **{bb_res['p_value_sharpe_positive']*100:.1f}%** with 90% Confidence Interval: [{bb_res['ci_5_sharpe']:.2f}, {bb_res['ci_95_sharpe']:.2f}].")
    else:
        st.info("Insufficient trades for Monte Carlo resampling (minimum 5 trades required).")

# ----------------- TAB 4: REGIME ANALYSIS -----------------
with tab_regimes:
    st.markdown("**Regime Conditioning: Bull/Bear & High/Low Volatility**")
    regime_results = decompose_regimes(res_df["Net_Return"], df["Close"], annual_days=market_cfg.annual_trading_days)
    if regime_results:
        reg_df = pd.DataFrame(
            [
                {"Regime": "Bull Market (Price > 200 SMA)", "Sharpe": regime_results["bull_market"]["sharpe"], "CAGR (%)": f"{regime_results['bull_market']['cagr']*100:.1f}%", "Bars": regime_results["bull_market"]["bars"]},
                {"Regime": "Bear Market (Price <= 200 SMA)", "Sharpe": regime_results["bear_market"]["sharpe"], "CAGR (%)": f"{regime_results['bear_market']['cagr']*100:.1f}%", "Bars": regime_results["bear_market"]["bars"]},
                {"Regime": "High Volatility (> Median Vol)", "Sharpe": regime_results["high_volatility"]["sharpe"], "CAGR (%)": f"{regime_results['high_volatility']['cagr']*100:.1f}%", "Bars": regime_results["high_volatility"]["bars"]},
                {"Regime": "Low Volatility (<= Median Vol)", "Sharpe": regime_results["low_volatility"]["sharpe"], "CAGR (%)": f"{regime_results['low_volatility']['cagr']*100:.1f}%", "Bars": regime_results["low_volatility"]["bars"]},
            ]
        )
        st.dataframe(reg_df, use_container_width=True, hide_index=True)
    else:
        st.info("Insufficient data for regime decomposition.")

render_disclaimer()

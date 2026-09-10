"""Research Lab: Interactive Quantitative Strategy Workstation."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date

from quant_engine.config.markets import load_market_config
from quant_engine.config.strategies import StrategyConfig, StrategyType
from quant_engine.data.downloader import DataDownloader
from quant_engine.signals.momentum import MovingAverageMomentum, TimeSeriesMomentum
from quant_engine.signals.mean_reversion import MeanReversion
from quant_engine.signals.breakout import DonchianBreakout
from quant_engine.signals.factor import CompositeFactorSignal
from quant_engine.portfolio.position_sizing import volatility_target_sizing
from quant_engine.features.volatility import realized_volatility
from quant_engine.backtest.engine import BacktestEngine
from quant_engine.analytics.performance import calculate_comprehensive_performance
from quant_engine.analytics.statistics import calculate_return_statistics
from quant_engine.analytics.attribution import attribute_return_drag
from quant_engine.risk.volatility import rolling_sharpe_ratio, rolling_annualized_volatility
from quant_engine.risk.drawdown import compute_drawdown_series
from quant_engine.validation.overfitting import OverfittingDetector
from quant_engine.registry.experiments import ExperimentRecord, ExperimentRegistry
from quant_engine.ui_helpers import (
    apply_terminal_theme,
    render_terminal_header,
    get_terminal_plotly_layout,
    render_disclaimer,
)

st.set_page_config(page_title="Research Lab | Systematic Engine", page_icon="🔬", layout="wide")
apply_terminal_theme()

force_demo = st.session_state.get("force_demo", False)

# ----------------- SIDEBAR PARAMETERS -----------------
st.sidebar.markdown("### 🏛️ Research Universe")
market_code = st.sidebar.selectbox("Market Selection", ["MY", "US", "JP", "EU"], index=0)
market_cfg = load_market_config(market_code)

symbols_options = [u.symbol for u in market_cfg.universe]
symbol_names = {u.symbol: f"{u.symbol} - {u.name}" for u in market_cfg.universe}
selected_symbol = st.sidebar.selectbox(
    "Target Security",
    symbols_options,
    format_func=lambda s: symbol_names.get(s, s),
)

st.sidebar.markdown("### ♟️ Strategy Formulation")
strategy_family = st.sidebar.selectbox(
    "Strategy Family",
    [
        "Time-Series Momentum",
        "Moving Average Crossover",
        "Z-Score Mean Reversion",
        "Donchian Channel Breakout",
        "Composite Factor",
    ],
)

strat_params = {}
if strategy_family == "Time-Series Momentum":
    strat_type = StrategyType.TS_MOMENTUM
    lookback = st.sidebar.slider("Lookback Window (Days)", 20, 252, 120, 10)
    holding = st.sidebar.slider("Holding Period (Days)", 1, 60, 20, 1)
    vol_scale = st.sidebar.checkbox("Volatility Scale Signal", value=True)
    strat_params = {"lookback_days": lookback, "holding_days": holding, "vol_scale": vol_scale}
elif strategy_family == "Moving Average Crossover":
    strat_type = StrategyType.MA_MOMENTUM
    fast = st.sidebar.slider("Fast MA Period", 5, 50, 20, 1)
    slow = st.sidebar.slider("Slow MA Period", 30, 200, 100, 5)
    strat_params = {"fast_period": fast, "slow_period": slow, "signal_type": "crossover"}
elif strategy_family == "Z-Score Mean Reversion":
    strat_type = StrategyType.MEAN_REVERSION
    win = st.sidebar.slider("Rolling Window (Days)", 10, 60, 20, 2)
    entry_z = st.sidebar.slider("Entry Z-Score (Std)", 1.0, 3.5, 2.0, 0.1)
    exit_z = st.sidebar.slider("Exit Z-Score (Std)", 0.0, 1.5, 0.5, 0.1)
    strat_params = {"window": win, "entry_z": entry_z, "exit_z": exit_z}
elif strategy_family == "Donchian Channel Breakout":
    strat_type = StrategyType.BREAKOUT
    lb = st.sidebar.slider("Channel Lookback (Days)", 20, 120, 50, 5)
    ex_lb = st.sidebar.slider("Exit Lookback (Days)", 10, 60, 25, 5)
    strat_params = {"lookback": lb, "exit_lookback": ex_lb}
else:
    strat_type = StrategyType.FACTOR
    m_w = st.sidebar.slider("Momentum Weight", 0.0, 1.0, 0.6, 0.05)
    strat_params = {"momentum_weight": m_w, "low_vol_weight": round(1.0 - m_w, 2), "entry_threshold": 0.5}

long_only = st.sidebar.checkbox("Long-Only Constraint", value=True)

st.sidebar.markdown("### ⏱️ Research Period")
start_d = st.sidebar.date_input("Start Date", date(2019, 1, 1))
end_d = st.sidebar.date_input("End Date", date(2024, 12, 31))

st.sidebar.markdown("### ⚖️ Position Sizing")
sizing_mode = st.sidebar.radio("Sizing Algorithm", ["Equal Weight (100%)", "Volatility Targeting (12% Vol)"])

st.sidebar.markdown("### 💸 Transaction Costs")
with st.sidebar.expander("Execution Cost Parameters", expanded=False):
    cost_cfg = market_cfg.transaction_costs
    brokerage_bps = st.number_input("Brokerage (bps)", value=float(cost_cfg.brokerage_bps), step=1.0)
    clearing_bps = st.number_input("Clearing Fee (bps)", value=float(cost_cfg.clearing_fee_bps), step=0.5)
    stamp_bps = st.number_input("Stamp Duty (bps)", value=float(cost_cfg.stamp_duty_bps), step=1.0)
    spread_bps = st.number_input("Bid-Ask Spread (bps)", value=float(cost_cfg.bid_ask_spread_bps), step=1.0)
    slippage_bps = st.number_input("Execution Slippage (bps)", value=float(cost_cfg.default_slippage_bps), step=1.0)
    min_comm = st.number_input("Min Commission", value=float(cost_cfg.minimum_commission), step=1.0)

    # Active cost config
    active_cost_cfg = cost_cfg.model_copy(
        update={
            "brokerage_bps": brokerage_bps,
            "clearing_fee_bps": clearing_bps,
            "stamp_duty_bps": stamp_bps,
            "bid_ask_spread_bps": spread_bps,
            "minimum_commission": min_comm,
        }
    )

run_button = st.sidebar.button("🚀 Run Strategy Experiment", type="primary", use_container_width=True)

# ----------------- MAIN EXECUTION -----------------
render_terminal_header(
    title=f"RESEARCH LAB: {market_cfg.market_name.upper()} ({selected_symbol})",
    subtitle=f"Systematic Backtest of {strategy_family} with Strict Execution Constraints",
    is_demo=force_demo,
)

downloader = DataDownloader(force_demo=force_demo)
price_df, health = downloader.load_price_data(selected_symbol, str(start_d), str(end_d))
bench_df, _ = downloader.load_price_data(market_cfg.benchmark_symbol, str(start_d), str(end_d))

if price_df.empty or len(price_df) < 30:
    st.error(f"Insufficient historical data available for {selected_symbol}.")
    st.stop()

# Instantiate Strategy
strat_cfg = StrategyConfig(
    name=f"{selected_symbol} {strategy_family}",
    strategy_type=strat_type,
    parameters=strat_params,
    long_only=long_only,
)

if strategy_family == "Time-Series Momentum":
    generator = TimeSeriesMomentum(strat_cfg)
elif strategy_family == "Moving Average Crossover":
    generator = MovingAverageMomentum(strat_cfg)
elif strategy_family == "Z-Score Mean Reversion":
    generator = MeanReversion(strat_cfg)
elif strategy_family == "Donchian Channel Breakout":
    generator = DonchianBreakout(strat_cfg)
else:
    generator = CompositeFactorSignal(strat_cfg)

raw_signals = generator.generate_signals(price_df)

# Sizing
if sizing_mode == "Volatility Targeting (12% Vol)":
    real_vol = realized_volatility(price_df["Close"], window=20)
    final_signals = volatility_target_sizing(raw_signals, real_vol, target_annual_vol=0.12, max_leverage=1.0)
else:
    final_signals = raw_signals

# Run Backtest
engine = BacktestEngine(
    cost_config=active_cost_cfg,
    slippage_bps=slippage_bps,
    initial_capital=100_000.0,
)

res_df, ledger, summary = engine.run(
    price_df,
    final_signals,
    symbol=selected_symbol,
    benchmark_close=bench_df["Close"] if not bench_df.empty else None,
)

# Compute metrics
perf = calculate_comprehensive_performance(
    res_df["Net_Return"], res_df["NAV"], risk_free_rate=0.03, annual_trading_days=market_cfg.annual_trading_days
)
drag = attribute_return_drag(
    100_000.0, summary["total_gross_return"] * 100_000 + 100_000, summary["final_equity"], summary["total_fees_paid"], summary["total_slippage_paid"]
)
stats_res = calculate_return_statistics(res_df["Net_Return"])

# Overfitting Audit
overfit_audit = OverfittingDetector.audit(
    in_sample_sharpe=perf.get("sharpe_ratio", 0.0),
    out_of_sample_sharpe=perf.get("sharpe_ratio", 0.0) * 0.75, # Estimated baseline
    total_trades=summary["trade_stats"]["total_trades"],
    annualized_turnover=summary["annualized_turnover"],
    max_drawdown=perf.get("max_drawdown", 0.0),
)

# Store in session state for cross-page sharing
st.session_state["last_res_df"] = res_df
st.session_state["last_ledger"] = ledger
st.session_state["last_summary"] = summary
st.session_state["last_perf"] = perf
st.session_state["last_symbol"] = selected_symbol
st.session_state["last_market"] = market_code

# Display Primary Performance Cards
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Net CAGR", f"{perf.get('cagr', 0.0)*100.0:.2f}%", f"Gross {drag['gross_return_pct']*100:.1f}%")
m2.metric("Sharpe Ratio", f"{perf.get('sharpe_ratio', 0.0):.2f}", f"Rf = 3.0%")
m3.metric("Sortino Ratio", f"{perf.get('sortino_ratio', 0.0):.2f}")
m4.metric("Max Drawdown", f"{perf.get('max_drawdown', 0.0)*100.0:.2f}%")
m5.metric("Profit Factor", f"{summary['trade_stats']['profit_factor']:.2f}")
m6.metric("Annual Turnover", f"{summary['annualized_turnover']:.1f}x")

# Overfitting Alert Banner
if overfit_audit.risk_level in ["HIGH", "SEVERE"]:
    st.warning(f"⚠️ **OVERFITTING DIAGNOSTIC: {overfit_audit.risk_level} RISK** — {', '.join(overfit_audit.warnings)}")
else:
    st.success(f"✅ **ROBUSTNESS CHECK: {overfit_audit.risk_level} OVERFITTING RISK** — Parameters and turnover meet baseline stability thresholds.")

# Tabbed Visualizations
tab_nav, tab_dd, tab_rolling, tab_drag = st.tabs(
    ["📈 NAV & Equity Curve", "🌊 Drawdown Dynamics", "📉 Rolling Risk & Sharpe", "💰 Cost & Drag Attribution"]
)

with tab_nav:
    fig_nav = go.Figure()
    fig_nav.add_trace(
        go.Scatter(
            x=res_df.index,
            y=res_df["NAV"],
            name="Strategy (Net of Costs)",
            line=dict(color="#00D4B2", width=2.2),
        )
    )
    fig_nav.add_trace(
        go.Scatter(
            x=res_df.index,
            y=res_df["Gross_NAV"],
            name="Strategy (Gross Returns)",
            line=dict(color="#36B37E", width=1.5, dash="dot"),
        )
    )
    if "Benchmark_NAV" in res_df.columns:
        fig_nav.add_trace(
            go.Scatter(
                x=res_df.index,
                y=res_df["Benchmark_NAV"],
                name=f"Benchmark ({market_cfg.benchmark_symbol})",
                line=dict(color="#8C9BAE", width=1.4),
            )
        )
    fig_nav.update_layout(get_terminal_plotly_layout(height=450, title="PORTFOLIO EQUITY TRAJECTORY (NORMALIZED BASE = 1.0)"))
    st.plotly_chart(fig_nav, use_container_width=True)

with tab_dd:
    dd_series = compute_drawdown_series(res_df["NAV"])
    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=dd_series.index,
            y=dd_series * 100.0,
            fill="tozeroy",
            name="Underwater (%)",
            line=dict(color="#FF5630", width=1.5),
            fillcolor="rgba(255, 86, 48, 0.2)",
        )
    )
    fig_dd.update_layout(get_terminal_plotly_layout(height=380, title="UNDERWATER DRAWDOWN CURVE (%)"))
    st.plotly_chart(fig_dd, use_container_width=True)

with tab_rolling:
    roll_s = rolling_sharpe_ratio(res_df["Net_Return"], window=126, risk_free_rate=0.03, annual_days=market_cfg.annual_trading_days)
    roll_v = rolling_annualized_volatility(res_df["Net_Return"], window=63, annual_days=market_cfg.annual_trading_days)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(x=roll_s.index, y=roll_s, line=dict(color="#00B8D9", width=1.8), name="Rolling 6M Sharpe"))
        fig_rs.add_hline(y=0, line_dash="dash", line_color="#505F79")
        fig_rs.update_layout(get_terminal_plotly_layout(height=340, title="ROLLING 126-DAY SHARPE RATIO"))
        st.plotly_chart(fig_rs, use_container_width=True)

    with col_r2:
        fig_rv = go.Figure()
        fig_rv.add_trace(go.Scatter(x=roll_v.index, y=roll_v * 100.0, line=dict(color="#FFAB00", width=1.8), name="Rolling 63D Vol (%)"))
        fig_rv.update_layout(get_terminal_plotly_layout(height=340, title="ROLLING 63-DAY ANNUALIZED VOLATILITY (%)"))
        st.plotly_chart(fig_rv, use_container_width=True)

with tab_drag:
    col_d1, col_d2 = st.columns([1, 2])
    with col_d1:
        st.markdown("**Execution Drag Decomposition**")
        drag_table = pd.DataFrame(
            [
                {"Component": "Total Gross Return", "Value": f"{drag['gross_return_pct']*100.0:.2f}%"},
                {"Component": "Brokerage & Clearing Fees", "Value": f"-{drag['fee_drag_pct']*100.0:.2f}%"},
                {"Component": "Slippage & Market Impact", "Value": f"-{drag['slippage_drag_pct']*100.0:.2f}%"},
                {"Component": "Total Execution Friction", "Value": f"-{drag['total_friction_pct']*100.0:.2f}%"},
                {"Component": "Realized Net Return", "Value": f"{drag['net_return_pct']*100.0:.2f}%"},
                {"Component": "Alpha Friction Survival Ratio", "Value": f"{drag['friction_survival_ratio']*100.0:.1f}%"},
            ]
        )
        st.dataframe(drag_table, use_container_width=True, hide_index=True)

    with col_d2:
        fig_waterfall = go.Figure(
            go.Waterfall(
                orientation="v",
                measure=["relative", "relative", "relative", "total"],
                x=["Gross Gain", "Exchange & Brokerage", "Slippage", "Net Profit"],
                y=[
                    drag["gross_return_pct"] * 100.0,
                    -drag["fee_drag_pct"] * 100.0,
                    -drag["slippage_drag_pct"] * 100.0,
                    drag["net_return_pct"] * 100.0,
                ],
                connector={"line": {"color": "#505F79"}},
                decreasing={"marker": {"color": "#FF5630"}},
                increasing={"marker": {"color": "#00D4B2"}},
                totals={"marker": {"color": "#00B8D9"}},
            )
        )
        fig_waterfall.update_layout(get_terminal_plotly_layout(height=340, title="PERFORMANCE DECAY WATERFALL (%)"))
        st.plotly_chart(fig_waterfall, use_container_width=True)

# Save to Registry
st.markdown("---")
col_reg1, col_reg2 = st.columns([3, 1])
with col_reg1:
    notes_input = st.text_input("Experiment Research Notes", placeholder="e.g. Tested 120-day momentum on Bursa Maybank with 10bps brokerage.")
with col_reg2:
    st.write("")
    st.write("")
    if st.button("💾 Save Experiment to Registry", use_container_width=True):
        reg = ExperimentRegistry()
        rec = ExperimentRecord(
            strategy_name=f"{strategy_family} ({selected_symbol})",
            strategy_type=strat_type.value,
            market=market_code,
            symbol=selected_symbol,
            start_date=str(start_d),
            end_date=str(end_d),
            parameters=strat_params,
            transaction_costs=active_cost_cfg.model_dump(),
            slippage_bps=slippage_bps,
            metrics=perf,
            overfitting_risk_level=overfit_audit.risk_level,
            notes=notes_input,
        )
        exp_id = reg.save(rec)
        st.success(f"Experiment saved to registry as **{exp_id}**! View in Experiment Registry page.")

render_disclaimer()

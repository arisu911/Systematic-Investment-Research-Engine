"""Risk Engine: Non-Gaussian Tail Risk Modeling & Nominal Cash-at-Risk Workstation.

Displays:
1. Nominal Cash-at-Risk (VaR & CVaR) across 1-Day, 5-Day, and 21-Day (1-Month) Horizons at 95% & 99%.
2. Empirical Fat-Tail Distribution & Cornish-Fisher Cutoff Overlay.
3. Quantile-Quantile (QQ) Diagnostic Plot exposing non-Gaussian excess kurtosis and skewness.
4. Component VaR & Risk Attribution: Capital Weight vs. Actual Risk Contribution (%RC and Nominal Cash).
5. Nominal Macroeconomic Shock & Historical Crisis Simulator.
"""

import sys
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data.aligner import load_universe_registry, get_tradable_tickers
from data.loader import get_cached_universe_prices, compute_lookback_dates, LOOKBACK_HORIZONS
from data.fx_engine import get_currency_symbol, FXEngine, SUPPORTED_CURRENCIES
from research.strategies import StrategyDispatcher, STRATEGY_REGISTRY
from research.risk import RiskEngine
from research.utils import inject_metric_css, format_money, render_data_freshness_badge, render_backfill_warning_badge
from experiments.stress_test import StressTestEngine

try:
    st.set_page_config(page_title="Nominal Risk Engine", page_icon="🛡️", layout="wide")
except Exception:
    pass

# Inject global metric CSS to prevent ellipsis truncation
inject_metric_css()

# Retrieve Global Session State or Defaults
capital = float(st.session_state.get("capital_amount", 100_000.0))
base_curr = st.session_state.get("selected_currency", "USD")
curr_sym = get_currency_symbol(base_curr)
active_strategy = st.session_state.get("selected_strategy", "max_sharpe")
is_hedged = st.session_state.get("hedged_toggle", False)
risk_free_rate = float(st.session_state.get("risk_free_rate", 0.040))
if "lookback_horizon" not in st.session_state:
    st.session_state["lookback_horizon"] = "5Y"
if "start_date" not in st.session_state or "end_date" not in st.session_state:
    s_date, e_date = compute_lookback_dates(st.session_state["lookback_horizon"])
    st.session_state["start_date"] = s_date
    st.session_state["end_date"] = e_date

st.markdown(
    f"""
    <div style="background-color: #131722; border: 1px solid #2a2e39; border-left: 5px solid #f59e0b;
                padding: 14px 18px; border-radius: 4px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 11px; font-weight: 800; color: #f59e0b; letter-spacing: 1.5px; text-transform: uppercase;">
                    NON-GAUSSIAN TAIL MODELING • NOMINAL CASH-AT-RISK & RISK ATTRIBUTION
                </span>
                <h2 style="margin: 4px 0 0 0; color: #ffffff; font-size: 22px; font-weight: 700;">
                    Nominal Cash-at-Risk & Macro Stress Workstation
                </h2>
            </div>
            <div style="text-align: right; color: #848e9c; font-family: monospace; font-size: 12px;">
                CAPITAL: <span style="color:#f59e0b; font-weight:700;">{format_money(capital, base_curr)}</span> | BASE: <span style="color:#fff;">{base_curr}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render Data Freshness & Proxy Backfill Warning Badges
render_data_freshness_badge()
render_backfill_warning_badge()

# Pinned interactive control bar for on-page adjustments
with st.expander("⚙️ Fine-Tune Risk Parameters & Capital Controls", expanded=False):
    ctl1, ctl2, ctl3, ctl4 = st.columns(4)
    
    curr_options = SUPPORTED_CURRENCIES + ["LOCAL"]
    new_curr = ctl1.selectbox("Base Currency", curr_options, index=curr_options.index(base_curr) if base_curr in curr_options else 0)
    if new_curr != base_curr:
        st.session_state["selected_currency"] = new_curr
        st.rerun()

    new_cap = ctl2.number_input("Portfolio Capital", min_value=100.0, value=capital, step=10_000.0, format="%.2f")
    if new_cap != capital:
        st.session_state["capital_amount"] = new_cap
        st.rerun()

    strat_keys = list(STRATEGY_REGISTRY.keys())
    strat_names = list(STRATEGY_REGISTRY.values())
    new_strat_name = ctl3.selectbox("Strategy Model", strat_names, index=strat_keys.index(active_strategy) if active_strategy in strat_keys else 0)
    new_strat_key = strat_keys[strat_names.index(new_strat_name)]
    if new_strat_key != active_strategy:
        st.session_state["selected_strategy"] = new_strat_key
        st.rerun()

    cur_lb = st.session_state.get("lookback_horizon", "5Y")
    cur_lb_idx = LOOKBACK_HORIZONS.index(cur_lb) if cur_lb in LOOKBACK_HORIZONS else 3
    new_lb = ctl4.selectbox("Historical Horizon", LOOKBACK_HORIZONS, index=cur_lb_idx, key="risk_horizon")
    if new_lb != cur_lb:
        st.session_state["lookback_horizon"] = new_lb
        s_date, e_date = compute_lookback_dates(new_lb)
        st.session_state["start_date"] = s_date
        st.session_state["end_date"] = e_date
        st.rerun()

# Load prices
with st.spinner("Triangulating market data and solving portfolio tail risk profiles..."):
    prices_df, is_demo = get_cached_universe_prices(
        start_date=st.session_state.get("start_date"),
        end_date=st.session_state.get("end_date"),
        base_currency=base_curr,
        lookback_horizon=st.session_state.get("lookback_horizon", "5Y"),
        ttl_seconds=st.session_state.get("cache_ttl_seconds", 14400),
        force_reload=st.session_state.get("force_reload", False),
    )

registry = load_universe_registry()
tradable_symbols = get_tradable_tickers(registry)
available_tradable = [s for s in tradable_symbols if s in prices_df.columns]

if len(available_tradable) < 3:
    st.error("Insufficient tradable assets for risk modeling.")
    st.stop()

# Returns calculation (accounting for hedged vs unhedged returns)
returns_df = FXEngine.calculate_asset_returns(prices_df[available_tradable], target_currency=base_curr, is_hedged=is_hedged)

# Dispatch active strategy weights
strat_res = StrategyDispatcher.dispatch(
    strategy_name=active_strategy,
    returns_df=returns_df,
    prices_df=prices_df[available_tradable],
    risk_free_rate=risk_free_rate,
)
weights = strat_res["weights"]
assets = strat_res["assets"]

port_daily_ret = pd.Series(np.dot(returns_df[assets].values, weights), index=returns_df.index)

# Compute Historical Max Drawdown
cum_ret = (1.0 + port_daily_ret).cumprod()
peak = cum_ret.cummax()
dd_series = (cum_ret - peak) / peak
max_dd_pct = float(dd_series.min())
nominal_max_dd = max_dd_pct * capital

# 1. Nominal Cash-at-Risk Metrics (1D, 5D, 21D)
st.markdown("#### 📐 Nominal Cash-at-Risk (VaR) & Expected Shortfall (CVaR) Matrix")

multi_var_df = RiskEngine.calculate_multi_horizon_nominal_var(
    port_daily_ret,
    capital=capital,
    horizons=[1, 5, 21],
    confidence_levels=[0.95, 0.99],
)

var_1d_95 = multi_var_df[(multi_var_df["Horizon_Days"] == 1) & (multi_var_df["Confidence"] == "95%")].iloc[0]
var_1d_99 = multi_var_df[(multi_var_df["Horizon_Days"] == 1) & (multi_var_df["Confidence"] == "99%")].iloc[0]
var_5d_99 = multi_var_df[(multi_var_df["Horizon_Days"] == 5) & (multi_var_df["Confidence"] == "99%")].iloc[0]
var_21d_99 = multi_var_df[(multi_var_df["Horizon_Days"] == 21) & (multi_var_df["Confidence"] == "99%")].iloc[0]

c_m1, c_m2, c_m3, c_m4 = st.columns(4)
c_m1.metric(
    "95% 1-Day Cornish-Fisher VaR",
    format_money(-var_1d_95["Nominal_Cornish_Fisher_VaR"], base_curr),
    f"{var_1d_95['Cornish_Fisher_VaR_Pct']:.2%} of capital",
    delta_color="inverse",
)
c_m2.metric(
    "99% 1-Day Cornish-Fisher VaR",
    format_money(-var_1d_99["Nominal_Cornish_Fisher_VaR"], base_curr),
    f"{var_1d_99['Cornish_Fisher_VaR_Pct']:.2%} of capital",
    delta_color="inverse",
)
c_m3.metric(
    "99% 5-Day (1-Wk) mVaR",
    format_money(-var_5d_99["Nominal_Cornish_Fisher_VaR"], base_curr),
    f"{var_5d_99['Cornish_Fisher_VaR_Pct']:.2%} of capital",
    delta_color="inverse",
)
c_m4.metric(
    "99% 21-Day (1-Mo) mVaR",
    format_money(-var_21d_99["Nominal_Cornish_Fisher_VaR"], base_curr),
    f"{var_21d_99['Cornish_Fisher_VaR_Pct']:.2%} of capital",
    delta_color="inverse",
)

# Highlight Callout Banner
st.markdown(
    f"""
    <div style="background-color: #1a1e29; border: 1px solid #374151; padding: 12px 18px; border-radius: 4px; margin: 14px 0; font-family: monospace; font-size: 13px;">
        🛡️ <b>INSTITUTIONAL RISK CALLOUT:</b> 99% 1-Day Cornish-Fisher VaR is <span style="color:#ef4444; font-weight:700;">{format_money(-var_1d_99['Nominal_Cornish_Fisher_VaR'], base_curr)}</span>
        on <span style="color:#ffffff; font-weight:700;">{format_money(capital, base_curr)}</span> active capital.
        (Parametric VaR: {format_money(-var_1d_99['Nominal_Parametric_VaR'], base_curr)} | Historical VaR: {format_money(-var_1d_99['Nominal_Historical_VaR'], base_curr)} | 1-Day mCVaR: {format_money(-var_1d_99['Nominal_Cornish_Fisher_CVaR'], base_curr)})
    </div>
    """,
    unsafe_allow_html=True,
)

# Detailed Multi-Horizon Cash-at-Risk Table with Strict Column Config
display_var_table = pd.DataFrame({
    "Horizon": multi_var_df["Horizon"],
    "Confidence": multi_var_df["Confidence"],
    "Cornish-Fisher mVaR": multi_var_df["Cornish_Fisher_VaR_Pct"],
    "Nominal mVaR": -multi_var_df["Nominal_Cornish_Fisher_VaR"],
    "Nominal mCVaR": -multi_var_df["Nominal_Cornish_Fisher_CVaR"],
    "Parametric VaR": -multi_var_df["Nominal_Parametric_VaR"],
    "Historical VaR": -multi_var_df["Nominal_Historical_VaR"],
})

st.dataframe(
    display_var_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Horizon": st.column_config.TextColumn("Horizon", width="medium"),
        "Confidence": st.column_config.TextColumn("Confidence", width="small"),
        "Cornish-Fisher mVaR": st.column_config.NumberColumn("Cornish-Fisher mVaR (%)", format="%.2f %%"),
        "Nominal mVaR": st.column_config.NumberColumn(f"Nominal mVaR ({base_curr})", format=f"{curr_sym}%,.2f"),
        "Nominal mCVaR": st.column_config.NumberColumn(f"Nominal mCVaR ({base_curr})", format=f"{curr_sym}%,.2f"),
        "Parametric VaR": st.column_config.NumberColumn(f"Parametric VaR ({base_curr})", format=f"{curr_sym}%,.2f"),
        "Historical VaR": st.column_config.NumberColumn(f"Historical VaR ({base_curr})", format=f"{curr_sym}%,.2f"),
    },
)

st.markdown("---")

# 2. Return Distribution & Fat-Tail QQ Diagnostics
c_dist, c_qq = st.columns([6, 4])

var_1d_data = RiskEngine.calculate_var_cvar(port_daily_ret, confidence_level=0.95, horizon_days=1)
skew = var_1d_data["skewness"]
kurt = var_1d_data["excess_kurtosis"]

with c_dist:
    st.markdown("#### 📊 Return Distribution & Cornish-Fisher Cutoffs")
    st.caption(f"Sample Skewness: {skew:+.3f} (Left asymmetry) | Excess Kurtosis: {kurt:+.3f} (Fat tails)")

    clean_ret = port_daily_ret.values
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=clean_ret,
            nbinsx=65,
            name="Daily Return Density",
            marker_color="#3b82f6",
            opacity=0.75,
            hovertemplate="Return: %{x:.2%}<br>Count: %{y}<extra></extra>",
        )
    )

    # Add VaR cutoff lines
    cf_cutoff = -var_1d_95["Cornish_Fisher_VaR_Pct"]
    param_cutoff = -var_1d_95["Parametric_VaR_Pct"]

    fig_hist.add_vline(
        x=cf_cutoff, line_dash="solid", line_color="#ef4444", line_width=2,
        annotation_text=f"CF VaR 95% ({format_money(-var_1d_95['Nominal_Cornish_Fisher_VaR'], base_curr)})",
        annotation_position="top left",
    )
    fig_hist.add_vline(
        x=param_cutoff, line_dash="dash", line_color="#94a3b8", line_width=1.5,
        annotation_text=f"Parametric ({format_money(-var_1d_95['Nominal_Parametric_VaR'], base_curr)})",
        annotation_position="bottom left",
    )

    fig_hist.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#131722",
        margin=dict(l=10, r=10, t=20, b=10),
        height=320,
        xaxis=dict(tickformat=".1%", gridcolor="#2a2e39"),
        yaxis=dict(gridcolor="#2a2e39"),
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False, "responsive": True})

with c_qq:
    st.markdown("#### 📉 Quantile-Quantile (QQ) Diagnostics")
    st.caption("Deviations from the red diagonal expose non-Gaussian tail risk.")

    qq_data = RiskEngine.generate_qq_plot_data(port_daily_ret)
    if len(qq_data["theoretical"]) > 0:
        fig_qq = go.Figure()
        fig_qq.add_trace(
            go.Scatter(
                x=qq_data["theoretical"],
                y=qq_data["empirical"],
                mode="markers",
                name="Standardized Returns",
                marker=dict(color="#f59e0b", size=4, opacity=0.7),
                hovertemplate="Theoretical: %{x:.2f}<br>Empirical: %{y:.2f}<extra></extra>",
            )
        )
        min_q = float(np.min(qq_data["theoretical"]))
        max_q = float(np.max(qq_data["theoretical"]))
        fig_qq.add_trace(
            go.Scatter(
                x=[min_q, max_q],
                y=[min_q, max_q],
                mode="lines",
                name="Normal 45° Line",
                line=dict(color="#ef4444", dash="dash"),
            )
        )
        fig_qq.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#131722",
            margin=dict(l=10, r=10, t=20, b=10),
            height=320,
            showlegend=False,
            xaxis=dict(title="Theoretical Normal Quantiles", gridcolor="#2a2e39"),
            yaxis=dict(title="Empirical Quantiles", gridcolor="#2a2e39"),
        )
        st.plotly_chart(fig_qq, use_container_width=True, config={"displayModeBar": False, "responsive": True})

st.markdown("---")

# 3. Component VaR & Risk Attribution
st.markdown("#### 🎯 Component VaR & Risk Attribution")
st.caption("Decomposes risk into Marginal Contribution (MCR), Percentage Contribution (%RC), and Nominal Cash Risk per asset.")

cov_matrix = returns_df[assets].cov().values * 252.0
rc_df = RiskEngine.calculate_risk_attribution(
    weights=weights,
    cov_matrix=cov_matrix,
    asset_names=assets,
    capital=capital,
)
rc_df["Name"] = [registry.get(a, {}).get("name", a) for a in rc_df["Asset"]]
rc_df = rc_df.sort_values("Pct_Risk_Contribution", ascending=False)

# Render interactive Plotly grouped bar chart: Capital Weight vs. %RC
fig_rc = go.Figure()
fig_rc.add_trace(
    go.Bar(
        x=rc_df["Asset"],
        y=rc_df["Weight"],
        name="Capital Weight",
        marker_color="#3b82f6",
        customdata=rc_df["Weight"] * capital,
        hovertemplate="<b>%{x}</b><br>Capital Weight: %{y:.2%}<br>Nominal Capital: " + curr_sym + "%{customdata:,.2f}<extra></extra>",
    )
)
fig_rc.add_trace(
    go.Bar(
        x=rc_df["Asset"],
        y=rc_df["Pct_Risk_Contribution"],
        name="Risk Contribution (%RC)",
        marker_color="#f59e0b",
        customdata=rc_df["Nominal_Risk_Contribution"],
        hovertemplate="<b>%{x}</b><br>Risk Contribution: %{y:.2%}<br>Nominal Annual Risk: " + curr_sym + "%{customdata:,.2f}<extra></extra>",
    )
)
fig_rc.update_layout(
    barmode="group",
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#131722",
    margin=dict(l=10, r=10, t=30, b=10),
    height=320,
    yaxis=dict(tickformat=".1%", gridcolor="#2a2e39"),
    xaxis=dict(gridcolor="#2a2e39"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_rc, use_container_width=True, config={"displayModeBar": False, "responsive": True})

# Summary Table of Risk Attribution with Strict Column Config
rc_table = pd.DataFrame({
    "Ticker": rc_df["Asset"],
    "Asset Name": rc_df["Name"],
    "Weight": rc_df["Weight"],
    "Nominal Capital": rc_df["Weight"] * capital,
    "Risk Contribution": rc_df["Pct_Risk_Contribution"],
    "Nominal Risk": rc_df["Nominal_Risk_Contribution"],
    "Marginal Volatility (MCR)": rc_df["Marginal_Risk_Contribution"],
})

st.dataframe(
    rc_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Asset Name": st.column_config.TextColumn("Asset Name", width="medium"),
        "Weight": st.column_config.NumberColumn("Weight (%)", format="%.2f %%"),
        "Nominal Capital": st.column_config.NumberColumn(f"Nominal Capital ({base_curr})", format=f"{curr_sym}%,.2f"),
        "Risk Contribution": st.column_config.NumberColumn("Risk Contribution (%RC)", format="%.2f %%"),
        "Nominal Risk": st.column_config.NumberColumn(f"Nominal Risk ({base_curr})", format=f"{curr_sym}%,.2f"),
        "Marginal Volatility (MCR)": st.column_config.NumberColumn("Marginal MCR", format="%.4f"),
    },
)

st.markdown("---")

# 4. Nominal Drawdown Exposure & Macroeconomic Scenario Stress Simulator
st.markdown("#### ⚡ Nominal Drawdown Exposure & Macro Scenario Stress Testing")
st.caption("Projects instant factor shocks and historical systemic crises into nominal monetary losses.")

c_dd1, c_dd2 = st.columns(2)
c_dd1.metric(
    "Historical Maximum Drawdown",
    f"{max_dd_pct:.2%}",
    f"{format_money(nominal_max_dd, base_curr)} Peak Loss",
    delta_color="inverse",
)
c_dd2.metric(
    "Total Portfolio Capital",
    format_money(capital, base_curr),
    f"Base Currency: {base_curr}",
)

# Extract macro series from prices_df
macro_cols = [c for c in ["^VIX", "^TNX", "BZ=F", "USDMYR=X", "JPY=X"] if c in prices_df.columns]
macro_df = prices_df[macro_cols] if macro_cols else None

stress_engine = StressTestEngine(returns_df[assets], macro_df=macro_df, assets=assets)

c_shock1, c_shock2 = st.columns([5, 5])

with c_shock1:
    st.markdown("##### Predefined Macro Shocks (Nominal Loss)")
    scenarios = stress_engine.run_standard_macro_scenarios(weights, capital=capital)
    scen_df = pd.DataFrame([
        {
            "Scenario": s["scenario_name"],
            "Factor": s["factor"],
            "Shock": s["shock_magnitude_pct"],
            "Impact": s["portfolio_impact_pct"],
            "Nominal Impact": s["nominal_portfolio_impact"],
        }
        for s in scenarios
    ])
    st.dataframe(
        scen_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Scenario": st.column_config.TextColumn("Scenario", width="medium"),
            "Factor": st.column_config.TextColumn("Factor", width="small"),
            "Shock": st.column_config.NumberColumn("Shock (%)", format="%+.0f %%"),
            "Impact": st.column_config.NumberColumn("Impact (%)", format="%+.2f %%"),
            "Nominal Impact": st.column_config.NumberColumn(f"Nominal Loss ({base_curr})", format=f"{curr_sym}%+,.2f"),
        },
    )

with c_shock2:
    st.markdown("##### Historical Crisis Replay (Nominal Loss)")
    crises = stress_engine.replay_historical_crises(weights, capital=capital)
    crisis_df = pd.DataFrame([
        {
            "Crisis": c["Crisis"],
            "Period": c["Period"],
            "Max Drawdown": c["Estimated_Max_Drawdown"],
            "Nominal Loss": c["Nominal_Max_Drawdown"],
            "Mode": c["Evaluation_Mode"],
        }
        for c in crises
    ])
    st.dataframe(
        crisis_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Crisis": st.column_config.TextColumn("Crisis Scenario", width="medium"),
            "Period": st.column_config.TextColumn("Period", width="medium"),
            "Max Drawdown": st.column_config.NumberColumn("Max Drawdown (%)", format="%.1f %%"),
            "Nominal Loss": st.column_config.NumberColumn(f"Nominal Drawdown ({base_curr})", format=f"{curr_sym}%,.2f"),
            "Mode": st.column_config.TextColumn("Evaluation", width="small"),
        },
    )

"""Risk Engine: Non-Gaussian Tail Risk Modeling & Macro Shock Simulation Workstation.

Displays:
1. VaR & CVaR Matrix: Parametric Gaussian vs Historical vs Cornish-Fisher Modified VaR/CVaR (95% & 99%).
2. Empirical Fat-Tail Distribution with Cornish-Fisher Critical Cutoff Overlay.
3. Quantile-Quantile (QQ) Diagnostic Plot exposing non-Gaussian excess kurtosis and skewness.
4. Percentage Risk Contribution (PCR) Decomposition across assets.
5. Interactive Macroeconomic Shock Simulator (VIX Doubling, US 10Y Spike, Oil Shock, Currency Surge).
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
from data.loader import get_cached_universe_prices
from research.optimization import PortfolioOptimizer
from research.risk import RiskEngine
from experiments.stress_test import StressTestEngine

try:
    st.set_page_config(page_title="Risk Engine", page_icon="🛡️", layout="wide")
except Exception:
    pass

st.markdown(
    """
    <div style="background-color: #131722; border: 1px solid #2a2e39; border-left: 5px solid #f59e0b;
                padding: 14px 18px; border-radius: 4px; margin-bottom: 20px;">
        <span style="font-size: 11px; font-weight: 800; color: #f59e0b; letter-spacing: 1.5px; text-transform: uppercase;">
            NON-GAUSSIAN TAIL MODELING • CORNISH-FISHER & MACRO STRESS
        </span>
        <h2 style="margin: 4px 0 0 0; color: #ffffff; font-size: 22px; font-weight: 700;">
            Multi-Market Tail Risk Engine & Macro Scenarios
        </h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load data
base_curr = st.session_state.get("base_currency", "USD")
prices_df, is_demo = get_cached_universe_prices(
    start_date="2020-01-01",
    end_date="2024-12-31",
    base_currency=base_curr,
)

registry = load_universe_registry()
tradable_symbols = get_tradable_tickers(registry)
available_tradable = [s for s in tradable_symbols if s in prices_df.columns]

if len(available_tradable) < 3:
    st.error("Insufficient tradable assets for risk modeling.")
    st.stop()

returns_df = prices_df[available_tradable].pct_change().dropna()

# Baseline Equal-Weight or Max-Sharpe portfolio
opt = PortfolioOptimizer(returns_df, filter_tradable=False)
opt_res = opt.optimize_mean_variance(objective="max_sharpe")
weights = opt_res["weights"]
assets = opt_res["assets"]

port_daily_ret = pd.Series(np.dot(returns_df[assets].values, weights), index=returns_df.index)

# 1. Non-Gaussian Tail Risk Metrics (95% & 99%)
st.markdown("#### 📐 Value-at-Risk (VaR) & Expected Shortfall (CVaR) Matrix")

var_95 = RiskEngine.calculate_var_cvar(port_daily_ret, confidence_level=0.95, horizon_days=1)
var_99 = RiskEngine.calculate_var_cvar(port_daily_ret, confidence_level=0.99, horizon_days=1)

c_m1, c_m2, c_m3, c_m4 = st.columns(4)
c_m1.metric("Sample Skewness", f"{var_95['skewness']:.3f}", "Negative = Left Tail Risk")
c_m2.metric("Excess Kurtosis", f"{var_95['excess_kurtosis']:.3f}", "Positive = Heavy Fat Tails")
c_m3.metric("Cornish-Fisher VaR (95%)", f"{var_95['cornish_fisher_var']:.2%}", f"vs {var_95['parametric_var']:.2%} Parametric")
c_m4.metric("Cornish-Fisher VaR (99%)", f"{var_99['cornish_fisher_var']:.2%}", f"vs {var_99['parametric_var']:.2%} Parametric")

tail_table = pd.DataFrame([
    {
        "Confidence Horizon": "95.0% (1-Day)",
        "Parametric Gaussian VaR": f"{var_95['parametric_var']:.2%}",
        "Historical Empirical VaR": f"{var_95['historical_var']:.2%}",
        "Cornish-Fisher Modified VaR": f"{var_95['cornish_fisher_var']:.2%}",
        "Parametric CVaR": f"{var_95['parametric_cvar']:.2%}",
        "Cornish-Fisher Modified CVaR": f"{var_95['cornish_fisher_cvar']:.2%}",
    },
    {
        "Confidence Horizon": "99.0% (1-Day)",
        "Parametric Gaussian VaR": f"{var_99['parametric_var']:.2%}",
        "Historical Empirical VaR": f"{var_99['historical_var']:.2%}",
        "Cornish-Fisher Modified VaR": f"{var_99['cornish_fisher_var']:.2%}",
        "Parametric CVaR": f"{var_99['parametric_cvar']:.2%}",
        "Cornish-Fisher Modified CVaR": f"{var_99['cornish_fisher_cvar']:.2%}",
    },
])
st.dataframe(tail_table, use_container_width=True, hide_index=True)

st.markdown("---")

# 2. Return Distribution & QQ-Plot Side by Side
c_dist, c_qq = st.columns([6, 4])

with c_dist:
    st.markdown("#### 📊 Return Distribution & Cornish-Fisher Cutoffs")

    clean_ret = port_daily_ret.values
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=clean_ret,
            nbinsx=60,
            name="Daily Return Density",
            marker_color="#3b82f6",
            opacity=0.75,
            hovertemplate="Return: %{x:.2%}<br>Count: %{y}<extra></extra>",
        )
    )

    # Add VaR cutoff lines
    cf_var_cutoff = -var_95["cornish_fisher_var"]
    param_var_cutoff = -var_95["parametric_var"]

    fig_hist.add_vline(x=cf_var_cutoff, line_dash="solid", line_color="#ef4444", line_width=2,
                       annotation_text=f"CF VaR 95% ({cf_var_cutoff:.2%})", annotation_position="top left")
    fig_hist.add_vline(x=param_var_cutoff, line_dash="dash", line_color="#94a3b8", line_width=1.5,
                       annotation_text=f"Parametric ({param_var_cutoff:.2%})", annotation_position="bottom left")

    fig_hist.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#131722",
        margin=dict(l=10, r=10, t=20, b=10),
        height=340,
        xaxis=dict(tickformat=".1%", gridcolor="#2a2e39"),
        yaxis=dict(gridcolor="#2a2e39"),
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False, "responsive": True})

with c_qq:
    st.markdown("#### 📉 Fat-Tail Quantile-Quantile (QQ) Plot")
    st.caption("Deviations from the red diagonal reveal non-Gaussian leptokurtosis.")

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
            height=340,
            showlegend=False,
            xaxis=dict(title="Theoretical Normal Quantiles", gridcolor="#2a2e39"),
            yaxis=dict(title="Empirical Quantiles", gridcolor="#2a2e39"),
        )
        st.plotly_chart(fig_qq, use_container_width=True, config={"displayModeBar": False, "responsive": True})

st.markdown("---")

# 3. Percentage Contribution to Risk (PCR)
st.markdown("#### 🎯 Percentage Risk Contribution (PCR) Breakdown")
cov_matrix = returns_df[assets].cov().values * 252.0
rc_res = RiskEngine.calculate_risk_contributions(weights, cov_matrix)

pcr_df = pd.DataFrame({
    "Asset": assets,
    "Name": [registry.get(a, {}).get("name", a) for a in assets],
    "Weight": weights,
    "Percentage_Risk_Contribution": rc_res["pcr"],
})
pcr_df = pcr_df.sort_values("Percentage_Risk_Contribution", ascending=False)

fig_pcr = px.bar(
    pcr_df,
    x="Asset",
    y=["Weight", "Percentage_Risk_Contribution"],
    barmode="group",
    labels=dict(value="Proportion", variable="Metric"),
    color_discrete_sequence=["#3b82f6", "#f59e0b"],
)
fig_pcr.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#131722",
    margin=dict(l=10, r=10, t=20, b=10),
    height=280,
    yaxis=dict(tickformat=".0%", gridcolor="#2a2e39"),
    xaxis=dict(gridcolor="#2a2e39"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_pcr, use_container_width=True, config={"displayModeBar": False, "responsive": True})

st.markdown("---")

# 4. Macroeconomic Scenario Stress Test Simulator
st.markdown("#### ⚡ Macroeconomic Scenario Shock Simulator")
st.caption("Replays instant factor spikes across volatility, sovereign rates, energy commodities, and FX.")

# Extract macro series from prices_df
macro_cols = [c for c in ["^VIX", "^TNX", "BZ=F", "USDMYR=X", "JPY=X"] if c in prices_df.columns]
macro_df = prices_df[macro_cols] if macro_cols else None

stress_engine = StressTestEngine(returns_df[assets], macro_df=macro_df, assets=assets)

c_shock1, c_shock2 = st.columns([5, 5])

with c_shock1:
    st.markdown("##### Predefined Macro Shocks")
    scenarios = stress_engine.run_standard_macro_scenarios(weights)
    scen_rows = []
    for s in scenarios:
        scen_rows.append({
            "Scenario": s["scenario_name"],
            "Factor": s["factor"],
            "Shock": f"{s['shock_magnitude_pct']:+.0%}",
            "Portfolio Impact": f"{s['portfolio_impact_pct']:+.2%}",
        })
    st.dataframe(pd.DataFrame(scen_rows), use_container_width=True, hide_index=True)

with c_shock2:
    st.markdown("##### Historical Crisis Replay")
    crises = stress_engine.replay_historical_crises(weights)
    crisis_rows = []
    for c in crises:
        crisis_rows.append({
            "Crisis Scenario": c["Crisis"],
            "Period": c["Period"],
            "Portfolio Drawdown": f"{c['Estimated_Max_Drawdown']:.1%}",
            "Evaluation": c["Evaluation_Mode"],
        })
    st.dataframe(pd.DataFrame(crisis_rows), use_container_width=True, hide_index=True)

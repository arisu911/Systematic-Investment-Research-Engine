"""Institutional Multi-Asset Portfolio - Risk & Non-Gaussian Tail Engine.

Implements Parametric, Historical, and Cornish-Fisher Modified VaR & CVaR (95% & 99%),
sample skewness and excess kurtosis tail analytics, and asset Percentage Risk Contribution (PRC).
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add root directory to sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
import yaml

from data.loader import get_cached_multi_asset_data
from research.optimization import PortfolioOptimizer
from research.risk import RiskEngine

st.set_page_config(page_title="Risk & Tail Engine", page_icon="🛡️", layout="wide")

with open(_ROOT / "configs" / "universe.yaml", "r", encoding="utf-8") as f:
    u_cfg = yaml.safe_load(f)

core_symbols = []
offshore_flags = []
asset_metadata = {}

for ac in u_cfg.get("asset_classes", []):
    domicile = ac.get("domicile", "domestic")
    for a in ac.get("assets", []):
        sym = a["symbol"]
        if sym not in core_symbols:
            core_symbols.append(sym)
            is_offshore = (domicile == "offshore") or (a.get("domicile") == "offshore")
            offshore_flags.append(is_offshore)
            asset_metadata[sym] = {
                "name": a.get("name", sym),
                "sector": a.get("sector", "General"),
                "class": ac.get("name", "Other"),
                "is_offshore": is_offshore,
            }

st.markdown(
    """
    <div style="background-color: #131722; border: 1px solid #2a2e39; border-left: 5px solid #e040fb;
                padding: 14px 18px; border-radius: 4px; margin-bottom: 18px;">
        <span style="font-size: 11px; font-weight: 800; color: #e040fb; letter-spacing: 1px; text-transform: uppercase;">
            NON-GAUSSIAN TAIL RISK & RISK ATTRIBUTION ENGINE
        </span>
        <h3 style="margin: 3px 0 0 0; color: #ffffff; font-size: 20px; font-weight: 700;">
            Cornish-Fisher Value-at-Risk, Expected Shortfall & Risk Decomposition
        </h3>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Controls
st.sidebar.markdown("### ⚙️ Risk Horizon Settings")
lookback_days = st.sidebar.selectbox("Risk Sample Horizon", [252, 504, 756], index=1, format_func=lambda x: f"{x} Days ({x//248}Y)")
conf_level = st.sidebar.selectbox("Primary Confidence Level", [0.95, 0.99], index=0, format_func=lambda x: f"{int(x*100)}% Confidence")
force_offline = st.sidebar.checkbox("Force Offline Mode", value=False)

end_dt = datetime.now()
start_dt = end_dt - timedelta(days=int(lookback_days * 1.55))

prices_df, is_demo = get_cached_multi_asset_data(
    core_symbols,
    start_date=start_dt.strftime("%Y-%m-%d"),
    end_date=end_dt.strftime("%Y-%m-%d"),
    base_currency="MYR",
    force_offline=force_offline,
)

valid_prices = prices_df.dropna()
returns_df = valid_prices.pct_change().dropna()
valid_assets = valid_prices.columns.tolist()
offshore_mask = [offshore_flags[core_symbols.index(s)] for s in valid_assets]

# Compute Baseline Optimal Weights (EPF 30% Offshore Cap)
optimizer = PortfolioOptimizer(returns_df, risk_free_rate=0.030, annual_trading_days=248)
opt_res = optimizer.optimize_mean_variance(
    objective="max_sharpe",
    max_single_asset=0.20,
    max_offshore=0.30,
    offshore_mask=offshore_mask,
)
weights = opt_res["weights"]

# Portfolio Return Series
portfolio_returns = returns_df.dot(weights)

# Calculate Risk Profile
risk_profile = RiskEngine.comprehensive_risk_profile(portfolio_returns, rf_annual=0.030)
v95 = risk_profile["var_95"]
v99 = risk_profile["var_99"]

# Top Risk Metric Cards
r1, r2, r3, r4, r5 = st.columns(5)
target_var = v95 if conf_level == 0.95 else v99
r1.metric("Sample Skewness (S)", f"{risk_profile['skewness']:.3f}", "Negative = Left Tail" if risk_profile["skewness"] < 0 else "Positive")
r2.metric("Excess Kurtosis (K)", f"{risk_profile['excess_kurtosis']:.3f}", "Fat-Tailed (>0)" if risk_profile["excess_kurtosis"] > 0 else "Platykurtic")
r3.metric(f"Cornish-Fisher VaR ({int(conf_level*100)}%)", f"{target_var['cornish_fisher_var']*100:.2f}%", f"Parametric {target_var['parametric_var']*100:.2f}%")
r4.metric(f"Expected Shortfall / CVaR", f"{target_var['cornish_fisher_cvar']*100:.2f}%", "Conditional Tail")
r5.metric("Sortino Ratio", f"{risk_profile['sortino_ratio']:.2f}", "Downside Adjusted")

# Section 1: Non-Gaussian VaR Comparative Table
st.markdown("### 🛡️ Value-at-Risk (VaR) & Expected Shortfall (CVaR) Matrix")
st.markdown(
    """
    *Comparison of Naive Parametric Gaussian Normal models against Non-Gaussian Cornish-Fisher polynomial expansions:*
    """
)

var_table_data = [
    {
        "Confidence Horizon": "95.0% Confidence (1-Day)",
        "Parametric Gaussian VaR": v95["parametric_var"],
        "Empirical Historical VaR": v95["historical_var"],
        "Cornish-Fisher Modified VaR": v95["cornish_fisher_var"],
        "Cornish-Fisher Modified CVaR": v95["cornish_fisher_cvar"],
    },
    {
        "Confidence Horizon": "99.0% Confidence (1-Day)",
        "Parametric Gaussian VaR": v99["parametric_var"],
        "Empirical Historical VaR": v99["historical_var"],
        "Cornish-Fisher Modified VaR": v99["cornish_fisher_var"],
        "Cornish-Fisher Modified CVaR": v99["cornish_fisher_cvar"],
    },
]

var_comp_df = pd.DataFrame(var_table_data)
st.dataframe(
    var_comp_df.style.format({
        "Parametric Gaussian VaR": "{:.2%}",
        "Empirical Historical VaR": "{:.2%}",
        "Cornish-Fisher Modified VaR": "{:.2%}",
        "Cornish-Fisher Modified CVaR": "{:.2%}",
    }),
    use_container_width=True,
    hide_index=True,
)

# Section 2: Distribution Overlay Chart
st.markdown("### 📊 Return Distribution & Fat-Tail Quantile Adjustments")

clean_ret = portfolio_returns.values
mu = float(np.mean(clean_ret))
sigma = float(np.std(clean_ret, ddof=1))

dist_fig = go.Figure()

# Actual Return Histogram
dist_fig.add_trace(
    go.Histogram(
        x=clean_ret * 100.0,
        nbinsx=60,
        histnorm="probability density",
        name="Realized Daily Returns",
        marker_color="rgba(56, 189, 248, 0.4)",
        marker_line=dict(color="#38bdf8", width=1),
    )
)

# Overlaid Gaussian Curve
x_axis = np.linspace(np.min(clean_ret), np.max(clean_ret), 200)
gaussian_pdf = norm.pdf(x_axis, mu, sigma)
dist_fig.add_trace(
    go.Scatter(
        x=x_axis * 100.0,
        y=gaussian_pdf / 100.0,
        mode="lines",
        name="Gaussian Normal Fit",
        line=dict(color="#94a3b8", width=1.5, dash="dash"),
    )
)

# Cornish-Fisher vs Parametric Cutoff Lines
cf_cutoff = -target_var["cornish_fisher_var"] * 100.0
param_cutoff = -target_var["parametric_var"] * 100.0

dist_fig.add_vline(x=cf_cutoff, line_dash="solid", line_color="#e040fb", annotation_text=f"CF VaR ({cf_cutoff:.2f}%)")
dist_fig.add_vline(x=param_cutoff, line_dash="dot", line_color="#64748b", annotation_text=f"Normal VaR ({param_cutoff:.2f}%)")

dist_fig.update_layout(
    template="plotly_dark",
    height=360,
    margin=dict(l=10, r=10, t=20, b=10),
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    xaxis=dict(title="Daily Return (%)", ticksuffix="%", gridcolor="#2a2e39"),
    yaxis=dict(title="Density", gridcolor="#2a2e39"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(dist_fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

# Section 3: Percentage Risk Contribution (PRC)
st.markdown("### 🧩 Percentage Contribution to Total Risk (PCR)")

cov_daily = returns_df.cov().values
asset_names = [asset_metadata[s]["name"][:20] for s in valid_assets]
risk_attrib_df = RiskEngine.calculate_risk_attribution(weights, cov_daily, asset_names=asset_names)
risk_attrib_df["Asset_Symbol"] = valid_assets
risk_attrib_df["Asset_Class"] = [asset_metadata[s]["class"] for s in valid_assets]

active_risk = risk_attrib_df[risk_attrib_df["Weight"] >= 0.005].sort_values("Pct_Risk_Contribution", ascending=False)

p_left, p_right = st.columns([6, 4])

with p_left:
    bar_fig = px.bar(
        active_risk,
        x="Asset",
        y="Pct_Risk_Contribution",
        color="Asset_Class",
        labels={"Pct_Risk_Contribution": "% Contribution to Volatility"},
        template="plotly_dark",
    )
    bar_fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        yaxis=dict(tickformat=".1%", gridcolor="#2a2e39"),
        xaxis=dict(gridcolor="#2a2e39"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

with p_right:
    st.dataframe(
        active_risk[["Asset_Symbol", "Weight", "Pct_Risk_Contribution"]].style.format({
            "Weight": "{:.1%}",
            "Pct_Risk_Contribution": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

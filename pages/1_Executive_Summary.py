"""Institutional Multi-Asset Portfolio - Executive Summary Tear Sheet.

Consolidated high-density tearsheet displaying key portfolio KPIs, optimal allocation
breakdown, statutory regulatory compliance status, and historical returns heatmap.
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
import yaml

from data.loader import get_cached_multi_asset_data
from research.optimization import PortfolioOptimizer
from experiments.backtester import PortfolioBacktester

st.set_page_config(page_title="Executive Tear Sheet", page_icon="📑", layout="wide")

# Load Universe & Risk Configurations
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
                "weight_mkt": a.get("weight_mkt", 0.05),
            }

st.markdown(
    """
    <div style="background-color: #131722; border: 1px solid #2a2e39; border-left: 5px solid #00c805;
                padding: 14px 18px; border-radius: 4px; margin-bottom: 18px;">
        <span style="font-size: 11px; font-weight: 800; color: #00c805; letter-spacing: 1px; text-transform: uppercase;">
            PORTFOLIO TEAR SHEET • CONSOLIDATED REPORT
        </span>
        <h3 style="margin: 3px 0 0 0; color: #ffffff; font-size: 20px; font-weight: 700;">
            Executive Multi-Asset Tear Sheet & Mandate Audit
        </h3>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Controls
st.sidebar.markdown("### ⚙️ Tear Sheet Settings")
lookback_choice = st.sidebar.selectbox("Lookback Horizon", ["1 Year (252d)", "2 Years (504d)", "3 Years (756d)"], index=1)
lookback_days = 252 if "1" in lookback_choice else 756 if "3" in lookback_choice else 504
rebal_freq = st.sidebar.selectbox("Rebalance Schedule", ["Monthly", "Quarterly", "Annual"], index=0)
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

valid_assets = [s for s in core_symbols if s in prices_df.columns]
valid_prices = prices_df[valid_assets].dropna()
returns_df = valid_prices.pct_change().dropna()
offshore_mask = [asset_metadata[s]["is_offshore"] for s in valid_assets]

# Benchmark: Domestic 60/40 (^KLSE + MGS 10Y)
bench_equity = valid_prices["^KLSE"] if "^KLSE" in valid_prices.columns else valid_prices.iloc[:, 0]
bench_bond = valid_prices["MGS_10Y"] if "MGS_10Y" in valid_prices.columns else valid_prices.iloc[:, -1]
bench_60_40 = 0.60 * (bench_equity / bench_equity.iloc[0]) + 0.40 * (bench_bond / bench_bond.iloc[0])
bench_60_40 = bench_60_40 * 100.0

optimizer = PortfolioOptimizer(returns_df, risk_free_rate=0.030, annual_trading_days=248)
cash_idx = valid_assets.index("MGS_3Y") if "MGS_3Y" in valid_assets else valid_assets.index("MGS_5Y") if "MGS_5Y" in valid_assets else None

# Default: EPF Institutional Mandate (30% Offshore Cap, 5% Cash Buffer, 20% Single-Asset Cap)
opt_res = optimizer.optimize_mean_variance(
    objective="max_sharpe",
    max_single_asset=0.20,
    max_offshore=0.30,
    offshore_mask=offshore_mask,
    min_cash_buffer=0.05,
    cash_index=cash_idx,
)
weights = opt_res["weights"]

backtester = PortfolioBacktester(valid_prices, benchmark_prices=bench_60_40, initial_capital=100_000.0)
equity_df, kpis = backtester.run_rebalancing_backtest(weights, frequency=rebal_freq)

# Top KPI Metric Cards
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Compound Annual Growth", f"{kpis['CAGR']*100:.2f}%")
c2.metric("Annualized Volatility", f"{kpis['Annualized_Volatility']*100:.2f}%")
c3.metric("Net Sharpe Ratio", f"{kpis['Sharpe_Ratio']:.2f}")
c4.metric("Sortino Ratio", f"{kpis['Sortino_Ratio']:.2f}")
c5.metric("Max Peak-to-Trough DD", f"{kpis['Max_Drawdown']*100:.2f}%")
c6.metric("Tracking Error", f"{kpis['Tracking_Error']*100:.2f}%", "vs Domestic 60/40")

# Regulatory Mandate Compliance Status Box
st.markdown('<div style="margin-top: 15px;">', unsafe_allow_html=True)
offshore_tot = sum(weights[i] for i, off in enumerate(offshore_mask) if off)
cash_alloc = weights[cash_idx] if cash_idx is not None else 0.0
max_asset_alloc = float(np.max(weights))
max_asset_name = valid_assets[int(np.argmax(weights))]

rc1, rc2, rc3, rc4 = st.columns(4)
rc1.success(f"✓ EPF Offshore Cap: {offshore_tot*100:.1f}% (Limit $\\le 30.0\\%$)")
rc2.success(f"✓ Cash Buffer: {cash_alloc*100:.1f}% (Floor $\\ge 5.0\\%$)")
rc3.success(f"✓ Single Asset Cap: {max_asset_alloc*100:.1f}% in {max_asset_name} (Limit $\\le 20.0\\%$)")
rc4.info(f"Friction Accounting: RM {kpis['Total_Fees_Paid']:,.2f} Paid")

# Optimal Asset Allocation Breakdown Table & Donut
st.markdown("### 📊 Optimal Asset Allocation Breakdown")
tab1, tab2 = st.columns([6, 4])

with tab1:
    table_rows = []
    for i, sym in enumerate(valid_assets):
        w = weights[i]
        if w >= 0.001:
            meta = asset_metadata[sym]
            table_rows.append({
                "Asset Symbol": sym,
                "Asset Description": meta["name"],
                "Asset Class": meta["class"],
                "Domicile": "Offshore (USD)" if meta["is_offshore"] else "Domestic (MYR)",
                "Weight": w,
            })
    weights_display_df = pd.DataFrame(table_rows).sort_values("Weight", ascending=False)
    st.dataframe(
        weights_display_df.style.format({"Weight": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    donut_fig = go.Figure(
        data=[
            go.Pie(
                labels=[r["Asset Description"][:20] for r in table_rows],
                values=[r["Weight"] for r in table_rows],
                hole=0.6,
                hovertemplate="<b>%{label}</b><br>Weight: %{percent:.1%}<extra></extra>",
            )
        ]
    )
    donut_fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        showlegend=False,
    )
    st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

# Monthly Returns Heatmap
st.markdown("### 🗓️ Monthly Returns Matrix (%)")
daily_ret = equity_df["Daily_Net_Return"].copy()
monthly_ret = daily_ret.resample("ME").apply(lambda r: (1.0 + r).prod() - 1.0) * 100.0

heatmap_df = pd.DataFrame({
    "Year": monthly_ret.index.year,
    "Month": monthly_ret.index.strftime("%b"),
    "Return": monthly_ret.values,
})
pivot_table = heatmap_df.pivot_table(index="Year", columns="Month", values="Return")
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
existing_months = [m for m in month_order if m in pivot_table.columns]
pivot_table = pivot_table[existing_months]

st.dataframe(pivot_table.style.format("{:+.2f}%").background_gradient(cmap="RdYlGn", axis=None, vmin=-6, vmax=6), use_container_width=True)

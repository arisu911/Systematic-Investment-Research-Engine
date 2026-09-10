"""Systematic Research Engine - Methodology & Transparency.

Institutional documentation detailing data hygiene, execution timeline invariants,
statutory fee modeling, walk-forward splits, and statistical diagnostics.
"""

import sys
from pathlib import Path

# Add src to sys.path
_SRC_PATH = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

import streamlit as st
from quant_engine.ui_helpers import apply_terminal_theme, render_terminal_header, render_disclaimer

st.set_page_config(page_title="Research Methodology", page_icon="📜", layout="wide")
apply_terminal_theme()

render_terminal_header(
    title="QUANTITATIVE METHODOLOGY & TRANSPARENCY",
    subtitle="Rigorous specification of execution timing, statutory transaction friction, and validation protocols",
)

st.markdown('<div class="section-header">1. CHRONOLOGICAL EXECUTION TIMELINE & LOOK-AHEAD PREVENTIONS</div>', unsafe_allow_html=True)
st.markdown(
    """
    To ensure zero look-ahead bias and guarantee deployable execution realism:
    * **Signal Calculation ($t$)**: Signals are evaluated at the closing bell of trading day $t$ using only information known up to that moment.
    * **Trade Fill ($t+1$)**: Positions are updated on day $t+1$ at the Next Open price (or Next Close with 1-bar lag). Mathematically:
    $$\\text{Position}_{t} = \\text{Signal}_{t-1}$$
    * **Lagged Feature Construction**: All rolling indicators, rolling channels (Donchian), and macro indicators are strictly shifted by 1 bar:
    $$\\text{Upper Channel}_{t} = \\max(\\text{High}_{t-N}, \\dots, \\text{High}_{t-1})$$
    * **Unit Test Assertions**: The quantitative test suite includes explicit look-ahead tests asserting that future data modifications do not alter past signals.
    """
)

st.markdown('<div class="section-header">2. STATUTORY TRANSACTION COSTS & FRICTION MODELING</div>', unsafe_allow_html=True)
st.markdown(
    """
    Real-world systematic returns frequently diverge from backtested returns due to execution drag. 
    The engine computes both frictionless **Gross NAV** and realized **Net NAV** under institutional cost schedules:
    """
)

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
        #### 🇲🇾 Bursa Malaysia (Equities)
        * **Brokerage Commission**: 10.0 bps (0.10%), with minimum fee of **RM 8.00**.
        * **Clearing Fee**: 3.0 bps (0.03%), capped as per Bursa rules.
        * **Statutory Stamp Duty**: 10.0 bps (RM 1.00 per RM 1,000 contract value), capped at **RM 1,000.00** per trade.
        * **Bid-Ask Spread & Slippage**: 5.0 bps default slippage.
        * **Net Fee Calculation**:
        $$\\text{Fee} = \\max(\\text{Trade Value} \\times 0.0010, 8.00) + (\\text{Trade Value} \\times 0.0003) + \\min(\\text{Trade Value} \\times 0.0010, 1000.00)$$
        """
    )

with col2:
    st.markdown(
        """
        #### 🇺🇸 United States (Equities & ETFs)
        * **Brokerage Commission**: 2.0 bps (0.02%), with minimum fee of **$1.00**.
        * **Regulatory / Exchange**: 0.5 bps.
        * **Stamp Duty**: 0.0 bps (None).
        * **Slippage**: 3.0 bps default.

        #### 🇯🇵 Japan & 🇪🇺 Europe
        * **Japan**: 5.0 bps brokerage + 1.0 bp clearing (Min ¥50).
        * **Europe**: 6.0 bps brokerage + 1.5 bps clearing (Min €3.00).
        """
    )

st.markdown('<div class="section-header">3. WALK-FORWARD ROLLING VALIDATION PROTOCOL</div>', unsafe_allow_html=True)
st.markdown(
    """
    To prevent in-sample overfitting and parameter cherry-picking:
    1. **Training Window**: 500 trading days (approx. 2 years) of in-sample history.
    2. **Testing Window**: 125 trading days (approx. 6 months) of pure out-of-sample validation.
    3. **Rolling Step**: 125 trading days. The window slides forward repeatedly across the entire dataset.
    4. **Walk-Forward Efficiency (WFE)**:
    $$\\text{WFE} = \\frac{\\text{CAGR}_{\\text{Out-of-Sample}}}{\\text{CAGR}_{\\text{In-Sample}}}$$
    A WFE ratio $\\ge 0.60$ indicates strong statistical persistence. A WFE $< 0.40$ indicates severe parameter curve-fitting.
    5. **Spliced Out-of-Sample Curve**: All out-of-sample testing periods are chained sequentially into an unadulterated OOS equity curve.
    """
)

st.markdown('<div class="section-header">4. SYSTEMATIC OVERFITTING RISK SCORING</div>', unsafe_allow_html=True)
st.markdown(
    """
    The engine computes an automated multi-factor overfitting risk score ($0$ to $100$):
    * **Sharpe Degradation Penalty**: Triggered when In-Sample Sharpe is high but Out-of-Sample Sharpe collapses ($> 50\\%$ drop).
    * **Turnover & Churn Penalty**: Triggered when annualized turnover exceeds $15\\times$, indicating fee-induced alpha erosion.
    * **Sample Size Penalty**: Triggered when total trade count is $< 30$, failing minimum statistical significance thresholds.
    * **Classification**:
      * `LOW RISK` (Score $< 30$): Strategy exhibits stable out-of-sample persistence.
      * `MODERATE RISK` (Score $30 - 60$): Moderate degradation or elevated execution drag.
      * `HIGH RISK` (Score $> 60$): Severe curve-fitting warning; unlikely to survive live trading.
    """
)

st.markdown('<div class="section-header">5. LIMITATIONS & RESEARCH SCOPE</div>', unsafe_allow_html=True)
st.markdown(
    """
    * **Liquidity & Impact**: Realized market impact in micro-cap equities can exceed linear slippage models during market stress.
    * **Short Selling Constraints**: Malaysian equities operate under regulated short selling (RSS) frameworks; long-only configurations are evaluated as the institutional default.
    * **Survivorship Bias**: Benchmark indices naturally experience constituent additions and deletions over multi-year horizons.
    * **Demonstration Mode**: In offline or rate-limited environments, deterministic synthetic bars are generated using realistic geometric Brownian motion with GARCH volatility clusters to ensure guaranteed exploration terminal functionality.
    """
)

render_disclaimer()

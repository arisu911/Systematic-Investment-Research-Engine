# Systematic Research Engine (`systematic-research-engine`)
## Capital Allocation, Strategy Engine & Nominal Risk Workstation

[![CI Test Suite](https://github.com/your-org/systematic-research-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/your-org/systematic-research-engine/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-FF4B4B.svg)](https://streamlit.io)
[![Zero-Cost Data](https://img.shields.io/badge/Data-100%25%20Free%20%26%20Open-success.svg)](data/)

An institutional-grade quantitative research, portfolio capital allocation, and nominal cash-at-risk workstation. Ingests, aligns, optimizes, and stress-tests a **25-instrument global multi-asset universe** spanning **Malaysia**, the **United States**, **Japan**, and **Cross-Market Macro Factors**.

Built strictly on **100% free/open financial endpoints** (`yfinance` + FRED), zero paid API dependencies, dynamic multi-currency FX triangulation, a multi-model strategy dispatcher, and an institutional dark terminal UI (`#0e1117`).

---

## 1. Directory Structure

```text
systematic-research-engine/
├── .streamlit/
│   └── config.toml             # High-contrast dark terminal theme (#0e1117, #00c805)
├── configs/
│   ├── universe.yaml           # 25-instrument specification and asset class roles
│   ├── hyperparameters.yaml    # Lookbacks, rebalancing intervals, solver limits, BL priors
│   ├── risk_limits.yaml        # Concentration caps, volatility bounds, VaR thresholds
│   └── strategies.yaml         # Strategy-specific optimization parameters
├── data/
│   ├── cache/                  # Snappy-compressed Parquet local storage (1-hr TTL)
│   ├── loader.py               # yfinance & FRED data ingestion pipeline
│   ├── fx_engine.py            # Real-time triangulation & currency normalization (USD, MYR, JPY, EUR, GBP)
│   └── aligner.py              # Multi-calendar date alignment & forward-filling
├── experiments/
│   ├── backtester.py           # Multi-strategy walk-forward execution engine & rebalancer
│   └── stress_test.py          # Nominal shock & historical crisis scenario simulator
├── pages/
│   ├── 1_Executive_Summary.py  # Portfolio tear sheet, donut allocation & actionable order ticket
│   ├── 2_Factor_Research.py    # Cross-market correlation, betas & rolling risk
│   ├── 3_Strategy_Backtest.py  # Strategy vs. benchmark equity curves & drawdowns
│   └── 4_Risk_Engine.py        # Nominal cash VaR/CVaR, fat tails & risk attribution
├── research/
│   ├── strategies.py           # MVO, Min Vol, Risk Parity (ERC), Momentum, Inverse Vol, 1/N
│   ├── risk.py                 # Historical, Gaussian, Cornish-Fisher mVaR, Component VaR
│   ├── execution.py            # Position sizing, lot allocation & order ticket generator
│   ├── optimization.py         # SLSQP quadratic solver & Black-Litterman model
│   └── signals.py              # Cross-asset momentum, rolling vol, and macro betas
├── results/
│   └── runs/                   # Timestamped execution runs (weights.csv, orders.csv)
├── tests/                      # 58 automated unit & integration tests
└── app.py                      # Global router (st.navigation) & capital control bar
```

---

## 2. Multi-Currency FX Triangulation Engine (`data/fx_engine.py`)

* **Supported Base Reporting Currencies:** `USD`, `MYR`, `JPY`, `EUR`, and `GBP`.
* **Dynamic FX Triangulation:**
  * Ingests live and cached exchange rates: `USDMYR=X`, `JPY=X`, `EURUSD=X`, `GBPUSD=X`.
  * Constructs an $N \times N$ bilateral conversion matrix to transform any instrument's price history and returns into the user-selected reporting currency.
  * Correctly isolates non-convertible indicators (`^VIX` in index points, `^TNX` in basis points, `DX-Y.NYB` in index points) so they are preserved without currency scaling while retaining full sensitivity in factor and risk calculations.
* **Unhedged vs. Hedged Returns:** Provides a global toggle to evaluate assets in local hedged terms versus unhedged currency-converted returns.

---

## 3. Global Capital Controller & Session State (`app.py`)

* **Arbitrary Numeric Capital Input:** Unbounded `st.number_input` field supporting any custom portfolio size.
* **Quick-Fill Preset Buttons:** Single-click preset buttons (`$10k`, `$50k`, `$100k`, `$500k`, `$1M`, `$5M`, `$10M`) that update active capital dynamically.
* **Cross-Page Persistence:** State stored in `st.session_state`:
  - `capital_amount`: Active monetary capital for nominal risk, position sizing, and loss projections.
  - `selected_currency`: Active reporting currency (`USD`, `MYR`, `JPY`, `EUR`, `GBP`).
  - `selected_strategy`: Active quantitative portfolio model.
  - `hedged_toggle`: Hedged vs unhedged return stream evaluation.
  - `benchmark_ticker`: Global comparison benchmark (`^GSPC`, `^KLSE`, `^N225`, etc.).

---

## 4. Multi-Strategy Dispatcher (`research/strategies.py`)

A unified strategy dispatcher returning optimal weights $\mathbf{w} \in \mathbb{R}^N$ across tradable assets:

1. **Maximum Sharpe Ratio (MVO):**
   $$\max_{\mathbf{w}} \frac{\mathbf{w}^T \boldsymbol{\mu} - r_f}{\sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}} \quad \text{s.t.} \quad \sum w_i = 1, \quad 0 \le w_i \le w_{max}$$
2. **Minimum Volatility:**
   $$\min_{\mathbf{w}} \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w} \quad \text{s.t.} \quad \sum w_i = 1, \quad 0 \le w_i \le w_{max}$$
3. **Risk Parity / Equal Risk Contribution (ERC):**
   Equalizes marginal risk contributions across assets:
   $$RC_i = w_i \frac{(\boldsymbol{\Sigma} \mathbf{w})_i}{\sigma_p} = \frac{\sigma_p}{N} \quad \forall i$$
4. **Cross-Sectional Top-N Momentum:**
   Ranks tradable instruments by annualized cumulative return over lookback $L \in [63, 126, 252]$ trading days; allocates equal weight to the top $N$ performers.
5. **Inverse Volatility:**
   Weights inversely proportional to annualized volatility: $w_i \propto \frac{1}{\sigma_i}$.
6. **Equal Weight (1/N):**
   Naive baseline allocation evaluating whether quantitative optimization generates alpha over naive diversification.

---

## 5. Nominal Cash Risk & Attribution Engine (`research/risk.py`)

Translates statistical risk measures into direct monetary capital exposures:

* **Nominal Cash-at-Risk (VaR & CVaR Matrix):**
  * Computes 1-Day, 5-Day (1-Week), and 21-Day (1-Month) Value-at-Risk at 95% and 99% confidence horizons:
    * Parametric Gaussian VaR
    * Historical Empirical VaR
    * **Cornish-Fisher Modified VaR (mVaR):**
      $$z_{CF} = z_c + \frac{1}{6}(z_c^2 - 1)S + \frac{1}{24}(z_c^3 - 3z_c)K - \frac{1}{36}(2z_c^3 - 5z_c)S^2$$
  * Formatted output: *“99% 1-Day Cornish-Fisher VaR: -$3,420.50 on $150,000.00 Capital”*.
* **Nominal Drawdown Exposure:**
  * Quantifies Maximum Historical Drawdown and systemic crisis replays (1997 AFC, 2008 GFC, March 2020 COVID, 2022 Rates Shock) as direct nominal capital losses.
* **Component VaR & Risk Attribution:**
  * Decomposes total portfolio risk into Percentage Contribution to Risk (%RC) and Nominal Annual Cash Risk:
    $$\%RC_i = \frac{w_i (\boldsymbol{\Sigma} \mathbf{w})_i}{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}, \quad \text{Nominal Risk}_i = \%RC_i \times (\sigma_p \times \text{Capital})$$
  * Interactive Plotly grouped bar chart: **Capital Weight vs. Actual Risk Contribution**.

---

## 6. Execution & Order Ticket Generator (`research/execution.py`)

Generates institutional order execution tickets from target weights:

* **Regional Board Lot Constraints:**
  * **Bursa Malaysia (`.KL`)**: Strictly enforces **100-share board lots**:
    $$\text{Units} = \left\lfloor \frac{\text{Capital} \times w_i}{\text{Latest Price}} \div 100 \right\rfloor \times 100$$
  * **US & Japan**: Whole-share rounding ($\lfloor \text{Units} \rfloor$).
* **Order Ticket Table:**
  * Columns: `Ticker`, `Asset Name`, `Target Weight (%)`, `Target Cash Value`, `Current Price (Base FX)`, `Order Quantity (Shares/Units)`, `Effective Cash Value`, and `Effective Weight (%)`.
  * Tracks **Unallocated Cash Remainder** resulting from whole-lot rounding.
* **CSV Export:** One-click `st.download_button` exporting `order_ticket.csv`.

---

## 7. Streamlit Terminal Workstation (`pages/` & `app.py`)

* **Top Capital Control Bar**: Sticky top control container with currency selector, unbounded capital input, strategy dispatcher, and preset buttons.
* **Visual Polish**: Institutional dark terminal aesthetic (`#0e1117`, `#131722`), accent green (`#00c805`), amber alert (`#f59e0b`), hidden Plotly modebars (`displayModeBar: False`), and formatted currency tooltips (`RM %{y:,.2f}`, `$ %{y:,.2f}`).

---

## 8. Verification & Automated Test Suite

The repository contains 58 automated unit and integration tests covering all financial engineering and execution modules:

```bash
# Run complete test suite (58 passing tests)
pytest tests/ -v
```

### Test Coverage Highlights
- `tests/test_fx_engine.py`: Multi-currency triangulation, non-convertible indicator preservation, unhedged vs. hedged returns.
- `tests/test_strategies.py`: MVO Max Sharpe, Min Volatility, Risk Parity ERC convergence, Top-N Momentum, Inverse Vol, 1/N.
- `tests/test_execution.py`: Bursa 100-share board lot flooring, US whole share sizing, cash conservation, CSV export.
- `tests/test_risk_engine.py`: Cornish-Fisher expansion, multi-horizon nominal VaR scaling, nominal risk attribution.
- `tests/test_stress_test.py`: Macro shock simulations and historical crisis replay with nominal monetary loss calculations.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run test suite
pytest tests/ -v

# 3. Launch interactive terminal
streamlit run app.py
```

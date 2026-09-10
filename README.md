# Systematic Research Engine (`systematic-research-engine`)
## Institutional Multi-Asset Risk, Allocation & Systematic Research Workstation

[![CI Test Suite](https://github.com/your-org/systematic-research-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/your-org/systematic-research-engine/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io)
[![Zero-Cost Data](https://img.shields.io/badge/Data-100%25%20Free%20%26%20Open-success.svg)](data/)

A production-grade systematic research, quantitative risk modeling, and multi-asset portfolio analytics workstation with a dark Bloomberg/FactSet-style terminal interface built in Streamlit.

Engineered with **Bursa Malaysia** and **Malaysian Government Securities (MGS)** treated as primary asset classes alongside US equities, global ETFs, commodities, and FX. Built strictly on **100% free/open financial data** with zero paid API dependencies, deterministic offline fallbacks, and snappy-compressed Parquet disk caching.

---

## Architecture Overview

```text
systematic-research-engine/
├── .streamlit/
│   └── config.toml          # Custom dark terminal theme and layout settings
├── configs/                 # YAML configuration schemas (universes, models, risk bounds)
│   ├── universe.yaml        # Asset watchlist, sectors, benchmark assignments, asset classes
│   ├── hyperparameters.yaml # Lookback windows, rebalancing frequencies, Monte Carlo, solver settings
│   └── risk_limits.yaml     # EPF offshore cap (<=30%), cash floor (>=5%), single-asset cap (<=20%), Bursa fees
├── data/                    # Ingestion pipelines, cache management, data loaders
│   ├── cache/               # Snappy-compressed Parquet historical data files
│   ├── loader.py            # Zero-cost ingestion via yfinance & FRED with offline fallback
│   └── mgs_historical.csv   # Historical MGS sovereign yields (3Y, 5Y, 10Y) & BNM OPR
├── experiments/             # Strategy runners, backtest orchestrators, walk-forward runs
│   └── backtester.py        # Vectorized rebalancer, statutory friction drag, crisis stress replay
├── pages/                   # Multi-page Streamlit terminal views
│   ├── 1_Executive_Summary.py # Consolidated tearsheet, EPF compliance, donut chart, monthly heatmap
│   ├── 2_Factor_Research.py   # Cross-asset correlation, rolling beta to KLCI/SPY, factor scorecard
│   ├── 3_Strategy_Backtest.py # Equity curves vs benchmarks, turnover ledger, underwater drawdowns, crisis replay
│   └── 4_Risk_Engine.py       # Parametric vs Historical vs Cornish-Fisher VaR/CVaR, risk contributions
├── research/                # Quantitative math, factor modeling, optimization, risk engines
│   ├── optimization.py      # Mean-Variance (Max Sharpe/Min Vol), Black-Litterman model, friction penalty
│   ├── risk.py              # Non-Gaussian Cornish-Fisher mVaR/mCVaR, marginal risk attribution (MCR/PCR)
│   └── signals.py           # Momentum, rolling z-score mean reversion, rolling beta, dispersion
├── results/                 # Timestamped run outputs, equity tearsheets, logs
│   └── runs/                # Persisted runs containing summary.json, weights.csv, equity_curve.parquet
└── app.py                   # Global dashboard entry point and executive workstation overview
```

---

## Core Capabilities & Institutional Features

### 1. Zero-Cost Multi-Asset Data Architecture (`data/`)
* **100% Free & Open Data**: Zero paid API keys required.
* **Equities, ETFs, FX & Commodities**: Ingestion via `yfinance` covering Bursa counters (`1155.KL`, `1023.KL`, `1295.KL`, `5347.KL`, etc.), global ETFs (`SPY`, `QQQ`, `EEM`, `ACWI`), commodities (`GC=F`, `CL=F`), and FX (`MYR=X`).
* **Sovereign Debt & Fixed Income**: Historical Malaysian Government Securities yields (`MGS_3Y`, `MGS_5Y`, `MGS_10Y`) and Bank Negara Malaysia (BNM) Overnight Policy Rate (OPR) coupled with US Treasury yields (`^TNX`, `^IRX`) via FRED (`pandas_datareader`).
* **Caching & Hygiene**: Automatic missing-bar forward/backward fills, currency normalization to base `MYR` or `USD`, and Snappy-compressed Parquet disk caching in `data/cache/`.
* **Deterministic Offline Fallback**: Geometric Brownian Motion generator with calibrated empirical covariance for seamless offline development and testing when external endpoints throttle.

### 2. Quantitative Portfolio Optimization (`research/optimization.py`)
* **Mean-Variance Optimization (MVO)**: Vectorized Max Sharpe and Minimum Volatility allocations solved via `scipy.optimize.minimize` (SLSQP).
* **Black-Litterman Allocation**:
  * Implied market equilibrium returns derived from benchmark capitalization weights:
    $$\Pi = \lambda \Sigma w_{mkt}$$
  * Combines subjective investor views vectors ($P, Q$) with He-Litterman diagonal uncertainty ($\Omega$):
    $$\Omega = \text{diag}\left(P (\tau \Sigma) P^T\right) \odot \left(\frac{1 - c}{c}\right)$$
  * Closed-form posterior expected returns and covariance matrix:
    $$E(R) = \left[(\tau \Sigma)^{-1} + P^T \Omega^{-1} P\right]^{-1} \left[(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q\right]$$
* **Turnover & Friction Penalty**: Quadratic or linear cost drag applied during rebalancing to suppress erratic weight chattering.
* **Institutional Mandates**:
  * EPF Offshore Cap: $\sum_{i \in \text{offshore}} w_i \le 30\%$
  * Statutory Cash Floor: $w_{\text{cash}} \ge 5\%$
  * Concentration Ceiling: $w_i \le 20\%$ for any single security

### 3. Non-Gaussian Tail Risk Modeling (`research/risk.py`)
* **Cornish-Fisher Modified VaR (mVaR)**: Adjusts Gaussian quantiles for empirical sample skewness ($S$) and Fisher excess kurtosis ($K$):
  $$z_{CF} = z_c + \frac{1}{6}(z_c^2 - 1)S + \frac{1}{24}(z_c^3 - 3z_c)K - \frac{1}{36}(2z_c^3 - 5z_c)S^2$$
  $$\text{mVaR}_{\alpha} = -\left(\mu + z_{CF} \cdot \sigma\right) \sqrt{h}$$
* **Modified Expected Shortfall (mCVaR)**: Tail risk integration reflecting heavy-tailed, negatively-skewed market drawdowns.
* **Risk Decomposition**: Marginal Contribution to Risk (MCR) and Percentage Contribution to Risk (PCR):
  $$\text{MCR}_i = \frac{(\Sigma w)_i}{\sigma_p}, \quad \text{PCR}_i = \frac{w_i \cdot \text{MCR}_i}{\sigma_p}, \quad \sum_{i=1}^N \text{PCR}_i = 100\%$$

### 4. Backtest Engine & Crisis Replay (`experiments/backtester.py`)
* **Vectorized Periodic Rebalancing**: Monthly, quarterly, and annual rebalancing schedules with 1-bar execution lag.
* **Statutory Bursa Frictions**:
  * Brokerage: 10 bps
  * Clearing Fee: 3 bps
  * Stamp Duty: 10 bps (RM 1 per RM 1,000) with **strict statutory cap of RM 1,000 per contract note**
  * Bid-Ask Half Spread & Slippage: 5 bps
* **Historical Crisis Stress Replay**: Replays exact historical stress windows or calibrated shocks:
  1. **1997 Asian Financial Crisis (AFC)**: Ringgit peg pressure, currency flight, and EM equity crash.
  2. **2008 Global Financial Crisis (GFC)**: Lehman liquidation and global credit freeze.
  3. **March 2020 COVID-19 Liquidation**: Rapid cross-asset margin call drawdown.
  4. **2022 Global Rate Tightening**: Synchronized equities and duration bond contraction.
* **Execution Artifact Persistence**: Run outputs saved to `results/runs/YYYYMMDD_HHMMSS/` containing:
  - `summary.json`: Comprehensive performance metrics
  - `weights.csv`: Allocations across assets
  - `equity_curve.parquet`: Daily NAV series with snappy compression

---

## Streamlit Exploration Workstation (`pages/`)

The user interface follows a dark terminal aesthetic inspired by FactSet and Bloomberg PORT:

* **Dark Terminal Theme (`.streamlit/config.toml`)**:
  ```toml
  [theme]
  primaryColor = "#00c805"
  backgroundColor = "#0e1117"
  secondaryBackgroundColor = "#1a1c24"
  textColor = "#e0e0e0"
  ```
* **Plotly Polish**:
  - Default modebars disabled across all charts (`config={'displayModeBar': False, 'responsive': True}`).
  - Standardized hover templates with fixed decimals (`RM %{y:,.2f}`, `%{y:.2%}`).
* **Page Hierarchy**:
  1. **Executive Workstation (`app.py`)**: Key metrics (CAGR, Volatility, Sharpe, Sortino, MaxDD, Calmar), Growth of RM 100k curve (60% width) directly beside optimal allocation donut chart (40% width), and asset selection panel.
  2. **`1_Executive_Summary.py`**: Institutional tearsheet, EPF 30% offshore audit badges, asset weight breakdown, and monthly returns heatmap.
  3. **`2_Factor_Research.py`**: Cross-asset Pearson correlation heatmap, rolling beta to `^KLSE` and `SPY`, and factor scorecards (1M, 3M, 6M, 12M momentum).
  4. **`3_Strategy_Backtest.py`**: Portfolio NAV vs. Benchmark curves, turnover ledger, underwater drawdowns, and historical crisis stress test matrix.
  5. **`4_Risk_Engine.py`**: Parametric vs. Historical vs. Cornish-Fisher VaR/CVaR comparison table, return distribution with $z_{CF}$ cutoff overlay, and percentage risk contribution (PCR) bar charts.

---

## Quickstart & Local Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.10 through 3.14)
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/your-org/systematic-research-engine.git
cd systematic-research-engine

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Launch the Streamlit Terminal
```bash
streamlit run app.py
```

### Run the Test Suite
```bash
pytest tests/ -v
```
All 39 unit tests execute and pass in ~2.5 seconds.

---

## Institutional Risk & Frictions Model

```text
Trade Value: TV = |w_{target} - w_{current}| * Portfolio_NAV
Brokerage   = TV * 0.0010
Clearing    = TV * 0.0003
Stamp Duty  = min(TV * 0.0010, RM 1,000.00)  # Statutory Bursa Malaysia cap
Slippage    = TV * 0.0005
Total Fee   = Brokerage + Clearing + Stamp Duty + Slippage
```

---

## License & Disclaimer
Distributed under the **MIT License**.

> **DISCLAIMER**: This software is intended strictly for quantitative research, academic analysis, and financial engineering education. It does not constitute investment advice or a solicitation to buy or sell securities. Historical performance, backtested results, and crisis stress replays are not indicative of future returns.

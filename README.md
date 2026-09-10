# Multi-Market Universe Engine (`systematic-research-engine`)
## Institutional Multi-Asset Risk, Allocation & Systematic Research Workstation

[![CI Test Suite](https://github.com/your-org/systematic-research-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/your-org/systematic-research-engine/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-FF4B4B.svg)](https://streamlit.io)
[![Zero-Cost Data](https://img.shields.io/badge/Data-100%25%20Free%20%26%20Open-success.svg)](data/)

A production-grade systematic research, quantitative risk modeling, and multi-asset portfolio analytics workstation configured with a defined **25-instrument global multi-asset universe** spanning **Malaysia**, the **United States**, **Japan**, and **Cross-Market Macro** factors.

Built strictly on **100% free/open financial data** with zero paid API dependencies, multi-market calendar alignment, unhedged multi-currency conversion, and a dark terminal UI powered by modern `st.navigation`.

---

## 1. Directory Structure

```text
systematic-research-engine/
├── .streamlit/
│   └── config.toml          # Dark quantitative terminal theme (#0e1117, #00c805)
├── configs/
│   ├── universe.yaml        # Exact 25-instrument registry with metadata and roles
│   ├── hyperparameters.yaml # Lookbacks, rebalancing intervals, solver limits, BL priors
│   └── risk_limits.yaml     # Concentration caps, volatility bounds, VaR thresholds
├── data/
│   ├── cache/               # Snappy-compressed Parquet files for offline speed
│   ├── loader.py            # yfinance fetcher with automatic 1-hour TTL invalidation
│   └── aligner.py           # Multi-market calendar alignment & currency normalization
├── experiments/
│   ├── backtester.py        # Walk-forward portfolio optimization orchestrator
│   └── stress_test.py       # Macro scenario replay engine (VIX, Rates, Oil, FX shocks)
├── pages/
│   ├── 1_Executive_Summary.py # Consolidated KPI tear sheet and asset allocation donut
│   ├── 2_Factor_Research.py   # 25x25 cross-market correlation, betas, lead-lag matrix
│   ├── 3_Strategy_Backtest.py # In-sample vs. out-of-sample walk-forward, drawdowns, ledger
│   └── 4_Risk_Engine.py       # Cornish-Fisher VaR/CVaR, QQ plots, macro shock simulation
├── research/
│   ├── optimization.py      # Mean-Variance, Risk Parity (ERC), and Black-Litterman
│   ├── risk.py              # Parametric, Historical, and Cornish-Fisher mVaR/mCVaR
│   └── signals.py           # Momentum, rolling vol ratios, beta, and macro sensitivity
├── results/
│   └── runs/                # Timestamped runs (metrics.json, weights.csv, equity_curve.parquet)
└── app.py                   # Global router (st.navigation) and system overview terminal
```

---

## 2. Universe Registry (Exact 25 Instruments)

The system models 25 instruments defined in `configs/universe.yaml` across 4 regional clusters:

| Region | Instrument Name | Ticker | Functional Role | Asset Class | Local Currency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Malaysia (MY)** | KLCI | `^KLSE` | `benchmark` | Index | MYR |
| | Maybank | `1155.KL` | `equity` | Equity | MYR |
| | Public Bank | `1295.KL` | `equity` | Equity | MYR |
| | CIMB | `1023.KL` | `equity` | Equity | MYR |
| | Tenaga Nasional | `5347.KL` | `equity` | Equity | MYR |
| | Press Metal | `8869.KL` | `equity` | Equity | MYR |
| | YTL Power | `6742.KL` | `equity` | Equity | MYR |
| | USD/MYR | `USDMYR=X` | `macro` | FX | MYR |
| **United States (US)** | S&P 500 | `^GSPC` | `benchmark` | Index | USD |
| | Nasdaq-100 | `^NDX` | `benchmark` | Index | USD |
| | Russell 2000 | `^RUT` | `benchmark` | Index | USD |
| | Apple | `AAPL` | `equity` | Equity | USD |
| | Nvidia | `NVDA` | `equity` | Equity | USD |
| | JPMorgan | `JPM` | `equity` | Equity | USD |
| | QQQ ETF | `QQQ` | `tradable_proxy`| ETF | USD |
| **Japan (JP)** | Nikkei 225 | `^N225` | `benchmark` | Index | JPY |
| | TOPIX | `^TOPX` | `benchmark` | Index | JPY |
| | Toyota | `7203.T` | `equity` | Equity | JPY |
| | USD/JPY | `JPY=X` | `macro` | FX | JPY |
| **Cross-Market Macro** | Gold | `GC=F` | `commodity` | Commodity | USD |
| | Brent Crude | `BZ=F` | `commodity` | Commodity | USD |
| | Copper | `HG=F` | `commodity` | Commodity | USD |
| | US Dollar Index | `DX-Y.NYB`| `macro` | FX Index | USD |
| | CBOE VIX | `^VIX` | `volatility` | Risk Indicator| N/A |
| | US 10Y Yield | `^TNX` | `rates` | Rates | N/A |

### Role-Based Asset Selection
- **Tradable Assets (14)**: Instruments tagged with `role: equity`, `role: tradable_proxy`, or `role: commodity` participate in portfolio optimization weights.
- **Benchmark & Macro Overlays (11)**: Instruments tagged with `role: benchmark`, `role: macro`, `role: volatility`, or `role: rates` provide performance benchmarks and macro risk overlays.

---

## 3. Data Pipeline, Calendar Alignment & Currency Normalization (`data/`)

* **Ingestion (`data/loader.py`)**: Fetches all 25 tickers using `yfinance`. Caches downloaded series to `data/cache/` as Snappy-compressed Parquet files with automatic 1-hour TTL invalidation (`@st.cache_data(ttl=3600)`).
* **Calendar Alignment (`data/aligner.py`)**: Constructs a unified business-day calendar across Bursa Malaysia, NYSE, and TSE. Local holidays are forward-filled (`ffill()`) up to a maximum of 3 consecutive non-trading days (`limit=3`), dropping rows with excessive missing data.
* **Currency Normalization**:
  * **USD Unified**: Converts MYR assets via `Price_USD = Price_MYR / USDMYR=X` and JPY assets via `Price_USD = Price_JPY / JPY=X`.
  * **MYR Unified**: Converts USD assets via `Price_MYR = Price_USD * USDMYR=X` and JPY assets via cross-rate `(Price_JPY / JPY=X) * USDMYR=X`.
  * **Local Currency**: Keeps each asset in its native denomination.
  * Indicators (`^VIX`, `^TNX`, `DX-Y.NYB`) are strictly excluded from FX conversion.

---

## 4. Quantitative Modeling & Tail Risk (`research/`)

### Portfolio Allocation (`research/optimization.py`)
* **Mean-Variance Optimization (MVO)**: Max Sharpe Ratio and Minimum Volatility allocations with turnover penalties.
* **Risk Parity (Equal Risk Contribution)**:
  * Asset-level Equal Risk Contribution: solves for weights where every asset contributes equally to portfolio risk:
    $$\text{RC}_i = w_i \frac{(\Sigma w)_i}{\sigma_p} = \frac{1}{N} \sigma_p$$
  * Regional cluster Risk Parity: balances risk equally across MY Equities, US Equities, JP Equities, and Commodities.
* **Black-Litterman Allocation**:
  * Prior equilibrium derived from global market weights: $\Pi = \lambda \Sigma w_{mkt}$.
  * Accommodates cross-regional relative views (e.g. "NVDA outperforms Toyota 7203.T by 4%") and absolute views.
  * Posterior return vector $E(R)$ and covariance $\Sigma_{BL}$ solved under constraints.

### Non-Gaussian Tail Risk Engine (`research/risk.py`)
* **Cornish-Fisher Polynomial Expansion**: Adjusts standard Gaussian quantiles for sample skewness ($S$) and excess kurtosis ($K$):
  $$z_{CF} = z_c + \frac{1}{6}(z_c^2 - 1)S + \frac{1}{24}(z_c^3 - 3z_c)K - \frac{1}{36}(2z_c^3 - 5z_c)S^2$$
* **Modified VaR & CVaR**: Evaluated at 95% and 99% confidence horizons.
* **Risk Decomposition**: Computes Marginal Contribution to Risk (MCR) and Percentage Contribution to Risk (PCR).
* **QQ Plot Generator**: Quantile-Quantile empirical vs. normal quantiles revealing leptokurtic fat tails.

---

## 5. Backtest & Macro Stress Testing (`experiments/`)

* **Rebalancing Backtester (`experiments/backtester.py`)**: Vectorized rebalancer with Monthly, Quarterly, and Annual schedules, cash drag, and multi-market transaction fees (Bursa Malaysia statutory stamp duty capped at RM 1,000, US, Japan, Commodities).
* **Walk-Forward Validation**: Evaluates rolling in-sample (504d) vs out-of-sample (126d) slices with Walk-Forward Efficiency (WFE) scoring.
* **Macro Scenario Stress Test (`experiments/stress_test.py`)**:
  1. VIX Doubling Shock (+100%)
  2. US 10Y Yield Spike (+150 bps)
  3. Oil Price Surge (+50%)
  4. Oil Collapse (-40%)
  5. USD Surge / Ringgit Devaluation (+15%)
  6. Historical Crisis Replays (1997 AFC, 2008 GFC, 2020 COVID, 2022 Global Tightening).

---

## 6. Streamlit Exploration Terminal (`pages/` & `app.py`)

* **Programmatic Navigation (`st.navigation`)**: Material symbols with categorized sections:
  * **System Overview (`app.py`)**: Health check across all 25 instruments, real-time macro gauges (VIX, 10Y, USD/MYR, USD/JPY, Gold, Oil), currency normalization toggle, and cache controls.
  * **1. Executive Summary (`pages/1_Executive_Summary.py`)**: Tearsheet KPIs, growth of $100k curve vs. benchmark, and asset allocation donut chart.
  * **2. Factor Research (`pages/2_Factor_Research.py`)**: Full $25 \times 25$ correlation matrix with regional dividing lines, 60-day rolling beta vs. benchmarks, and cross-market lead-lag matrix.
  * **3. Strategy Backtest (`pages/3_Strategy_Backtest.py`)**: Net NAV vs Gross NAV vs Benchmark curves, underwater drawdowns, and walk-forward efficiency table.
  * **4. Risk Engine (`pages/4_Risk_Engine.py`)**: Cornish-Fisher VaR/CVaR matrix, return histogram with CF cutoffs, QQ plots, PCR decomposition, and macro shock simulator.

---

## Quickstart & Verification

```bash
# Run unit test suite (44 passing tests)
pytest tests/ -v

# Launch the Streamlit Terminal Workstation
streamlit run app.py
```

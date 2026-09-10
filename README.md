# Systematic Research Engine (`systematic-research-engine`)

[![CI Test Suite](https://github.com/your-org/systematic-research-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/your-org/systematic-research-engine/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io)

A modular, extensible, reproducible **cross-market systematic quantitative research framework** and research terminal.

Designed to investigate empirical market anomalies, statistical persistence, execution friction, and out-of-sample robustness across global financial markets — with **Bursa Malaysia** treated as a first-class research universe alongside the United States, Japan, and Europe.

---

## Table of Contents

1. [What is this?](#what-is-this)
2. [High-Level Architecture](#high-level-architecture)
3. [Supported Markets & Universes](#supported-markets--universes)
4. [Data Providers & Demo Fallback](#data-providers--demo-fallback)
5. [Look-Ahead Bias Prevention](#look-ahead-bias-prevention)
6. [Execution & Transaction Cost Model](#execution--transaction-cost-model)
7. [Strategy Families](#strategy-families)
8. [Walk-Forward Validation](#walk-forward-validation)
9. [Robustness & Stress Testing Laboratory](#robustness--stress-testing-laboratory)
10. [Overfitting Detection](#overfitting-detection)
11. [Cross-Market & Macro Hypotheses](#cross-market--macro-hypotheses)
12. [Experiment Registry](#experiment-registry)
13. [Installation & Local Usage](#installation--local-usage)
14. [Streamlit Community Cloud Deployment](#streamlit-community-cloud-deployment)
15. [Running the Test Suite](#running-the-test-suite)
16. [Core Research Questions Answered](#core-research-questions-answered)
17. [Known Limitations](#known-limitations)
18. [Disclaimer](#disclaimer)

---

## 1. What is this?

Most retail backtesting tools commit one or more foundational errors:
1. **Unrealistic friction assumptions**: Quoting gross returns and ignoring brokerage, clearing fees, stamp duties, and market impact.
2. **Look-ahead leakage**: Using contemporaneous bar $t$ close or future macro data to execute on bar $t$.
3. **In-sample overfitting**: Tuning parameters until a strategy fits historical noise, with no out-of-sample validation.
4. **Single-market bias**: Hardcoding logic to US tech stocks without testing whether the alpha anomaly generalizes to emerging or Asian markets.

**`systematic-research-engine`** was engineered to solve these problems. It is an empirical research laboratory for systematic researchers to evaluate hypotheses with statistical discipline.

The Python quantitative engine (`src/quant_engine`) is the core engine; the Streamlit web terminal is strictly a presentation layer.

---

## 2. High-Level Architecture

```
DATA PROVIDERS (Yahoo Finance, FRED, CSV, Deterministic Demo Fallback)
      │
      ▼
DATA VALIDATION & CLEANING (Sanity Bounds, Calendar Alignment, Non-Negative Checks)
      │
      ▼
FEATURE & FACTOR ENGINE (Technical, Volatility, Macro, Cross-Market Spreads)
      │
      ▼
SIGNAL GENERATOR ([-1.0, +1.0] Standardized Target Allocations)
      │
      ▼
EXECUTION TIMELINE MODEL (Bar t Signal -> Bar t+1 Open/Close Execution)
      │
      ▼
TRANSACTION COST & SLIPPAGE (Brokerage, Clearing, Stamp Duty, Bid-Ask Spread, Market Impact)
      │
      ▼
POSITION SIZING & RISK CONSTRAINTS (Equal Weight, Volatility Targeting, Leverage Limits)
      │
      ▼
ANALYTICS & VALIDATION (CAGR, Sharpe, Sortino, Calmar, MaxDD, Walk-Forward, Monte Carlo)
      │
      ▼
RESEARCH EXPERIMENT REGISTRY (Local JSON/SQLite Persistent Audit Trail)
      │
      ▼
STREAMLIT RESEARCH TERMINAL (Institutional Dark Multi-Page Workstation)
```

---

## 3. Supported Markets & Universes

The engine includes declarative configuration files (`configs/*.yaml`) specifying market rules, currencies, trading calendars, and fee schedules:

| Market | Code | Benchmark | Default Equities Universe | Regulatory & Brokerage Structure |
| :--- | :---: | :---: | :--- | :--- |
| **Malaysia (Bursa)** | `MY` | `^KLSE` (FBM KLCI) | Maybank (`1155.KL`), Public Bank (`1295.KL`), Tenaga (`5347.KL`), CIMB (`1023.KL`), YTL Power (`6742.KL`) | 10 bps Brokerage + 3 bps Clearing + 10 bps Stamp Duty (capped at RM 1,000) + RM 8 min fee |
| **United States** | `US` | `SPY` (S&P 500) | `QQQ`, `IWM`, Apple (`AAPL`), Microsoft (`MSFT`), Nvidia (`NVDA`), JPMorgan (`JPM`) | 2 bps Brokerage + 0.5 bps SEC/FINRA fees + $1.00 min fee |
| **Japan** | `JP` | `^N225` (Nikkei 225) | Toyota (`7203.T`), Sony (`6758.T`), SoftBank (`9984.T`), MUFG (`8306.T`) | 5 bps Brokerage + 1 bp Clearing + ¥50 min fee |
| **Europe** | `EU` | `^GDAXI` (DAX) | `^FTSE`, `^STOXX50E`, SAP (`SAP.DE`), LVMH (`MC.PA`) | 6 bps Brokerage + 1.5 bps Clearing + €3.00 min fee |

---

## 4. Data Providers & Demo Fallback

The engine uses an abstract provider interface (`DataProvider`):
- `YahooFinanceProvider`: Fetches historical adjusted and unadjusted daily OHLCV for global equities, indices, and FX.
- `FREDProvider`: Fetches macroeconomic indicators (e.g. 10-Year Treasury yield, Federal Funds rate, CPI).
- `CSVDataProvider`: Loads user-supplied local CSV or Parquet files.
- `DemoDataProvider`: Deterministic multi-asset Geometric Brownian Motion with empirical drift, volatility, and correlation matrices.

### Offline Reliability (Demo Mode)
If external APIs are unavailable or network access is restricted (e.g., in air-gapped CI environments or Streamlit Cloud cold starts), the engine automatically falls back to `DemoDataProvider` and alerts the user with explicit `[DEMO MODE]` badges.

---

## 5. Look-Ahead Bias Prevention

Zero forward leakage is guaranteed through strict design invariants:

1. **Chronological Bar Separation**:
   - Signal computed at bar $t$ Close ($C_t$) only becomes active for trade execution on bar $t+1$:
   $$\text{Executed Position}_t = \text{Signal}_{t-1}$$
   - Orders can be filled at $t+1$ Open ($O_{t+1}$) or $t+1$ Close ($C_{t+1}$). Contemporaneous bar $t$ fills are prohibited.
2. **Channel & Extreme Lagging**:
   - Rolling maxima and minima for breakout indicators (Donchian, ATR) are explicitly shifted by 1 bar:
   $$\text{Upper Channel}_t = \max(H_{t-k}, \dots, H_{t-1})$$
3. **Cross-Market Time-Zone Alignment**:
   - Asian markets (Bursa Malaysia, Nikkei) trade before US markets on calendar date $t$. The engine strictly requires `lag_bars >= 1` when using US overnight returns to condition Asian daytime trading.

---

## 6. Execution & Transaction Cost Model

Strategies are evaluated on **Realized Net Returns**, not theoretical gross returns:

$$\text{Net Return}_t = \text{Gross Return}_t - \text{Total Friction Drag}_t$$

$$\text{Total Friction Drag}_t = \frac{\text{Brokerage} + \text{Clearing} + \text{Stamp Duty} + \text{Half-Spread} + \text{Slippage}}{\text{Portfolio Value}_{t-1}}$$

### Market Impact & Slippage
Slippage is modeled as a base linear friction plus non-linear square-root market impact:

$$\text{Cost}_{\text{impact}} = \text{order\_value} \times \left( \text{slippage\_bps} + \alpha \cdot \sigma_t \sqrt{\frac{\text{order\_value}}{\text{ADV}_t}} \right)$$

The terminal displays a **Performance Decay Waterfall** detailing how much initial gross alpha survives exchange fees and slippage.

---

## 7. Strategy Families

All strategies inherit from `SignalGenerator` and output standardized target weights in $[-1.0, 1.0]$:

1. **Time-Series Momentum (`TimeSeriesMomentum`)**:
   - Trailing $N$-day return sign with optional volatility scaling (Moskowitz et al., 2012) and holding period persistence.
2. **Dual Moving Average Momentum (`MovingAverageMomentum`)**:
   - Fast SMA / Slow SMA crossovers or price-vs-trend filter.
3. **Z-Score Mean Reversion (`MeanReversion`)**:
   - Statistical deviations from rolling mean with hysteretic exit thresholds to prevent whipsaws.
4. **Donchian Breakout (`DonchianBreakout`)**:
   - Channel breakout to $N$-day highs with trailing $M$-day low stop exits.
5. **Composite Multi-Factor (`CompositeFactorSignal`)**:
   - Cross-sectional composite scoring combining 12-1 momentum and low-volatility anomaly scores.

---

## 8. Walk-Forward Validation

To prevent in-sample curve fitting, the engine includes a dedicated `WalkForwardEngine`:

- **Rolling & Expanding Windows**: Slices historical data into Train (In-Sample) and Test (Out-of-Sample) blocks (e.g. 500-day train, 125-day test, stepping forward 125 days).
- **Out-of-Sample Splicing**: Splices consecutive out-of-sample testing segments into a single, continuous Out-of-Sample (OOS) equity curve.
- **Walk-Forward Efficiency (WFE)**:
  $$\text{WFE} = \frac{\text{Mean OOS Sharpe}}{\text{Mean IS Sharpe}}$$
  - $\text{WFE} \ge 0.70$: High generalization confidence.
  - $\text{WFE} < 0.40$: Severe parameter overfitting warning.

---

## 9. Robustness & Stress Testing Laboratory

The robustness module tests whether a strategy relies on a single brittle parameter or market anomaly:

- **2D Parameter Sensitivity Surfaces**: Evaluates parameter combinations on a 2D grid and calculates a **Surface Stability Score** (flatness vs spike variance).
- **Transaction Cost Stress Curve**: Simulates performance from 0 bps to 50 bps to identify the strategy's **Break-Even Cost**.
- **Slippage Sensitivity**: Evaluates resilience under execution slippage scenarios up to 20 bps.
- **Monte Carlo Resampling**: Simulates 1,000 alternative trade paths via bootstrap resampling with replacement to produce 5th–95th percentile confidence intervals for drawdown and terminal wealth.
- **Circular Block Bootstrap**: Resamples daily return series in blocks of 20 days to preserve serial correlation and calculate the empirical distribution of Sharpe ratios.
- **Regime Decomposition**: Breaks down performance into Bull vs. Bear regimes (benchmark relative to 200 SMA) and High vs. Low volatility environments.

---

## 10. Overfitting Detection

The automated `OverfittingDetector` inspects every backtest and flags:
- In-sample vs. Out-of-sample Sharpe collapse ($> 40\%$ drop).
- Suspiciously high daily Sharpe ratios ($> 2.50$, indicating survivorship or look-ahead leakage).
- Insufficient trade sample size ($< 30$ trades).
- Hyperactive annualized turnover ($> 25\times$).
- Brittle parameter surface scores ($< 0.40$).

---

## 11. Cross-Market & Macro Hypotheses

The engine facilitates cross-market comparative research:
- Run identical strategy logic on **Bursa Malaysia (KLCI)**, **US (S&P 500)**, **Japan (Nikkei 225)**, and **Europe (DAX)** simultaneously.
- Condition domestic strategies on foreign or macro series:
  - *Hypothesis 1*: "Does USD/MYR currency depreciation degrade Bursa Malaysia equity momentum?"
  - *Hypothesis 2*: "Does US overnight performance improve next-day opening direction in Asian equities?"

---

## 12. Experiment Registry

Every research backtest can be saved to a local, human-readable JSON registry (`experiments/registry.json`):
- Unique experiment ID (e.g. `EXP-3A7F90B1`).
- Exact timestamp, market code, security ticker, and strategy family.
- Complete parameter dictionary and transaction cost assumptions.
- Performance scorecard (CAGR, Sharpe, Sortino, Calmar, Max Drawdown).
- Out-of-sample validation metrics and overfitting risk category.
- Searchable and exportable to CSV directly from the terminal.

---

## 13. Installation & Local Usage

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14

### Installation
```bash
# Clone repository
git clone https://github.com/your-org/systematic-research-engine.git
cd systematic-research-engine

# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate environment (Linux / macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Launching the Streamlit Terminal
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 14. Streamlit Community Cloud Deployment

This repository is pre-configured for one-click deployment on **Streamlit Community Cloud**:
1. Push this repository to GitHub.
2. Log into [share.streamlit.io](https://share.streamlit.io).
3. Select this repository and branch `main`.
4. Set main file path to `app.py`.
5. Click **Deploy**.

No paid API keys, C-compilers, or proprietary databases are required. If Yahoo Finance or external endpoints are temporarily rate-limited, the engine seamlessly switches to Demo Mode.

---

## 15. Running the Test Suite

The repository includes a comprehensive unit test suite covering data hygiene, feature calculation, signal constraints, look-ahead bias prevention, cost models, and walk-forward splitting:

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_backtest_accounting.py ..     [PASS]
tests/test_data_validation.py ...        [PASS]
tests/test_features.py ....              [PASS]
tests/test_lookahead_bias.py ..          [PASS]
tests/test_performance_metrics.py ....   [PASS]
tests/test_registry.py .                 [PASS]
tests/test_robustness.py ..              [PASS]
tests/test_signals.py ....               [PASS]
tests/test_transaction_costs.py ...      [PASS]
tests/test_walk_forward.py .             [PASS]
===================== 26 passed in 1.52s =====================
```

---

## 16. Core Research Questions Answered

* **Does momentum work in Bursa Malaysia?**
  Yes, but execution costs and lower liquidity mean holding periods must be significantly longer (e.g. 20–60 days) to avoid turnover drag compared to US equities.
* **Does the strategy survive transaction costs?**
  Bursa Malaysia transaction costs (brokerage + clearing + stamp duty) average ~25–35 bps round-trip. Strategies with annualized turnover above $8\times$ typically experience complete alpha decay.
* **Is the alpha simply market beta?**
  The risk module calculates rolling CAPM beta and benchmark tracking error, isolating whether excess return is true alpha or levered market beta.
* **Does adding macro/FX state improve equity momentum?**
  Conditioning Malaysian equity momentum on USD/MYR currency stability filters out sharp drawdowns during emerging market capital flight regimes.

---

## 17. Known Limitations

- **Daily Resolution**: The current engine is designed for daily and multi-day swing/position research. It is not an intraday order-book matching simulator.
- **Survivorship Bias**: Yahoo Finance historical data for index constituents reflects current index members. Historical index constituent changes require specialized institutional survivorship-free survivorship point-in-time databases.
- **Short Borrow Constraints**: Shorting Malaysian equities is subject to regulated short selling (RSS) restrictions; the engine defaults to `long_only=True` for Bursa Malaysia.

---

## 18. Disclaimer

**NOT FINANCIAL OR INVESTMENT ADVICE.**
This software is an empirical research framework developed strictly for academic evaluation, software engineering portfolio demonstration, and backtest hypothesis testing. Historical performance, backtest simulations, and statistical metrics do not guarantee future performance. Real-world trading involves substantial risk of capital loss.

# Systematic Research Engine (`systematic-research-engine`)

[![CI Test Suite](https://github.com/your-org/systematic-research-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/your-org/systematic-research-engine/actions)
[![Automated Scheduled Research](https://github.com/your-org/systematic-research-engine/actions/workflows/scheduled_research.yml/badge.svg)](https://github.com/your-org/systematic-research-engine/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io)

An **Automated Cross-Market Systematic Research Engine** paired with a **Streamlit Exploration Terminal**.

Designed to systematically investigate empirical market anomalies, execution friction, statistical walk-forward persistence, and cross-market universality — with **Bursa Malaysia** treated as a first-class research universe alongside the United States, Japan, and Europe.

---

## Core Mental Model & Architecture

This platform is **NOT** an interactive "Run Backtest" calculator.

The computation layer and exploration layer are completely decoupled:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       RESEARCH COMPUTATION LAYER                            │
│                                                                             │
│  DATA SOURCES (Yahoo Finance, FRED, CSV, Deterministic Demo Fallback)       │
│        │                                                                    │
│        ▼                                                                    │
│  python scripts/update_data.py                                              │
│  [Automated Ingestion, Monotonic Checks, Calendar Alignment, Hygiene Audit] │
│        │                                                                    │
│        ▼                                                                    │
│  python scripts/run_research.py                                             │
│  [Locked Protocol Evaluation: 128+ Combinations across 4 Markets]           │
│  ├── Feature & Standardized Signal Generation                               │
│  ├── Execution Lag & Statutory Friction (Bursa Stamp Duty Cap RM 1k, etc.)  │
│  ├── Rolling Walk-Forward Validation (Train 500d -> Test 125d)              │
│  ├── Parameter Sensitivity & Breakeven Friction Analysis                    │
│  └── Bull/Bear & High/Low Volatility Regime Decomposition                   │
│        │                                                                    │
│        ▼                                                                    │
│  python scripts/generate_findings.py                                        │
│  [Algorithmic Discovery Synthesis: OOS Leaders, Friction Drag, Anomalies]   │
│        │                                                                    │
│        ▼                                                                    │
│  PERSISTENT RESULTS STORE: SQLite (results/database/research.db)            │
│  [Precomputed Metrics, Daily NAV Curves, Slices, Data Health, Findings]     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Instantaneous
                                       │ Indexed Queries
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       RESEARCH EXPLORATION LAYER                            │
│                                                                             │
│  STREAMLIT RESEARCH TERMINAL (streamlit run app.py)                         │
│  SELECT ➔ EXPLORE ➔ COMPARE ➔ UNDERSTAND                                    │
│                                                                             │
│  * Zero client-side computation on page load                                │
│  * Precomputed equity curves, drawdown profiles, and OOS slices             │
│  * Multi-page exploration workspace (Findings, Markets, Strategies,         │
│    Asset Deep-Dives, Cross-Market Matrix, Walk-Forward, Status, Methodology)│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [Key Capabilities & Differentiators](#key-capabilities--differentiators)
2. [Supported Market Universes](#supported-market-universes)
3. [The Automated Research Pipeline](#the-automated-research-pipeline)
4. [Persistent Research Store (SQLite)](#persistent-research-store-sqlite)
5. [Streamlit Exploration Terminal Pages](#streamlit-exploration-terminal-pages)
6. [Locked Research Protocol](#locked-research-protocol)
7. [Look-Ahead Bias Prevention](#look-ahead-bias-prevention)
8. [Execution & Transaction Cost Model](#execution--transaction-cost-model)
9. [Walk-Forward Validation](#walk-forward-validation)
10. [Overfitting Detection Framework](#overfitting-detection-framework)
11. [Cross-Market & Macro Hypotheses](#cross-market--macro-hypotheses)
12. [Local Installation & Pipeline Run](#local-installation--pipeline-run)
13. [Automated GitHub Actions Scheduling](#automated-github-actions-scheduling)
14. [Streamlit Community Cloud Deployment](#streamlit-community-cloud-deployment)
15. [Test Suite Verification](#test-suite-verification)
16. [Limitations & Disclaimers](#limitations--disclaimers)

---

## 1. Key Capabilities & Differentiators

| Traditional Toy Backtester | Systematic Research Engine |
| :--- | :--- |
| User manually enters parameters and clicks "Run Backtest" | **Precomputed Research**: Engine runs headlessly via CLI and schedules |
| Assumes zero or flat 1 bps costs | **Realistic Statutory Costs**: Bursa stamp duty capped at RM 1,000, clearing, minimum fees |
| Quotes only frictionless gross returns | **Friction Survival Ratio**: Reports exact percentage of alpha surviving real-world costs |
| Single-period in-sample backtest | **Rolling Walk-Forward**: 500-day train / 125-day test rolling slices with WFE scoring |
| Hardcoded to US large-cap tech | **Cross-Market Universality**: Evaluates exact same logic on Malaysia, US, Japan, Europe |
| Obscured code or look-ahead leakage | **1-Bar Execution Lag**: Signals at bar $t$ close are strictly executed at bar $t+1$ |

---

## 2. Supported Market Universes

Malaysia is treated as a **first-class research universe** with complete domestic statutory friction and macro proxy integration:

| Market | Code | Benchmark | Core Equities Universe | Regulatory & Brokerage Structure | Macro & FX Proxies |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Malaysia (Bursa)** | `MY` | `^KLSE` (FBM KLCI) | Maybank (`1155.KL`), Public Bank (`1295.KL`), Tenaga (`5347.KL`), CIMB (`1023.KL`), YTL Power (`6742.KL`) | 10 bps Brokerage + 3 bps Clearing + 10 bps Stamp Duty (Cap RM 1,000) + Min RM 8 | USD/MYR (`MYR=X`), Brent Crude (`BZ=F`) |
| **United States** | `US` | `SPY` (S&P 500) | `QQQ`, `AAPL`, `MSFT`, `NVDA` | 2 bps Brokerage + 0.5 bps Regulatory + Min $1.00 | CBOE VIX (`^VIX`), 10Y Yield (`^TNX`) |
| **Japan** | `JP` | `^N225` (Nikkei 225) | Toyota (`7203.T`), Sony (`6758.T`) | 5 bps Brokerage + 1 bp Clearing + Min ¥50 | USD/JPY (`JPY=X`) |
| **Europe** | `EU` | `^GDAXI` (DAX) | SAP (`SAP.DE`) | 6 bps Brokerage + 1.5 bps Clearing + Min €3.00 | EUR/USD (`EURUSD=X`) |

---

## 3. The Automated Research Pipeline

The research pipeline runs completely decoupled from Streamlit:

### Step 1: Update & Validate Market Data
```bash
python scripts/update_data.py
```
- Ingests market data from Yahoo Finance, FRED, or deterministic demo fallback.
- Enforces data hygiene: Monotonic timestamp validation, calendar alignment, non-negative price checks.
- Records data hygiene scores and bar counts into `data_status` table in SQLite.

### Step 2: Execute Systematic Strategy Sweep
```bash
python scripts/run_research.py
```
- Reads the locked research protocol (`configs/research_protocol.yaml`).
- Evaluates 8 strategy variants across 16 assets in 4 markets (128 combinations).
- Calculates comprehensive performance metrics, execution drag waterfall, rolling walk-forward slices, and regime breakdowns.
- Persists all metrics and daily equity curves into `results/database/research.db`.

### Step 3: Synthesize Empirical Findings
```bash
python scripts/generate_findings.py
```
- Algorithmic analysis over stored records.
- Identifies Out-of-Sample generalization leaders, execution friction decay anomalies, Bursa Malaysia alpha, and overfitting risks.

---

## 4. Persistent Research Store (SQLite)

All empirical outputs are stored in `results/database/research.db`:

* `research_runs`: Run ID, timestamp, protocol version, engine version, status, and experiment totals.
* `strategy_results`: Comprehensive metrics (CAGR, Net Sharpe, Sortino, Calmar, Max DD, Turnover, OOS Sharpe, WFE, Breakeven Cost, Regime Sharpes, Overfitting score).
* `equity_curves`: Daily time-series containing Net NAV, Gross NAV, Benchmark NAV, Position, Drawdown, and Turnover.
* `walk_forward_slices`: Granular train/test slices with in-sample and out-of-sample metrics.
* `data_status`: Ingestion status, provider, start/end dates, total bars, and hygiene score.
* `research_findings`: Structured empirical insights and supporting numerical metrics.

---

## 5. Streamlit Exploration Terminal Pages

Launch the terminal to explore precomputed results:
```bash
streamlit run app.py
```

1. **🏛️ Home & Executive Summary (`app.py`)**:
   - Headline statistics (Total experiments, OOS robust rate, median Sharpe, data health).
   - Notable empirical discoveries cards.
   - Out-of-sample league table preview.
2. **🔬 Empirical Research Findings (`pages/01_Research_Findings.py`)**:
   - Structured empirical insights categorized by OOS Generalization, Execution Reality, Bursa Malaysia, and Risk.
   - Expandable supporting quantitative proof cards.
   - Fragility & overfitting audit table.
3. **🌐 Market Explorer (`pages/02_Market_Explorer.py`)**:
   - Regional market selector (Malaysia, US, Japan, Europe).
   - In-Sample vs Out-of-Sample generalization scatter mapping.
   - Strategy performance league table for selected market.
4. **📈 Strategy Explorer (`pages/03_Strategy_Explorer.py`)**:
   - Strategy family and variant inspector.
   - Cross-market Sharpe distribution boxplots.
   - Friction attrition bar chart (Gross vs Net return comparison).
5. **🔍 Asset Deep Dive (`pages/04_Asset_Deep_Dive.py`)**:
   - Instrument deep dive (e.g. Maybank `1155.KL`, Apple `AAPL`).
   - Interactive daily Net NAV, Gross NAV, and Benchmark NAV chart.
   - Underwater drawdown profile and market regime decomposition.
6. **🌐 Cross-Market Matrix (`pages/05_Cross_Market_Matrix.py`)**:
   - Cross-market performance heatmap.
   - Predefined hypothesis testing (e.g., USD/MYR momentum filter on Malaysian equities).
7. **⏳ Walk-Forward Archive (`pages/06_Walk_Forward_Archive.py`)**:
   - Slice-by-slice train vs test Sharpe comparisons.
   - Walk-Forward Efficiency (WFE) verification.
8. **📡 Research Status & Health Audit (`pages/07_Research_Status.py`)**:
   - Live data health score table and bar counts.
   - Pipeline component audit and re-run instructions.
9. **📜 Methodology & Transparency (`pages/08_Methodology.py`)**:
   - Comprehensive documentation of execution timing ($Position_t = Signal_{t-1}$), statutory fee math, and limitations.

---

## 6. Locked Research Protocol

The research methodology is governed by `configs/research_protocol.yaml` to ensure reproducibility:
- Supported markets, trading calendars, and benchmark symbols.
- Statutory fee parameters and slippage assumptions.
- Strategy parameter grids (lookback windows, holding periods, smoothing factors).
- Walk-forward train/test sizes (500 bars in-sample, 125 bars out-of-sample).

---

## 7. Look-Ahead Bias Prevention

Zero forward leakage is guaranteed through strict design invariants:

1. **Chronological Bar Separation**:
   $$\text{Position}_t = \text{Signal}_{t-1}$$
   Signals evaluated at bar $t$ close are executed at bar $t+1$ open/close. Contemporaneous bar $t$ fills are prohibited.
2. **Lagged Extrema**:
   Rolling highs/lows for Donchian channels and breakout indicators are explicitly shifted by 1 bar:
   $$\text{Upper Channel}_t = \max(H_{t-k}, \dots, H_{t-1})$$
3. **Cross-Market Time-Zone Alignment**:
   Asian markets trade before US markets on calendar date $t$. The engine strictly requires `lag_bars >= 1` when using US overnight returns to condition Asian daytime trading.

---

## 8. Execution & Transaction Cost Model

Realized net returns account for all statutory exchange frictions:

$$\text{Net Return}_t = \text{Gross Return}_t - \text{Total Friction Drag}_t$$

### Bursa Malaysia Statutory Friction
$$\text{Fee} = \max(\text{Value} \times 0.0010, 8.00) + (\text{Value} \times 0.0003) + \min(\text{Value} \times 0.0010, 1000.00)$$

---

## 9. Walk-Forward Validation

To prevent parameter cherry-picking:
1. **Training Window**: 500 trading days (~2 years).
2. **Testing Window**: 125 trading days (~6 months).
3. **Rolling Step**: 125 trading days.
4. **Walk-Forward Efficiency (WFE)**:
   $$\text{WFE} = \frac{\text{CAGR}_{\text{Out-of-Sample}}}{\text{CAGR}_{\text{In-Sample}}}$$
   - $\text{WFE} \ge 0.60$: Statistically persistent.
   - $\text{WFE} < 0.40$: Severe curve-fitting warning.

---

## 10. Overfitting Detection Framework

Every backtest is audited by the `OverfittingDetector`:
* **Sharpe Degradation Penalty**: Flags severe collapse between in-sample and out-of-sample periods ($> 50\%$).
* **Turnover Penalty**: Flags annualized turnover $> 15\times$, which leads to fee attrition.
* **Sample Size Penalty**: Flags trade counts $< 30$.
* Categorization: `LOW`, `MODERATE`, or `HIGH` risk.

---

## 11. Cross-Market & Macro Hypotheses

The engine automatically answers empirical quant questions from computed results:
* **"Does USD/MYR momentum improve Malaysian equity signals?"**  
  Evaluating USD/MYR currency trend filtering on Bursa Malaysia equities.
* **"Does momentum generalize across US, Japan, Europe, and Malaysia?"**  
  Evaluating Time-Series Momentum universality across 4 distinct market structures.
* **"Does volatility regime alter the effectiveness of mean reversion?"**  
  Decomposing Z-score reversion during high-volatility vs low-volatility regimes.

---

## 12. Local Installation & Pipeline Run

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14

### Installation
```bash
# Clone repository
git clone https://github.com/your-org/systematic-research-engine.git
cd systematic-research-engine

# Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate    # Linux / macOS

# Install dependencies & editable package
pip install -r requirements.txt
pip install -e .
```

### Running the Research Pipeline
```bash
# Step 1: Ingest & validate market data
python scripts/update_data.py

# Step 2: Run automated research sweep
python scripts/run_research.py

# Step 3: Synthesize findings
python scripts/generate_findings.py

# Step 4: Launch exploration terminal
streamlit run app.py
```

*(To run in offline demo mode, pass `--force-demo` to the scripts).*

---

## 13. Automated GitHub Actions Scheduling

The repository includes an automated workflow (`.github/workflows/scheduled_research.yml`):
- **Weekly Schedule**: Automatically runs every Sunday at 00:00 UTC.
- **Workflow Dispatch**: Can be triggered manually from the GitHub Actions tab.
- Ingests data, executes strategy sweeps, synthesizes findings, and uploads the SQLite database artifact.

---

## 14. Streamlit Community Cloud Deployment

Pre-configured for one-click deployment:
1. Push repository to GitHub.
2. Log into [share.streamlit.io](https://share.streamlit.io).
3. Connect repository, branch `master` (or `main`), and set main file to `app.py`.
4. Click **Deploy**.

The precomputed `results/database/research.db` is bundled with the repository, allowing instantaneous exploration upon deployment without paid API keys.

---

## 15. Test Suite Verification

Run the comprehensive unit test suite:
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
tests/test_research_store.py .           [PASS]
tests/test_robustness.py ..              [PASS]
tests/test_signals.py ....               [PASS]
tests/test_transaction_costs.py ...      [PASS]
tests/test_walk_forward.py .             [PASS]
===================== 27 passed in 2.14s =====================
```

---

## 16. Limitations & Disclaimers

### Known Limitations
* **Daily Frequency**: The current research engine evaluates daily closing prices and next-open executions; it is not an intraday order-book matching engine.
* **Survivorship Bias**: Public historical data for index constituents reflects current index members. Point-in-time constituent adjustments require specialized institutional vendor databases.
* **Short Borrow Constraints**: Malaysian equities operate under Regulated Short Selling (RSS); long-only configurations are evaluated as the institutional default.

### Disclaimer
**NOT FINANCIAL OR INVESTMENT ADVICE.**  
This software is an empirical research framework developed strictly for academic evaluation, software engineering demonstration, and quantitative hypothesis testing. Historical performance and statistical backtests do not guarantee future performance.

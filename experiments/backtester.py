"""Multi-Asset Portfolio Backtest Engine & Historical Crisis Replay Module.

Simulates periodic rebalancing strategies under realistic execution constraints,
statutory Bursa transaction costs, cash drag, benchmark tracking, and historical crisis stress tests.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RUNS_DIR = _PROJECT_ROOT / "results" / "runs"


class PortfolioBacktester:
    """Vectorized portfolio simulation engine with friction modeling and crisis replay."""

    def __init__(
        self,
        prices_df: pd.DataFrame,
        benchmark_prices: Optional[pd.Series] = None,
        annual_trading_days: int = 248,
        brokerage_bps: float = 10.0,
        clearing_bps: float = 3.0,
        stamp_duty_bps: float = 10.0,
        stamp_duty_cap: float = 1000.0,
        slippage_bps: float = 5.0,
        initial_capital: float = 100_000.0,
    ):
        """Initialize backtest simulator."""
        self.prices_df = prices_df.dropna()
        self.returns_df = self.prices_df.pct_change().dropna()
        self.assets = self.prices_df.columns.tolist()
        self.annual_days = annual_trading_days

        self.brokerage_bps = brokerage_bps
        self.clearing_bps = clearing_bps
        self.stamp_duty_bps = stamp_duty_bps
        self.stamp_duty_cap = stamp_duty_cap
        self.slippage_bps = slippage_bps
        self.initial_capital = initial_capital

        # Align benchmark
        if benchmark_prices is not None:
            aligned_bench = benchmark_prices.reindex(self.prices_df.index).ffill().bfill()
            self.bench_returns = aligned_bench.pct_change().fillna(0.0)
        else:
            # Synthetic 60/40 benchmark default: Equal return of first asset or market proxy
            self.bench_returns = self.returns_df.iloc[:, 0]

    def compute_transaction_cost(self, trade_value: float, is_domestic: bool = True) -> float:
        """Calculate realistic trade fee including statutory stamp duty cap."""
        if trade_value <= 1.0:
            return 0.0

        brokerage = trade_value * (self.brokerage_bps / 10000.0)
        clearing = trade_value * (self.clearing_bps / 10000.0)
        slippage = trade_value * (self.slippage_bps / 10000.0)

        stamp = 0.0
        if is_domestic:
            stamp = min(trade_value * (self.stamp_duty_bps / 10000.0), self.stamp_duty_cap)

        return float(brokerage + clearing + stamp + slippage)

    def run_rebalancing_backtest(
        self,
        weights: np.ndarray,
        frequency: str = "Monthly",
        rebalance_dates: Optional[List[pd.Timestamp]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run backtest with scheduled rebalancing and cash friction drag.
        
        Args:
            weights: Target asset weights vector summing to 1.0
            frequency: 'Daily', 'Monthly', 'Quarterly', 'Annual'
        """
        w_target = np.array(weights) / np.sum(weights)
        dates = self.returns_df.index
        n_days = len(dates)
        n_assets = len(self.assets)

        # Identify rebalance timestamps
        if rebalance_dates is None:
            if frequency == "Daily":
                rebal_mask = np.ones(n_days, dtype=bool)
            elif frequency == "Monthly":
                rebal_mask = np.array([True] + [dates[i].month != dates[i-1].month for i in range(1, n_days)])
            elif frequency == "Quarterly":
                rebal_mask = np.array([True] + [dates[i].quarter != dates[i-1].quarter for i in range(1, n_days)])
            else:  # Annual
                rebal_mask = np.array([True] + [dates[i].year != dates[i-1].year for i in range(1, n_days)])
        else:
            rebal_mask = np.isin(dates, rebalance_dates)

        # Time series tracking
        nav = np.zeros(n_days)
        gross_nav = np.zeros(n_days)
        bench_nav = np.zeros(n_days)
        portfolio_weights = np.zeros((n_days, n_assets))
        fee_ledger = []

        # Initialization
        curr_equity = self.initial_capital
        curr_gross_equity = self.initial_capital
        curr_bench_equity = self.initial_capital
        curr_holdings = curr_equity * w_target

        nav[0] = curr_equity
        gross_nav[0] = curr_gross_equity
        bench_nav[0] = curr_bench_equity
        portfolio_weights[0] = w_target

        total_fees = 0.0

        for t in range(1, n_days):
            asset_ret = self.returns_df.iloc[t].values
            bench_ret = float(self.bench_returns.iloc[t])

            # Update asset holdings with daily price returns
            curr_holdings = curr_holdings * (1.0 + asset_ret)
            curr_equity = np.sum(curr_holdings)
            curr_gross_equity = curr_gross_equity * (1.0 + np.dot(w_target, asset_ret))
            curr_bench_equity = curr_bench_equity * (1.0 + bench_ret)

            # Check if rebalancing day
            if rebal_mask[t]:
                target_holdings = curr_equity * w_target
                trades = np.abs(target_holdings - curr_holdings)
                rebal_cost = sum(self.compute_transaction_cost(tr) for tr in trades)

                curr_equity -= rebal_cost
                total_fees += rebal_cost
                curr_holdings = curr_equity * w_target

                if rebal_cost > 0:
                    fee_ledger.append({
                        "Date": str(dates[t].date()),
                        "Turnover_Value": float(np.sum(trades) / 2.0),
                        "Cost_Paid": float(rebal_cost),
                    })

            nav[t] = curr_equity
            gross_nav[t] = curr_gross_equity
            bench_nav[t] = curr_bench_equity
            portfolio_weights[t] = curr_holdings / curr_equity if curr_equity > 0 else 0.0

        # Normalized to base 1.0
        res_df = pd.DataFrame({
            "NAV": nav / self.initial_capital,
            "Gross_NAV": gross_nav / self.initial_capital,
            "Benchmark_NAV": bench_nav / self.initial_capital,
            "Daily_Net_Return": pd.Series(nav / self.initial_capital, index=dates).pct_change().fillna(0.0),
            "Daily_Bench_Return": pd.Series(bench_nav / self.initial_capital, index=dates).pct_change().fillna(0.0),
        }, index=dates)

        # Drawdowns
        peaks = res_df["NAV"].cummax()
        res_df["Drawdown"] = (res_df["NAV"] - peaks) / peaks

        bench_peaks = res_df["Benchmark_NAV"].cummax()
        res_df["Bench_Drawdown"] = (res_df["Benchmark_NAV"] - bench_peaks) / bench_peaks

        # Summary KPIs
        total_years = max(0.1, len(dates) / self.annual_days)
        final_nav = float(res_df["NAV"].iloc[-1])
        cagr = (final_nav ** (1.0 / total_years)) - 1.0
        ann_vol = float(res_df["Daily_Net_Return"].std() * np.sqrt(self.annual_days))
        sharpe = (cagr - 0.030) / ann_vol if ann_vol > 1e-6 else 0.0
        max_dd = float(res_df["Drawdown"].min())
        calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-4 else 0.0

        # Tracking error & Sortino
        excess_ret = res_df["Daily_Net_Return"] - res_df["Daily_Bench_Return"]
        tracking_error = float(excess_ret.std() * np.sqrt(self.annual_days))
        downside_ret = res_df["Daily_Net_Return"][res_df["Daily_Net_Return"] < 0.0]
        downside_dev = float(downside_ret.std() * np.sqrt(self.annual_days)) if len(downside_ret) > 0 else 1e-6
        sortino = (cagr - 0.030) / downside_dev if downside_dev > 1e-6 else 0.0

        summary = {
            "CAGR": cagr,
            "Annualized_Volatility": ann_vol,
            "Sharpe_Ratio": sharpe,
            "Sortino_Ratio": sortino,
            "Max_Drawdown": max_dd,
            "Calmar_Ratio": calmar,
            "Tracking_Error": tracking_error,
            "Total_Fees_Paid": total_fees,
            "Rebalance_Count": int(np.sum(rebal_mask)),
            "Total_Trades_Logged": len(fee_ledger),
        }

        return res_df, summary

    def replay_historical_crisis(self, weights: np.ndarray) -> List[Dict[str, Any]]:
        """Stress-test asset weights against four landmark historical shocks."""
        crisis_definitions = [
            {
                "name": "1997 Asian Financial Crisis (Shock Proxy)",
                "start": "1997-07-01",
                "end": "1998-09-01",
                "description": "Severe Ringgit peg pressure, currency flight, and emerging market equity collapse.",
                "synth_equity_shock": -0.65,
                "synth_fx_shock": 0.45,   # USD/MYR appreciation
                "synth_bond_shock": -0.15,
            },
            {
                "name": "2008 Global Financial Crisis",
                "start": "2007-10-01",
                "end": "2009-03-09",
                "description": "Lehman bankruptcy, worldwide credit contraction, and global equity liquidation.",
                "synth_equity_shock": -0.48,
                "synth_fx_shock": 0.15,
                "synth_bond_shock": 0.08,
            },
            {
                "name": "March 2020 COVID-19 Crash",
                "start": "2020-02-19",
                "end": "2020-03-23",
                "description": "Unprecedented global liquidity dash across all asset classes.",
                "synth_equity_shock": -0.28,
                "synth_fx_shock": 0.08,
                "synth_bond_shock": -0.04,
            },
            {
                "name": "2022 Global Rate Tightening Cycle",
                "start": "2022-01-03",
                "end": "2022-10-14",
                "description": "Aggressive central bank rate hikes causing simultaneous drawdown in stocks and bonds.",
                "synth_equity_shock": -0.22,
                "synth_fx_shock": 0.12,
                "synth_bond_shock": -0.18,
            },
        ]

        results = []
        for crisis in crisis_definitions:
            # Check if dates exist in real returns
            sub_returns = self.returns_df.loc[crisis["start"]:crisis["end"]]
            if len(sub_returns) >= 15:
                # Actual empirical replay
                cum_ret = (1.0 + sub_returns).cumprod()
                port_cum = np.dot(cum_ret.values, weights)
                peak = np.maximum.accumulate(port_cum)
                max_dd = float(np.min((port_cum - peak) / peak))
                total_ret = float(port_cum[-1] / port_cum[0]) - 1.0
                mode = "HISTORICAL REPLAY"
            else:
                # Synthetic calibrated stress projection
                port_ret = 0.0
                for i, sym in enumerate(self.assets):
                    w = weights[i]
                    if "MGS" in sym or "US_10Y" in sym:
                        shock = crisis["synth_bond_shock"]
                    elif "MYR=X" in sym:
                        shock = -crisis["synth_fx_shock"]
                    elif "=F" in sym:
                        shock = crisis["synth_equity_shock"] * 0.8
                    else:
                        shock = crisis["synth_equity_shock"]
                    port_ret += w * shock
                total_ret = port_ret
                max_dd = min(total_ret, total_ret * 1.25)
                mode = "STRESS SHOCK PROJECTION"

            results.append({
                "Crisis": crisis["name"],
                "Period": f"{crisis['start']} to {crisis['end']}",
                "Description": crisis["description"],
                "Estimated_Return": total_ret,
                "Estimated_Max_Drawdown": max_dd,
                "Evaluation_Mode": mode,
            })

        return results

    def save_run_artifacts(
        self,
        weights: np.ndarray,
        summary: Dict[str, Any],
        equity_df: pd.DataFrame,
        run_name: str = "portfolio_run",
    ) -> Path:
        """Persist execution artifacts into results/runs/YYYYMMDD_HHMMSS/."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        target_dir = _RUNS_DIR / f"{timestamp}_{run_name}"
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. summary.json
        summary_path = target_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # 2. weights.csv
        weights_df = pd.DataFrame({"Asset": self.assets, "Weight": weights})
        weights_path = target_dir / "weights.csv"
        weights_df.to_csv(weights_path, index=False)

        # 3. equity_curve.parquet
        equity_path = target_dir / "equity_curve.parquet"
        equity_df.to_parquet(equity_path, compression="snappy")

        return target_dir

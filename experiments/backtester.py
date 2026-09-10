"""Multi-Market Portfolio Backtest Engine & Walk-Forward Orchestrator.

Simulates periodic rebalancing strategies under realistic execution constraints,
multi-market statutory transaction costs, cash drag, benchmark tracking, and
walk-forward out-of-sample validation.
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
    """Vectorized portfolio simulation and walk-forward validation engine."""

    def __init__(
        self,
        prices_df: pd.DataFrame,
        benchmark_prices: Optional[pd.Series] = None,
        annual_trading_days: int = 252,
        brokerage_bps: Optional[float] = None,
        bursa_brokerage_bps: float = 10.0,
        clearing_bps: float = 3.0,
        stamp_duty_bps: float = 10.0,
        stamp_duty_cap: Optional[float] = None,
        bursa_stamp_duty_cap: float = 1000.0,
        us_brokerage_bps: float = 2.0,
        jp_brokerage_bps: float = 5.0,
        slippage_bps: float = 4.0,
        initial_capital: float = 100_000.0,
        **kwargs,
    ):
        """Initialize backtest simulator."""
        self.prices_df = prices_df.dropna()
        self.returns_df = self.prices_df.pct_change().dropna()
        self.assets = self.prices_df.columns.tolist()
        self.annual_days = annual_trading_days

        self.bursa_brokerage_bps = brokerage_bps if brokerage_bps is not None else bursa_brokerage_bps
        self.bursa_stamp_duty_cap = stamp_duty_cap if stamp_duty_cap is not None else bursa_stamp_duty_cap
        self.clearing_bps = clearing_bps
        self.stamp_duty_bps = stamp_duty_bps
        self.us_brokerage_bps = us_brokerage_bps
        self.jp_brokerage_bps = jp_brokerage_bps
        self.slippage_bps = slippage_bps
        self.initial_capital = initial_capital

        if benchmark_prices is not None:
            aligned_bench = benchmark_prices.reindex(self.prices_df.index).ffill().bfill()
            self.bench_returns = aligned_bench.pct_change().fillna(0.0)
        else:
            self.bench_returns = self.returns_df.mean(axis=1)

    def compute_transaction_cost(self, trade_value: float, is_domestic: bool = True) -> float:
        """Calculate realistic trade fee including statutory stamp duty cap."""
        if trade_value <= 1.0:
            return 0.0

        brokerage = trade_value * (self.bursa_brokerage_bps / 10000.0)
        clearing = trade_value * (self.clearing_bps / 10000.0)
        slippage = trade_value * (self.slippage_bps / 10000.0)

        stamp = 0.0
        if is_domestic:
            stamp = min(trade_value * (self.stamp_duty_bps / 10000.0), self.bursa_stamp_duty_cap)

        return float(brokerage + clearing + stamp + slippage)

    def compute_asset_transaction_cost(self, asset: str, trade_value: float) -> float:
        """Calculate realistic trade fee including statutory stamp duty for Malaysian counters."""
        if trade_value <= 1.0:
            return 0.0

        if ".KL" in asset:
            return self.compute_transaction_cost(trade_value, is_domestic=True)
        elif ".T" in asset:
            fee = trade_value * (self.jp_brokerage_bps / 10000.0)
            slippage = trade_value * (self.slippage_bps / 10000.0)
            return float(fee + slippage)
        else:
            fee = trade_value * (self.us_brokerage_bps / 10000.0)
            slippage = trade_value * (self.slippage_bps / 10000.0)
            return float(fee + slippage)

    def replay_historical_crisis(self, weights: np.ndarray) -> List[Dict[str, Any]]:
        """Replay crisis periods for backtester."""
        from experiments.stress_test import StressTestEngine
        ste = StressTestEngine(self.returns_df, assets=self.assets)
        return ste.replay_historical_crises(weights)

    def run_rebalancing_backtest(
        self,
        weights: np.ndarray,
        frequency: str = "Monthly",
        friction_bps: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run periodic rebalancing simulation with transaction cost friction and cash drag.
        
        Args:
            weights: Target portfolio weights vector summing to 1.0
            frequency: 'Daily', 'Monthly', 'Quarterly', 'Annual'
            friction_bps: Optional interactive friction/slippage in basis points.
                          If None, applies regional statutory costs (Bursa stamp duty, etc.).
        """
        w_target = np.array(weights) / np.sum(weights)
        dates = self.returns_df.index
        n_days = len(dates)
        n_assets = len(self.assets)

        if frequency == "Daily":
            rebal_mask = np.ones(n_days, dtype=bool)
        elif frequency == "Monthly":
            rebal_mask = np.array([True] + [dates[i].month != dates[i-1].month for i in range(1, n_days)])
        elif frequency == "Quarterly":
            rebal_mask = np.array([True] + [dates[i].quarter != dates[i-1].quarter for i in range(1, n_days)])
        else:  # Annual
            rebal_mask = np.array([True] + [dates[i].year != dates[i-1].year for i in range(1, n_days)])

        nav = np.zeros(n_days)
        gross_nav = np.zeros(n_days)
        nav[0] = self.initial_capital
        gross_nav[0] = self.initial_capital

        curr_values = self.initial_capital * w_target
        gross_curr_values = self.initial_capital * w_target

        total_costs = 0.0
        rebalance_records = []

        # Initial allocation fee
        if friction_bps is not None:
            initial_friction = float(self.initial_capital * np.sum(np.abs(w_target)) * (friction_bps / 10000.0) / 2.0)
            total_costs += initial_friction
            nav[0] -= initial_friction
        else:
            for i, a in enumerate(self.assets):
                total_costs += self.compute_asset_transaction_cost(a, curr_values[i])
            nav[0] -= total_costs

        for t in range(1, n_days):
            day_ret = self.returns_df.iloc[t].values

            # Asset growth
            curr_values = curr_values * (1.0 + day_ret)
            gross_curr_values = gross_curr_values * (1.0 + day_ret)

            port_value = np.sum(curr_values)
            gross_port_value = np.sum(gross_curr_values)

            # Rebalancing execution
            if rebal_mask[t]:
                target_values = port_value * w_target
                trade_diffs = target_values - curr_values

                if friction_bps is not None:
                    w_curr = curr_values / port_value if port_value > 1e-6 else w_target
                    turnover_t = float(np.sum(np.abs(w_target - w_curr)))
                    day_friction = float(port_value * (turnover_t / 2.0) * (friction_bps / 10000.0))
                    turnover_amt = (turnover_t / 2.0) * port_value
                else:
                    day_friction = sum(
                        self.compute_asset_transaction_cost(self.assets[i], abs(trade_diffs[i]))
                        for i in range(n_assets)
                    )
                    turnover_amt = np.sum(np.abs(trade_diffs)) / 2.0

                port_value -= day_friction
                total_costs += day_friction
                curr_values = port_value * w_target

                rebalance_records.append({
                    "Date": dates[t],
                    "Turnover": float(turnover_amt),
                    "Friction_Cost": float(day_friction),
                    "Portfolio_Value": float(port_value),
                })

            nav[t] = port_value
            gross_nav[t] = gross_port_value

        # Build time-series DataFrame
        aligned_bench_ret = self.bench_returns.reindex(dates).fillna(0.0)
        bench_nav = self.initial_capital * (1.0 + aligned_bench_ret).cumprod()

        peak = np.maximum.accumulate(nav)
        drawdown = (nav - peak) / peak

        daily_port_ret = pd.Series(nav, index=dates).pct_change().fillna(0.0)

        equity_df = pd.DataFrame({
            "NAV": nav,
            "Gross_NAV": gross_nav,
            "Benchmark_NAV": bench_nav.values,
            "Drawdown": drawdown,
            "Daily_Return": daily_port_ret.values,
        }, index=dates)

        # Performance summary metrics
        total_ret = float(nav[-1] / nav[0]) - 1.0
        gross_ending = float(gross_nav[-1])
        gross_tot = float(gross_ending / gross_nav[0]) - 1.0

        n_years = max(1.0 / self.annual_days, n_days / self.annual_days)
        cagr = float((1.0 + total_ret) ** (1.0 / n_years)) - 1.0
        gross_cagr = float((1.0 + gross_tot) ** (1.0 / n_years)) - 1.0
        friction_drag_bps = float((gross_cagr - cagr) * 10000.0)

        ann_vol = float(daily_port_ret.std() * np.sqrt(self.annual_days))
        sharpe = (cagr - 0.040) / ann_vol if ann_vol > 1e-4 else 0.0

        downside = daily_port_ret[daily_port_ret < 0.0]
        downside_vol = float(downside.std() * np.sqrt(self.annual_days)) if len(downside) > 1 else ann_vol
        sortino = (cagr - 0.040) / downside_vol if downside_vol > 1e-4 else 0.0

        max_dd = float(np.min(drawdown))
        calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-4 else 0.0

        bench_total_ret = float(bench_nav.iloc[-1] / bench_nav.iloc[0]) - 1.0
        bench_cagr = float((1.0 + bench_total_ret) ** (1.0 / n_years)) - 1.0

        summary = {
            "initial_capital": self.initial_capital,
            "ending_nav": float(nav[-1]),
            "gross_ending_nav": gross_ending,
            "total_return": total_ret,
            "gross_total_return": gross_tot,
            "cagr": cagr,
            "CAGR": cagr,
            "gross_cagr": gross_cagr,
            "friction_drag_bps": friction_drag_bps,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "benchmark_cagr": bench_cagr,
            "total_friction_drag": float(total_costs),
            "Total_Fees_Paid": float(total_costs),
            "rebalance_count": len(rebalance_records),
            "total_turnover": float(sum(r["Turnover"] for r in rebalance_records)),
            "start_date": str(dates[0].strftime("%Y-%m-%d")),
            "end_date": str(dates[-1].strftime("%Y-%m-%d")),
        }

        return equity_df, summary

    def run_walk_forward_validation(
        self,
        weights_func: Any,
        train_window: int = 504,
        test_window: int = 126,
    ) -> List[Dict[str, Any]]:
        """Perform rolling walk-forward out-of-sample testing across time slices.
        
        Args:
            weights_func: Callable accepting in-sample returns DataFrame and returning weights np.ndarray.
            train_window: Number of training bars (e.g. 504 days = 2 years).
            test_window: Number of out-of-sample test bars (e.g. 126 days = 6 months).
        """
        slices = []
        n_days = len(self.returns_df)
        step = test_window

        idx = 0
        while idx + train_window + test_window <= n_days:
            train_ret = self.returns_df.iloc[idx : idx + train_window]
            test_ret = self.returns_df.iloc[idx + train_window : idx + train_window + test_window]

            # In-sample weights
            w_is = weights_func(train_ret)

            # In-sample performance
            port_train = np.dot(train_ret.values, w_is)
            is_cagr = float(np.mean(port_train) * self.annual_days)
            is_vol = float(np.std(port_train) * np.sqrt(self.annual_days))
            is_sharpe = (is_cagr - 0.04) / is_vol if is_vol > 1e-4 else 0.0

            # Out-of-sample performance
            port_test = np.dot(test_ret.values, w_is)
            oos_cagr = float(np.mean(port_test) * self.annual_days)
            oos_vol = float(np.std(port_test) * np.sqrt(self.annual_days))
            oos_sharpe = (oos_cagr - 0.04) / oos_vol if oos_vol > 1e-4 else 0.0

            wfe = oos_sharpe / is_sharpe if is_sharpe > 0.0 else 0.0

            slices.append({
                "slice_id": len(slices) + 1,
                "train_start": str(train_ret.index[0].strftime("%Y-%m-%d")),
                "train_end": str(train_ret.index[-1].strftime("%Y-%m-%d")),
                "test_start": str(test_ret.index[0].strftime("%Y-%m-%d")),
                "test_end": str(test_ret.index[-1].strftime("%Y-%m-%d")),
                "in_sample_sharpe": is_sharpe,
                "out_of_sample_sharpe": oos_sharpe,
                "walk_forward_efficiency": wfe,
            })

            idx += step

        return slices

    def save_run_artifacts(
        self,
        weights: np.ndarray,
        summary: Dict[str, Any],
        equity_df: pd.DataFrame,
        run_name: str = "multi_market_run",
    ) -> Path:
        """Persist execution artifacts into results/runs/YYYYMMDD_HHMMSS/."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        target_dir = _RUNS_DIR / f"{timestamp}_{run_name}"
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. metrics.json & summary.json
        for fname in ["metrics.json", "summary.json"]:
            with open(target_dir / fname, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        # 2. weights.csv
        weights_df = pd.DataFrame({"Asset": self.assets, "Weight": weights})
        weights_df.to_csv(target_dir / "weights.csv", index=False)

        # 3. equity_curve.parquet
        equity_df.to_parquet(target_dir / "equity_curve.parquet", compression="snappy")

        return target_dir

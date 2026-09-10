"""Monte Carlo trade simulation and path distribution analysis."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


class MonteCarloSimulator:
    """Simulates thousands of alternative equity paths via trade resampling."""

    def __init__(self, num_simulations: int = 1000, seed: int = 42):
        self.num_simulations = num_simulations
        self.seed = seed

    def simulate_trades(
        self, trade_return_pcts: List[float], initial_capital: float = 100_000.0
    ) -> Dict[str, Any]:
        """Perform bootstrap resampling with replacement over historical trade returns.
        
        Returns confidence intervals for Max Drawdown, Final Equity, and Win Rates.
        """
        if not trade_return_pcts or len(trade_return_pcts) < 5:
            return {
                "median_final_equity": initial_capital,
                "ci_5_equity": initial_capital,
                "ci_95_equity": initial_capital,
                "median_max_dd": 0.0,
                "ci_95_max_dd": 0.0,
                "simulated_paths": [],
            }

        rng = np.random.RandomState(self.seed)
        n_trades = len(trade_return_pcts)
        ret_array = np.array(trade_return_pcts)

        final_equities = []
        max_drawdowns = []
        sample_paths = []

        for sim_idx in range(self.num_simulations):
            # Resample trade returns with replacement
            sampled_rets = rng.choice(ret_array, size=n_trades, replace=True)
            equity_path = initial_capital * np.cumprod(1.0 + sampled_rets)
            equity_path = np.insert(equity_path, 0, initial_capital)

            peak = np.maximum.accumulate(equity_path)
            dd = (equity_path - peak) / peak

            final_equities.append(equity_path[-1])
            max_drawdowns.append(abs(dd.min()))

            # Keep first 25 paths for charting
            if sim_idx < 25:
                sample_paths.append(equity_path.tolist())

        final_eq_arr = np.array(final_equities)
        max_dd_arr = np.array(max_drawdowns)

        return {
            "median_final_equity": round(float(np.median(final_eq_arr)), 2),
            "ci_5_equity": round(float(np.percentile(final_eq_arr, 5)), 2),
            "ci_95_equity": round(float(np.percentile(final_eq_arr, 95)), 2),
            "median_max_dd": round(float(np.median(max_dd_arr)), 4),
            "ci_95_max_dd": round(float(np.percentile(max_dd_arr, 95)), 4),
            "sample_paths": sample_paths,
            "num_simulations": self.num_simulations,
        }

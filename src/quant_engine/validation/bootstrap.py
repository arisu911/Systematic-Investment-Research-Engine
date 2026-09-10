"""Circular Block Bootstrap for dependent financial time series."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from quant_engine.analytics.performance import calculate_sharpe_ratio


class BlockBootstrap:
    """Circular Block Bootstrap preserving return autocorrelation and volatility clustering."""

    def __init__(self, block_size: int = 20, num_samples: int = 500, seed: int = 42):
        self.block_size = block_size
        self.num_samples = num_samples
        self.seed = seed

    def resample_sharpe_distribution(self, returns: pd.Series) -> Dict[str, Any]:
        """Generate empirical distribution of annualized Sharpe ratios under block bootstrap."""
        clean_rets = returns.dropna().values
        n = len(clean_rets)
        if n < self.block_size * 2:
            return {"median_sharpe": 0.0, "ci_5_sharpe": 0.0, "ci_95_sharpe": 0.0, "p_value_sharpe_positive": 0.5}

        rng = np.random.RandomState(self.seed)
        num_blocks = int(np.ceil(n / self.block_size))
        sharpe_distribution = []

        # Circular indexing
        for _ in range(self.num_samples):
            sampled_indices = []
            for _ in range(num_blocks):
                start_idx = rng.randint(0, n)
                indices = [(start_idx + k) % n for k in range(self.block_size)]
                sampled_indices.extend(indices)

            sampled_series = pd.Series(clean_rets[sampled_indices[:n]])
            s = calculate_sharpe_ratio(sampled_series)
            sharpe_distribution.append(s)

        sharpe_arr = np.array(sharpe_distribution)
        p_positive = float((sharpe_arr > 0).mean())

        return {
            "median_sharpe": round(float(np.median(sharpe_arr)), 2),
            "ci_5_sharpe": round(float(np.percentile(sharpe_arr, 5)), 2),
            "ci_95_sharpe": round(float(np.percentile(sharpe_arr, 95)), 2),
            "p_value_sharpe_positive": round(p_positive, 3),
            "distribution": sharpe_arr.tolist(),
        }

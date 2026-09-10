"""Factor & Signal Generation Engine.

Vectorized calculation of cross-asset momentum, rolling z-score mean reversion,
realized volatility, beta to domestic/global benchmarks, and dispersion matrices.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class SignalEngine:
    """Computes technical, momentum, mean-reversion, and cross-sectional factor signals."""

    @staticmethod
    def calculate_momentum_scores(prices_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trailing 1-Month (21d), 3-Month (63d), 6-Month (126d), and 12-Month (252d) momentum."""
        scores = {}
        for col in prices_df.columns:
            s = prices_df[col]
            m1 = (s / s.shift(21)) - 1.0
            m3 = (s / s.shift(63)) - 1.0
            m6 = (s / s.shift(126)) - 1.0
            m12 = (s / s.shift(252)) - 1.0
            
            # Composite blend: 40% 12M + 30% 6M + 20% 3M + 10% 1M
            composite = 0.40 * m12.iloc[-1] + 0.30 * m6.iloc[-1] + 0.20 * m3.iloc[-1] + 0.10 * m1.iloc[-1]

            scores[col] = {
                "1M_Return": float(m1.iloc[-1]) if not pd.isna(m1.iloc[-1]) else 0.0,
                "3M_Return": float(m3.iloc[-1]) if not pd.isna(m3.iloc[-1]) else 0.0,
                "6M_Return": float(m6.iloc[-1]) if not pd.isna(m6.iloc[-1]) else 0.0,
                "12M_Return": float(m12.iloc[-1]) if not pd.isna(m12.iloc[-1]) else 0.0,
                "Composite_Momentum": float(composite) if not pd.isna(composite) else 0.0,
            }

        return pd.DataFrame.from_dict(scores, orient="index")

    @staticmethod
    def calculate_mean_reversion_zscores(prices_df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate current standardized z-score deviation from rolling SMA."""
        z_scores = {}
        for col in prices_df.columns:
            s = prices_df[col]
            sma = s.rolling(window=window).mean()
            std = s.rolling(window=window).std()
            z = (s - sma) / std
            z_scores[col] = float(z.iloc[-1]) if not pd.isna(z.iloc[-1]) else 0.0
        return pd.Series(z_scores, name=f"ZScore_{window}d")

    @staticmethod
    def calculate_rolling_beta(returns_df: pd.DataFrame, benchmark_series: pd.Series, window: int = 60) -> pd.DataFrame:
        """Compute rolling CAPM beta for each asset relative to a benchmark series."""
        aligned_bench = benchmark_series.reindex(returns_df.index).dropna()
        betas = pd.DataFrame(index=returns_df.index)

        bench_var = aligned_bench.rolling(window).var()
        for col in returns_df.columns:
            cov = returns_df[col].rolling(window).cov(aligned_bench)
            betas[col] = cov / bench_var

        return betas.dropna()

    @staticmethod
    def calculate_cross_asset_correlation(returns_df: pd.DataFrame) -> pd.DataFrame:
        """Compute full pairwise Pearson correlation matrix."""
        return returns_df.corr()

    @staticmethod
    def calculate_cross_sectional_dispersion(returns_df: pd.DataFrame) -> pd.Series:
        """Calculate daily cross-sectional return dispersion across the asset universe."""
        return returns_df.std(axis=1)

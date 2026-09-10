"""Cross-Market Factor & Signal Generation Engine.

Vectorized calculation of:
1. Cross-asset momentum (1M, 3M, 6M, 12M, composite).
2. Rolling volatility ratios (21d / 63d).
3. Rolling beta against regional benchmarks (^KLSE, ^GSPC, ^N225).
4. Full 25x25 cross-market correlation and dispersion matrices.
5. Cross-market lead-lag cross-correlation analysis.
6. Macro factor sensitivities (^VIX, ^TNX, BZ=F, DX-Y.NYB).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class SignalEngine:
    """Computes technical, factor, cross-sectional, and macro sensitivity signals."""

    @staticmethod
    def calculate_momentum_scores(prices_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trailing 1-Month (21d), 3-Month (63d), 6-Month (126d), and 12-Month (252d) momentum."""
        scores = {}
        for col in prices_df.columns:
            s = prices_df[col].dropna()
            if len(s) < 25:
                continue

            m1 = (s.iloc[-1] / s.iloc[-22] - 1.0) if len(s) >= 22 else 0.0
            m3 = (s.iloc[-1] / s.iloc[-64] - 1.0) if len(s) >= 64 else m1
            m6 = (s.iloc[-1] / s.iloc[-127] - 1.0) if len(s) >= 127 else m3
            m12 = (s.iloc[-1] / s.iloc[-253] - 1.0) if len(s) >= 253 else m6

            composite = 0.40 * m12 + 0.30 * m6 + 0.20 * m3 + 0.10 * m1

            scores[col] = {
                "1M_Return": float(m1),
                "3M_Return": float(m3),
                "6M_Return": float(m6),
                "12M_Return": float(m12),
                "Composite_Momentum": float(composite),
            }

        return pd.DataFrame.from_dict(scores, orient="index")

    @staticmethod
    def calculate_rolling_vol_ratios(returns_df: pd.DataFrame, short_window: int = 21, long_window: int = 63) -> pd.Series:
        """Calculate ratio of short-term volatility to long-term volatility (vol spike indicator)."""
        ratios = {}
        for col in returns_df.columns:
            s = returns_df[col].dropna()
            if len(s) >= long_window:
                vol_short = s.iloc[-short_window:].std()
                vol_long = s.iloc[-long_window:].std()
                ratio = (vol_short / vol_long) if vol_long > 1e-6 else 1.0
                ratios[col] = float(ratio)
            else:
                ratios[col] = 1.0
        return pd.Series(ratios, name=f"VolRatio_{short_window}d_{long_window}d")

    @staticmethod
    def calculate_rolling_beta(returns_df: pd.DataFrame, benchmark_series: pd.Series, window: int = 60) -> pd.DataFrame:
        """Compute rolling CAPM beta for each asset relative to a benchmark series."""
        aligned_bench = benchmark_series.reindex(returns_df.index).dropna()
        betas = pd.DataFrame(index=returns_df.index)

        bench_var = aligned_bench.rolling(window).var()
        for col in returns_df.columns:
            cov = returns_df[col].rolling(window).cov(aligned_bench)
            betas[col] = cov / bench_var

        return betas.dropna(how="all")

    @staticmethod
    def calculate_cross_market_correlation(returns_df: pd.DataFrame) -> pd.DataFrame:
        """Compute full pairwise Pearson correlation matrix."""
        return returns_df.corr()

    @staticmethod
    def calculate_lead_lag_cross_correlation(
        series_x: pd.Series, series_y: pd.Series, max_lag: int = 5
    ) -> pd.Series:
        """Calculate cross-correlation between series_x and shifted series_y from -max_lag to +max_lag.
        
        Positive lag means series_y leads series_x.
        """
        aligned = pd.concat([series_x, series_y], axis=1).dropna()
        x = aligned.iloc[:, 0]
        y = aligned.iloc[:, 1]

        lags = range(-max_lag, max_lag + 1)
        corrs = {}
        for lag in lags:
            if lag < 0:
                c = x.iloc[-lag:].corr(y.iloc[:lag])
            elif lag > 0:
                c = x.iloc[:-lag].corr(y.iloc[lag:])
            else:
                c = x.corr(y)
            corrs[lag] = float(c) if not pd.isna(c) else 0.0

        return pd.Series(corrs, name="CrossCorrelation")

    @staticmethod
    def calculate_macro_sensitivities(returns_df: pd.DataFrame, macro_returns_df: pd.DataFrame) -> pd.DataFrame:
        """Compute regression beta sensitivities of each asset to macro drivers (^VIX, ^TNX, BZ=F, DX-Y.NYB)."""
        sensitivities = pd.DataFrame(index=returns_df.columns, columns=macro_returns_df.columns)

        for macro_col in macro_returns_df.columns:
            m_series = macro_returns_df[macro_col]
            m_var = m_series.var()
            if m_var < 1e-8:
                continue
            for asset_col in returns_df.columns:
                cov = returns_df[asset_col].cov(m_series)
                sensitivities.loc[asset_col, macro_col] = cov / m_var

        return sensitivities.astype(float)

    @staticmethod
    def calculate_cross_sectional_dispersion(returns_df: pd.DataFrame) -> pd.Series:
        """Calculate daily cross-sectional return dispersion across the asset universe."""
        return returns_df.std(axis=1)

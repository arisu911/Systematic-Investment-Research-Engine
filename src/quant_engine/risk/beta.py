"""Market beta, CAPM alpha decomposition, and tracking error analytics."""

from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy import stats


def calculate_beta_and_alpha(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.03,
    annual_trading_days: int = 252,
) -> Dict[str, float]:
    """Perform OLS regression of strategy excess returns on benchmark excess returns."""
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 30:
        return {"beta": 1.0, "alpha_annualized": 0.0, "r_squared": 0.0, "information_ratio": 0.0}

    strat_ret = aligned.iloc[:, 0]
    bench_ret = aligned.iloc[:, 1]

    daily_rf = (1.0 + risk_free_rate) ** (1.0 / annual_trading_days) - 1.0
    strat_excess = strat_ret - daily_rf
    bench_excess = bench_ret - daily_rf

    slope, intercept, r_value, p_value, std_err = stats.linregress(bench_excess, strat_excess)
    alpha_ann = intercept * annual_trading_days
    r_squared = r_value ** 2

    # Tracking Error & Information Ratio
    active_return = strat_ret - bench_ret
    tracking_error = float(active_return.std() * np.sqrt(annual_trading_days))
    info_ratio = (
        float((active_return.mean() * annual_trading_days) / tracking_error)
        if tracking_error > 1e-6
        else 0.0
    )

    return {
        "beta": round(float(slope), 3),
        "alpha_annualized": round(float(alpha_ann), 4),
        "r_squared": round(float(r_squared), 4),
        "tracking_error": round(tracking_error, 4),
        "information_ratio": round(info_ratio, 3),
    }


def compute_rolling_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Calculate rolling 63-day (~quarterly) market beta."""
    cov = strategy_returns.rolling(window).cov(benchmark_returns)
    var = benchmark_returns.rolling(window).var()
    return cov / (var + 1e-12)

"""Statistical distribution properties, tail risk, and Value-at-Risk analytics."""

from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy import stats


def calculate_return_statistics(returns: pd.Series) -> Dict[str, Any]:
    """Compute moments, skewness, excess kurtosis, VaR, and CVaR (Expected Shortfall)."""
    clean_rets = returns.dropna()
    if len(clean_rets) < 10:
        return {
            "skewness": 0.0,
            "excess_kurtosis": 0.0,
            "var_95_daily": 0.0,
            "var_99_daily": 0.0,
            "cvar_95_daily": 0.0,
            "best_day": 0.0,
            "worst_day": 0.0,
            "positive_days_pct": 0.0,
        }

    skew = float(stats.skew(clean_rets))
    kurt = float(stats.kurtosis(clean_rets))  # Excess kurtosis (Fisher definition, normal = 0)

    # Historical Value at Risk (VaR)
    var_95 = float(np.percentile(clean_rets, 5))
    var_99 = float(np.percentile(clean_rets, 1))

    # Conditional VaR (Expected Shortfall): mean of returns below 5th percentile
    tail_returns = clean_rets[clean_rets <= var_95]
    cvar_95 = float(tail_returns.mean()) if len(tail_returns) > 0 else var_95

    return {
        "skewness": round(skew, 3),
        "excess_kurtosis": round(kurt, 3),
        "var_95_daily": round(var_95, 4),
        "var_99_daily": round(var_99, 4),
        "cvar_95_daily": round(cvar_95, 4),
        "best_day": round(float(clean_rets.max()), 4),
        "worst_day": round(float(clean_rets.min()), 4),
        "positive_days_pct": round(float((clean_rets > 0).mean() * 100.0), 2),
    }

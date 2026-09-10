"""Rolling volatility and downside deviation risk analytics."""

import numpy as np
import pandas as pd


def rolling_annualized_volatility(
    returns: pd.Series, window: int = 63, annual_days: int = 252
) -> pd.Series:
    """Rolling annualized volatility."""
    return returns.rolling(window).std() * np.sqrt(annual_days)


def downside_semi_deviation(
    returns: pd.Series, target_return: float = 0.0, annual_days: int = 252
) -> float:
    """Downside semi-deviation (for Sortino Ratio calculation)."""
    downside_diff = returns[returns < target_return] - target_return
    if len(downside_diff) == 0:
        return 0.0
    semi_variance = (downside_diff ** 2).mean()
    return float(np.sqrt(semi_variance) * np.sqrt(annual_days))


def rolling_sharpe_ratio(
    returns: pd.Series,
    window: int = 126,
    risk_free_rate: float = 0.03,
    annual_days: int = 252,
) -> pd.Series:
    """Rolling 6-month annualized Sharpe ratio."""
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / annual_days) - 1.0
    excess_ret = returns - daily_rf
    mean_excess = excess_ret.rolling(window).mean()
    roll_vol = returns.rolling(window).std() + 1e-6
    return (mean_excess / roll_vol) * np.sqrt(annual_days)

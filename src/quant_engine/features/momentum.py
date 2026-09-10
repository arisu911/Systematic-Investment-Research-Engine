"""Time-series and cross-sectional momentum feature extractors."""

import numpy as np
import pandas as pd


def time_series_momentum(
    price: pd.Series,
    lookback_days: int = 120,
    vol_adjusted: bool = True,
    vol_window: int = 60,
) -> pd.Series:
    """Calculate time-series momentum return.
    
    If vol_adjusted is True, scales the return by annualized rolling realized volatility
    following Moskowitz, Ooi, Pedersen (2012).
    """
    raw_return = (price / price.shift(lookback_days)) - 1.0
    if not vol_adjusted:
        return raw_return

    daily_ret = price.pct_change()
    roll_vol = daily_ret.rolling(vol_window).std() * np.sqrt(252)
    # Volatility-scaled signal: Target 10% annualized volatility
    scaled_signal = raw_return / (roll_vol + 1e-6)
    return scaled_signal


def cross_sectional_momentum_rank(prices: pd.DataFrame, lookback_days: int = 120) -> pd.DataFrame:
    """Compute cross-sectional percentile rank [0.0 to 1.0] across an asset universe."""
    returns = (prices / prices.shift(lookback_days)) - 1.0
    ranks = returns.rank(axis=1, pct=True)
    return ranks


def momentum_acceleration(price: pd.Series, fast: int = 20, slow: int = 60) -> pd.Series:
    """Momentum derivative / rate of momentum acceleration (Fast ROC - Slow ROC)."""
    fast_roc = (price / price.shift(fast)) - 1.0
    slow_roc = (price / price.shift(slow)) - 1.0
    return fast_roc - slow_roc

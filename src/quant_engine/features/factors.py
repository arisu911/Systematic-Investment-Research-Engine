"""Composite factor score generators."""

import pandas as pd
from quant_engine.features.momentum import time_series_momentum
from quant_engine.features.volatility import realized_volatility


def composite_momentum_score(close: pd.Series) -> pd.Series:
    """Standard academic 12-1 momentum (12-month return skipping last 1 month to avoid short-term reversion)."""
    ret_12m = (close.shift(21) / close.shift(252)) - 1.0
    ret_6m = (close.shift(21) / close.shift(126)) - 1.0
    ret_3m = (close.shift(21) / close.shift(63)) - 1.0
    
    score = (0.5 * ret_12m) + (0.3 * ret_6m) + (0.2 * ret_3m)
    return score


def low_volatility_score(close: pd.Series, window: int = 126) -> pd.Series:
    """Low-volatility factor score (inverse annualized volatility, normalized)."""
    vol = realized_volatility(close, window=window)
    inv_vol = 1.0 / (vol + 1e-6)
    score = (inv_vol - inv_vol.rolling(252).mean()) / (inv_vol.rolling(252).std() + 1e-6)
    return score

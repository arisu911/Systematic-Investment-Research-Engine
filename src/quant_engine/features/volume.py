"""Volume indicators, liquidity filters, and volume-weighted features."""

import numpy as np
import pandas as pd


def volume_sma_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Ratio of current volume to trailing average volume."""
    vol_sma = volume.rolling(period).mean()
    return volume / (vol_sma + 1e-6)


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Cumulative On-Balance Volume (OBV)."""
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv


def volume_weighted_average_price_proxy(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """Rolling multi-day VWAP approximation."""
    typical_price = (high + low + close) / 3.0
    cum_pv = (typical_price * volume).rolling(window).sum()
    cum_v = volume.rolling(window).sum()
    return cum_pv / (cum_v + 1e-6)

"""Volatility models, Parkinson/Garman-Klass estimators, and regime classifiers."""

import numpy as np
import pandas as pd


def realized_volatility(
    close: pd.Series, window: int = 20, annualization_factor: int = 252
) -> pd.Series:
    """Standard close-to-close annualized realized volatility."""
    returns = close.pct_change()
    return returns.rolling(window=window).std() * np.sqrt(annualization_factor)


def parkinson_volatility(
    high: pd.Series, low: pd.Series, window: int = 20, annualization_factor: int = 252
) -> pd.Series:
    """Parkinson (1980) High-Low volatility estimator.
    
    Approximately 5x more efficient than standard close-to-close volatility.
    """
    hl_ratio = np.log(high / low) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    variance = factor * hl_ratio.rolling(window=window).mean()
    return np.sqrt(variance) * np.sqrt(annualization_factor)


def garman_klass_volatility(
    open_p: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    annualization_factor: int = 252,
) -> pd.Series:
    """Garman-Klass (1980) OHLC volatility estimator.
    
    Accounts for intraday drift and extreme ranges, ~8x more efficient than close-to-close.
    """
    hl = 0.5 * (np.log(high / low) ** 2)
    co = (2.0 * np.log(2.0) - 1.0) * (np.log(close / open_p) ** 2)
    variance = (hl - co).rolling(window=window).mean()
    variance = variance.clip(lower=1e-8)
    return np.sqrt(variance) * np.sqrt(annualization_factor)


def volatility_percentile(vol_series: pd.Series, lookback: int = 252) -> pd.Series:
    """Rolling percentile rank (0.0 to 1.0) of current volatility vs past history."""
    return vol_series.rolling(lookback).rank(pct=True)


def volatility_regime(
    vol_series: pd.Series,
    lookback: int = 252,
    low_thresh: float = 0.33,
    high_thresh: float = 0.66,
) -> pd.Series:
    """Classify into volatility regimes: -1 (Low Vol), 0 (Normal Vol), 1 (High Vol)."""
    pct = volatility_percentile(vol_series, lookback)
    regime = pd.Series(0, index=vol_series.index)
    regime[pct <= low_thresh] = -1
    regime[pct >= high_thresh] = 1
    return regime

"""Macroeconomic indicators, rate shifts, and currency momentum."""

import pandas as pd


def yield_change(yield_series: pd.Series, period: int = 20) -> pd.Series:
    """Absolute change in interest rate / yield in percentage points over period."""
    return yield_series - yield_series.shift(period)


def fx_momentum(fx_series: pd.Series, lookback: int = 20) -> pd.Series:
    """Percentage return of currency pair over lookback.
    
    Positive for USD/MYR indicates USD appreciation / MYR depreciation.
    """
    return (fx_series / fx_series.shift(lookback)) - 1.0


def rate_regime(yield_series: pd.Series, ma_period: int = 50) -> pd.Series:
    """Rate regime indicator: +1 (rising yields / tightening), -1 (falling yields / easing)."""
    ma = yield_series.rolling(ma_period).mean()
    regime = pd.Series(0, index=yield_series.index)
    regime[yield_series > ma] = 1
    regime[yield_series < ma] = -1
    return regime

"""Cross-market feature extractors with rigorous chronological time-zone lagging."""

import pandas as pd


def overnight_us_return(
    us_close: pd.Series, local_dates: pd.DatetimeIndex, lag_bars: int = 1
) -> pd.Series:
    """Extract prior-day US return aligned to local Asian trading calendar.
    
    IMPORTANT LOOK-AHEAD PREVENTION:
    Asian markets (Bursa Malaysia, Nikkei, STI, HSI) trade before US markets on calendar date t.
    Therefore, the US close available to a Malaysian trader on morning t is from session t-1.
    We strictly enforce lag_bars >= 1.
    """
    assert lag_bars >= 1, "Cross-market features must have lag_bars >= 1 to prevent forward leakage."
    us_daily_ret = us_close.pct_change().shift(lag_bars)
    aligned = us_daily_ret.reindex(local_dates).ffill()
    aligned.name = "US_Overnight_Return_Lagged"
    return aligned


def cross_market_spread(series1: pd.Series, series2: pd.Series, normalize: bool = True) -> pd.Series:
    """Spread or relative return between two cross-market assets."""
    ret1 = series1.pct_change()
    ret2 = series2.pct_change()
    spread = ret1 - ret2
    if normalize:
        spread = (spread - spread.rolling(60).mean()) / (spread.rolling(60).std() + 1e-6)
    return spread


def macro_conditioned_signal(
    raw_signal: pd.Series, macro_filter: pd.Series, threshold: float = 0.0, lag_macro: int = 1
) -> pd.Series:
    """Filter or gate a trading signal based on a macro state (strictly lagged).
    
    Example: Only take long equity signals when USD/MYR trailing return is below threshold (strengthening MYR).
    """
    lagged_macro = macro_filter.shift(lag_macro)
    conditioned = raw_signal.copy()
    # Mask out signals when macro regime is unfavorable
    conditioned[lagged_macro > threshold] = 0.0
    return conditioned

"""Drawdown analysis, underwater series, and recovery duration metrics."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


def compute_drawdown_series(nav_series: pd.Series) -> pd.Series:
    """Calculate percentage drawdown from rolling peak: (NAV - Peak) / Peak."""
    peak = nav_series.cummax()
    drawdown = (nav_series - peak) / peak
    return drawdown


def analyze_drawdowns(nav_series: pd.Series) -> Dict[str, Any]:
    """Extract maximum drawdown, average drawdown, underwater durations, and recovery times."""
    if nav_series.empty or len(nav_series) < 2:
        return {
            "max_drawdown": 0.0,
            "avg_drawdown": 0.0,
            "max_drawdown_duration_bars": 0,
            "longest_recovery_bars": 0,
            "current_drawdown": 0.0,
        }

    dd = compute_drawdown_series(nav_series)
    max_dd = float(dd.min())
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
    current_dd = float(dd.iloc[-1])

    # Calculate underwater periods
    is_underwater = dd < 0
    durations = []
    current_duration = 0

    for underwater in is_underwater:
        if underwater:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
            current_duration = 0

    if current_duration > 0:
        durations.append(current_duration)

    max_duration = max(durations) if durations else 0

    return {
        "max_drawdown": round(max_dd, 4),
        "avg_drawdown": round(avg_dd, 4),
        "max_drawdown_duration_bars": int(max_duration),
        "current_drawdown": round(current_dd, 4),
    }

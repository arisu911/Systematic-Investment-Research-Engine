"""Exposure and capital utilization analytics."""

from typing import Dict, Any
import numpy as np
import pandas as pd


def analyze_exposure(positions: pd.Series) -> Dict[str, Any]:
    """Calculate market exposure statistics."""
    if positions.empty:
        return {
            "time_in_market_pct": 0.0,
            "long_exposure_pct": 0.0,
            "short_exposure_pct": 0.0,
            "cash_pct": 100.0,
            "avg_gross_exposure": 0.0,
            "max_gross_exposure": 0.0,
        }

    n = len(positions)
    in_market = (positions != 0).sum()
    long_bars = (positions > 0).sum()
    short_bars = (positions < 0).sum()
    cash_bars = (positions == 0).sum()

    gross_exposure = positions.abs()

    return {
        "time_in_market_pct": round(float((in_market / n) * 100.0), 2),
        "long_exposure_pct": round(float((long_bars / n) * 100.0), 2),
        "short_exposure_pct": round(float((short_bars / n) * 100.0), 2),
        "cash_pct": round(float((cash_bars / n) * 100.0), 2),
        "avg_gross_exposure": round(float(gross_exposure.mean()), 3),
        "max_gross_exposure": round(float(gross_exposure.max()), 3),
    }

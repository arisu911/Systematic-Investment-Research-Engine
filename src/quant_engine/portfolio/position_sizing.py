"""Position sizing algorithms: Volatility Targeting, Inverse Volatility, and Kelly Criterion."""

from typing import Optional
import numpy as np
import pandas as pd


def equal_weight_sizing(signals: pd.Series, max_weight: float = 1.0) -> pd.Series:
    """Equal weight sizing bounded by maximum position limit."""
    return signals.clip(lower=-max_weight, upper=max_weight)


def volatility_target_sizing(
    signals: pd.Series,
    realized_vol: pd.Series,
    target_annual_vol: float = 0.12,
    max_leverage: float = 1.0,
) -> pd.Series:
    """Size positions inversely with realized volatility to maintain constant portfolio risk.
    
    Formula: weight_t = signal_t * min(max_leverage, target_vol / realized_vol_t)
    """
    scale = target_annual_vol / (realized_vol + 1e-6)
    scale = scale.clip(lower=0.1, upper=max_leverage)
    sized = signals * scale
    return sized


def inverse_volatility_weights(volatilities: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional inverse volatility weights across assets."""
    inv_vol = 1.0 / (volatilities + 1e-6)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    return weights


def fractional_kelly_sizing(
    win_rate: float,
    payoff_ratio: float,
    fraction: float = 0.25,
    max_cap: float = 0.30,
) -> float:
    """Calculate conservative fractional Kelly criterion sizing.
    
    NOTE: Kelly sizing is a theoretical upper-bound optimization for independent Bernoulli bets.
    In continuous financial time-series with fat tails, full Kelly leads to catastrophic drawdowns.
    We enforce a default quarter-Kelly (fraction=0.25) with a hard cap.
    """
    if payoff_ratio <= 0:
        return 0.0
    
    # Kelly f* = (p * b - (1 - p)) / b = p - (1 - p) / b
    q = 1.0 - win_rate
    kelly_full = (win_rate * payoff_ratio - q) / payoff_ratio
    
    if kelly_full <= 0:
        return 0.0

    sized = kelly_full * fraction
    return min(max_cap, round(sized, 4))

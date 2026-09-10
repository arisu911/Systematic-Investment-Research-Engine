"""Portfolio risk limits and weight constraint enforcers."""

import pandas as pd


def apply_position_limits(weights: pd.DataFrame, max_weight: float = 0.30) -> pd.DataFrame:
    """Enforce individual asset maximum concentration limits."""
    return weights.clip(lower=-max_weight, upper=max_weight)


def apply_leverage_limit(weights: pd.DataFrame, max_gross_leverage: float = 1.0) -> pd.DataFrame:
    """Scale down weights when total gross exposure exceeds maximum leverage limit."""
    gross = weights.abs().sum(axis=1)
    scale = (max_gross_leverage / gross).clip(upper=1.0)
    return weights.mul(scale, axis=0)


def apply_long_only_constraint(weights: pd.DataFrame) -> pd.DataFrame:
    """Zero out any negative (short) allocations."""
    return weights.clip(lower=0.0)

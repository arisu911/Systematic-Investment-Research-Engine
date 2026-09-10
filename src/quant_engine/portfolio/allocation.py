"""Multi-asset portfolio weight aggregator."""

import pandas as pd
from quant_engine.portfolio.constraints import apply_leverage_limit, apply_position_limits


def combine_signals_to_portfolio_weights(
    signal_dict: dict[str, pd.Series],
    sizing_method: str = "equal",
    max_weight_per_asset: float = 0.35,
    max_gross_leverage: float = 1.0,
) -> pd.DataFrame:
    """Aggregate individual asset signals into normalized portfolio weights."""
    signals_df = pd.DataFrame(signal_dict).fillna(0.0)
    if signals_df.empty:
        return signals_df

    n_assets = len(signals_df.columns)
    if sizing_method == "equal":
        raw_weights = signals_df / max(1, n_assets)
    else:
        raw_weights = signals_df.copy()

    constrained = apply_position_limits(raw_weights, max_weight=max_weight_per_asset)
    constrained = apply_leverage_limit(constrained, max_gross_leverage=max_gross_leverage)
    return constrained

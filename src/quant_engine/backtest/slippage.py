"""Slippage and market impact models."""

import numpy as np


class SlippageModel:
    """Execution slippage model combining fixed bps and square-root market impact."""

    def __init__(self, base_slippage_bps: float = 5.0, impact_constant: float = 0.1):
        self.base_slippage_bps = base_slippage_bps
        self.impact_constant = impact_constant

    def calculate_slippage(
        self,
        order_value: float,
        price: float,
        daily_volume: float = 1_000_000.0,
        daily_volatility: float = 0.015,
    ) -> float:
        """Calculate total execution slippage in monetary currency."""
        if order_value <= 0:
            return 0.0

        # Linear fixed slippage
        linear_slippage = order_value * (self.base_slippage_bps / 10000.0)

        # Non-linear square-root market impact: impact = eta * sigma * sqrt(OrderSize / ADV)
        adv_monetary = daily_volume * price
        participation = order_value / max(1.0, adv_monetary)
        impact_pct = self.impact_constant * daily_volatility * np.sqrt(min(1.0, participation))
        impact_cost = order_value * impact_pct

        return float(linear_slippage + impact_cost)

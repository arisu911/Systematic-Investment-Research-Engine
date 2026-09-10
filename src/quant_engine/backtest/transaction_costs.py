"""Comprehensive transaction fee and regulatory cost models."""

from typing import Dict
from quant_engine.config.markets import TransactionCostConfig


class TransactionCostModel:
    """Calculates brokerage commissions, regulatory clearing fees, and stamp duties."""

    def __init__(self, config: TransactionCostConfig):
        self.config = config

    def calculate_trade_fees(self, traded_value: float) -> Dict[str, float]:
        """Calculate one-way explicit transaction fees given gross traded monetary value."""
        if traded_value <= 0:
            return {
                "brokerage": 0.0,
                "clearing_fee": 0.0,
                "stamp_duty": 0.0,
                "half_spread": 0.0,
                "total_fees": 0.0,
            }

        # Brokerage with minimum commission rule
        brokerage = max(
            self.config.minimum_commission,
            traded_value * (self.config.brokerage_bps / 10000.0),
        )

        # Clearing fee
        clearing = traded_value * (self.config.clearing_fee_bps / 10000.0)

        # Stamp duty (Bursa Malaysia cap is RM 1000 per contract note)
        stamp_duty = min(1000.0, traded_value * (self.config.stamp_duty_bps / 10000.0))

        # Bid-ask half-spread cost
        half_spread = traded_value * (self.config.bid_ask_spread_bps / 20000.0)

        total = brokerage + clearing + stamp_duty + half_spread

        return {
            "brokerage": round(brokerage, 4),
            "clearing_fee": round(clearing, 4),
            "stamp_duty": round(stamp_duty, 4),
            "half_spread": round(half_spread, 4),
            "total_fees": round(total, 4),
        }

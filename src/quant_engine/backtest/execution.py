"""Execution price models and strict next-bar filling conventions."""

from enum import Enum
import pandas as pd


class ExecutionType(str, Enum):
    """Timing convention for filling strategy signals."""

    NEXT_BAR_OPEN = "NEXT_BAR_OPEN"    # Signal at t Close -> Execute at t+1 Open
    NEXT_BAR_CLOSE = "NEXT_BAR_CLOSE"  # Signal at t Close -> Execute at t+1 Close


class ExecutionModel:
    """Computes fill prices and executed target positions with zero forward leakage."""

    def __init__(self, execution_type: ExecutionType = ExecutionType.NEXT_BAR_OPEN):
        self.execution_type = execution_type

    def get_fill_prices_and_positions(
        self, df: pd.DataFrame, signals: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """Align signals to execution prices.
        
        Returns:
            fill_prices: Series of prices where positions were entered/rebalanced
            executed_positions: Series of active target position weights in market for each bar
        """
        if self.execution_type == ExecutionType.NEXT_BAR_OPEN:
            # Signal generated at t Close executes at t+1 Open
            # The position is held from t+1 Open until next rebalance
            fill_prices = df["Open"].copy()
            executed_positions = signals.shift(1).fillna(0.0)
        else:
            # Signal generated at t Close executes at t+1 Close
            fill_prices = df["Close"].copy()
            executed_positions = signals.shift(1).fillna(0.0)

        return fill_prices, executed_positions

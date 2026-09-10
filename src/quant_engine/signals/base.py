"""Abstract base class and utilities for signal generation."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from quant_engine.config.strategies import StrategyConfig


class SignalGenerator(ABC):
    """Abstract base class for quantitative signal generators.
    
    A signal is a normalized continuous or discrete target weight in [-1.0, 1.0]:
      +1.0 = Full Long
       0.0 = Cash / Flat
      -1.0 = Full Short
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy/signal name."""
        pass

    @abstractmethod
    def generate_raw_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate unconstrained signals from price history.
        
        CRITICAL: All indicator calculations must only use data up to bar t.
        The signal series returned is indexed by date t.
        """
        pass

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate final actionable signals with strategy constraints applied."""
        raw = self.generate_raw_signals(df)
        
        # Apply Long-Only constraint if enabled
        if self.config.long_only:
            raw = raw.clip(lower=0.0)

        # Ensure index matches and NaNs are filled with 0.0 (Cash)
        signals = raw.reindex(df.index).fillna(0.0)
        signals.name = "Signal"
        return signals

    @staticmethod
    def apply_holding_period(signals: pd.Series, holding_days: int) -> pd.Series:
        """Extend discrete entry signals for a fixed holding duration."""
        if holding_days <= 1:
            return signals
        
        held = signals.replace(0, np.nan).ffill(limit=holding_days - 1).fillna(0.0)
        return held

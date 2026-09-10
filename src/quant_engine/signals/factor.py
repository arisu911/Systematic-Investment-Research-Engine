"""Multi-factor composite scoring strategy."""

import numpy as np
import pandas as pd
from quant_engine.signals.base import SignalGenerator
from quant_engine.features.factors import composite_momentum_score, low_volatility_score


class CompositeFactorSignal(SignalGenerator):
    """Composite multi-factor signal combining Momentum and Low-Volatility."""

    @property
    def name(self) -> str:
        return "Composite Factor Signal"

    def generate_raw_signals(self, df: pd.DataFrame) -> pd.Series:
        params = self.config.parameters
        mom_weight = float(params.get("momentum_weight", 0.6))
        vol_weight = float(params.get("low_vol_weight", 0.4))
        entry_threshold = float(params.get("entry_threshold", 0.5))

        close = df["Close"]
        mom_score = composite_momentum_score(close)
        vol_score = low_volatility_score(close)

        # Normalize factors to z-scores over 1-year rolling window
        mom_z = (mom_score - mom_score.rolling(252).mean()) / (mom_score.rolling(252).std() + 1e-6)
        vol_z = (vol_score - vol_score.rolling(252).mean()) / (vol_score.rolling(252).std() + 1e-6)

        composite = (mom_weight * mom_z) + (vol_weight * vol_z)

        signal = pd.Series(0.0, index=close.index)
        signal[composite > entry_threshold] = 1.0
        signal[composite < -entry_threshold] = -1.0
        return signal

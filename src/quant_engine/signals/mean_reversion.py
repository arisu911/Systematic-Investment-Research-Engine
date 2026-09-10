"""Mean reversion signal generators: Z-Score and Bollinger Band reversals."""

import numpy as np
import pandas as pd
from quant_engine.signals.base import SignalGenerator
from quant_engine.features.technical import bollinger_bands


class MeanReversion(SignalGenerator):
    """Z-score statistical deviation mean reversion with hysteretic entry/exit."""

    @property
    def name(self) -> str:
        return "Z-Score Mean Reversion"

    def generate_raw_signals(self, df: pd.DataFrame) -> pd.Series:
        params = self.config.parameters
        window = int(params.get("window", 20))
        entry_z = float(params.get("entry_z", 2.0))
        exit_z = float(params.get("exit_z", 0.5))

        close = df["Close"]
        rolling_mean = close.rolling(window).mean()
        rolling_std = close.rolling(window).std() + 1e-6
        z_score = (close - rolling_mean) / rolling_std

        signal = pd.Series(0.0, index=close.index)
        current_state = 0.0

        # State machine to prevent whipsaw
        for i, z in enumerate(z_score):
            if np.isnan(z):
                continue
            if current_state == 0.0:
                if z <= -entry_z:
                    current_state = 1.0  # Oversold: Enter Long
                elif z >= entry_z:
                    current_state = -1.0 # Overbought: Enter Short
            elif current_state == 1.0:
                if z >= -exit_z:
                    current_state = 0.0  # Exit Long
            elif current_state == -1.0:
                if z <= exit_z:
                    current_state = 0.0  # Exit Short

            signal.iloc[i] = current_state

        return signal


class BollingerReversion(SignalGenerator):
    """Bollinger Band bounce mean-reversion strategy."""

    @property
    def name(self) -> str:
        return "Bollinger Band Reversion"

    def generate_raw_signals(self, df: pd.DataFrame) -> pd.Series:
        params = self.config.parameters
        period = int(params.get("period", 20))
        num_std = float(params.get("num_std", 2.0))

        close = df["Close"]
        middle, upper, lower = bollinger_bands(close, period=period, num_std=num_std)

        signal = pd.Series(0.0, index=close.index)
        current_state = 0.0

        for i in range(len(close)):
            c = close.iloc[i]
            m = middle.iloc[i]
            u = upper.iloc[i]
            l = lower.iloc[i]

            if np.isnan(m):
                continue

            if current_state == 0.0:
                if c <= l:
                    current_state = 1.0
                elif c >= u:
                    current_state = -1.0
            elif current_state == 1.0:
                if c >= m:
                    current_state = 0.0
            elif current_state == -1.0:
                if c <= m:
                    current_state = 0.0

            signal.iloc[i] = current_state

        return signal

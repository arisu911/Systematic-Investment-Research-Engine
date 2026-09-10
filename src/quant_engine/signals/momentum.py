"""Momentum signal generators: Moving Average Crossover and Time-Series Momentum."""

import numpy as np
import pandas as pd
from quant_engine.signals.base import SignalGenerator
from quant_engine.features.technical import sma, ema
from quant_engine.features.momentum import time_series_momentum


class MovingAverageMomentum(SignalGenerator):
    """Dual moving average crossover or single moving average trend filter."""

    @property
    def name(self) -> str:
        return "Moving Average Momentum"

    def generate_raw_signals(self, df: pd.DataFrame) -> pd.Series:
        params = self.config.parameters
        fast_period = int(params.get("fast_period", 20))
        slow_period = int(params.get("slow_period", 100))
        signal_type = params.get("signal_type", "crossover")  # 'crossover' or 'filter'

        close = df["Close"]
        fast_ma = sma(close, fast_period)
        slow_ma = sma(close, slow_period)

        if signal_type == "filter":
            # Long when price > slow_ma, short when price < slow_ma
            signal = pd.Series(0.0, index=close.index)
            signal[close > slow_ma] = 1.0
            signal[close < slow_ma] = -1.0
        else:
            # Classic fast/slow crossover
            signal = pd.Series(0.0, index=close.index)
            signal[fast_ma > slow_ma] = 1.0
            signal[fast_ma < slow_ma] = -1.0

        return signal


class TimeSeriesMomentum(SignalGenerator):
    """Time-Series (Absolute) Momentum with optional volatility scaling and holding period."""

    @property
    def name(self) -> str:
        return "Time-Series Momentum"

    def generate_raw_signals(self, df: pd.DataFrame) -> pd.Series:
        params = self.config.parameters
        lookback_days = int(params.get("lookback_days", 120))
        vol_scale = bool(params.get("vol_scale", True))
        holding_days = int(params.get("holding_days", 20))

        ts_mom = time_series_momentum(df["Close"], lookback_days=lookback_days, vol_adjusted=vol_scale)

        # Discrete sign signal: +1 if positive trailing return, -1 if negative
        raw_signal = np.sign(ts_mom).fillna(0.0)

        # Apply holding period
        signal = self.apply_holding_period(raw_signal, holding_days)
        return signal

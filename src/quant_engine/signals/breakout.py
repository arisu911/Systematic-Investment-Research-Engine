"""Breakout signal generators: Donchian Channel and ATR Volatility breakouts."""

import numpy as np
import pandas as pd
from quant_engine.signals.base import SignalGenerator
from quant_engine.features.technical import atr, sma


class DonchianBreakout(SignalGenerator):
    """Classic Donchian Channel Trend Following Breakout."""

    @property
    def name(self) -> str:
        return "Donchian Breakout"

    def generate_raw_signals(self, df: pd.DataFrame) -> pd.Series:
        params = self.config.parameters
        lookback = int(params.get("lookback", 50))
        exit_lookback = int(params.get("exit_lookback", max(10, lookback // 2)))
        use_atr_filter = bool(params.get("atr_filter", False))

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # Shift by 1 so breakout is evaluated against PRIOR window's extreme (STRICT LOOK-AHEAD PREVENTION)
        upper_channel = high.rolling(lookback).max().shift(1)
        lower_channel = low.rolling(exit_lookback).min().shift(1)

        signal = pd.Series(0.0, index=close.index)
        current_state = 0.0

        for i in range(len(close)):
            c = close.iloc[i]
            u = upper_channel.iloc[i]
            l = lower_channel.iloc[i]

            if np.isnan(u) or np.isnan(l):
                continue

            if current_state == 0.0:
                if c > u:
                    current_state = 1.0  # Breakout to new high
                elif c < l:
                    current_state = -1.0 # Breakdown to new low
            elif current_state == 1.0:
                if c < l:
                    current_state = 0.0  # Exit long
            elif current_state == -1.0:
                if c > u:
                    current_state = 0.0  # Exit short

            signal.iloc[i] = current_state

        return signal


class ATRBreakout(SignalGenerator):
    """Volatility envelope breakout strategy using Keltner/ATR bands."""

    @property
    def name(self) -> str:
        return "ATR Volatility Breakout"

    def generate_raw_signals(self, df: pd.DataFrame) -> pd.Series:
        params = self.config.parameters
        ma_period = int(params.get("ma_period", 20))
        atr_period = int(params.get("atr_period", 14))
        multiplier = float(params.get("multiplier", 2.0))

        close = df["Close"]
        mid = sma(close, ma_period).shift(1)
        band = atr(df["High"], df["Low"], close, atr_period).shift(1) * multiplier

        upper = mid + band
        lower = mid - band

        signal = pd.Series(0.0, index=close.index)
        signal[close > upper] = 1.0
        signal[close < lower] = -1.0
        return signal

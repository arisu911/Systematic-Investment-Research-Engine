"""Unit tests for baseline strategy signal generators."""

import numpy as np
import pandas as pd
from quant_engine.config.strategies import StrategyConfig
from quant_engine.signals.momentum import MovingAverageMomentum, TimeSeriesMomentum
from quant_engine.signals.mean_reversion import MeanReversion
from quant_engine.signals.breakout import DonchianBreakout


def get_mock_price_df(n=250):
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    np.random.seed(42)
    drift = np.linspace(100, 150, n)
    noise = np.cumsum(np.random.randn(n) * 1.5)
    close = drift + noise
    high = close + np.random.uniform(0.5, 2.0, n)
    low = close - np.random.uniform(0.5, 2.0, n)
    open_p = close + np.random.uniform(-0.5, 0.5, n)
    volume = np.random.randint(10000, 50000, n)
    
    return pd.DataFrame(
        {"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def test_moving_average_momentum_signals():
    df = get_mock_price_df()
    cfg = StrategyConfig(parameters={"fast_period": 10, "slow_period": 50}, long_only=True)
    strategy = MovingAverageMomentum(cfg)
    signals = strategy.generate_signals(df)

    assert len(signals) == len(df)
    assert set(signals.unique()).issubset({0.0, 1.0})


def test_time_series_momentum_signals():
    df = get_mock_price_df()
    cfg = StrategyConfig(parameters={"lookback_days": 30, "holding_days": 5}, long_only=False)
    strategy = TimeSeriesMomentum(cfg)
    signals = strategy.generate_signals(df)

    assert len(signals) == len(df)
    assert set(signals.unique()).issubset({-1.0, 0.0, 1.0})


def test_mean_reversion_signals():
    df = get_mock_price_df()
    cfg = StrategyConfig(parameters={"window": 15, "entry_z": 1.5, "exit_z": 0.5}, long_only=True)
    strategy = MeanReversion(cfg)
    signals = strategy.generate_signals(df)

    assert len(signals) == len(df)
    assert (signals >= 0.0).all()


def test_donchian_breakout_signals():
    df = get_mock_price_df()
    cfg = StrategyConfig(parameters={"lookback": 20, "exit_lookback": 10}, long_only=True)
    strategy = DonchianBreakout(cfg)
    signals = strategy.generate_signals(df)

    assert len(signals) == len(df)
    assert set(signals.unique()).issubset({0.0, 1.0})

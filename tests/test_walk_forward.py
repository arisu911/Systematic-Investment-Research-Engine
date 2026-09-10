"""Unit tests for walk-forward chronological splitting and out-of-sample assembly."""

import pandas as pd
import numpy as np
from quant_engine.config.strategies import StrategyConfig
from quant_engine.signals.momentum import MovingAverageMomentum
from quant_engine.validation.walk_forward import WalkForwardEngine


def test_walk_forward_execution_and_no_overlap():
    # 700 trading days
    dates = pd.date_range("2020-01-01", periods=700, freq="B")
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.randn(700))
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 1.0,
            "Low": prices - 1.0,
            "Close": prices,
            "Volume": 100000,
        },
        index=dates,
    )

    cfg = StrategyConfig(parameters={"fast_period": 10, "slow_period": 30}, long_only=True)
    wf_engine = WalkForwardEngine(train_window_bars=250, test_window_bars=100, step_bars=100)
    result, oos_curve = wf_engine.run(df, MovingAverageMomentum, cfg)

    assert len(result.slices) >= 3
    for s in result.slices:
        assert pd.to_datetime(s.train_end) <= pd.to_datetime(s.test_start)
    assert not oos_curve.empty
    assert len(oos_curve) >= 300

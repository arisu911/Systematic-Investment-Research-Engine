"""Unit tests designed to prove the absence of look-ahead bias."""

import numpy as np
import pandas as pd
from quant_engine.backtest.engine import BacktestEngine
from quant_engine.backtest.execution import ExecutionModel, ExecutionType


def test_signal_shift_prevents_contemporaneous_execution():
    """Verify that a signal generated on bar t Close is only executed on bar t+1."""
    dates = pd.date_range("2021-01-01", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100.0, 105.0, 110.0, 100.0, 95.0],
            "High": [101.0, 106.0, 111.0, 101.0, 96.0],
            "Low": [99.0, 104.0, 109.0, 99.0, 94.0],
            "Close": [100.0, 105.0, 110.0, 100.0, 95.0],
            "Volume": 10000,
        },
        index=dates,
    )

    # Signal triggers only on Bar index 1 (date 2021-01-04)
    signals = pd.Series([0.0, 1.0, 0.0, 0.0, 0.0], index=dates)

    exec_model = ExecutionModel(ExecutionType.NEXT_BAR_OPEN)
    fill_prices, active_positions = exec_model.get_fill_prices_and_positions(df, signals)

    # On Bar 1, active position must still be 0 (cannot trade on bar 1's signal during bar 1)
    assert active_positions.iloc[1] == 0.0
    # On Bar 2, active position becomes 1.0
    assert active_positions.iloc[2] == 1.0
    # Fill price for Bar 2 must be Bar 2's Open (110.0)
    assert fill_prices.iloc[2] == 110.0


def test_perfect_future_knowledge_is_lagged_by_engine():
    """If an omniscient signal knows tomorrow's return, the execution engine lag must delay it by 1 bar."""
    dates = pd.date_range("2021-01-01", periods=10, freq="B")
    prices = [100.0, 110.0, 90.0, 120.0, 80.0, 130.0, 70.0, 140.0, 60.0, 150.0]
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": 10000,
        },
        index=dates,
    )

    # Cheating signal: buys only when tomorrow's close is higher than today's
    future_return = pd.Series(prices).pct_change().shift(-1)
    cheating_signal = (future_return > 0).astype(float)
    cheating_signal.index = dates

    # Run backtest
    engine = BacktestEngine()
    res_df, ledger, summary = engine.run(df, cheating_signal)

    # Because engine executes with 1 bar lag (signals.shift(1)), the cheating signal
    # is delayed by 1 full bar, destroying the artificial clairvoyance!
    # Verified by checking that positions[t] != cheating_signal[t]
    assert (res_df["Position"].iloc[1:] == cheating_signal.shift(1).iloc[1:]).all()

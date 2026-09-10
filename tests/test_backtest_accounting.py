"""Unit tests for backtest accounting invariants and conservation of capital."""

import numpy as np
import pandas as pd
from quant_engine.config.markets import TransactionCostConfig
from quant_engine.backtest.engine import BacktestEngine


def test_accounting_conservation_and_no_nan():
    dates = pd.date_range("2021-01-01", periods=150, freq="B")
    np.random.seed(42)
    drift = np.linspace(100, 120, 150)
    noise = np.cumsum(np.random.randn(150))
    close = drift + noise
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 100000,
        },
        index=dates,
    )

    # Signal toggles between 1.0 and 0.0 every 10 days
    signals = pd.Series([1.0 if (i // 10) % 2 == 0 else 0.0 for i in range(150)], index=dates)

    engine = BacktestEngine(initial_capital=50_000.0)
    res_df, ledger, summary = engine.run(df, signals)

    # Invariants
    assert not res_df["Equity"].isna().any()
    assert (res_df["Equity"] > 0).all()
    assert np.isclose(res_df["Equity"].iloc[0], 50_000.0)
    assert np.isclose(summary["initial_capital"], 50_000.0)
    assert np.isclose(summary["final_equity"], res_df["Equity"].iloc[-1])
    assert len(ledger.trades) >= 5


def test_zero_turnover_preserves_cash():
    dates = pd.date_range("2021-01-01", periods=50, freq="B")
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Volume": 10000,
        },
        index=dates,
    )
    # Signal is 0 throughout (100% Cash)
    signals = pd.Series(0.0, index=dates)

    engine = BacktestEngine(initial_capital=100_000.0)
    res_df, ledger, summary = engine.run(df, signals)

    # If always in cash, equity remains exactly 100,000 and total fees = 0
    assert (res_df["Equity"] == 100_000.0).all()
    assert summary["total_fees_paid"] == 0.0
    assert len(ledger.trades) == 0

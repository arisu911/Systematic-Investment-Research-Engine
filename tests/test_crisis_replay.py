"""Unit tests for crisis stress replay, transaction friction, and backtesting."""

import numpy as np
import pandas as pd
from experiments.backtester import PortfolioBacktester


def test_bursa_transaction_cost_with_stamp_cap():
    # Small trade: RM 10,000
    bt = PortfolioBacktester(
        pd.DataFrame({"A": [10.0, 11.0]}),
        brokerage_bps=10.0,
        clearing_bps=3.0,
        stamp_duty_bps=10.0,
        stamp_duty_cap=1000.0,
        slippage_bps=5.0,
    )
    cost_small = bt.compute_transaction_cost(10_000.0, is_domestic=True)
    # Brokerage: 10, Clearing: 3, Stamp: 10, Slippage: 5 -> Total 28 bps = RM 28.00
    assert np.isclose(cost_small, 28.0, atol=1e-2)

    # Large trade: RM 2,000,000 (Stamp duty 10 bps would be RM 2,000, but capped at RM 1,000)
    cost_large = bt.compute_transaction_cost(2_000_000.0, is_domestic=True)
    # Brokerage: 2000, Clearing: 600, Stamp: capped at 1000, Slippage: 1000 -> Total RM 4600
    assert np.isclose(cost_large, 4600.0, atol=1e-2)


def test_rebalancing_backtest_simulation():
    dates = pd.date_range("2021-01-01", periods=100, freq="B")
    prices = pd.DataFrame({
        "Asset_A": np.linspace(10.0, 12.0, 100),
        "Asset_B": np.linspace(20.0, 22.0, 100),
    }, index=dates)

    bt = PortfolioBacktester(prices, initial_capital=100_000.0)
    weights = np.array([0.60, 0.40])
    res_df, summary = bt.run_rebalancing_backtest(weights, frequency="Monthly")

    assert "NAV" in res_df.columns
    assert "Gross_NAV" in res_df.columns
    assert "Drawdown" in res_df.columns
    assert summary["CAGR"] > 0.0
    assert summary["Total_Fees_Paid"] >= 0.0


def test_historical_crisis_stress_replay():
    dates = pd.date_range("2019-01-01", periods=500, freq="B")
    prices = pd.DataFrame({
        "1155.KL": np.linspace(8.0, 9.0, 500),
        "SPY": np.linspace(300.0, 400.0, 500),
        "MGS_10Y": np.linspace(100.0, 104.0, 500),
    }, index=dates)

    bt = PortfolioBacktester(prices)
    weights = np.array([0.40, 0.30, 0.30])
    crisis_results = bt.replay_historical_crisis(weights)

    assert len(crisis_results) == 4
    for c in crisis_results:
        assert "Crisis" in c
        assert "Estimated_Return" in c
        assert "Estimated_Max_Drawdown" in c

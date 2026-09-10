"""Unit tests for transaction cost calculations and regulatory fee deductions."""

import numpy as np
import pandas as pd
from quant_engine.config.markets import TransactionCostConfig
from quant_engine.backtest.transaction_costs import TransactionCostModel
from quant_engine.backtest.engine import BacktestEngine


def test_malaysia_transaction_costs():
    cfg = TransactionCostConfig(
        brokerage_bps=10.0,
        clearing_fee_bps=3.0,
        stamp_duty_bps=10.0,
        bid_ask_spread_bps=12.0,
        default_slippage_bps=8.0,
        minimum_commission=8.0,
    )
    model = TransactionCostModel(cfg)

    # Trade of RM 100,000
    fees = model.calculate_trade_fees(100_000.0)
    # Brokerage: 100,000 * 0.0010 = RM 100
    assert np.isclose(fees["brokerage"], 100.0)
    # Clearing: 100,000 * 0.0003 = RM 30
    assert np.isclose(fees["clearing_fee"], 30.0)
    # Stamp duty: 100,000 * 0.0010 = RM 100
    assert np.isclose(fees["stamp_duty"], 100.0)
    # Half spread: 100,000 * (12/20000) = RM 60
    assert np.isclose(fees["half_spread"], 60.0)
    assert np.isclose(fees["total_fees"], 290.0)


def test_minimum_commission_rule():
    cfg = TransactionCostConfig(
        brokerage_bps=10.0,
        minimum_commission=8.0,
    )
    model = TransactionCostModel(cfg)
    # Small trade of RM 1,000: 1,000 * 0.0010 = RM 1.0, but minimum is RM 8.0
    fees = model.calculate_trade_fees(1_000.0)
    assert fees["brokerage"] == 8.0


def test_net_return_never_exceeds_gross_return():
    dates = pd.date_range("2021-01-01", periods=100, freq="B")
    df = pd.DataFrame(
        {
            "Open": 100.0 + np.arange(100),
            "High": 102.0 + np.arange(100),
            "Low": 99.0 + np.arange(100),
            "Close": 101.0 + np.arange(100),
            "Volume": 1_000_000,
        },
        index=dates,
    )
    # Alternating signals to generate turnover
    signals = pd.Series([1.0, 0.0] * 50, index=dates)

    engine = BacktestEngine(
        cost_config=TransactionCostConfig(brokerage_bps=10.0, bid_ask_spread_bps=10.0),
        slippage_bps=5.0,
    )
    res_df, ledger, summary = engine.run(df, signals)

    # In every bar where turnover occurs, Net Return must be strictly less than Gross Return
    active_turnover = res_df["Turnover"] > 0
    assert (res_df.loc[active_turnover, "Net_Return"] < res_df.loc[active_turnover, "Gross_Return"]).all()
    assert summary["total_net_return"] < summary["total_gross_return"]

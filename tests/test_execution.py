"""Unit tests for ExecutionEngine and actionable order ticket generation."""

import numpy as np
import pandas as pd
import pytest

from research.execution import ExecutionEngine


def test_bursa_board_lot_rounding():
    # 100-share board lot rule for Bursa Malaysia
    lot_size_kl = ExecutionEngine.get_lot_size("1155.KL")
    assert lot_size_kl == 100

    # 350 shares should floor to 300
    qty_kl = ExecutionEngine.round_to_lot("1155.KL", raw_shares=350.8)
    assert qty_kl == 300

    # 95 shares should floor to 0 (cannot buy fractional lot)
    qty_small = ExecutionEngine.round_to_lot("1155.KL", raw_shares=95.0)
    assert qty_small == 0


def test_us_and_global_lot_rounding():
    # 1-share lot rule for US / Global counters
    lot_size_us = ExecutionEngine.get_lot_size("SPY")
    assert lot_size_us == 1

    # 12.8 shares should floor to 12
    qty_us = ExecutionEngine.round_to_lot("SPY", raw_shares=12.8)
    assert qty_us == 12


def test_order_ticket_generation_and_cash_conservation():
    assets = ["SPY", "1155.KL", "TLT"]
    weights = np.array([0.50, 0.30, 0.20])
    prices = {
        "SPY": 500.0,       # Base FX price
        "1155.KL": 10.0,     # Base FX price
        "TLT": 95.0,        # Base FX price
    }
    capital = 100_000.0

    res = ExecutionEngine.generate_order_ticket(
        weights=weights,
        assets=assets,
        latest_prices=prices,
        capital=capital,
        base_currency="USD",
    )

    ticket_df = res["order_ticket"]
    allocated = res["allocated_cash"]
    unallocated = res["unallocated_cash"]

    # Basic structure checks
    assert len(ticket_df) == 3
    expected_cols = [
        "Ticker",
        "Asset Name",
        "Target Weight (%)",
        "Target Cash Value",
        "Current Price (Base FX)",
        "Order Quantity (Shares/Units)",
        "Effective Cash Value",
        "Effective Weight (%)",
    ]
    for col in expected_cols:
        assert col in ticket_df.columns

    # SPY: 50,000 / 500 = 100 shares
    spy_row = ticket_df[ticket_df["Ticker"] == "SPY"].iloc[0]
    assert spy_row["Order Quantity (Shares/Units)"] == 100

    # 1155.KL: 30,000 / 10 = 3000 shares (multiple of 100)
    kl_row = ticket_df[ticket_df["Ticker"] == "1155.KL"].iloc[0]
    assert kl_row["Order Quantity (Shares/Units)"] == 3000

    # Cash conservation: allocated + unallocated must exactly equal total capital
    assert np.isclose(allocated + unallocated, capital, atol=1e-4)
    assert unallocated >= 0.0


def test_csv_export_format():
    assets = ["SPY", "TLT"]
    weights = np.array([0.60, 0.40])
    prices = {"SPY": 400.0, "TLT": 100.0}
    capital = 50_000.0

    res = ExecutionEngine.generate_order_ticket(
        weights=weights,
        assets=assets,
        latest_prices=prices,
        capital=capital,
    )
    csv_data = ExecutionEngine.export_order_ticket_csv(res["order_ticket"])

    assert "Ticker" in csv_data
    assert "SPY" in csv_data
    assert "TLT" in csv_data
    assert "Order Quantity (Shares/Units)" in csv_data

"""Unit tests for Multi-Market Calendar Alignment & Currency Normalization Engine."""

import numpy as np
import pandas as pd
import pytest
from data.aligner import (
    load_universe_registry,
    get_tradable_tickers,
    align_cross_market_calendars,
    normalize_currency,
    compute_unhedged_returns,
)


def test_universe_registry_structure():
    reg = load_universe_registry()
    assert len(reg) == 25
    assert "^KLSE" in reg
    assert "^GSPC" in reg
    assert "^N225" in reg
    assert "GC=F" in reg

    tradable = get_tradable_tickers(reg)
    assert len(tradable) == 14
    assert "1155.KL" in tradable
    assert "AAPL" in tradable
    assert "7203.T" in tradable
    assert "GC=F" in tradable
    assert "^KLSE" not in tradable
    assert "^VIX" not in tradable


def test_calendar_alignment_and_imputation():
    # Simulate a raw prices dataframe with a 2-day holiday in Malaysia and 1-day holiday in US
    dates = pd.date_range("2023-01-02", periods=10, freq="B")
    data = {
        "1155.KL": [10.0, np.nan, np.nan, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8],
        "AAPL": [150.0, 151.0, 152.0, np.nan, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0],
        "^VIX": [20.0, 21.0, 20.5, 20.2, 19.8, 19.5, 19.2, 19.0, 18.8, 18.5],
    }
    raw_df = pd.DataFrame(data, index=dates)

    aligned = align_cross_market_calendars(raw_df, max_ffill_days=3)
    assert not aligned.empty
    assert aligned["1155.KL"].isna().sum() == 0
    # Day 1 and 2 should be forward filled with 10.0
    assert aligned.loc[dates[1], "1155.KL"] == 10.0
    assert aligned.loc[dates[2], "1155.KL"] == 10.0
    # AAPL day 3 should be forward filled with 152.0
    assert aligned.loc[dates[3], "AAPL"] == 152.0


def test_currency_normalization():
    dates = pd.date_range("2023-01-02", periods=5, freq="B")
    prices_df = pd.DataFrame({
        "1155.KL": [10.0, 10.0, 10.0, 10.0, 10.0],       # MYR 10.00
        "AAPL": [200.0, 200.0, 200.0, 200.0, 200.0],       # USD 200.00
        "7203.T": [3000.0, 3000.0, 3000.0, 3000.0, 3000.0], # JPY 3000.00
        "USDMYR=X": [4.0, 4.0, 4.0, 4.0, 4.0],             # 1 USD = 4.0 MYR
        "JPY=X": [150.0, 150.0, 150.0, 150.0, 150.0],      # 1 USD = 150.0 JPY
        "^VIX": [20.0, 20.0, 20.0, 20.0, 20.0],            # Indicator
    }, index=dates)

    # 1. USD Normalization:
    # 1155.KL should be 10.0 / 4.0 = 2.50 USD
    # 7203.T should be 3000.0 / 150.0 = 20.00 USD
    # AAPL remains 200.0 USD
    # ^VIX remains 20.0
    usd_norm = normalize_currency(prices_df, target_currency="USD")
    assert np.isclose(usd_norm["1155.KL"].iloc[0], 2.50)
    assert np.isclose(usd_norm["7203.T"].iloc[0], 20.00)
    assert np.isclose(usd_norm["AAPL"].iloc[0], 200.00)
    assert np.isclose(usd_norm["^VIX"].iloc[0], 20.0)

    # 2. MYR Normalization:
    # 1155.KL remains 10.00 MYR
    # AAPL becomes 200.0 * 4.0 = 800.00 MYR
    # 7203.T becomes (3000 / 150) * 4.0 = 80.00 MYR
    myr_norm = normalize_currency(prices_df, target_currency="MYR")
    assert np.isclose(myr_norm["1155.KL"].iloc[0], 10.00)
    assert np.isclose(myr_norm["AAPL"].iloc[0], 800.00)
    assert np.isclose(myr_norm["7203.T"].iloc[0], 80.00)
    assert np.isclose(myr_norm["^VIX"].iloc[0], 20.0)

    # 3. Local currency: unchanged
    local_norm = normalize_currency(prices_df, target_currency="LOCAL")
    assert np.isclose(local_norm["1155.KL"].iloc[0], 10.00)
    assert np.isclose(local_norm["AAPL"].iloc[0], 200.00)

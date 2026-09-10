"""Unit tests for Multi-Currency FX Engine and triangulation logic."""

import numpy as np
import pandas as pd
import pytest

from data.fx_engine import (
    FXEngine,
    SUPPORTED_CURRENCIES,
    CURRENCY_SYMBOLS,
    INDICATOR_SYMBOLS,
    get_currency_symbol,
)


def test_supported_currencies_and_symbols():
    assert "USD" in SUPPORTED_CURRENCIES
    assert "MYR" in SUPPORTED_CURRENCIES
    assert "JPY" in SUPPORTED_CURRENCIES
    assert "EUR" in SUPPORTED_CURRENCIES
    assert "GBP" in SUPPORTED_CURRENCIES

    assert get_currency_symbol("USD") == "$"
    assert get_currency_symbol("MYR") == "RM "
    assert get_currency_symbol("JPY") == "¥"
    assert get_currency_symbol("EUR") == "€"
    assert get_currency_symbol("GBP") == "£"


def test_fx_triangulation_matrix():
    # Synthetic quotes
    live_quotes = {
        "USDMYR=X": 4.50,   # 1 USD = 4.50 MYR
        "JPY=X": 150.0,     # 1 USD = 150 JPY
        "EURUSD=X": 1.08,   # 1 EUR = 1.08 USD
        "GBPUSD=X": 1.25,   # 1 GBP = 1.25 USD
    }

    rates = FXEngine.triangulate_rates(live_quotes)

    # Base USD
    assert np.isclose(rates["USD"]["USD"], 1.0)
    assert np.isclose(rates["USD"]["MYR"], 4.50)
    assert np.isclose(rates["USD"]["JPY"], 150.0)
    assert np.isclose(rates["USD"]["EUR"], 1.0 / 1.08)
    assert np.isclose(rates["USD"]["GBP"], 1.0 / 1.25)

    # Base MYR
    assert np.isclose(rates["MYR"]["USD"], 1.0 / 4.50)
    assert np.isclose(rates["MYR"]["MYR"], 1.0)
    # EUR to MYR: 1 EUR = 1.08 USD = 1.08 * 4.50 = 4.86 MYR
    assert np.isclose(rates["EUR"]["MYR"], 1.08 * 4.50)


def test_convert_prices_indicator_exclusion():
    # Synthetic price dataframe with an equity, a bond index, and VIX
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    prices_df = pd.DataFrame({
        "SPY": [450.0, 452.0, 455.0, 453.0, 458.0],      # USD
        "1155.KL": [9.0, 9.1, 9.05, 9.15, 9.2],          # MYR
        "^VIX": [14.5, 15.0, 14.2, 16.1, 15.5],          # Points
        "^TNX": [4.25, 4.30, 4.28, 4.35, 4.32],          # Yield %
        "USDMYR=X": [4.50, 4.50, 4.50, 4.50, 4.50],
        "JPY=X": [150.0, 150.0, 150.0, 150.0, 150.0],
        "EURUSD=X": [1.08, 1.08, 1.08, 1.08, 1.08],
        "GBPUSD=X": [1.25, 1.25, 1.25, 1.25, 1.25],
    }, index=dates)

    # Convert to MYR
    converted_df = FXEngine.convert_prices_to_base(prices_df, target_currency="MYR")

    # SPY should be scaled by USDMYR (450 * 4.5 = 2025.0)
    assert np.isclose(converted_df["SPY"].iloc[0], 450.0 * 4.50)

    # 1155.KL is already in MYR, so it should be unscaled
    assert np.isclose(converted_df["1155.KL"].iloc[0], 9.0)

    # ^VIX and ^TNX must be strictly preserved without currency scaling
    assert np.isclose(converted_df["^VIX"].iloc[0], 14.5)
    assert np.isclose(converted_df["^TNX"].iloc[0], 4.25)


def test_unhedged_vs_hedged_returns():
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    prices_df = pd.DataFrame({
        "SPY": [100.0, 105.0, 102.0, 110.0],
        "1155.KL": [10.0, 10.2, 10.1, 10.4],
    }, index=dates)

    ret_unhedged = FXEngine.calculate_asset_returns(prices_df, hedged=False)
    ret_hedged = FXEngine.calculate_asset_returns(prices_df, hedged=True)

    assert ret_unhedged.shape == (3, 2)
    assert ret_hedged.shape == (3, 2)
    assert np.all(np.isfinite(ret_unhedged.values))
    assert np.all(np.isfinite(ret_hedged.values))

"""Unit tests for data schemas, hygiene checks, and demo provider."""

import pandas as pd
import numpy as np
from quant_engine.data.schemas import validate_ohlcv_dataframe
from quant_engine.data.providers.demo import DemoDataProvider
from quant_engine.data.cleaner import DataCleaner


def test_demo_provider_generates_valid_ohlcv():
    provider = DemoDataProvider(seed=123)
    df = provider.get_prices("^KLSE", start_date="2020-01-01", end_date="2021-12-31")
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert all(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"])
    assert (df["High"] >= df["Low"]).all()
    assert (df["Close"] > 0).all()
    assert len(df) > 400


def test_data_validator_catches_zero_and_nan_prices():
    dates = pd.date_range("2021-01-01", periods=100, freq="B")
    bad_df = pd.DataFrame(
        {
            "Open": np.linspace(10, 20, 100),
            "High": np.linspace(11, 21, 100),
            "Low": np.linspace(9, 19, 100),
            "Close": np.linspace(10, 20, 100),
            "Volume": 1000,
        },
        index=dates,
    )
    # Insert bad records
    bad_df.iloc[10, bad_df.columns.get_loc("Close")] = 0.0
    bad_df.iloc[20, bad_df.columns.get_loc("Close")] = np.nan

    report = validate_ohlcv_dataframe(bad_df, symbol="TEST_BAD")
    assert report.zero_negative_prices_count >= 1
    assert report.nan_values_count >= 1
    assert report.data_health_score < 100.0


def test_data_cleaner_remedies_invalids():
    dates = pd.date_range("2021-01-01", periods=50, freq="B")
    dirty = pd.DataFrame(
        {
            "Open": [10.0] * 50,
            "High": [10.0] * 50,
            "Low": [10.0] * 50,
            "Close": [10.0] * 50,
            "Volume": [100] * 50,
        },
        index=dates,
    )
    # Introduce negative and zero
    dirty.iloc[5, dirty.columns.get_loc("Close")] = -5.0
    dirty.iloc[6, dirty.columns.get_loc("Close")] = 0.0

    cleaned = DataCleaner.clean_ohlcv(dirty)
    assert (cleaned["Close"] > 0).all()
    assert not cleaned["Close"].isna().any()

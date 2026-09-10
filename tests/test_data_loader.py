"""Unit tests for multi-asset data loader, MGS pipeline, and Parquet caching."""

import pandas as pd
from data.loader import (
    load_multi_asset_dataset,
    load_mgs_fixed_income,
    generate_deterministic_synthetic_prices,
    _hash_cache_key,
)


def test_mgs_fixed_income_loading():
    df = load_mgs_fixed_income()
    assert not df.empty
    assert "MGS_10Y" in df.columns
    assert "BNM_OPR" in df.columns
    assert df["MGS_10Y"].mean() > 2.0
    assert df["BNM_OPR"].mean() >= 1.5


def test_deterministic_synthetic_generation():
    syms = ["1155.KL", "SPY", "MGS_10Y", "MYR=X"]
    df = generate_deterministic_synthetic_prices(syms, "2020-01-01", "2021-12-31")
    assert df.shape[1] == 4
    assert not df.isna().any().any()
    # Prices must be strictly positive
    assert (df > 0).all().all()


def test_parquet_caching_lifecycle():
    syms = ["1155.KL", "MGS_10Y"]
    df1, is_demo1 = load_multi_asset_dataset(syms, "2021-01-01", "2021-06-30", force_offline=True)
    assert not df1.empty

    # Re-call should load cached data without issue
    df2, is_demo2 = load_multi_asset_dataset(syms, "2021-01-01", "2021-06-30", force_offline=True)
    assert df1.equals(df2)

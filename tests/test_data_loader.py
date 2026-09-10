"""Unit tests for Multi-Market 25-instrument data loader, caching, and fallback."""

import pandas as pd
from data.loader import (
    load_universe_prices,
    get_all_universe_tickers,
    generate_deterministic_multi_market_prices,
    _hash_cache_key,
)


def test_universe_tickers_count():
    tickers = get_all_universe_tickers()
    assert len(tickers) == 25
    assert "^KLSE" in tickers
    assert "^GSPC" in tickers
    assert "^N225" in tickers
    assert "GC=F" in tickers


def test_deterministic_multi_market_prices_generation():
    tickers = ["1155.KL", "AAPL", "7203.T", "GC=F", "^VIX", "^TNX"]
    df = generate_deterministic_multi_market_prices(tickers, "2021-01-01", "2021-12-31")
    assert df.shape[1] == 6
    assert not df.isna().any().any()
    # Prices must be strictly positive
    assert (df > 0).all().all()


def test_universe_prices_loading_and_parquet_caching():
    tickers = ["1155.KL", "AAPL", "7203.T", "^VIX"]
    df1, is_demo1 = load_universe_prices(tickers, "2021-01-01", "2021-06-30", force_offline=True)
    assert not df1.empty
    assert set(tickers).issubset(set(df1.columns))

    # Re-call should load cached data without issue
    df2, is_demo2 = load_universe_prices(tickers, "2021-01-01", "2021-06-30", force_offline=True)
    assert df1.equals(df2)

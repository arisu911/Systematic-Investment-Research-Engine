"""Unit tests for Multi-Market 25-instrument data loader, caching, multi-horizon lookbacks, and fallback."""

import time
import pandas as pd
from datetime import datetime
from data.loader import (
    load_universe_prices,
    get_all_universe_tickers,
    generate_deterministic_multi_market_prices,
    compute_lookback_dates,
    LOOKBACK_HORIZONS,
    CACHE_TTL_POLICIES,
    save_cache_metadata,
    load_cache_metadata,
    is_cache_expired,
    purge_cache,
    backfill_shorter_history_with_proxies,
    fetch_universe_data,
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


def test_compute_lookback_dates_horizons():
    """Verify that compute_lookback_dates correctly offsets calendar years for 2Y, 3Y, 4Y, 5Y, 10Y."""
    fixed_ref = datetime(2026, 9, 10)
    
    assert LOOKBACK_HORIZONS == ["2Y", "3Y", "4Y", "5Y", "10Y"]
    
    s_2y, e_2y = compute_lookback_dates("2Y", ref_date=fixed_ref)
    assert s_2y == "2024-09-10"
    assert e_2y == "2026-09-10"

    s_3y, e_3y = compute_lookback_dates("3Y", ref_date=fixed_ref)
    assert s_3y == "2023-09-10"
    assert e_3y == "2026-09-10"

    s_4y, e_4y = compute_lookback_dates("4Y", ref_date=fixed_ref)
    assert s_4y == "2022-09-10"
    assert e_4y == "2026-09-10"

    s_5y, e_5y = compute_lookback_dates("5Y", ref_date=fixed_ref)
    assert s_5y == "2021-09-10"
    assert e_5y == "2026-09-10"

    s_10y, e_10y = compute_lookback_dates("10Y", ref_date=fixed_ref)
    assert s_10y == "2016-09-10"
    assert e_10y == "2026-09-10"


def test_cache_metadata_persistence_and_ttl_expiration():
    """Test metadata save, load, and configurable TTL expiration checks."""
    meta = save_cache_metadata(
        ttl_seconds=3600,
        ttl_label="1 Hour",
        data_source=["yfinance", "FRED"],
        status="Live / Synced",
    )
    assert meta["active_ttl_seconds"] == 3600
    assert meta["active_ttl_label"] == "1 Hour"
    assert meta["status"] == "Live / Synced"

    loaded = load_cache_metadata()
    assert loaded["active_ttl_seconds"] == 3600
    assert loaded["active_ttl_label"] == "1 Hour"
    assert loaded["last_refresh_timestamp"] != "N/A"

    # Immediately should not be expired for positive TTL
    assert not is_cache_expired(ttl_seconds=3600)
    # Negative TTL simulates past expiration
    assert is_cache_expired(ttl_seconds=-10)


def test_backfill_shorter_history_with_proxies():
    """Verify that shorter-history IPO assets are gracefully backfilled with scaled regional proxies."""
    dates = pd.date_range("2016-01-01", "2026-01-01", freq="B")
    
    # Regional proxy
    proxy_series = pd.Series(range(100, 100 + len(dates)), index=dates, dtype=float)
    
    # Newer IPO asset starting in 2021 (NaNs prior to 2021)
    ipo_series = pd.Series(index=dates, dtype=float)
    ipo_start = pd.Timestamp("2021-01-01")
    ipo_series.loc[ipo_series.index >= ipo_start] = 50.0
    
    raw_df = pd.DataFrame({
        "^KLSE": proxy_series,
        "NEW_CO.KL": ipo_series,
    })
    
    backfilled_df, info = backfill_shorter_history_with_proxies(
        raw_df,
        registry={"NEW_CO.KL": {"region": "MY"}},
    )
    
    assert "NEW_CO.KL" in info
    assert info["NEW_CO.KL"]["proxy"] == "^KLSE"
    # Backfilled series should have zero NaNs
    assert not backfilled_df["NEW_CO.KL"].isna().any()
    # Post-IPO values should be preserved
    assert backfilled_df.loc[ipo_start, "NEW_CO.KL"] == 50.0


def test_universe_prices_loading_and_parquet_caching():
    tickers = ["1155.KL", "AAPL", "7203.T", "^VIX"]
    df1, is_demo1 = load_universe_prices(tickers, "2021-01-01", "2021-06-30", force_offline=True)
    assert not df1.empty
    assert set(tickers).issubset(set(df1.columns))

    # Re-call should load cached data without issue
    df2, is_demo2 = load_universe_prices(tickers, "2021-01-01", "2021-06-30", force_offline=True)
    assert df1.equals(df2)


def test_memory_slicing_performance_and_accuracy():
    """Verify that toggling between 2Y, 5Y, and 10Y slices in memory executes in < 50ms."""
    # Ensure master cache is populated
    load_universe_prices(force_offline=True, lookback_horizon="10Y")
    
    # Measure slicing time for 2Y
    t0 = time.perf_counter()
    df_2y, _ = load_universe_prices(force_offline=True, lookback_horizon="2Y")
    elapsed_2y = (time.perf_counter() - t0) * 1000.0

    # Measure slicing time for 5Y
    t0 = time.perf_counter()
    df_5y, _ = load_universe_prices(force_offline=True, lookback_horizon="5Y")
    elapsed_5y = (time.perf_counter() - t0) * 1000.0

    # Measure slicing time for 10Y
    t0 = time.perf_counter()
    df_10y, _ = load_universe_prices(force_offline=True, lookback_horizon="10Y")
    elapsed_10y = (time.perf_counter() - t0) * 1000.0

    assert elapsed_2y < 50.0, f"2Y slice took {elapsed_2y:.2f}ms (expected < 50ms)"
    assert elapsed_5y < 50.0, f"5Y slice took {elapsed_5y:.2f}ms (expected < 50ms)"
    assert elapsed_10y < 50.0, f"10Y slice took {elapsed_10y:.2f}ms (expected < 50ms)"

    assert len(df_2y) < len(df_5y) < len(df_10y)
    assert len(df_2y) > 400
    assert len(df_5y) > 1000
    assert len(df_10y) > 2000


def test_purge_cache_and_high_level_fetch():
    """Verify that purge_cache clears disk files and fetch_universe_data repopulates smoothly."""
    count = purge_cache()
    assert count >= 0
    meta = load_cache_metadata()
    assert "Purged" in meta.get("status", "")

    df, is_demo = fetch_universe_data(lookback_horizon="2Y", force_offline=True)
    assert not df.empty
    assert len(df.columns) == 25

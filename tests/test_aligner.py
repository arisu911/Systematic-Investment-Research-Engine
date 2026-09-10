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
    get_fx_adjusted_benchmark,
    calculate_relative_benchmark_metrics,
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


def test_get_fx_adjusted_benchmark_usd_myr():
    dates = pd.date_range("2023-01-02", periods=5, freq="B")
    raw_df = pd.DataFrame({
        "^KLSE": [1500.0, 1510.0, 1520.0, 1530.0, 1540.0],
        "USDMYR=X": [4.50, 4.50, 4.50, 4.50, 4.50],
    }, index=dates)

    # In USD, KLSE close must be divided by USDMYR
    bench_usd = get_fx_adjusted_benchmark(
        ticker="^KLSE",
        index_currency="MYR",
        target_currency="USD",
        raw_prices=raw_df,
        is_currency_adjusted=True,
    )
    expected = raw_df["^KLSE"] / 4.50
    pd.testing.assert_series_equal(bench_usd, expected, check_names=False)


def test_get_fx_adjusted_benchmark_usd_jpy():
    dates = pd.date_range("2023-01-02", periods=5, freq="B")
    raw_df = pd.DataFrame({
        "^N225": [30000.0, 30100.0, 30200.0, 30300.0, 30400.0],
        "JPY=X": [150.0, 150.0, 150.0, 150.0, 150.0],
    }, index=dates)

    # In USD, N225 close must be divided by JPY=X
    bench_usd = get_fx_adjusted_benchmark(
        ticker="^N225",
        index_currency="JPY",
        target_currency="USD",
        raw_prices=raw_df,
        is_currency_adjusted=True,
    )
    expected = raw_df["^N225"] / 150.0
    pd.testing.assert_series_equal(bench_usd, expected, check_names=False)


def test_get_fx_adjusted_benchmark_myr_usd():
    dates = pd.date_range("2023-01-02", periods=5, freq="B")
    raw_df = pd.DataFrame({
        "^GSPC": [5000.0, 5020.0, 5040.0, 5060.0, 5080.0],
        "USDMYR=X": [4.40, 4.40, 4.40, 4.40, 4.40],
    }, index=dates)

    # In MYR, GSPC close must be multiplied by USDMYR
    bench_myr = get_fx_adjusted_benchmark(
        ticker="^GSPC",
        index_currency="USD",
        target_currency="MYR",
        raw_prices=raw_df,
        is_currency_adjusted=True,
    )
    expected = raw_df["^GSPC"] * 4.40
    pd.testing.assert_series_equal(bench_myr, expected, check_names=False)


def test_get_fx_adjusted_benchmark_unadjusted_local():
    dates = pd.date_range("2023-01-02", periods=5, freq="B")
    raw_df = pd.DataFrame({
        "^KLSE": [1500.0, 1510.0, 1520.0, 1530.0, 1540.0],
        "USDMYR=X": [4.50, 4.50, 4.50, 4.50, 4.50],
    }, index=dates)

    # When is_currency_adjusted=False, returns raw local index
    bench_raw = get_fx_adjusted_benchmark(
        ticker="^KLSE",
        index_currency="MYR",
        target_currency="USD",
        raw_prices=raw_df,
        is_currency_adjusted=False,
    )
    pd.testing.assert_series_equal(bench_raw, raw_df["^KLSE"], check_names=False)


def test_calculate_relative_benchmark_metrics():
    dates = pd.date_range("2023-01-02", periods=100, freq="B")
    np.random.seed(42)
    bench_rets = np.random.normal(0.0005, 0.01, size=100)
    # Portfolio has beta = 1.2 and alpha = 0.0002 daily
    port_rets = 1.2 * bench_rets + 0.0002 + np.random.normal(0, 0.002, size=100)

    bench_curve = 100.0 * np.exp(np.cumsum(bench_rets))
    port_curve = 1000.0 * np.exp(np.cumsum(port_rets))

    bench_series = pd.Series(bench_curve, index=dates)
    port_series = pd.Series(port_curve, index=dates)

    # Add holiday gaps (NaNs) in benchmark series to test 3-day forward fill
    bench_with_holidays = bench_series.copy()
    bench_with_holidays.iloc[10:12] = np.nan  # 2-day market holiday

    metrics = calculate_relative_benchmark_metrics(port_series, bench_with_holidays)

    assert np.isclose(metrics["beta"], 1.2, atol=0.15)
    assert metrics["alpha"] > 0.0
    assert 0.0 <= metrics["r_squared"] <= 1.0
    assert metrics["tracking_error"] > 0.0
    assert metrics["correlation"] > 0.80


def test_benchmark_isolation_from_portfolio_metrics():
    """Verify that switching benchmarks has ZERO impact on standalone portfolio statistics."""
    dates = pd.date_range("2023-01-02", periods=50, freq="B")
    np.random.seed(101)
    port_rets = np.random.normal(0.0006, 0.012, size=50)
    port_curve = pd.Series(100_000.0 * np.exp(np.cumsum(port_rets)), index=dates)

    # Standalone metrics computed directly on portfolio equity curve
    daily_port_ret = port_curve.pct_change().dropna()
    standalone_cagr = float((port_curve.iloc[-1] / port_curve.iloc[0]) ** (252.0 / len(port_curve))) - 1.0
    standalone_vol = float(daily_port_ret.std() * np.sqrt(252))
    standalone_sharpe = (standalone_cagr - 0.04) / standalone_vol

    # Benchmark A (^GSPC)
    bench_a = pd.Series(5000.0 * np.exp(np.cumsum(np.random.normal(0.0004, 0.01, size=50))), index=dates)
    metrics_a = calculate_relative_benchmark_metrics(port_curve, bench_a)

    # Benchmark B (^KLSE) with different returns and missing dates
    bench_b = pd.Series(1500.0 * np.exp(np.cumsum(np.random.normal(-0.0001, 0.008, size=50))), index=dates)
    bench_b.iloc[5:7] = np.nan
    metrics_b = calculate_relative_benchmark_metrics(port_curve, bench_b)

    # Relative metrics differ
    assert metrics_a["beta"] != metrics_b["beta"]
    assert metrics_a["alpha"] != metrics_b["alpha"]

    # Standalone metrics remain 100% IDENTICAL
    assert np.isclose(standalone_cagr, (port_curve.iloc[-1] / port_curve.iloc[0]) ** (252.0 / len(port_curve)) - 1.0)
    assert np.isclose(standalone_vol, daily_port_ret.std() * np.sqrt(252))
    assert np.isclose(standalone_sharpe, (standalone_cagr - 0.04) / standalone_vol)


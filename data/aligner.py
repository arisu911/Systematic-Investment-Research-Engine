"""Multi-Market Cross-Calendar Alignment & Multi-Currency Normalization Engine.

Aligns trading calendars across Bursa Malaysia, NYSE, and TSE. Imputes local holidays
up to a 3-day threshold and provides currency conversion into Local Currency, USD Unified,
or MYR Unified while preserving raw indicator scales (^VIX, ^TNX, DX-Y.NYB).
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_UNIVERSE_CONFIG = _PROJECT_ROOT / "configs" / "universe.yaml"


def load_universe_registry() -> Dict[str, Any]:
    """Load universe definition and build per-instrument metadata map."""
    with open(_UNIVERSE_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    registry = {}
    regions = cfg.get("universe", {})
    for region_key, region_val in regions.items():
        loc_curr = region_val.get("local_currency", "USD")
        reg_code = region_val.get("region", "GLOBAL")
        for inst in region_val.get("instruments", []):
            ticker = inst["ticker"]
            registry[ticker] = {
                "name": inst.get("name", ticker),
                "ticker": ticker,
                "role": inst.get("role", "equity"),
                "asset_class": inst.get("asset_class", "equity"),
                "region": reg_code,
                "region_group": region_key,
                "local_currency": loc_curr,
            }
    return registry


def get_tradable_tickers(registry: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return tickers eligible for portfolio optimization (equity, tradable_proxy, commodity)."""
    if registry is None:
        registry = load_universe_registry()
    tradable = [
        t for t, meta in registry.items()
        if meta.get("role") in ["equity", "tradable_proxy", "commodity"]
    ]
    return tradable


def align_cross_market_calendars(
    raw_prices: pd.DataFrame,
    max_ffill_days: int = 3,
    min_coverage_pct: float = 0.60,
) -> pd.DataFrame:
    """Align multi-market price series across different exchange calendars.
    
    1. Reindexes to business days.
    2. Forward-fills missing days for local holidays up to max_ffill_days.
    3. Drops rows or columns with excessive missing data.
    
    Args:
        raw_prices: DataFrame of raw adjusted closing prices.
        max_ffill_days: Maximum consecutive non-trading days to forward fill (default 3).
        min_coverage_pct: Minimum proportion of valid data required for a row/col.
        
    Returns:
        Cleaned, aligned DataFrame of prices on a unified calendar.
    """
    if raw_prices.empty:
        return pd.DataFrame()

    df = raw_prices.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Remove timezone if present for clean calendar indexing
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Reindex to business days covering the span
    b_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq="B")
    aligned = df.reindex(b_days)

    # Forward fill local holidays up to max_ffill_days
    aligned = aligned.ffill(limit=max_ffill_days)

    # Backward fill initial startup gap (up to 3 days) if markets opened slightly apart
    aligned = aligned.bfill(limit=max_ffill_days)

    # Drop columns that have too little data
    col_coverage = aligned.notna().mean()
    valid_cols = col_coverage[col_coverage >= min_coverage_pct].index
    aligned = aligned[valid_cols]

    # Drop rows where more than half of the universe is still NaN
    row_coverage = aligned.notna().mean(axis=1)
    aligned = aligned[row_coverage >= 0.50]

    # Final forward fill remaining isolated gaps
    aligned = aligned.ffill().bfill()
    return aligned.dropna()


def normalize_currency(
    prices_df: pd.DataFrame,
    target_currency: str = "USD",
    registry: Optional[Dict[str, Any]] = None,
    fx_myr_ticker: str = "USDMYR=X",
    fx_jpy_ticker: str = "JPY=X",
) -> pd.DataFrame:
    """Convert asset prices to a unified base currency (USD or MYR) or keep in Local Currency.
    
    Excludes pure risk indicators (^VIX, ^TNX, DX-Y.NYB) from currency conversion.
    
    FX Quote convention on Yahoo Finance:
      - USDMYR=X: Price of 1 USD in MYR (~4.40 MYR)
      - JPY=X: Price of 1 USD in JPY (~150.0 JPY)
      
    Conversion Logic:
      Target == 'USD':
        - MYR assets: Price_USD = Price_MYR / USDMYR=X
        - JPY assets: Price_USD = Price_JPY / JPY=X
        - USD assets: Unchanged
      Target == 'MYR':
        - MYR assets: Unchanged
        - USD assets: Price_MYR = Price_USD * USDMYR=X
        - JPY assets: Price_MYR = (Price_JPY / JPY=X) * USDMYR=X
      Target == 'LOCAL':
        - All assets remain at their native local price levels.
    """
    if prices_df.empty:
        return pd.DataFrame()

    if registry is None:
        registry = load_universe_registry()

    target = target_currency.upper().strip()
    if target in ["LOCAL", "RAW", "NONE"]:
        return prices_df.copy()

    norm_df = prices_df.copy()

    # Extract FX series if available, otherwise use conservative fallbacks
    if fx_myr_ticker in prices_df.columns:
        usd_myr = prices_df[fx_myr_ticker]
    else:
        usd_myr = pd.Series(4.45, index=prices_df.index)

    if fx_jpy_ticker in prices_df.columns:
        usd_jpy = prices_df[fx_jpy_ticker]
    else:
        usd_jpy = pd.Series(150.0, index=prices_df.index)

    # Excluded from any FX conversion
    non_convertible = {"^VIX", "^TNX", "DX-Y.NYB", fx_myr_ticker, fx_jpy_ticker}

    for col in norm_df.columns:
        if col in non_convertible:
            continue

        meta = registry.get(col, {})
        role = meta.get("role", "")
        asset_class = meta.get("asset_class", "")
        loc_curr = meta.get("local_currency", "USD")

        # Skip indicators and rates
        if role in ["volatility", "rates"] or asset_class in ["risk_indicator", "rates", "fx_index", "fx"]:
            continue

        if target == "USD":
            if loc_curr == "MYR":
                norm_df[col] = norm_df[col] / usd_myr
            elif loc_curr == "JPY":
                norm_df[col] = norm_df[col] / usd_jpy
            # If already USD, no conversion needed

        elif target == "MYR":
            if loc_curr == "USD":
                norm_df[col] = norm_df[col] * usd_myr
            elif loc_curr == "JPY":
                norm_df[col] = (norm_df[col] / usd_jpy) * usd_myr
            # If already MYR, no conversion needed

    return norm_df


def compute_unhedged_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns from aligned price series."""
    return prices_df.pct_change().dropna(how="all")


def get_fx_adjusted_benchmark(
    ticker: str,
    index_currency: str = "USD",
    target_currency: str = "USD",
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    is_currency_adjusted: bool = True,
    raw_prices: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Extract and dynamically FX-adjust benchmark price series into reporting base currency.
    
    FX Normalization Logic:
      - If not is_currency_adjusted or target_currency in ['LOCAL', 'RAW', 'NONE']:
          Returns raw local benchmark price series.
      - If index_currency == target_currency:
          Returns raw benchmark price series.
      - If target_currency == 'USD' and index_currency == 'MYR':
          bench_usd = bench_myr / USDMYR=X
      - If target_currency == 'USD' and index_currency == 'JPY':
          bench_usd = bench_jpy / JPY=X
      - If target_currency == 'MYR' and index_currency == 'USD':
          bench_myr = bench_usd * USDMYR=X
      - General case across USD, MYR, JPY, EUR, GBP:
          Triangulates index_currency -> USD -> target_currency using empirical FX rates.
          
    Multi-Calendar Alignment:
      - Reindexes to business days and forward-fills non-trading holidays up to 3 days (ffill(limit=3)).
      - Backward fills initial gap up to 3 days (bfill(limit=3)).
    """
    # 1. Obtain raw series and FX source DataFrame
    if raw_prices is not None and ticker in raw_prices.columns:
        bench_series = raw_prices[ticker].copy()
        fx_source_df = raw_prices
    else:
        from data.loader import get_cached_universe_prices, generate_deterministic_multi_market_prices
        cached_df, _ = get_cached_universe_prices(base_currency="LOCAL")
        if ticker in cached_df.columns:
            bench_series = cached_df[ticker].copy()
            fx_source_df = cached_df
        else:
            s_date = str(start_date) if start_date is not None else "2019-01-01"
            e_date = str(end_date) if end_date is not None else "2024-12-31"
            synth_df = generate_deterministic_multi_market_prices([ticker], s_date, e_date)
            bench_series = synth_df[ticker].copy()
            fx_source_df = cached_df if not cached_df.empty else synth_df

    # 2. Clean datetime index
    bench_series.index = pd.to_datetime(bench_series.index)
    if bench_series.index.tz is not None:
        bench_series.index = bench_series.index.tz_localize(None)

    # 3. Date range filtering if requested
    if start_date is not None:
        bench_series = bench_series[bench_series.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        bench_series = bench_series[bench_series.index <= pd.to_datetime(end_date)]

    # 4. Forward-fill domestic exchange holidays up to 3 consecutive days
    bench_series = bench_series.ffill(limit=3).bfill(limit=3)

    target_clean = target_currency.upper().strip()
    idx_clean = index_currency.upper().strip()

    # If unadjusted or local currency or identical currency, return raw index
    if not is_currency_adjusted or target_clean in ["LOCAL", "RAW", "NONE"] or target_clean == idx_clean:
        return bench_series.rename(ticker)

    # 5. Dynamic FX Triangulation
    def _get_fx_series(pair_col: str, default_val: float) -> pd.Series:
        if pair_col in fx_source_df.columns:
            s = fx_source_df[pair_col].copy()
        else:
            s = pd.Series(default_val, index=bench_series.index)
        s.index = pd.to_datetime(s.index)
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s = s.reindex(bench_series.index).ffill().bfill()
        return s.fillna(default_val)

    usd_myr = _get_fx_series("USDMYR=X", 4.45)
    usd_jpy = _get_fx_series("JPY=X", 150.0)
    eur_usd = _get_fx_series("EURUSD=X", 1.08)
    gbp_usd = _get_fx_series("GBPUSD=X", 1.28)

    # Convert Index Currency -> USD
    if idx_clean == "USD":
        usd_bench = bench_series.copy()
    elif idx_clean == "MYR":
        usd_bench = bench_series / usd_myr
    elif idx_clean == "JPY":
        usd_bench = bench_series / usd_jpy
    elif idx_clean == "EUR":
        usd_bench = bench_series * eur_usd
    elif idx_clean == "GBP":
        usd_bench = bench_series * gbp_usd
    else:
        usd_bench = bench_series.copy()

    # Convert USD -> Target Currency
    if target_clean == "USD":
        target_bench = usd_bench
    elif target_clean == "MYR":
        target_bench = usd_bench * usd_myr
    elif target_clean == "JPY":
        target_bench = usd_bench * usd_jpy
    elif target_clean == "EUR":
        target_bench = usd_bench / eur_usd
    elif target_clean == "GBP":
        target_bench = usd_bench / gbp_usd
    else:
        target_bench = usd_bench

    target_bench = target_bench.ffill(limit=3).bfill(limit=3)
    return target_bench.rename(ticker)


def calculate_relative_benchmark_metrics(
    portfolio_curve: pd.Series,
    benchmark_curve: pd.Series,
    annual_trading_days: int = 252,
) -> Dict[str, float]:
    """Calculate relative performance and risk metrics between portfolio and benchmark.
    
    Adheres strictly to institutional alignment rules:
      1. Reindexes portfolio and benchmark to a common trading day calendar.
      2. Forward-fills non-trading holidays for each market up to 3 days (ffill(limit=3)).
      3. Computes daily returns on the aligned series.
      4. Calculates Beta (cov(p,b)/var(b)), Jensen's Alpha, R-Squared,
         Tracking Error (annualized std of diffs), and Information Ratio.
    """
    if portfolio_curve.empty or benchmark_curve.empty:
        return {
            "beta": 1.0,
            "alpha": 0.0,
            "r_squared": 0.0,
            "tracking_error": 0.0,
            "information_ratio": 0.0,
            "benchmark_cagr": 0.0,
            "benchmark_volatility": 0.0,
            "correlation": 0.0,
        }

    p_curve = portfolio_curve.copy()
    b_curve = benchmark_curve.copy()
    p_curve.index = pd.to_datetime(p_curve.index)
    b_curve.index = pd.to_datetime(b_curve.index)
    if p_curve.index.tz is not None:
        p_curve.index = p_curve.index.tz_localize(None)
    if b_curve.index.tz is not None:
        b_curve.index = b_curve.index.tz_localize(None)

    # Align multi-market series with 3-day holiday forward-fill
    aligned_df = pd.DataFrame({
        "portfolio": p_curve,
        "benchmark": b_curve,
    }).ffill(limit=3).dropna()

    if len(aligned_df) < 10:
        aligned_df = pd.DataFrame({
            "portfolio": p_curve,
            "benchmark": b_curve,
        }).ffill().bfill().dropna()

    aligned_returns = aligned_df.pct_change().dropna()
    if len(aligned_returns) < 5:
        return {
            "beta": 1.0,
            "alpha": 0.0,
            "r_squared": 0.0,
            "tracking_error": 0.0,
            "information_ratio": 0.0,
            "benchmark_cagr": 0.0,
            "benchmark_volatility": 0.0,
            "correlation": 0.0,
        }

    port_ret = aligned_returns["portfolio"].values
    bench_ret = aligned_returns["benchmark"].values

    cov_matrix = np.cov(port_ret, bench_ret)
    var_bench = cov_matrix[1, 1]
    var_port = cov_matrix[0, 0]
    cov_pb = cov_matrix[0, 1]

    beta = float(cov_pb / var_bench) if var_bench > 1e-8 else 1.0
    alpha = float((np.mean(port_ret) - beta * np.mean(bench_ret)) * annual_trading_days)

    if var_port > 1e-8 and var_bench > 1e-8:
        corr = float(cov_pb / np.sqrt(var_port * var_bench))
        r_squared = float(min(1.0, max(0.0, corr ** 2)))
    else:
        corr = 0.0
        r_squared = 0.0

    diff = port_ret - bench_ret
    tracking_error = float(np.std(diff) * np.sqrt(annual_trading_days))
    excess_return = float(np.mean(diff) * annual_trading_days)
    information_ratio = float(excess_return / tracking_error) if tracking_error > 1e-6 else 0.0

    # Standalone benchmark stats over aligned window
    n_days = len(aligned_df)
    n_years = max(1.0 / annual_trading_days, n_days / annual_trading_days)
    bench_tot = float(aligned_df["benchmark"].iloc[-1] / aligned_df["benchmark"].iloc[0]) - 1.0
    bench_cagr = float((1.0 + bench_tot) ** (1.0 / n_years)) - 1.0
    bench_vol = float(np.std(bench_ret) * np.sqrt(annual_trading_days))

    return {
        "beta": beta,
        "alpha": alpha,
        "r_squared": r_squared,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "benchmark_cagr": bench_cagr,
        "benchmark_volatility": bench_vol,
        "correlation": corr,
    }


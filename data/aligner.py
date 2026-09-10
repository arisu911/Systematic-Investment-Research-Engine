"""Multi-Market Cross-Calendar Alignment & Multi-Currency Normalization Engine.

Aligns trading calendars across Bursa Malaysia, NYSE, and TSE. Imputes local holidays
up to a 3-day threshold and provides currency conversion into Local Currency, USD Unified,
or MYR Unified while preserving raw indicator scales (^VIX, ^TNX, DX-Y.NYB).
"""

from typing import Dict, List, Optional, Tuple, Any
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

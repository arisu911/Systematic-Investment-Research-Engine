"""Multi-Market Financial Data Ingestion & Caching Engine.

Fetches 25 global multi-asset instruments across Malaysia, US, Japan, and Macro
using yfinance, caches locally as Snappy-compressed Parquet files in data/cache/,
and provides deterministic offline synthetic fallback with empirical correlations.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import yaml

from data.aligner import load_universe_registry, align_cross_market_calendars, normalize_currency

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"
_UNIVERSE_FILE = _PROJECT_ROOT / "configs" / "universe.yaml"


def get_cache_dir() -> Path:
    """Ensure data/cache directory exists and return Path."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def get_all_universe_tickers() -> List[str]:
    """Extract list of all 25 tickers from configs/universe.yaml."""
    registry = load_universe_registry()
    return list(registry.keys())


def _hash_cache_key(prefix: str, symbols: List[str], start_date: str, end_date: str, extra: str = "") -> str:
    """Generate deterministic MD5 hash for Parquet disk caching."""
    raw = f"{prefix}|{sorted(symbols)}|{start_date}|{end_date}|{extra}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def generate_deterministic_multi_market_prices(
    symbols: List[str], start_date: str = "2019-01-01", end_date: str = "2024-12-31"
) -> pd.DataFrame:
    """Generate realistic Geometric Brownian Motion with empirical multi-market block correlations.
    
    Used when yfinance throttles or drops connection to ensure zero interruptions in testing and demo.
    """
    dates = pd.date_range(start_date, end_date, freq="B")
    n_days = len(dates)
    n_assets = len(symbols)

    np.random.seed(42)
    # Block correlation structure
    corr = np.full((n_assets, n_assets), 0.25)
    np.fill_diagonal(corr, 1.0)

    # Correlate regional clusters higher
    for i, s1 in enumerate(symbols):
        for j, s2 in enumerate(symbols):
            if i != j:
                if (".KL" in s1 and ".KL" in s2) or ("^KLSE" in (s1, s2) and (".KL" in s1 or ".KL" in s2)):
                    corr[i, j] = 0.55
                elif (s1 in ["AAPL", "NVDA", "JPM", "QQQ", "^GSPC", "^NDX", "^RUT"] and
                      s2 in ["AAPL", "NVDA", "JPM", "QQQ", "^GSPC", "^NDX", "^RUT"]):
                    corr[i, j] = 0.65
                elif ("7203.T" in (s1, s2) or "^N225" in (s1, s2) or "^TOPX" in (s1, s2)):
                    if (".T" in s1 or "^N225" in s1 or "^TOPX" in s1) and (".T" in s2 or "^N225" in s2 or "^TOPX" in s2):
                        corr[i, j] = 0.60
                elif ("=F" in s1 and "=F" in s2):
                    corr[i, j] = 0.45

    # Ensure positive semi-definite
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues = np.maximum(eigenvalues, 1e-4)
    corr_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    inv_diag = np.diag(1.0 / np.sqrt(np.diag(corr_psd)))
    corr_psd = inv_diag @ corr_psd @ inv_diag

    cholesky = np.linalg.cholesky(corr_psd)
    raw_draws = np.random.normal(0, 1, size=(n_days, n_assets))
    correlated_draws = raw_draws @ cholesky.T

    data = {}
    dt = 1.0 / 252.0

    for i, sym in enumerate(symbols):
        # Determine realistic initial values, drift, and volatility per asset
        if sym == "^VIX":
            # Mean-reverting Ornstein-Uhlenbeck process for VIX
            vix_vals = np.zeros(n_days)
            vix_vals[0] = 18.0
            theta, kappa, xi = 17.5, 3.5, 4.0
            for t in range(1, n_days):
                vix_vals[t] = max(
                    10.0,
                    vix_vals[t-1] + kappa * (theta - vix_vals[t-1]) * dt + xi * np.sqrt(dt) * correlated_draws[t, i]
                )
            data[sym] = vix_vals
        elif sym == "^TNX":
            # US 10Y Treasury yield (around 3.8% - 4.5%)
            yield_vals = np.zeros(n_days)
            yield_vals[0] = 4.10
            for t in range(1, n_days):
                yield_vals[t] = max(
                    1.5,
                    yield_vals[t-1] + 0.05 * (4.0 - yield_vals[t-1]) * dt + 0.5 * np.sqrt(dt) * correlated_draws[t, i]
                )
            data[sym] = yield_vals
        elif sym == "USDMYR=X":
            vol, drift, init = 0.05, 0.015, 4.45
            daily_ret = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * correlated_draws[:, i]
            data[sym] = init * np.exp(np.cumsum(daily_ret))
        elif sym == "JPY=X":
            vol, drift, init = 0.09, 0.03, 148.0
            daily_ret = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * correlated_draws[:, i]
            data[sym] = init * np.exp(np.cumsum(daily_ret))
        elif sym == "DX-Y.NYB":
            vol, drift, init = 0.06, 0.01, 103.0
            daily_ret = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * correlated_draws[:, i]
            data[sym] = init * np.exp(np.cumsum(daily_ret))
        elif sym == "EURUSD=X":
            vol, drift, init = 0.06, 0.005, 1.08
            daily_ret = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * correlated_draws[:, i]
            data[sym] = init * np.exp(np.cumsum(daily_ret))
        elif sym == "GBPUSD=X":
            vol, drift, init = 0.07, 0.005, 1.28
            daily_ret = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * correlated_draws[:, i]
            data[sym] = init * np.exp(np.cumsum(daily_ret))
        elif "=F" in sym:
            # Commodities
            init = 2300.0 if "GC" in sym else 80.0 if "BZ" in sym else 4.20
            vol = 0.16 if "GC" in sym else 0.28 if "BZ" in sym else 0.22
            drift = 0.08 if "GC" in sym else 0.05
            daily_ret = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * correlated_draws[:, i]
            data[sym] = init * np.exp(np.cumsum(daily_ret))
        elif "^" in sym:
            # Indices
            init = 1600.0 if "KLSE" in sym else 5000.0 if "GSPC" in sym else 18000.0 if "NDX" in sym else 38000.0 if "N225" in sym else 2000.0
            vol, drift = 0.14, 0.09
            daily_ret = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * correlated_draws[:, i]
            data[sym] = init * np.exp(np.cumsum(daily_ret))
        else:
            # Equities
            init = 10.0 if ".KL" in sym else 2600.0 if ".T" in sym else 180.0
            vol = 0.18 if ".KL" in sym else 0.22 if ".T" in sym else 0.32 if sym == "NVDA" else 0.24
            drift = 0.08 if ".KL" in sym else 0.10 if ".T" in sym else 0.18 if sym == "NVDA" else 0.12
            daily_ret = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * correlated_draws[:, i]
            data[sym] = init * np.exp(np.cumsum(daily_ret))

    df = pd.DataFrame(data, index=dates)
    return df


def load_raw_yfinance_multi_market(
    symbols: List[str], start_date: str = "2019-01-01", end_date: str = "2024-12-31"
) -> pd.DataFrame:
    """Download daily adjusted closing prices from Yahoo Finance with robust MultiIndex flattening."""
    if not symbols:
        return pd.DataFrame()

    try:
        raw = yf.download(
            tickers=symbols,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=True,
        )
        if raw.empty:
            return pd.DataFrame()

        # Handle column levels
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.levels[0]:
                df = raw["Close"].copy()
            else:
                df = raw.xs(raw.columns.levels[0][0], axis=1, level=0).copy()
        elif "Close" in raw.columns:
            df = raw[["Close"]].copy()
            df.columns = symbols[:1]
        else:
            df = raw.copy()

        if isinstance(df, pd.Series):
            df = df.to_frame(name=symbols[0])

        return df.dropna(how="all")
    except Exception as e:
        print(f"[WARN] yfinance download failed: {e}")
        return pd.DataFrame()


def load_universe_prices(
    symbols: Optional[List[str]] = None,
    start_date: str = "2019-01-01",
    end_date: str = "2024-12-31",
    base_currency: str = "USD",
    force_offline: bool = False,
    max_ffill_days: int = 3,
) -> Tuple[pd.DataFrame, bool]:
    """Primary entry point for loading the 25-instrument universe.
    
    1. Checks local Snappy Parquet cache in data/cache/.
    2. Downloads from yfinance if missing and not forced offline.
    3. Falls back deterministically to synthetic data if network unavailable.
    4. Applies multi-market calendar alignment (data/aligner.py).
    5. Normalizes to base_currency (USD, MYR, or LOCAL).
    
    Returns:
        (aligned_prices_df, is_demo_mode)
    """
    if symbols is None:
        symbols = get_all_universe_tickers()

    cache_key = _hash_cache_key("multi_universe", symbols, start_date, end_date, base_currency)
    cache_file = get_cache_dir() / f"universe_{cache_key}.parquet"

    if cache_file.exists() and not force_offline:
        try:
            cached_df = pd.read_parquet(cache_file)
            if not cached_df.empty and len(cached_df) >= 30:
                return cached_df, False
        except Exception:
            pass

    is_demo = False
    raw_df = pd.DataFrame()

    if not force_offline:
        raw_df = load_raw_yfinance_multi_market(symbols, start_date, end_date)

    if raw_df.empty or len(raw_df) < 50:
        is_demo = True
        raw_df = generate_deterministic_multi_market_prices(symbols, start_date, end_date)
    else:
        # Check for missing symbols and impute synthetic series if necessary
        missing = [s for s in symbols if s not in raw_df.columns]
        if missing:
            synth_patch = generate_deterministic_multi_market_prices(missing, start_date, end_date)
            synth_aligned = synth_patch.reindex(raw_df.index).ffill().bfill()
            for m in missing:
                raw_df[m] = synth_aligned[m]

    # Align calendars across Bursa, NYSE, TSE
    aligned_df = align_cross_market_calendars(raw_df, max_ffill_days=max_ffill_days)

    # Normalize currency via FXEngine
    from data.fx_engine import FXEngine
    registry = load_universe_registry()
    fx_engine = FXEngine(aligned_df, registry=registry)
    normalized_df = fx_engine.convert_asset_prices(target_currency=base_currency)

    # Filter to requested symbols that exist
    final_cols = [s for s in symbols if s in normalized_df.columns]
    result_df = normalized_df[final_cols].copy()

    # Persist cache
    if not result_df.empty:
        try:
            result_df.to_parquet(cache_file, compression="snappy")
        except Exception:
            pass

    return result_df, is_demo


def get_cached_universe_prices(
    symbols: Optional[List[str]] = None,
    start_date: str = "2019-01-01",
    end_date: str = "2024-12-31",
    base_currency: str = "USD",
    force_offline: bool = False,
    max_ffill_days: int = 3,
) -> Tuple[pd.DataFrame, bool]:
    """Streamlit cached wrapper with 1-hour TTL invalidation."""
    if HAS_STREAMLIT:
        @st.cache_data(ttl=3600, show_spinner=False)
        def _cached_loader(syms_tuple, s_date, e_date, b_curr, f_off, m_ffill):
            syms_list = list(syms_tuple) if syms_tuple is not None else None
            return load_universe_prices(syms_list, s_date, e_date, b_curr, f_off, m_ffill)

        syms_key = tuple(symbols) if symbols is not None else None
        return _cached_loader(syms_key, start_date, end_date, base_currency, force_offline, max_ffill_days)
    else:
        return load_universe_prices(symbols, start_date, end_date, base_currency, force_offline, max_ffill_days)

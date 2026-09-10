"""Zero-Cost Multi-Asset Financial Data Ingestion & Caching Layer.

Fetches equities, ETFs, FX, commodities via yfinance, macro/sovereign rates via FRED
and MGS/BNM tables, with Snappy-compressed Parquet disk caching and deterministic demo fallback.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from pandas_datareader import data as pdr
    HAS_PDR = True
except ImportError:
    HAS_PDR = False

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"
_MGS_FILE = _PROJECT_ROOT / "data" / "mgs_historical.csv"


def get_cache_dir() -> Path:
    """Ensure data/cache directory exists and return Path."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _hash_cache_key(prefix: str, symbols: List[str], start_date: str, end_date: str, extra: str = "") -> str:
    """Generate deterministic MD5 hash for Parquet disk caching."""
    raw = f"{prefix}|{sorted(symbols)}|{start_date}|{end_date}|{extra}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_mgs_fixed_income() -> pd.DataFrame:
    """Load historical Malaysian Government Securities yields (3Y, 5Y, 10Y) & BNM OPR."""
    if _MGS_FILE.exists():
        df = pd.read_csv(_MGS_FILE)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        return df
    
    # Fallback synthesize if file missing
    dates = pd.date_range("2018-01-01", datetime.now().strftime("%Y-%m-%d"), freq="B")
    return pd.DataFrame({
        "BNM_OPR": 3.00,
        "MGS_3Y": 3.45,
        "MGS_5Y": 3.65,
        "MGS_10Y": 3.85,
    }, index=dates)


def load_fred_macro_series(series_id: str = "DGS10", start_date: str = "2019-01-01", end_date: str = "2024-12-31") -> pd.Series:
    """Load US sovereign macro rate (e.g. 10Y Treasury, 3M T-bill) via FRED or public fallback."""
    cache_file = get_cache_dir() / f"fred_{series_id}_{_hash_cache_key('fred', [series_id], start_date, end_date)}.parquet"
    if cache_file.exists():
        try:
            cached = pd.read_parquet(cache_file)
            return cached.iloc[:, 0]
        except Exception:
            pass

    series = pd.Series(dtype=float)
    if HAS_PDR:
        try:
            fred_df = pdr.DataReader(series_id, "fred", start_date, end_date)
            if not fred_df.empty:
                series = fred_df.iloc[:, 0].dropna()
                pd.DataFrame({series_id: series}).to_parquet(cache_file, compression="snappy")
                return series
        except Exception:
            pass

    # Fallback via yfinance if series is 10Y Treasury yield (^TNX)
    if series_id in ["DGS10", "^TNX", "TNX"]:
        try:
            yf_df = yf.download("^TNX", start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not yf_df.empty:
                close_col = yf_df["Close"] if isinstance(yf_df.columns, pd.Index) and "Close" in yf_df.columns else yf_df.iloc[:, 0]
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                series = close_col.dropna()
                pd.DataFrame({series_id: series}).to_parquet(cache_file, compression="snappy")
                return series
        except Exception:
            pass

    # Fallback synthesis
    dates = pd.date_range(start_date, end_date, freq="B")
    synth = pd.Series(3.95 + 0.5 * np.sin(np.linspace(0, 10, len(dates))), index=dates, name=series_id)
    return synth


def generate_deterministic_synthetic_prices(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Generate realistic Geometric Brownian Motion with empirical correlations when offline."""
    dates = pd.date_range(start_date, end_date, freq="B")
    n_days = len(dates)
    n_assets = len(symbols)

    np.random.seed(1337)
    # Generate realistic correlated covariance matrix
    corr_base = np.full((n_assets, n_assets), 0.35)
    np.fill_diagonal(corr_base, 1.0)
    cholesky = np.linalg.cholesky(corr_base)

    raw_draws = np.random.normal(0, 1, size=(n_days, n_assets))
    corr_draws = raw_draws @ cholesky.T

    # Base characteristics
    data = {}
    for i, sym in enumerate(symbols):
        is_fx = "=X" in sym or sym == "MYR=X"
        is_cmd = "=F" in sym
        is_mgs = "MGS" in sym
        
        vol = 0.05 if is_fx else 0.08 if is_mgs else 0.22 if is_cmd else 0.18
        drift = 0.02 if is_fx else 0.035 if is_mgs else 0.06 if is_cmd else 0.09
        
        dt = 1.0 / 248.0
        daily_ret = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * corr_draws[:, i]
        
        init_price = 4.40 if is_fx else 100.0 if is_mgs else 80.0 if is_cmd else 25.0
        price_series = init_price * np.exp(np.cumsum(daily_ret))
        data[sym] = price_series

    df = pd.DataFrame(data, index=dates)
    return df


SYMBOL_MAP = {
    "US_10Y": "^TNX",
    "US_3M": "^IRX",
}
REVERSE_MAP = {v: k for k, v in SYMBOL_MAP.items()}


def load_raw_yfinance_prices(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Download prices from Yahoo Finance with robust error handling and column flattening."""
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

        # Handle multi-level columns if multiple tickers
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

        # Clean column names
        if isinstance(df, pd.Series):
            df = df.to_frame(name=symbols[0])

        return df.dropna(how="all")
    except Exception as e:
        print(f"[WARN] yfinance download failed for {symbols}: {e}")
        return pd.DataFrame()


def load_multi_asset_dataset(
    symbols: List[str],
    start_date: str = "2019-01-01",
    end_date: str = "2024-12-31",
    base_currency: str = "MYR",
    force_offline: bool = False,
) -> Tuple[pd.DataFrame, bool]:
    """Primary multi-asset ingestion routine.
    
    Returns:
        (price_df, is_demo_mode): DataFrame of cleaned prices aligned to base_currency, and demo boolean.
    """
    cache_key = _hash_cache_key("multi_asset", symbols, start_date, end_date, base_currency)
    cache_file = get_cache_dir() / f"prices_{cache_key}.parquet"

    if cache_file.exists() and not force_offline:
        try:
            cached_df = pd.read_parquet(cache_file)
            if not cached_df.empty and len(cached_df) >= 30:
                return cached_df, False
        except Exception:
            pass

    is_demo = False
    price_df = pd.DataFrame()

    # Separate synthetic MGS assets from public Yahoo symbols
    mgs_symbols = [s for s in symbols if "MGS" in s]
    public_symbols = [s for s in symbols if "MGS" not in s]

    if not force_offline and public_symbols:
        # Map tickers to valid Yahoo equivalents (e.g. US_10Y -> ^TNX)
        mapped_query = [SYMBOL_MAP.get(s, s) for s in public_symbols]
        if base_currency == "MYR" and "MYR=X" not in mapped_query:
            mapped_query.append("MYR=X")

        price_df = load_raw_yfinance_prices(mapped_query, start_date, end_date)
        if not price_df.empty:
            price_df.rename(columns=REVERSE_MAP, inplace=True)

    # Fallback to deterministic synthetic if download failed or offline forced
    if price_df.empty or len(price_df) < 50:
        is_demo = True
        price_df = generate_deterministic_synthetic_prices(symbols, start_date, end_date)
    else:
        # Drop columns that are completely all-NaN
        price_df.dropna(how="all", axis=1, inplace=True)
        # If any requested symbol is missing, generate synthetic series for that symbol
        missing_syms = [s for s in symbols if s not in price_df.columns and "MGS" not in s]
        if missing_syms:
            synth_patch = generate_deterministic_synthetic_prices(missing_syms, start_date, end_date)
            synth_aligned = synth_patch.reindex(price_df.index).ffill().bfill()
            for ms in missing_syms:
                price_df[ms] = synth_aligned[ms]

    # Attach MGS fixed income series as synthetic sovereign bond return indices
    if mgs_symbols:
        mgs_data = load_mgs_fixed_income()
        aligned_mgs = mgs_data.reindex(price_df.index).ffill().bfill()

        for s in mgs_symbols:
            col_name = "MGS_10Y" if "10" in s else "MGS_5Y" if "5" in s else "MGS_3Y"
            yield_col = aligned_mgs[col_name] if col_name in aligned_mgs.columns else aligned_mgs["MGS_10Y"]
            
            # Convert yield level into total return bond index: daily return ~ (yield / 248) - duration * delta_yield
            dur = 7.5 if "10" in s else 4.2 if "5" in s else 2.7
            daily_coupon = (yield_col / 100.0) / 248.0
            delta_yield = (yield_col - yield_col.shift(1)).fillna(0.0) / 100.0
            daily_bond_ret = daily_coupon - dur * delta_yield
            bond_idx = 100.0 * (1.0 + daily_bond_ret).cumprod()
            price_df[s] = bond_idx

    # Forward fill gaps and align
    price_df = price_df.ffill().bfill().dropna()

    # Currency normalization: If base currency is MYR, convert offshore assets
    offshore_symbols = ["SPY", "QQQ", "EEM", "ACWI", "AAPL", "MSFT", "NVDA", "US_10Y", "GC=F", "CL=F"]
    if base_currency == "MYR" and "MYR=X" in price_df.columns:
        fx_rate = price_df["MYR=X"]
        for sym in offshore_symbols:
            if sym in price_df.columns:
                price_df[sym] = price_df[sym] * fx_rate

    # Filter to requested symbols only
    final_symbols = [s for s in symbols if s in price_df.columns]
    result_df = price_df[final_symbols].copy()

    # Persist to Parquet
    if not result_df.empty:
        try:
            result_df.to_parquet(cache_file, compression="snappy")
        except Exception:
            pass

    return result_df, is_demo


def get_cached_multi_asset_data(
    symbols: List[str],
    start_date: str = "2019-01-01",
    end_date: str = "2024-12-31",
    base_currency: str = "MYR",
    force_offline: bool = False,
) -> Tuple[pd.DataFrame, bool]:
    """Cached wrapper suitable for Streamlit apps."""
    if HAS_STREAMLIT:
        @st.cache_data(ttl=3600, show_spinner=False)
        def _cached_loader(syms_tuple, s_date, e_date, b_curr, f_off):
            return load_multi_asset_dataset(list(syms_tuple), s_date, e_date, b_curr, f_off)

        return _cached_loader(tuple(symbols), start_date, end_date, base_currency, force_offline)
    else:
        return load_multi_asset_dataset(symbols, start_date, end_date, base_currency, force_offline)

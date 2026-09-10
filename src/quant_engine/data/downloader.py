"""Unified data loader and caching orchestrator with automatic demo fallback."""

import logging
from typing import Dict, Any, Tuple, Optional
import pandas as pd
from quant_engine.data.providers.base import DataProvider
from quant_engine.data.providers.yahoo import YahooFinanceProvider
from quant_engine.data.providers.fred import FREDProvider
from quant_engine.data.providers.demo import DemoDataProvider
from quant_engine.data.cache import DataCache
from quant_engine.data.cleaner import DataCleaner
from quant_engine.data.schemas import DataHealthReport, validate_ohlcv_dataframe

logger = logging.getLogger(__name__)


class DataDownloader:
    """High-level multi-asset loader handling caching, API queries, and demo fallback."""

    def __init__(
        self,
        use_cache: bool = True,
        force_demo: bool = False,
        cache_ttl_hours: int = 24,
    ):
        self.use_cache = use_cache
        self.force_demo = force_demo
        self.cache = DataCache(ttl_hours=cache_ttl_hours)
        self.yahoo_provider = YahooFinanceProvider()
        self.fred_provider = FREDProvider()
        self.demo_provider = DemoDataProvider()

    def load_price_data(
        self,
        symbol: str,
        start_date: str = "2018-01-01",
        end_date: str = "2025-12-31",
    ) -> Tuple[pd.DataFrame, DataHealthReport]:
        """Fetch, clean, validate, and cache price data for a single symbol."""
        cache_key = f"price_{symbol}_{start_date}_{end_date}"

        # 1. Check Cache
        if self.use_cache and not self.force_demo:
            cached_df = self.cache.get(cache_key)
            if cached_df is not None and not cached_df.empty:
                report = validate_ohlcv_dataframe(cached_df, symbol=symbol)
                report.provider = "Parquet Cache"
                report.is_demo = False
                return cached_df, report

        # 2. Try primary live provider unless force_demo
        df = pd.DataFrame()
        provider_name = self.yahoo_provider.name
        is_demo = False

        if not self.force_demo:
            try:
                df = self.yahoo_provider.get_prices(symbol, start_date, end_date)
            except Exception as e:
                logger.warning("Primary provider fetch failed for %s: %s", symbol, e)
                df = pd.DataFrame()

        # 3. Fallback to DemoDataProvider if live data is empty
        if df.empty:
            logger.info("Using DemoDataProvider fallback for %s", symbol)
            df = self.demo_provider.get_prices(symbol, start_date, end_date)
            provider_name = self.demo_provider.name
            is_demo = True

        # 4. Clean data
        cleaned_df = DataCleaner.clean_ohlcv(df)

        # 5. Validate hygiene
        report = validate_ohlcv_dataframe(cleaned_df, symbol=symbol)
        report.provider = provider_name
        report.is_demo = is_demo

        # 6. Cache if healthy
        if self.use_cache and not is_demo and report.is_usable:
            self.cache.put(cache_key, cleaned_df)

        return cleaned_df, report

    def load_macro_data(
        self,
        series_id: str,
        start_date: str = "2018-01-01",
        end_date: str = "2025-12-31",
    ) -> Tuple[pd.Series, bool]:
        """Load macroeconomic series with demo fallback."""
        if not self.force_demo:
            try:
                # Try FRED first
                s = self.fred_provider.get_macro(series_id, start_date, end_date)
                if not s.empty:
                    return s, False
            except Exception:
                pass

            try:
                # Try Yahoo proxy
                s = self.yahoo_provider.get_macro(series_id, start_date, end_date)
                if not s.empty:
                    return s, False
            except Exception:
                pass

        # Demo fallback
        s = self.demo_provider.get_macro(series_id, start_date, end_date)
        return s, True

"""Yahoo Finance market data provider."""

import logging
from typing import List, Dict, Any, Union, Optional
import pandas as pd
import yfinance as yf
from quant_engine.data.providers.base import DataProvider

logger = logging.getLogger(__name__)


class YahooFinanceProvider(DataProvider):
    """Yahoo Finance API wrapper for global equities, indices, FX, and commodities."""

    @property
    def name(self) -> str:
        return "Yahoo Finance"

    def _fetch_ticker(self, symbol: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
            if df.empty:
                logger.warning("Empty response from Yahoo Finance for %s", symbol)
                return pd.DataFrame()
            
            # Standardize index and strip timezone to avoid multi-market tz mismatch
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index = pd.to_datetime(df.index).normalize()
            df.index.name = "Date"

            # Rename or retain standard columns
            expected = ["Open", "High", "Low", "Close", "Volume"]
            for col in expected:
                if col not in df.columns:
                    logger.warning("Column %s missing in Yahoo response for %s", col, symbol)

            if "Adj Close" not in df.columns:
                df["Adj Close"] = df["Close"]

            return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].sort_index()

        except Exception as e:
            logger.error("Error fetching %s from Yahoo Finance: %s", symbol, e)
            return pd.DataFrame()

    def get_prices(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        if isinstance(symbols, str):
            return self._fetch_ticker(symbols, start_date, end_date)
        
        results = {}
        for sym in symbols:
            df = self._fetch_ticker(sym, start_date, end_date)
            if not df.empty:
                results[sym] = df
        return results

    def get_macro(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.Series:
        df = self._fetch_ticker(series_id, start_date, end_date)
        if df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float, name=series_id)
        series = df["Close"].copy()
        series.name = series_id
        return series

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            return {
                "symbol": symbol,
                "name": info.get("shortName") or info.get("longName") or symbol,
                "currency": info.get("currency", "USD"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "provider": self.name,
                "is_demo": False,
            }
        except Exception:
            return {"symbol": symbol, "name": symbol, "provider": self.name, "is_demo": False}

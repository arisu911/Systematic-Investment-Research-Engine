"""Federal Reserve Economic Data (FRED) macroeconomic provider."""

import os
import logging
from typing import Dict, Any, Union, List, Optional
import pandas as pd
import requests
from quant_engine.data.providers.base import DataProvider

logger = logging.getLogger(__name__)


class FREDProvider(DataProvider):
    """Macroeconomic data provider using St. Louis Fed FRED API or public endpoints."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY")

    @property
    def name(self) -> str:
        return "FRED (Federal Reserve Economic Data)"

    def get_prices(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """FRED provides macroeconomic time series, not equity OHLCV.
        
        Returns price-like DataFrame with 'Close' set to the series observation.
        """
        if isinstance(symbols, str):
            series = self.get_macro(symbols, start_date, end_date)
            df = pd.DataFrame({"Close": series, "Open": series, "High": series, "Low": series, "Volume": 0})
            return df
        return {sym: self.get_prices(sym, start_date, end_date) for sym in symbols}

    def get_macro(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.Series:
        """Fetch macroeconomic series by FRED ID (e.g., DGS10, FEDFUNDS, CPIAUCSL)."""
        if self.api_key:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={self.api_key}&file_type=json"
            if start_date:
                url += f"&observation_start={start_date}"
            if end_date:
                url += f"&observation_end={end_date}"
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json().get("observations", [])
                    records = []
                    for obs in data:
                        val = obs.get("value")
                        if val != ".":
                            records.append({"Date": obs["date"], series_id: float(val)})
                    if records:
                        df = pd.DataFrame(records).set_index("Date")
                        df.index = pd.to_datetime(df.index)
                        return df[series_id]
            except Exception as e:
                logger.warning("Failed to fetch %s via FRED API: %s", series_id, e)

        # Fallback to direct public CSV endpoint (no API key required)
        try:
            csv_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            df = pd.read_csv(csv_url, index_col=0, parse_dates=True, na_values=".")
            df = df.dropna()
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            if not df.empty:
                series = df.iloc[:, 0]
                series.name = series_id
                return series
        except Exception as e:
            logger.warning("Failed to fetch %s via public FRED CSV: %s", series_id, e)

        return pd.Series(dtype=float, name=series_id)

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "name": f"FRED Series: {symbol}",
            "provider": self.name,
            "category": "Macroeconomic",
            "is_demo": False,
        }

"""Custom CSV and Parquet file data provider."""

from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import pandas as pd
from quant_engine.data.providers.base import DataProvider


class CSVDataProvider(DataProvider):
    """Provider that loads user-supplied CSV or Parquet market data files."""

    def __init__(self, data_directory: Optional[Union[str, Path]] = None):
        self.data_directory = Path(data_directory) if data_directory else Path("data")

    @property
    def name(self) -> str:
        return "CSV/Parquet File Provider"

    def _find_file(self, symbol: str) -> Optional[Path]:
        sanitized = symbol.replace("^", "").replace("=", "").replace(".", "_")
        candidates = [
            self.data_directory / f"{symbol}.csv",
            self.data_directory / f"{symbol}.parquet",
            self.data_directory / f"{sanitized}.csv",
            self.data_directory / f"{sanitized}.parquet",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def get_prices(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        if isinstance(symbols, str):
            path = self._find_file(symbols)
            if not path:
                return pd.DataFrame()
            
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                
            df.index = pd.to_datetime(df.index)
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            return df.sort_index()

        return {sym: self.get_prices(sym, start_date, end_date) for sym in symbols}

    def get_macro(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.Series:
        df = self.get_prices(series_id, start_date, end_date)
        if df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float, name=series_id)
        series = df["Close"].copy()
        series.name = series_id
        return series

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        path = self._find_file(symbol)
        return {
            "symbol": symbol,
            "name": symbol,
            "provider": self.name,
            "file_path": str(path) if path else None,
            "is_demo": False,
        }

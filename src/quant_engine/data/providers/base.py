"""Abstract base class for all data providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
import pandas as pd
from quant_engine.data.schemas import DataHealthReport, validate_ohlcv_dataframe


class DataProvider(ABC):
    """Abstract interface for historical price, macro, and metadata providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        pass

    @abstractmethod
    def get_prices(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Fetch historical OHLCV pricing data.
        
        If a single symbol is provided, returns DataFrame with columns [Open, High, Low, Close, Volume].
        If multiple symbols are provided, returns Dict[symbol, DataFrame].
        """
        pass

    @abstractmethod
    def get_macro(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.Series:
        """Fetch macroeconomic or external reference time-series."""
        pass

    @abstractmethod
    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """Fetch descriptive metadata for a given symbol."""
        pass

    def validate_data(self, df: pd.DataFrame, symbol: str = "Unknown") -> DataHealthReport:
        """Run hygiene checks on data fetched from this provider."""
        report = validate_ohlcv_dataframe(df, symbol=symbol)
        report.provider = self.name
        return report

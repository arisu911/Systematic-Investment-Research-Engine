"""Data schemas, health report structures, and validation rules."""

from typing import List, Dict, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field


STANDARD_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
STANDARD_ADJ_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


class DataHealthReport(BaseModel):
    """Diagnostic report on data quality and hygiene."""
    
    symbol: str
    total_bars: int
    start_date: str
    end_date: str
    missing_dates_count: int = 0
    duplicate_dates_count: int = 0
    zero_negative_prices_count: int = 0
    nan_values_count: int = 0
    stale_bars_count: int = 0
    extreme_spikes_count: int = 0
    data_health_score: float = Field(default=100.0, ge=0.0, le=100.0)
    warnings: List[str] = Field(default_factory=list)
    is_usable: bool = True
    provider: str = "unknown"
    is_demo: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def validate_ohlcv_dataframe(df: pd.DataFrame, symbol: str = "Unknown") -> DataHealthReport:
    """Run rigorous sanity checks on an OHLCV price DataFrame."""
    warnings = []
    
    if df.empty:
        return DataHealthReport(
            symbol=symbol,
            total_bars=0,
            start_date="",
            end_date="",
            data_health_score=0.0,
            is_usable=False,
            warnings=["DataFrame is completely empty."],
        )
        
    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        warnings.append("Index is not a DatetimeIndex; chronological checks may fail.")
        
    total_bars = len(df)
    start_date = str(df.index[0].date()) if len(df) > 0 else ""
    end_date = str(df.index[-1].date()) if len(df) > 0 else ""
    
    # Check duplicate dates
    duplicate_dates_count = int(df.index.duplicated().sum())
    if duplicate_dates_count > 0:
        warnings.append(f"Found {duplicate_dates_count} duplicate timestamp rows.")
        
    # Check required columns
    missing_cols = [c for c in ["Open", "High", "Low", "Close"] if c not in df.columns]
    if missing_cols:
        warnings.append(f"Missing essential price columns: {missing_cols}")
        
    # Count NaNs
    nan_count = int(df[["Open", "High", "Low", "Close"]].isna().sum().sum()) if not missing_cols else int(df.isna().sum().sum())
    if nan_count > 0:
        warnings.append(f"Found {nan_count} NaN values in price columns.")
        
    # Count zero or negative prices
    zero_neg_count = 0
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            zero_neg_count += int((df[col] <= 0).sum())
    if zero_neg_count > 0:
        warnings.append(f"Found {zero_neg_count} zero or negative price entries.")
        
    # Stale bars (flat price for 5+ consecutive bars)
    stale_bars = 0
    if "Close" in df.columns and len(df) > 5:
        consecutive_flat = (df["Close"].diff() == 0).astype(int)
        stale_bars = int((consecutive_flat.rolling(5).sum() >= 4).sum())
        if stale_bars > 0:
            warnings.append(f"Detected {stale_bars} potentially stale consecutive price bars.")

    # Extreme single-bar spikes (> 50% price change in 1 bar without split)
    extreme_spikes = 0
    if "Close" in df.columns and len(df) > 2:
        returns = df["Close"].pct_change().abs()
        extreme_spikes = int((returns > 0.50).sum())
        if extreme_spikes > 0:
            warnings.append(f"Detected {extreme_spikes} extreme single-day price jumps (>50%).")
            
    # Calculate health score deduction
    score = 100.0
    score -= min(30.0, duplicate_dates_count * 5.0)
    score -= min(40.0, (nan_count / max(1, total_bars)) * 100.0)
    score -= min(40.0, (zero_neg_count / max(1, total_bars)) * 100.0)
    score -= min(15.0, stale_bars * 1.5)
    score -= min(20.0, extreme_spikes * 5.0)
    score = max(0.0, round(score, 2))
    
    is_usable = score >= 50.0 and len(missing_cols) == 0 and total_bars >= 20
    
    return DataHealthReport(
        symbol=symbol,
        total_bars=total_bars,
        start_date=start_date,
        end_date=end_date,
        missing_dates_count=0,
        duplicate_dates_count=duplicate_dates_count,
        zero_negative_prices_count=zero_neg_count,
        nan_values_count=nan_count,
        stale_bars_count=stale_bars,
        extreme_spikes_count=extreme_spikes,
        data_health_score=score,
        warnings=warnings,
        is_usable=is_usable,
    )

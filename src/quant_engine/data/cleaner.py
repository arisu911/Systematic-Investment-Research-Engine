"""Data cleaning, timezone normalization, and corporate adjustment sanitization."""

import numpy as np
import pandas as pd


class DataCleaner:
    """Sanitizes raw OHLCV datasets and aligns multi-asset calendars."""

    @staticmethod
    def clean_ohlcv(df: pd.DataFrame, fill_limit: int = 3) -> pd.DataFrame:
        """Clean an individual asset OHLCV DataFrame."""
        if df.empty:
            return df.copy()

        cleaned = df.copy()

        # 1. Normalize index to tz-naive DatetimeIndex
        if not isinstance(cleaned.index, pd.DatetimeIndex):
            cleaned.index = pd.to_datetime(cleaned.index)
        if cleaned.index.tz is not None:
            cleaned.index = cleaned.index.tz_localize(None)
        cleaned.index = cleaned.index.normalize()
        cleaned.index.name = "Date"

        # 2. Drop duplicate dates (keep last)
        cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
        cleaned = cleaned.sort_index()

        # 3. Replace zero and negative values with NaN
        price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in cleaned.columns]
        for col in price_cols:
            cleaned.loc[cleaned[col] <= 0, col] = np.nan

        # 4. Conservative forward-fill (max fill_limit bars)
        cleaned[price_cols] = cleaned[price_cols].ffill(limit=fill_limit)

        # 5. Drop any remaining leading NaNs
        cleaned = cleaned.dropna(subset=["Close"])

        # 6. Sanity bounds: High >= max(Open, Close), Low <= min(Open, Close)
        if all(c in cleaned.columns for c in ["Open", "High", "Low", "Close"]):
            cleaned["High"] = np.maximum(cleaned["High"], np.maximum(cleaned["Open"], cleaned["Close"]))
            cleaned["Low"] = np.minimum(cleaned["Low"], np.minimum(cleaned["Open"], cleaned["Close"]))

        # 7. Clean Volume
        if "Volume" in cleaned.columns:
            cleaned["Volume"] = cleaned["Volume"].fillna(0).astype(np.int64)

        return cleaned

    @staticmethod
    def align_multi_asset(
        asset_dict: dict[str, pd.DataFrame],
        method: str = "inner",
        ffill_macro: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Align multiple assets to a common trading calendar.
        
        Args:
            asset_dict: Mapping of symbol -> DataFrame
            method: 'inner' (intersection) or 'outer' (union with forward-fill)
            ffill_macro: If True, macroeconomic and FX series are forward-filled across equity holidays.
        """
        if not asset_dict:
            return {}

        cleaned_dict = {sym: DataCleaner.clean_ohlcv(df) for sym, df in asset_dict.items() if not df.empty}
        if not cleaned_dict:
            return {}

        if method == "inner":
            common_idx = None
            for df in cleaned_dict.values():
                if common_idx is None:
                    common_idx = df.index
                else:
                    common_idx = common_idx.intersection(df.index)
            return {sym: df.loc[common_idx].copy() for sym, df in cleaned_dict.items()}

        # Method == "outer"
        all_dates = None
        for df in cleaned_dict.values():
            if all_dates is None:
                all_dates = df.index
            else:
                all_dates = all_dates.union(df.index)

        all_dates = all_dates.sort_values()
        aligned = {}
        for sym, df in cleaned_dict.items():
            reindexed = df.reindex(all_dates)
            if ffill_macro:
                price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in reindexed.columns]
                reindexed[price_cols] = reindexed[price_cols].ffill(limit=5)
            aligned[sym] = reindexed.dropna(subset=["Close"])
        return aligned

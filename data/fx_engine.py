"""Multi-Currency FX Engine & Triangulation Layer.

Supports real-time and historical conversion across:
- USD (US Dollar)
- MYR (Malaysian Ringgit)
- JPY (Japanese Yen)
- EUR (Euro)
- GBP (British Pound)

Provides unhedged vs. hedged returns modeling and excludes pure volatility
and rate indicators (^VIX, ^TNX, DX-Y.NYB) from currency scaling.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from data.aligner import load_universe_registry


SUPPORTED_CURRENCIES = ["USD", "MYR", "JPY", "EUR", "GBP"]

CURRENCY_SYMBOLS: Dict[str, str] = {
    "USD": "$",
    "MYR": "RM ",
    "JPY": "¥",
    "EUR": "€",
    "GBP": "£",
    "LOCAL": "",
}

FX_TICKERS: Dict[str, str] = {
    "USDMYR": "USDMYR=X",
    "USDJPY": "JPY=X",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
}

INDICATOR_SYMBOLS = {"^VIX", "^TNX", "DX-Y.NYB"}

NON_CONVERTIBLE_ASSETS = {
    "^VIX", "^TNX", "DX-Y.NYB",
    "USDMYR=X", "JPY=X", "EURUSD=X", "GBPUSD=X"
}


def get_currency_symbol(currency_code: str) -> str:
    """Return display symbol for currency (e.g. '$', 'RM ', '¥', '€', '£')."""
    return CURRENCY_SYMBOLS.get(currency_code.upper().strip(), "$")


class FXEngine:
    """Triangulation and multi-currency conversion engine."""

    @staticmethod
    def triangulate_rates(live_quotes: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Construct full bilateral FX triangulation matrix across supported currencies."""
        usd_myr = float(live_quotes.get("USDMYR=X", live_quotes.get("USDMYR", 4.45)))
        usd_jpy = float(live_quotes.get("JPY=X", live_quotes.get("USDJPY", 150.0)))
        eur_usd = float(live_quotes.get("EURUSD=X", live_quotes.get("EURUSD", 1.08)))
        gbp_usd = float(live_quotes.get("GBPUSD=X", live_quotes.get("GBPUSD", 1.28)))

        # USD rates: 1 USD = X Target
        usd_rates = {
            "USD": 1.0,
            "MYR": usd_myr,
            "JPY": usd_jpy,
            "EUR": 1.0 / eur_usd if eur_usd > 0 else 1.0,
            "GBP": 1.0 / gbp_usd if gbp_usd > 0 else 1.0,
        }

        # Build full NxN matrix: rate from base -> target
        matrix = {}
        for base in SUPPORTED_CURRENCIES:
            matrix[base] = {}
            base_to_usd = 1.0 / usd_rates[base]
            for target in SUPPORTED_CURRENCIES:
                matrix[base][target] = base_to_usd * usd_rates[target]

        return matrix

    @classmethod
    def convert_prices_to_base(
        cls,
        prices_df: pd.DataFrame,
        target_currency: str = "USD",
        registry: Optional[Dict[str, Any]] = None,
        is_hedged: bool = False,
    ) -> pd.DataFrame:
        """Convenience method to convert prices to base currency."""
        engine = cls(prices_df=prices_df, registry=registry)
        return engine.convert_asset_prices(target_currency=target_currency, is_hedged=is_hedged)

    @classmethod
    def calculate_asset_returns(
        cls,
        prices_df: pd.DataFrame,
        target_currency: str = "USD",
        is_hedged: bool = False,
        hedged: Optional[bool] = None,
        registry: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Convenience method to compute asset returns in target reporting currency."""
        if hedged is not None:
            is_hedged = hedged
        engine = cls(prices_df=prices_df, registry=registry)
        return engine.compute_asset_returns(target_currency=target_currency, is_hedged=is_hedged)

    def __init__(
        self,
        prices_df: pd.DataFrame,
        registry: Optional[Dict[str, Any]] = None,
    ):
        """Initialize FX engine with aligned price DataFrame."""
        self.prices_df = prices_df.copy()
        self.registry = registry if registry is not None else load_universe_registry()
        self.fx_rates = self._extract_fx_rates()

    def _extract_fx_rates(self) -> Dict[str, pd.Series]:
        """Extract or synthesize the core FX rate series."""
        rates = {}
        index = self.prices_df.index if not self.prices_df.empty else pd.date_range("2021-01-01", periods=10, freq="B")

        # 1. USDMYR=X (1 USD = X MYR)
        if "USDMYR=X" in self.prices_df.columns:
            rates["USDMYR"] = self.prices_df["USDMYR=X"]
        else:
            rates["USDMYR"] = pd.Series(4.45, index=index)

        # 2. JPY=X (1 USD = Y JPY)
        if "JPY=X" in self.prices_df.columns:
            rates["USDJPY"] = self.prices_df["JPY=X"]
        else:
            rates["USDJPY"] = pd.Series(150.0, index=index)

        # 3. EURUSD=X (1 EUR = E USD)
        if "EURUSD=X" in self.prices_df.columns:
            rates["EURUSD"] = self.prices_df["EURUSD=X"]
        else:
            rates["EURUSD"] = pd.Series(1.08, index=index)

        # 4. GBPUSD=X (1 GBP = G USD)
        if "GBPUSD=X" in self.prices_df.columns:
            rates["GBPUSD"] = self.prices_df["GBPUSD=X"]
        else:
            rates["GBPUSD"] = pd.Series(1.28, index=index)

        return rates

    def convert_price_to_usd(self, series: pd.Series, local_currency: str) -> pd.Series:
        """Convert a price series from local currency into USD."""
        loc = local_currency.upper().strip()
        if loc == "USD":
            return series
        elif loc == "MYR":
            return series / self.fx_rates["USDMYR"]
        elif loc == "JPY":
            return series / self.fx_rates["USDJPY"]
        elif loc == "EUR":
            return series * self.fx_rates["EURUSD"]
        elif loc == "GBP":
            return series * self.fx_rates["GBPUSD"]
        return series

    def convert_usd_to_target(self, usd_series: pd.Series, target_currency: str) -> pd.Series:
        """Convert a USD price series into target currency."""
        target = target_currency.upper().strip()
        if target == "USD":
            return usd_series
        elif target == "MYR":
            return usd_series * self.fx_rates["USDMYR"]
        elif target == "JPY":
            return usd_series * self.fx_rates["USDJPY"]
        elif target == "EUR":
            return usd_series / self.fx_rates["EURUSD"]
        elif target == "GBP":
            return usd_series / self.fx_rates["GBPUSD"]
        return usd_series

    def convert_asset_prices(
        self,
        target_currency: str = "USD",
        is_hedged: bool = False,
    ) -> pd.DataFrame:
        """Convert all asset prices in the DataFrame to the target reporting currency.
        
        Args:
            target_currency: One of 'USD', 'MYR', 'JPY', 'EUR', 'GBP', or 'LOCAL'.
            is_hedged: If True, returns prices in local terms (hedging out FX moves).
        """
        target = target_currency.upper().strip()
        if target == "LOCAL" or is_hedged:
            return self.prices_df.copy()

        converted_df = self.prices_df.copy()

        for col in converted_df.columns:
            if col in NON_CONVERTIBLE_ASSETS:
                continue

            meta = self.registry.get(col, {})
            role = meta.get("role", "")
            asset_class = meta.get("asset_class", "")
            local_curr = meta.get("local_currency", "USD")

            if role in ["volatility", "rates", "macro"] or asset_class in ["risk_indicator", "rates", "fx_index", "fx"]:
                continue

            # Triangulate: Local -> USD -> Target
            usd_series = self.convert_price_to_usd(converted_df[col], local_curr)
            converted_df[col] = self.convert_usd_to_target(usd_series, target)

        return converted_df

    def compute_asset_returns(
        self,
        target_currency: str = "USD",
        is_hedged: bool = False,
    ) -> pd.DataFrame:
        """Compute percentage returns in either unhedged target currency or hedged local currency."""
        prices = self.convert_asset_prices(target_currency=target_currency, is_hedged=is_hedged)
        return prices.pct_change().dropna(how="all")

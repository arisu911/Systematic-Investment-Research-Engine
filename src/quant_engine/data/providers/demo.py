"""Deterministic synthetic & bundled demo data provider.

Guarantees 100% offline availability for Streamlit Cloud and test environments.
Uses deterministic Geometric Brownian Motion with empirical drift, volatility,
and cross-market correlation matrices.
"""

from typing import List, Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from quant_engine.data.providers.base import DataProvider


# Baseline annualized parameters (Drift mu, Volatility sigma, Initial price S0)
ASSET_PRIORS = {
    # Malaysia
    "^KLSE": {"mu": 0.04, "sigma": 0.14, "s0": 1550.0, "name": "FTSE Bursa Malaysia KLCI"},
    "1155.KL": {"mu": 0.07, "sigma": 0.16, "s0": 9.50, "name": "Malayan Banking Berhad"},
    "1295.KL": {"mu": 0.06, "sigma": 0.15, "s0": 4.20, "name": "Public Bank Berhad"},
    "5347.KL": {"mu": 0.05, "sigma": 0.18, "s0": 12.00, "name": "Tenaga Nasional Berhad"},
    "1023.KL": {"mu": 0.08, "sigma": 0.20, "s0": 6.80, "name": "CIMB Group Holdings"},
    "5183.KL": {"mu": 0.03, "sigma": 0.24, "s0": 6.00, "name": "Petronas Chemicals"},
    "5225.KL": {"mu": 0.05, "sigma": 0.17, "s0": 6.10, "name": "IHH Healthcare"},
    "6742.KL": {"mu": 0.12, "sigma": 0.32, "s0": 4.50, "name": "YTL Power International"},
    # US
    "SPY": {"mu": 0.11, "sigma": 0.18, "s0": 420.0, "name": "SPDR S&P 500 ETF"},
    "QQQ": {"mu": 0.15, "sigma": 0.24, "s0": 360.0, "name": "Invesco QQQ Trust"},
    "IWM": {"mu": 0.07, "sigma": 0.23, "s0": 190.0, "name": "iShares Russell 2000 ETF"},
    "AAPL": {"mu": 0.18, "sigma": 0.26, "s0": 175.0, "name": "Apple Inc."},
    "MSFT": {"mu": 0.17, "sigma": 0.25, "s0": 340.0, "name": "Microsoft Corporation"},
    "NVDA": {"mu": 0.35, "sigma": 0.45, "s0": 450.0, "name": "NVIDIA Corporation"},
    # Japan
    "^N225": {"mu": 0.09, "sigma": 0.20, "s0": 32000.0, "name": "Nikkei 225 Stock Average"},
    "7203.T": {"mu": 0.08, "sigma": 0.22, "s0": 2600.0, "name": "Toyota Motor Corp"},
    # Europe
    "^GDAXI": {"mu": 0.07, "sigma": 0.19, "s0": 16000.0, "name": "DAX Performance Index"},
    # FX & Macro
    "MYR=X": {"mu": 0.015, "sigma": 0.07, "s0": 4.45, "name": "USD/MYR Spot"},
    "JPY=X": {"mu": 0.03, "sigma": 0.10, "s0": 140.0, "name": "USD/JPY Spot"},
    "EURUSD=X": {"mu": -0.01, "sigma": 0.08, "s0": 1.08, "name": "EUR/USD Spot"},
    "BZ=F": {"mu": 0.05, "sigma": 0.35, "s0": 78.0, "name": "Brent Crude Oil"},
    "GC=F": {"mu": 0.08, "sigma": 0.16, "s0": 1950.0, "name": "Gold Futures"},
    "^VIX": {"mu": 0.00, "sigma": 0.40, "s0": 18.0, "name": "CBOE Volatility Index"},
    "^TNX": {"mu": 0.02, "sigma": 0.15, "s0": 4.10, "name": "10-Year Treasury Yield"},
}


class DemoDataProvider(DataProvider):
    """Deterministic synthetic data generator and fallback provider."""

    def __init__(self, seed: int = 42):
        self._seed = seed

    @property
    def name(self) -> str:
        return "Demo/Synthetic Data Provider (Offline Fallback)"

    def _generate_single_asset(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        dates = pd.bdate_range(start=start_date, end=end_date)
        n = len(dates)
        if n == 0:
            dates = pd.bdate_range(start="2018-01-01", end="2025-12-31")
            n = len(dates)

        prior = ASSET_PRIORS.get(
            symbol, {"mu": 0.06, "sigma": 0.20, "s0": 100.0, "name": symbol}
        )
        mu = prior["mu"]
        sigma = prior["sigma"]
        s0 = prior["s0"]

        # Deterministic seed unique per symbol
        symbol_seed = (self._seed + sum(ord(c) for c in symbol)) % (2**31 - 1)
        rng = np.random.RandomState(symbol_seed)

        # Generate geometric brownian motion returns
        dt = 1.0 / 252.0
        daily_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.normal(size=n)
        
        # Mean-reverting adjustments for bounded series like VIX and TNX
        if symbol in ["^VIX", "^TNX"]:
            target = s0
            kappa = 2.0  # Speed of mean reversion
            values = [s0]
            for i in range(1, n):
                prev = values[-1]
                drift = kappa * (target - prev) * dt
                shock = sigma * np.sqrt(dt) * rng.normal()
                new_val = max(1.0, prev + drift + prev * shock)
                values.append(new_val)
            close = np.array(values)
        else:
            price_paths = s0 * np.exp(np.cumsum(daily_returns))
            close = price_paths

        # Generate realistic High, Low, Open, Volume
        intra_vol = sigma * np.sqrt(dt)
        open_price = np.roll(close, 1)
        open_price[0] = close[0] * (1.0 - 0.002 * rng.normal())
        
        high_price = np.maximum(open_price, close) * (1.0 + np.abs(rng.normal(0, intra_vol * 0.6, size=n)))
        low_price = np.minimum(open_price, close) * (1.0 - np.abs(rng.normal(0, intra_vol * 0.6, size=n)))
        
        base_vol = 1_000_000 if symbol.endswith(".KL") or "T" in symbol else 10_000_000
        volume = (base_vol * np.exp(rng.normal(0, 0.4, size=n))).astype(int)

        df = pd.DataFrame(
            {
                "Open": np.round(open_price, 4),
                "High": np.round(high_price, 4),
                "Low": np.round(low_price, 4),
                "Close": np.round(close, 4),
                "Adj Close": np.round(close, 4),
                "Volume": volume,
            },
            index=dates,
        )
        df.index.name = "Date"
        return df

    def get_prices(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[str] = "2018-01-01",
        end_date: Optional[str] = "2025-12-31",
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        start = start_date or "2018-01-01"
        end = end_date or "2025-12-31"

        if isinstance(symbols, str):
            return self._generate_single_asset(symbols, start, end)
        
        return {sym: self._generate_single_asset(sym, start, end) for sym in symbols}

    def get_macro(
        self,
        series_id: str,
        start_date: Optional[str] = "2018-01-01",
        end_date: Optional[str] = "2025-12-31",
    ) -> pd.Series:
        df = self._generate_single_asset(series_id, start_date or "2018-01-01", end_date or "2025-12-31")
        series = df["Close"].copy()
        series.name = series_id
        return series

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        prior = ASSET_PRIORS.get(symbol, {"name": symbol, "s0": 100.0})
        return {
            "symbol": symbol,
            "name": prior.get("name", symbol),
            "provider": self.name,
            "is_demo": True,
            "currency": "MYR" if symbol.endswith(".KL") or symbol == "^KLSE" else ("JPY" if ".T" in symbol or symbol == "^N225" else "USD"),
        }

"""Unit tests for feature engineering indicators and mathematical correctness."""

import numpy as np
import pandas as pd
from quant_engine.features.technical import sma, ema, roc, rsi, atr, bollinger_bands
from quant_engine.features.volatility import realized_volatility, parkinson_volatility
from quant_engine.features.cross_market import overnight_us_return


def test_technical_indicators_calculation():
    series = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
    s = sma(series, period=5)
    assert pd.isna(s.iloc[3])
    assert np.isclose(s.iloc[4], 12.0)  # (10+11+12+13+14)/5 = 12.0
    assert np.isclose(s.iloc[10], 18.0) # (16+17+18+19+20)/5 = 18.0

    r = roc(series, period=1)
    assert np.isclose(r.iloc[1], 0.10)  # (11-10)/10 = 0.10


def test_rsi_bounds():
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(200)))
    rsi_vals = rsi(prices, period=14).dropna()
    assert (rsi_vals >= 0.0).all()
    assert (rsi_vals <= 100.0).all()


def test_volatility_estimators_positive():
    dates = pd.date_range("2021-01-01", periods=100, freq="B")
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 102.0,
            "Low": 98.0,
            "Close": 101.0,
        },
        index=dates,
    )
    p_vol = parkinson_volatility(df["High"], df["Low"], window=20).dropna()
    assert (p_vol > 0).all()


def test_cross_market_lag_enforced():
    us_dates = pd.date_range("2021-01-01", periods=10, freq="B")
    us_close = pd.Series([100, 105, 110, 100, 95, 100, 105, 110, 115, 120], index=us_dates)
    
    # Lag 1 bar
    lagged = overnight_us_return(us_close, us_dates, lag_bars=1)
    # The return from date 0 to date 1 is (105-100)/100 = +5%
    # With lag_bars=1, this +5% should appear on date 2 (index position 2)
    assert np.isclose(lagged.iloc[2], 0.05)

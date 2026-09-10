"""Technical indicators with strict chronological computation."""

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def roc(series: pd.Series, period: int = 20) -> pd.Series:
    """Rate of Change (percentage return over period bars)."""
    return (series / series.shift(period)) - 1.0


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands (Middle, Upper, Lower)."""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    return middle, upper, lower


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Moving Average Convergence Divergence."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def rolling_beta(asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 63) -> pd.Series:
    """Rolling Market Beta (Covariance / Benchmark Variance)."""
    cov = asset_returns.rolling(window).cov(benchmark_returns)
    var = benchmark_returns.rolling(window).var()
    beta = cov / (var + 1e-12)
    return beta


def rolling_correlation(s1: pd.Series, s2: pd.Series, window: int = 63) -> pd.Series:
    """Rolling Pearson correlation coefficient."""
    return s1.rolling(window).corr(s2)

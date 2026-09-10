"""Performance metrics: CAGR, Sharpe, Sortino, Calmar, and Omega ratios."""

from typing import Dict, Any
import numpy as np
import pandas as pd
from quant_engine.risk.volatility import downside_semi_deviation
from quant_engine.risk.drawdown import analyze_drawdowns


def calculate_cagr(nav_series: pd.Series, annual_trading_days: int = 252) -> float:
    """Compound Annual Growth Rate (CAGR).
    
    Formula: (NAV_end / NAV_start) ** (annual_days / N) - 1.0
    """
    if len(nav_series) < 2 or nav_series.iloc[0] <= 0:
        return 0.0
    total_return_ratio = nav_series.iloc[-1] / nav_series.iloc[0]
    if total_return_ratio <= 0:
        return -1.0
    num_years = max(1.0 / annual_trading_days, len(nav_series) / annual_trading_days)
    return float(total_return_ratio ** (1.0 / num_years) - 1.0)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    annual_trading_days: int = 252,
) -> float:
    """Annualized Sharpe Ratio with user-specified risk-free rate."""
    if len(returns) < 2:
        return 0.0
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / annual_trading_days) - 1.0
    excess = returns - daily_rf
    mean_excess = excess.mean() * annual_trading_days
    ann_vol = returns.std() * np.sqrt(annual_trading_days)
    if ann_vol <= 1e-8:
        return 0.0
    return float(mean_excess / ann_vol)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    annual_trading_days: int = 252,
) -> float:
    """Annualized Sortino Ratio using downside semi-deviation."""
    if len(returns) < 2:
        return 0.0
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / annual_trading_days) - 1.0
    excess = returns - daily_rf
    mean_excess = excess.mean() * annual_trading_days
    downside_vol = downside_semi_deviation(returns, target_return=daily_rf, annual_days=annual_trading_days)
    if downside_vol <= 1e-8:
        return 0.0
    return float(mean_excess / downside_vol)


def calculate_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """Calmar Ratio: CAGR / abs(Max Drawdown)."""
    abs_dd = abs(max_drawdown)
    if abs_dd <= 1e-4:
        return 999.0 if cagr > 0 else 0.0
    return float(cagr / abs_dd)


def calculate_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Omega Ratio: probability-weighted ratio of gains vs losses above threshold."""
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = abs(excess[excess < 0].sum())
    if losses <= 1e-8:
        return 999.0 if gains > 0 else 1.0
    return float(gains / losses)


def calculate_comprehensive_performance(
    returns: pd.Series,
    nav_series: pd.Series,
    risk_free_rate: float = 0.03,
    annual_trading_days: int = 252,
) -> Dict[str, Any]:
    """Compute complete institutional performance scorecard."""
    clean_rets = returns.dropna()
    clean_nav = nav_series.dropna()

    if len(clean_rets) < 2:
        return {}

    total_net_return = float((clean_nav.iloc[-1] / clean_nav.iloc[0]) - 1.0)
    cagr = calculate_cagr(clean_nav, annual_trading_days=annual_trading_days)
    ann_vol = float(clean_rets.std() * np.sqrt(annual_trading_days))
    sharpe = calculate_sharpe_ratio(clean_rets, risk_free_rate, annual_trading_days)
    sortino = calculate_sortino_ratio(clean_rets, risk_free_rate, annual_trading_days)
    
    dd_stats = analyze_drawdowns(clean_nav)
    calmar = calculate_calmar_ratio(cagr, dd_stats["max_drawdown"])
    omega = calculate_omega_ratio(clean_rets, threshold=0.0)

    win_days = (clean_rets > 0).sum()
    total_days = len(clean_rets)
    win_rate_days = float(win_days / total_days) if total_days > 0 else 0.0

    return {
        "total_net_return": round(total_net_return, 4),
        "cagr": round(cagr, 4),
        "annualized_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "omega_ratio": round(omega, 3),
        "max_drawdown": dd_stats["max_drawdown"],
        "avg_drawdown": dd_stats["avg_drawdown"],
        "max_drawdown_duration_bars": dd_stats["max_drawdown_duration_bars"],
        "daily_win_rate": round(win_rate_days, 4),
        "total_trading_days": total_days,
    }

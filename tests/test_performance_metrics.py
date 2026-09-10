"""Unit tests for performance ratios, drawdowns, and risk calculations."""

import numpy as np
import pandas as pd
from quant_engine.analytics.performance import (
    calculate_cagr,
    calculate_sharpe_ratio,
    calculate_calmar_ratio,
    calculate_comprehensive_performance,
)
from quant_engine.risk.drawdown import compute_drawdown_series, analyze_drawdowns


def test_cagr_calculation():
    # 252 trading days (1 year), starting at 100, ending at 120 -> CAGR = +20%
    dates = pd.date_range("2021-01-01", periods=253, freq="B")
    nav = pd.Series(np.linspace(100.0, 120.0, 253), index=dates)
    cagr = calculate_cagr(nav, annual_trading_days=252)
    assert np.isclose(cagr, 0.20, atol=0.01)


def test_drawdown_calculation():
    # Peak at 100, drops to 80, recovers to 110
    nav = pd.Series([100.0, 90.0, 80.0, 95.0, 105.0, 110.0])
    dd = compute_drawdown_series(nav)
    # At index 2 (80.0), drawdown is (80-100)/100 = -0.20 (-20%)
    assert np.isclose(dd.min(), -0.20)
    
    stats = analyze_drawdowns(nav)
    assert np.isclose(stats["max_drawdown"], -0.20)


def test_sharpe_ratio_zero_volatility_safe():
    # Flat return series should return 0.0 without divide-by-zero exception
    returns = pd.Series([0.0] * 50)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.03)
    assert sharpe == 0.0


def test_comprehensive_metrics_structure():
    dates = pd.date_range("2021-01-01", periods=100, freq="B")
    returns = pd.Series(np.random.normal(0.0005, 0.01, 100), index=dates)
    nav = (1.0 + returns).cumprod()

    perf = calculate_comprehensive_performance(returns, nav)
    assert "cagr" in perf
    assert "sharpe_ratio" in perf
    assert "max_drawdown" in perf
    assert "sortino_ratio" in perf

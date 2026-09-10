"""Unit tests for PortfolioOptimizer and Black-Litterman allocation model."""

import numpy as np
import pandas as pd
import pytest
from research.optimization import PortfolioOptimizer


@pytest.fixture
def synthetic_returns():
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=252, freq="B")
    ret_a = np.random.normal(0.0004, 0.012, len(dates))  # Domestic Bursa equity
    ret_b = np.random.normal(0.0006, 0.015, len(dates))  # Offshore Tech equity
    ret_c = np.random.normal(0.00015, 0.003, len(dates)) # Sovereign MGS bond
    return pd.DataFrame({"1155.KL": ret_a, "SPY": ret_b, "MGS_10Y": ret_c}, index=dates)


def test_mean_variance_optimization(synthetic_returns):
    opt = PortfolioOptimizer(synthetic_returns, risk_free_rate=0.030, annual_trading_days=248)
    
    # Max Sharpe
    res_sharpe = opt.optimize_mean_variance(objective="max_sharpe")
    assert res_sharpe["success"] is True
    weights = res_sharpe["weights"]
    assert len(weights) == 3
    assert np.isclose(np.sum(weights), 1.0, atol=1e-3)
    assert np.all(weights >= -1e-4)

    # Min Volatility
    res_vol = opt.optimize_mean_variance(objective="min_volatility")
    assert res_vol["success"] is True
    assert res_vol["expected_volatility"] <= res_sharpe["expected_volatility"] + 1e-4


def test_institutional_constraints(synthetic_returns):
    opt = PortfolioOptimizer(synthetic_returns, risk_free_rate=0.030, annual_trading_days=248)
    
    # Offshore mask: SPY is index 1
    offshore_mask = [False, True, False]
    max_offshore = 0.30
    single_cap = 0.50
    min_cash = 0.10
    cash_idx = 2  # MGS_10Y as cash/sovereign

    res = opt.optimize_mean_variance(
        objective="max_sharpe",
        max_single_asset=single_cap,
        max_offshore=max_offshore,
        offshore_mask=offshore_mask,
        min_cash_buffer=min_cash,
        cash_index=cash_idx,
    )
    w = res["weights"]
    assert w[1] <= max_offshore + 1e-3  # Offshore cap <= 30%
    assert w[cash_idx] >= min_cash - 1e-3  # Cash buffer >= 10%
    assert np.all(w <= single_cap + 1e-3)  # Single asset <= 50%
    assert np.isclose(np.sum(w), 1.0, atol=1e-3)


def test_black_litterman_model(synthetic_returns):
    opt = PortfolioOptimizer(synthetic_returns, risk_free_rate=0.030, annual_trading_days=248)
    mkt_weights = np.array([0.40, 0.40, 0.20])

    # View: 1155.KL will return 12% with high confidence
    views = {"1155.KL": (0.12, 0.80)}
    pi, post_returns, post_cov = opt.compute_black_litterman(mkt_weights, views, tau=0.05, risk_aversion=2.50)

    assert len(pi) == 3
    assert len(post_returns) == 3
    assert post_cov.shape == (3, 3)

    # Asset 0 (1155.KL) posterior return should be tilted upward compared to prior equilibrium
    assert post_returns[0] > pi[0] - 0.05
    # Covariance matrix must remain symmetric positive semi-definite
    assert np.allclose(post_cov, post_cov.T, atol=1e-5)
    eigenvalues = np.linalg.eigvals(post_cov)
    assert np.all(eigenvalues > -1e-7)

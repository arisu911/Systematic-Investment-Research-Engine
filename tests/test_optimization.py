"""Unit tests for PortfolioOptimizer, Risk Parity, and Black-Litterman allocation models."""

import numpy as np
import pandas as pd
import pytest
from research.optimization import PortfolioOptimizer


@pytest.fixture
def multi_market_returns():
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=252, freq="B")
    ret_my = np.random.normal(0.0004, 0.012, len(dates))   # 1155.KL
    ret_us = np.random.normal(0.0008, 0.018, len(dates))   # NVDA
    ret_jp = np.random.normal(0.0005, 0.014, len(dates))   # 7203.T
    ret_cmd = np.random.normal(0.0003, 0.010, len(dates))  # GC=F
    return pd.DataFrame({
        "1155.KL": ret_my,
        "NVDA": ret_us,
        "7203.T": ret_jp,
        "GC=F": ret_cmd,
    }, index=dates)


def test_mean_variance_optimization(multi_market_returns):
    opt = PortfolioOptimizer(multi_market_returns, risk_free_rate=0.040, filter_tradable=False)
    
    # Max Sharpe
    res_sharpe = opt.optimize_mean_variance(objective="max_sharpe")
    assert res_sharpe["success"] is True
    weights = res_sharpe["weights"]
    assert len(weights) == 4
    assert np.isclose(np.sum(weights), 1.0, atol=1e-3)
    assert np.all(weights >= -1e-4)

    # Min Volatility
    res_vol = opt.optimize_mean_variance(objective="min_volatility")
    assert res_vol["success"] is True
    assert res_vol["expected_volatility"] <= res_sharpe["expected_volatility"] + 1e-4


def test_risk_parity_optimization(multi_market_returns):
    opt = PortfolioOptimizer(multi_market_returns, filter_tradable=False)
    
    # Asset level risk parity
    res_rp = opt.optimize_risk_parity(mode="asset_level")
    assert res_rp["success"] is True
    w = res_rp["weights"]
    assert len(w) == 4
    assert np.isclose(np.sum(w), 1.0, atol=1e-3)
    assert np.all(w > 0.0)  # All assets must have positive weight in ERC

    # Verify risk contributions are approximately equal
    cov = multi_market_returns.cov().values * 252.0
    port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    marginal = np.dot(cov, w)
    risk_contribs = w * marginal / port_vol
    
    # Check that maximum difference among risk contributions is small
    rc_diff = np.max(risk_contribs) - np.min(risk_contribs)
    assert rc_diff < 0.05


def test_black_litterman_relative_and_absolute_views(multi_market_returns):
    opt = PortfolioOptimizer(multi_market_returns, risk_free_rate=0.040, filter_tradable=False)
    
    views = [
        # Relative view: NVDA outperforms 7203.T by 4%
        {"asset_long": "NVDA", "asset_short": "7203.T", "relative_return": 0.04, "confidence": 0.75},
        # Absolute view: 1155.KL returns 8%
        {"asset_long": "1155.KL", "view_return": 0.08, "confidence": 0.70},
    ]

    pi, post_ret, post_cov = opt.compute_black_litterman(views_list=views)

    assert len(pi) == 4
    assert len(post_ret) == 4
    assert post_cov.shape == (4, 4)

    # Covariance matrix must remain symmetric positive semi-definite
    assert np.allclose(post_cov, post_cov.T, atol=1e-5)
    eigenvalues = np.linalg.eigvals(post_cov)
    assert np.all(eigenvalues > -1e-7)

    # Solve optimal weights with Black-Litterman
    bl_res = opt.optimize_black_litterman(views_list=views)
    assert bl_res["success"] is True
    assert np.isclose(np.sum(bl_res["weights"]), 1.0, atol=1e-3)

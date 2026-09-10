"""Unit tests for non-Gaussian Cornish-Fisher tail risk and risk attribution."""

import numpy as np
import pandas as pd
from scipy.stats import norm
from research.risk import RiskEngine


def test_cornish_fisher_expansion():
    # Symmetric distribution with zero skew and zero excess kurtosis must match standard normal
    z_c = norm.ppf(0.05)
    z_cf_normal = RiskEngine.cornish_fisher_quantile(z_c, skewness=0.0, excess_kurtosis=0.0)
    assert np.isclose(z_c, z_cf_normal, atol=1e-5)

    # Negative skewness (left tail risk) must make z_cf more negative (larger loss)
    z_cf_neg_skew = RiskEngine.cornish_fisher_quantile(z_c, skewness=-0.8, excess_kurtosis=0.0)
    assert z_cf_neg_skew < z_c

    # Fat tails (positive excess kurtosis) must also amplify left-tail quantile
    z_cf_fat_tail = RiskEngine.cornish_fisher_quantile(z_c, skewness=-0.5, excess_kurtosis=2.5)
    assert z_cf_fat_tail < z_c


def test_var_and_cvar_calculations():
    np.random.seed(42)
    # Fat-tailed student-t-like returns
    returns = np.random.standard_t(df=5, size=1000) * 0.015

    res_95 = RiskEngine.calculate_var_cvar(returns, confidence_level=0.95)
    res_99 = RiskEngine.calculate_var_cvar(returns, confidence_level=0.99)

    # 99% VaR must exceed 95% VaR
    assert res_99["cornish_fisher_var"] > res_95["cornish_fisher_var"]
    assert res_99["parametric_var"] > res_95["parametric_var"]
    assert res_99["historical_var"] > res_95["historical_var"]

    # CVaR (Expected Shortfall) must exceed VaR
    assert res_95["cornish_fisher_cvar"] >= res_95["cornish_fisher_var"]
    assert res_99["cornish_fisher_cvar"] >= res_99["cornish_fisher_var"]


def test_risk_attribution_sums_to_one():
    cov = np.array([
        [0.04, 0.015, 0.005],
        [0.015, 0.06, 0.008],
        [0.005, 0.008, 0.01],
    ])
    weights = np.array([0.50, 0.30, 0.20])

    attrib_df = RiskEngine.calculate_risk_attribution(weights, cov)
    assert len(attrib_df) == 3
    # Percentage risk contributions must sum exactly to 1.0 (100%)
    pcr_sum = attrib_df["Pct_Risk_Contribution"].sum()
    assert np.isclose(pcr_sum, 1.0, atol=1e-4)

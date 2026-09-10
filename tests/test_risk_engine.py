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


def test_nominal_var_cvar_scaling():
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.012, 1000)
    
    cap_100k = 100_000.0
    cap_500k = 500_000.0
    
    res_100k = RiskEngine.calculate_nominal_var_cvar(returns, capital=cap_100k, confidence_level=0.95, horizon_days=1)
    res_500k = RiskEngine.calculate_nominal_var_cvar(returns, capital=cap_500k, confidence_level=0.95, horizon_days=1)
    
    # Percentages must be identical
    assert np.isclose(res_100k["cornish_fisher_var"], res_500k["cornish_fisher_var"])
    
    # Nominal cash amounts must scale 5x
    assert np.isclose(res_500k["nominal_cornish_fisher_var"], 5.0 * res_100k["nominal_cornish_fisher_var"], rtol=1e-4)
    assert np.isclose(res_500k["nominal_parametric_var"], 5.0 * res_100k["nominal_parametric_var"], rtol=1e-4)
    assert np.isclose(res_500k["nominal_historical_var"], 5.0 * res_100k["nominal_historical_var"], rtol=1e-4)


def test_multi_horizon_nominal_var():
    np.random.seed(42)
    returns = np.random.normal(0.0002, 0.015, 1000)
    
    df = RiskEngine.calculate_multi_horizon_nominal_var(returns, capital=250_000.0, horizons=[1, 5, 21], confidence_levels=[0.95, 0.99])
    
    assert len(df) == 6 # 3 horizons x 2 confidence levels
    assert set(df["Horizon_Days"]) == {1, 5, 21}
    assert set(df["Confidence"]) == {"95%", "99%"}
    
    # 21-day nominal VaR must be greater than 1-day nominal VaR
    var_1d_95 = df[(df["Horizon_Days"] == 1) & (df["Confidence"] == "95%")]["Nominal_Cornish_Fisher_VaR"].iloc[0]
    var_21d_95 = df[(df["Horizon_Days"] == 21) & (df["Confidence"] == "95%")]["Nominal_Cornish_Fisher_VaR"].iloc[0]
    assert var_21d_95 > var_1d_95


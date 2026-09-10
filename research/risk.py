"""Non-Gaussian Tail Risk Modeling & Risk Attribution Engine.

Implements Parametric, Historical, and Cornish-Fisher Modified Value-at-Risk (mVaR)
and Modified Expected Shortfall (mCVaR) at 95% and 99% confidence horizons,
along with asset marginal risk attribution and drawdown analytics.
"""

from typing import Dict, Tuple, List, Union, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis


class RiskEngine:
    """Quantitative risk engine specialized for non-normal asset return distributions."""

    @staticmethod
    def calculate_moments(returns: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """Compute mean, volatility, skewness, and Fisher excess kurtosis."""
        clean_ret = np.asarray(returns)[~np.isnan(returns)]
        if len(clean_ret) < 10:
            return {"mean": 0.0, "volatility": 0.0, "skewness": 0.0, "kurtosis": 0.0}

        mu = float(np.mean(clean_ret))
        sigma = float(np.std(clean_ret, ddof=1))
        s = float(skew(clean_ret))
        # Fisher definition: Normal distribution has kurtosis = 0
        k = float(kurtosis(clean_ret, fisher=True))

        return {
            "mean": mu,
            "volatility": sigma,
            "skewness": s,
            "kurtosis": k,
        }

    @staticmethod
    def cornish_fisher_quantile(z_c: float, skewness: float, excess_kurtosis: float) -> float:
        """Calculate adjusted quantile critical value using the Cornish-Fisher polynomial expansion."""
        s = skewness
        k = excess_kurtosis

        z_cf = (
            z_c
            + (1.0 / 6.0) * (z_c**2 - 1.0) * s
            + (1.0 / 24.0) * (z_c**3 - 3.0 * z_c) * k
            - (1.0 / 36.0) * (2.0 * z_c**3 - 5.0 * z_c) * (s**2)
        )
        return float(z_cf)

    @classmethod
    def calculate_var_cvar(
        cls,
        returns: Union[pd.Series, np.ndarray],
        confidence_level: float = 0.95,
        horizon_days: int = 1,
    ) -> Dict[str, float]:
        """Calculate Parametric, Historical, and Cornish-Fisher Modified VaR and CVaR.
        
        Returned as positive loss percentages (e.g. 0.025 = 2.5% loss).
        """
        clean_ret = np.asarray(returns)[~np.isnan(returns)]
        if len(clean_ret) < 10:
            return {
                "parametric_var": 0.0,
                "historical_var": 0.0,
                "cornish_fisher_var": 0.0,
                "parametric_cvar": 0.0,
                "historical_cvar": 0.0,
                "cornish_fisher_cvar": 0.0,
                "skewness": 0.0,
                "excess_kurtosis": 0.0,
            }

        moments = cls.calculate_moments(clean_ret)
        mu = moments["mean"]
        sigma = moments["volatility"]
        s = moments["skewness"]
        k = moments["kurtosis"]

        alpha = 1.0 - confidence_level

        # 1. Parametric Gaussian VaR / CVaR
        z_norm = norm.ppf(alpha)  # Negative z for left tail
        param_var = -(mu + z_norm * sigma) * np.sqrt(horizon_days)
        param_cvar = -(mu - (norm.pdf(z_norm) / alpha) * sigma) * np.sqrt(horizon_days)

        # 2. Historical Empirical VaR / CVaR
        hist_quantile = np.percentile(clean_ret, alpha * 100.0)
        hist_var = -hist_quantile * np.sqrt(horizon_days)
        tail_losses = clean_ret[clean_ret <= hist_quantile]
        hist_cvar = -np.mean(tail_losses) * np.sqrt(horizon_days) if len(tail_losses) > 0 else hist_var

        # 3. Cornish-Fisher Modified VaR / CVaR
        z_cf = cls.cornish_fisher_quantile(z_norm, s, k)
        cf_var = -(mu + z_cf * sigma) * np.sqrt(horizon_days)

        # Cornish-Fisher Expected Shortfall approximation
        # Adjusting CVaR for the fat-tail expansion
        cf_tail_factor = max(1.0, abs(z_cf / z_norm)) if abs(z_norm) > 1e-4 else 1.0
        cf_cvar = max(cf_var, param_cvar * cf_tail_factor)

        return {
            "parametric_var": max(0.0, float(param_var)),
            "historical_var": max(0.0, float(hist_var)),
            "cornish_fisher_var": max(0.0, float(cf_var)),
            "parametric_cvar": max(0.0, float(param_cvar)),
            "historical_cvar": max(0.0, float(hist_cvar)),
            "cornish_fisher_cvar": max(0.0, float(cf_cvar)),
            "skewness": s,
            "excess_kurtosis": k,
        }

    @classmethod
    def comprehensive_risk_profile(cls, returns: Union[pd.Series, np.ndarray], rf_annual: float = 0.030) -> Dict[str, Any]:
        """Compute full risk report across 95% and 99% horizons with downside ratios."""
        clean_ret = np.asarray(returns)[~np.isnan(returns)]
        var_95 = cls.calculate_var_cvar(clean_ret, confidence_level=0.95)
        var_99 = cls.calculate_var_cvar(clean_ret, confidence_level=0.99)

        rf_daily = rf_annual / 248.0
        downside_ret = clean_ret[clean_ret < rf_daily] - rf_daily
        downside_dev = np.sqrt(np.mean(downside_ret**2)) * np.sqrt(248) if len(downside_ret) > 0 else 1e-6

        ann_ret = float(np.mean(clean_ret) * 248)
        ann_vol = float(np.std(clean_ret, ddof=1) * np.sqrt(248))
        sortino = (ann_ret - rf_annual) / downside_dev if downside_dev > 1e-6 else 0.0

        # Cumulative peak & drawdowns
        cum = np.cumprod(1.0 + clean_ret)
        peaks = np.maximum.accumulate(cum)
        dd = (cum - peaks) / peaks
        max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0
        calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-4 else 0.0

        return {
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "max_drawdown": max_dd,
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "skewness": var_95["skewness"],
            "excess_kurtosis": var_95["excess_kurtosis"],
            "var_95": var_95,
            "var_99": var_99,
        }

    @staticmethod
    def calculate_risk_attribution(
        weights: np.ndarray, cov_matrix: np.ndarray, asset_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Calculate Marginal Contribution to Risk (MCR) and Percentage Contribution to Risk (PCR)."""
        w = np.asarray(weights)
        cov = np.asarray(cov_matrix)

        port_var = float(np.dot(w.T, np.dot(cov, w)))
        port_vol = np.sqrt(port_var) if port_var > 1e-6 else 1e-6

        # Marginal Contribution to Risk: MCR_i = (cov * w)_i / port_vol
        mcr = np.dot(cov, w) / port_vol

        # Absolute Contribution to Risk: ACR_i = w_i * MCR_i
        acr = w * mcr

        # Percentage Contribution to Risk: PCR_i = ACR_i / port_vol
        pcr = acr / port_vol

        names = asset_names if asset_names is not None else [f"Asset_{i}" for i in range(len(w))]
        return pd.DataFrame({
            "Asset": names,
            "Weight": w,
            "Marginal_Risk": mcr,
            "Absolute_Risk": acr,
            "Pct_Risk_Contribution": pcr,
        })

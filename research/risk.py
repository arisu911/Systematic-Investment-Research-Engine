"""Non-Gaussian Tail Risk Modeling & Risk Attribution Engine.

Implements:
1. Parametric Gaussian VaR and CVaR.
2. Historical Empirical VaR and CVaR.
3. Cornish-Fisher Modified VaR (mVaR) and Modified Expected Shortfall (mCVaR) at 95% and 99%.
4. Marginal Contribution to Risk (MCR) and Percentage Contribution to Risk (PCR).
5. QQ-Plot data generation for fat-tail diagnostic exploration.
"""

from typing import Dict, Tuple, List, Union, Optional, Any
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis


class RiskEngine:
    """Quantitative risk engine specialized for non-Gaussian asset distributions."""

    @staticmethod
    def calculate_moments(returns: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """Compute mean, volatility, skewness, and Fisher excess kurtosis."""
        clean_ret = np.asarray(returns)[~np.isnan(returns)]
        if len(clean_ret) < 10:
            return {"mean": 0.0, "volatility": 0.0, "skewness": 0.0, "kurtosis": 0.0}

        mu = float(np.mean(clean_ret))
        sigma = float(np.std(clean_ret, ddof=1))
        s = float(skew(clean_ret))
        # Fisher definition: Normal distribution has excess kurtosis = 0
        k = float(kurtosis(clean_ret, fisher=True))

        return {
            "mean": mu,
            "volatility": sigma,
            "skewness": s,
            "kurtosis": k,
        }

    @staticmethod
    def cornish_fisher_quantile(z_c: float, skewness: float, excess_kurtosis: float) -> float:
        """Calculate adjusted quantile critical value using the Cornish-Fisher polynomial expansion.
        
        Formula:
          z_CF = z_c + (1/6)*(z_c^2 - 1)*S + (1/24)*(z_c^3 - 3*z_c)*K - (1/36)*(2*z_c^3 - 5*z_c)*(S^2)
        """
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
        
        Returns metrics as positive loss percentages (e.g. 0.025 = 2.5% loss).
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
        cf_tail_factor = max(1.0, abs(z_cf / z_norm)) if abs(z_norm) > 1e-4 else 1.0
        cf_cvar = max(cf_var, hist_cvar * cf_tail_factor)

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

    @staticmethod
    def calculate_risk_attribution(
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        asset_names: Optional[List[str]] = None,
        capital: float = 100_000.0,
    ) -> pd.DataFrame:
        """Compute asset-level Marginal Risk Contribution, Percentage Contribution, and Nominal Cash Risk."""
        w = np.asarray(weights)
        cov = np.asarray(cov_matrix)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        n = len(w)
        if port_vol < 1e-8:
            mcr = np.zeros(n)
            pcr = np.zeros(n)
            nominal_rc = np.zeros(n)
        else:
            mcr = np.dot(cov, w) / port_vol
            pcr = (w * mcr) / port_vol
            nominal_rc = pcr * (port_vol * capital)

        names = asset_names if asset_names is not None else [f"Asset_{i+1}" for i in range(n)]
        return pd.DataFrame({
            "Asset": names,
            "Weight": w,
            "Marginal_Risk_Contribution": mcr,
            "Pct_Risk_Contribution": pcr,
            "Nominal_Risk_Contribution": nominal_rc,
        })

    @classmethod
    def calculate_nominal_var_cvar(
        cls,
        returns: Union[pd.Series, np.ndarray],
        capital: float = 100_000.0,
        confidence_level: float = 0.95,
        horizon_days: int = 1,
    ) -> Dict[str, float]:
        """Calculate percentage and nominal cash Value-at-Risk and Expected Shortfall."""
        res = cls.calculate_var_cvar(returns, confidence_level=confidence_level, horizon_days=horizon_days)
        res["capital"] = capital
        res["horizon_days"] = horizon_days
        res["confidence_level"] = confidence_level
        res["nominal_parametric_var"] = res["parametric_var"] * capital
        res["nominal_historical_var"] = res["historical_var"] * capital
        res["nominal_cornish_fisher_var"] = res["cornish_fisher_var"] * capital
        res["nominal_parametric_cvar"] = res["parametric_cvar"] * capital
        res["nominal_historical_cvar"] = res["historical_cvar"] * capital
        res["nominal_cornish_fisher_cvar"] = res["cornish_fisher_cvar"] * capital
        return res

    @classmethod
    def calculate_multi_horizon_nominal_var(
        cls,
        returns: Union[pd.Series, np.ndarray],
        capital: float = 100_000.0,
        horizons: Optional[List[int]] = None,
        confidence_levels: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Generate matrix of Cash-at-Risk across 1-Day, 5-Day, and 21-Day horizons."""
        if horizons is None:
            horizons = [1, 5, 21]
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]

        rows = []
        for h in horizons:
            for conf in confidence_levels:
                data = cls.calculate_nominal_var_cvar(returns, capital=capital, confidence_level=conf, horizon_days=h)
                horizon_label = f"{h}-Day (1 Month)" if h == 21 else f"{h}-Day (1 Week)" if h == 5 else f"{h}-Day"
                rows.append({
                    "Horizon": horizon_label,
                    "Horizon_Days": h,
                    "Confidence": f"{conf:.0%}",
                    "Cornish_Fisher_VaR_Pct": data["cornish_fisher_var"],
                    "Nominal_Cornish_Fisher_VaR": data["nominal_cornish_fisher_var"],
                    "Parametric_VaR_Pct": data["parametric_var"],
                    "Nominal_Parametric_VaR": data["nominal_parametric_var"],
                    "Historical_VaR_Pct": data["historical_var"],
                    "Nominal_Historical_VaR": data["nominal_historical_var"],
                    "Nominal_Cornish_Fisher_CVaR": data["nominal_cornish_fisher_cvar"],
                })
        return pd.DataFrame(rows)

    @staticmethod
    def calculate_risk_contributions(weights: np.ndarray, cov_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute Marginal Risk Contribution (MCR) and Percentage Contribution to Risk (PCR)."""
        w = np.asarray(weights)
        cov = np.asarray(cov_matrix)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))

        if port_vol < 1e-8:
            n = len(w)
            return {"mcr": np.zeros(n), "pcr": np.zeros(n)}

        # Marginal contribution to risk
        mcr = np.dot(cov, w) / port_vol
        # Percentage contribution to risk: w_i * MCR_i / port_vol (sums to 1.0)
        pcr = (w * mcr) / port_vol

        return {
            "mcr": mcr,
            "pcr": pcr,
            "portfolio_volatility": port_vol,
        }

    @staticmethod
    def generate_qq_plot_data(returns: Union[pd.Series, np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate empirical vs theoretical standard normal quantiles for QQ-plot."""
        clean_ret = np.asarray(returns)[~np.isnan(returns)]
        n = len(clean_ret)
        if n < 10:
            return {"theoretical": np.array([]), "empirical": np.array([])}

        # Standardize returns
        mu = np.mean(clean_ret)
        sigma = np.std(clean_ret, ddof=1)
        std_ret = (clean_ret - mu) / sigma if sigma > 1e-8 else clean_ret

        sorted_empirical = np.sort(std_ret)
        # Theoretical standard normal quantiles
        probs = (np.arange(1, n + 1) - 0.5) / n
        theoretical_quantiles = norm.ppf(probs)

        return {
            "theoretical": theoretical_quantiles,
            "empirical": sorted_empirical,
        }

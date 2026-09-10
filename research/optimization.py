"""Institutional Multi-Asset Portfolio Optimization & Allocation Engine.

Implements Classical Mean-Variance Optimization (Max Sharpe, Min Volatility),
Black-Litterman allocation model with investor views, and institutional constraints
(EPF 30% offshore cap, 5% cash floor, 20% single-asset cap, and transaction friction drag).
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PortfolioOptimizer:
    """Vectorized quantitative portfolio optimization and Black-Litterman allocation engine."""

    def __init__(
        self,
        returns_df: pd.DataFrame,
        risk_free_rate: float = 0.030,
        annual_trading_days: int = 248,
    ):
        """Initialize optimizer with historical daily asset returns."""
        self.returns_df = returns_df.dropna()
        self.assets = returns_df.columns.tolist()
        self.num_assets = len(self.assets)
        self.rf = risk_free_rate
        self.annual_days = annual_trading_days

        # Precompute empirical moments
        self.mean_returns = self.returns_df.mean() * self.annual_days
        self.cov_matrix = self.returns_df.cov() * self.annual_days

    def portfolio_performance(
        self, weights: np.ndarray, mean_returns: Optional[np.ndarray] = None, cov_matrix: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        """Calculate annualized expected return, volatility, and Sharpe ratio."""
        mu = mean_returns if mean_returns is not None else self.mean_returns.values
        cov = cov_matrix if cov_matrix is not None else self.cov_matrix.values

        port_ret = np.dot(weights, mu)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        sharpe = (port_ret - self.rf) / port_vol if port_vol > 1e-6 else 0.0
        return port_ret, port_vol, sharpe

    def optimize_mean_variance(
        self,
        objective: str = "max_sharpe",
        max_single_asset: float = 0.20,
        max_offshore: float = 0.30,
        offshore_mask: Optional[List[bool]] = None,
        min_cash_buffer: float = 0.05,
        cash_index: Optional[int] = None,
        expected_returns: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
        current_weights: Optional[np.ndarray] = None,
        transaction_cost_bps: float = 0.0,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """Solve constrained portfolio optimization problem.
        
        Args:
            objective: 'max_sharpe' or 'min_volatility'
            max_single_asset: Upper bound per security (e.g. 0.20)
            max_offshore: Regulatory maximum for offshore assets (e.g. 0.30 EPF mandate)
            offshore_mask: Boolean list indicating which assets are offshore
            min_cash_buffer: Minimum cash allocation (e.g. 0.05)
            cash_index: Index of cash/sovereign short proxy in assets
            expected_returns: Optional expected return vector (e.g. Black-Litterman posterior)
            cov_matrix: Optional covariance matrix (e.g. Black-Litterman posterior)
            current_weights: Current portfolio weights for turnover penalty
            transaction_cost_bps: Friction cost in basis points
        """
        mu = expected_returns if expected_returns is not None else self.mean_returns.values
        cov = cov_matrix if cov_matrix is not None else self.cov_matrix.values
        n = self.num_assets

        # Objective functions
        def neg_sharpe(w):
            ret, vol, sharpe = self.portfolio_performance(w, mu, cov)
            # Apply turnover penalty if current weights provided
            penalty = 0.0
            if current_weights is not None and transaction_cost_bps > 0:
                turnover = np.sum(np.abs(w - current_weights)) / 2.0
                cost_drag = turnover * (transaction_cost_bps / 10000.0)
                sharpe = (ret - cost_drag - self.rf) / vol if vol > 1e-6 else 0.0
            return -sharpe

        def min_vol(w):
            return np.sqrt(np.dot(w.T, np.dot(cov, w)))

        obj_func = neg_sharpe if objective == "max_sharpe" else min_vol

        # Constraints
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Offshore exposure constraint
        if offshore_mask is not None and any(offshore_mask) and max_offshore < 1.0:
            mask_arr = np.array(offshore_mask, dtype=float)
            constraints.append({"type": "ineq", "fun": lambda w: max_offshore - np.dot(w, mask_arr)})

        # Cash floor constraint
        if cash_index is not None and min_cash_buffer > 0:
            constraints.append({"type": "ineq", "fun": lambda w: w[cash_index] - min_cash_buffer})

        # Bounds per asset (ensuring mathematical feasibility when n * max_single_asset < 1.0)
        effective_cap = max(max_single_asset, 1.0 / n) if (n * max_single_asset < 1.0) else max_single_asset
        bounds = tuple((0.0, effective_cap) for _ in range(n))

        # Initial guess: Equal weights normalized
        init_guess = np.ones(n) / n

        res = minimize(
            obj_func,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        weights = res.x if res.success else init_guess
        # Clean small precision floats
        weights = np.where(weights < 1e-4, 0.0, weights)
        weights = weights / np.sum(weights)

        ret, vol, sharpe = self.portfolio_performance(weights, mu, cov)

        return {
            "weights": weights,
            "expected_return": ret,
            "expected_volatility": vol,
            "sharpe_ratio": sharpe,
            "success": res.success,
        }

    def compute_black_litterman(
        self,
        market_weights: np.ndarray,
        views_dict: Dict[str, Tuple[float, float]],
        tau: float = 0.05,
        risk_aversion: float = 2.50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Black-Litterman posterior expected return vector and covariance matrix.
        
        Args:
            market_weights: Benchmark or market capitalization weights w_mkt
            views_dict: Dict mapping asset symbol -> (expected_return_view, confidence_level in (0, 1])
            tau: Scaling constant for prior covariance uncertainty
            risk_aversion: Market risk aversion lambda
            
        Returns:
            (pi_equilibrium, posterior_returns, posterior_covariance)
        """
        n = self.num_assets
        sigma = self.cov_matrix.values
        w_mkt = np.array(market_weights)
        w_mkt = w_mkt / np.sum(w_mkt)

        # 1. Reverse optimization for Implied Equilibrium Excess Returns: Pi = lambda * Sigma * w_mkt
        pi = risk_aversion * np.dot(sigma, w_mkt)

        if not views_dict:
            # If no views, posterior equals prior equilibrium
            return pi + self.rf, pi + self.rf, sigma

        # 2. Construct View Matrix P and View Vector Q
        k = len(views_dict)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)

        for row_idx, (sym, (view_ret, conf)) in enumerate(views_dict.items()):
            if sym in self.assets:
                asset_idx = self.assets.index(sym)
                P[row_idx, asset_idx] = 1.0
                # Excess view return
                Q[row_idx] = view_ret - self.rf
                
                # He-Litterman uncertainty formulation scaled by confidence
                view_variance = np.dot(P[row_idx, :], np.dot(tau * sigma, P[row_idx, :].T))
                conf_clamped = max(0.01, min(0.99, conf))
                # Inverse confidence scale: Higher confidence -> lower uncertainty
                omega_diag[row_idx] = view_variance * ((1.0 - conf_clamped) / conf_clamped)

        Omega = np.diag(omega_diag)

        # 3. Compute Posterior Expected Return:
        # E(R) = [(tau*Sigma)^-1 + P^T * Omega^-1 * P]^-1 * [(tau*Sigma)^-1 * Pi + P^T * Omega^-1 * Q]
        tau_sigma = tau * sigma
        inv_tau_sigma = np.linalg.pinv(tau_sigma)
        inv_omega = np.linalg.pinv(Omega)

        M_inv = inv_tau_sigma + np.dot(P.T, np.dot(inv_omega, P))
        M = np.linalg.pinv(M_inv)

        rhs = np.dot(inv_tau_sigma, pi) + np.dot(P.T, np.dot(inv_omega, Q))
        er_excess = np.dot(M, rhs)
        posterior_returns = er_excess + self.rf

        # 4. Posterior Covariance: Sigma_BL = Sigma + M
        sigma_bl = sigma + M

        return pi + self.rf, posterior_returns, sigma_bl

    def generate_efficient_frontier(
        self, num_points: int = 30, expected_returns: Optional[np.ndarray] = None, cov_matrix: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Generate smooth Efficient Frontier curves for charting."""
        mu = expected_returns if expected_returns is not None else self.mean_returns.values
        cov = cov_matrix if cov_matrix is not None else self.cov_matrix.values

        # Find min vol and max return bounds
        min_vol_res = self.optimize_mean_variance(objective="min_volatility", expected_returns=mu, cov_matrix=cov)
        max_ret = np.max(mu)
        min_ret = min_vol_res["expected_return"]

        target_returns = np.linspace(min_ret, max_ret * 0.98, num_points)
        frontier_vols = []
        frontier_sharpes = []

        for target in target_returns:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w: np.dot(w, mu) - target},
            ]
            bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
            res = minimize(
                lambda w: np.sqrt(np.dot(w.T, np.dot(cov, w))),
                np.ones(self.num_assets) / self.num_assets,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            if res.success:
                vol = res.fun
                sharpe = (target - self.rf) / vol if vol > 1e-6 else 0.0
                frontier_vols.append(vol)
                frontier_sharpes.append(sharpe)
            else:
                frontier_vols.append(np.nan)
                frontier_sharpes.append(np.nan)

        return pd.DataFrame({
            "Target_Return": target_returns,
            "Volatility": frontier_vols,
            "Sharpe_Ratio": frontier_sharpes,
        }).dropna()

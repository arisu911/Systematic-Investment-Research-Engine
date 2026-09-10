"""Cross-Market Portfolio Optimization & Allocation Engine.

Implements:
1. Role-based universe filtering (only equity, tradable_proxy, commodity enter optimization).
2. Mean-Variance Optimization (Max Sharpe Ratio, Minimum Volatility).
3. Risk Parity (Equal Risk Contribution at asset level and regional cluster level).
4. Black-Litterman allocation model with cross-regional relative & absolute views.
5. Turnover penalty and transaction friction modeling.
"""

from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from data.aligner import load_universe_registry, get_tradable_tickers


class PortfolioOptimizer:
    """Vectorized multi-market portfolio optimization engine."""

    def __init__(
        self,
        returns_df: pd.DataFrame,
        risk_free_rate: float = 0.040,
        annual_trading_days: int = 252,
        registry: Optional[Dict[str, Any]] = None,
        filter_tradable: bool = True,
    ):
        """Initialize optimizer with daily returns.
        
        Args:
            returns_df: Clean DataFrame of daily returns.
            risk_free_rate: Annualized risk-free rate proxy.
            annual_trading_days: Number of trading bars per year.
            registry: Universe metadata mapping.
            filter_tradable: If True, restricts optimization to equities, proxies, and commodities.
        """
        self.registry = registry if registry is not None else load_universe_registry()
        
        if filter_tradable:
            tradable_list = get_tradable_tickers(self.registry)
            valid_cols = [c for c in returns_df.columns if c in tradable_list]
            if len(valid_cols) >= 2:
                self.returns_df = returns_df[valid_cols].dropna()
            else:
                self.returns_df = returns_df.dropna()
        else:
            self.returns_df = returns_df.dropna()

        self.assets = self.returns_df.columns.tolist()
        self.num_assets = len(self.assets)
        self.rf = risk_free_rate
        self.annual_days = annual_trading_days

        # Empirical annualized moments
        self.mean_returns = self.returns_df.mean() * self.annual_days
        self.cov_matrix = self.returns_df.cov() * self.annual_days

        # Map assets to regions for cluster risk parity
        self.asset_regions = {}
        for a in self.assets:
            meta = self.registry.get(a, {})
            reg = meta.get("region", "OTHER")
            ac = meta.get("asset_class", "equity")
            if ac == "commodity":
                self.asset_regions[a] = "COMMODITY"
            else:
                self.asset_regions[a] = reg

    def portfolio_performance(
        self,
        weights: np.ndarray,
        mean_returns: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float]:
        """Calculate annualized expected return, volatility, and Sharpe ratio."""
        mu = mean_returns if mean_returns is not None else self.mean_returns.values
        cov = cov_matrix if cov_matrix is not None else self.cov_matrix.values

        port_ret = float(np.dot(weights, mu))
        port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))
        sharpe = (port_ret - self.rf) / port_vol if port_vol > 1e-6 else 0.0
        return port_ret, port_vol, sharpe

    def optimize_mean_variance(
        self,
        objective: str = "max_sharpe",
        max_single_asset: float = 0.25,
        max_region_exposure: float = 0.50,
        expected_returns: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
        current_weights: Optional[np.ndarray] = None,
        transaction_cost_bps: float = 0.0,
    ) -> Dict[str, Any]:
        """Solve constrained Mean-Variance Optimization (Max Sharpe or Min Volatility)."""
        mu = expected_returns if expected_returns is not None else self.mean_returns.values
        cov = cov_matrix if cov_matrix is not None else self.cov_matrix.values
        n = self.num_assets

        def neg_sharpe(w):
            ret, vol, sharpe = self.portfolio_performance(w, mu, cov)
            penalty = 0.0
            if current_weights is not None and transaction_cost_bps > 0:
                turnover = np.sum(np.abs(w - current_weights)) / 2.0
                cost_drag = turnover * (transaction_cost_bps / 10000.0)
                sharpe = (ret - cost_drag - self.rf) / vol if vol > 1e-6 else 0.0
            return -sharpe

        def min_vol(w):
            return np.sqrt(np.dot(w.T, np.dot(cov, w)))

        obj_func = neg_sharpe if objective == "max_sharpe" else min_vol

        # Constraints: sum to 1.0
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Regional exposure constraint if applicable
        if max_region_exposure < 1.0:
            unique_regions = set(self.asset_regions.values())
            for reg in unique_regions:
                reg_mask = np.array([1.0 if self.asset_regions.get(a) == reg else 0.0 for a in self.assets])
                if np.sum(reg_mask) > 0:
                    constraints.append({
                        "type": "ineq",
                        "fun": lambda w, m=reg_mask: max_region_exposure - np.dot(w, m)
                    })

        effective_cap = max(max_single_asset, 1.0 / n) if (n * max_single_asset < 1.0) else max_single_asset
        bounds = tuple((0.0, effective_cap) for _ in range(n))
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
        weights = np.where(weights < 1e-4, 0.0, weights)
        weights = weights / np.sum(weights)

        ret, vol, sharpe = self.portfolio_performance(weights, mu, cov)

        return {
            "weights": weights,
            "expected_return": ret,
            "expected_volatility": vol,
            "sharpe_ratio": sharpe,
            "success": res.success,
            "method": f"MVO ({objective})",
            "assets": self.assets,
        }

    def optimize_risk_parity(
        self,
        mode: str = "asset_level",
        cov_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Solve Equal Risk Contribution (Risk Parity) portfolio.
        
        Args:
            mode: 'asset_level' (each asset contributes 1/N risk) or
                  'group_level' (each asset class/region contributes equally).
        """
        cov = cov_matrix if cov_matrix is not None else self.cov_matrix.values
        n = self.num_assets

        if mode == "group_level":
            # Identify groups
            groups = list(set(self.asset_regions.values()))
            num_groups = len(groups)
            group_indices = {g: [i for i, a in enumerate(self.assets) if self.asset_regions[a] == g] for g in groups}

            def group_risk_budget_obj(w):
                port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
                if port_vol < 1e-8:
                    return 0.0
                marginal_contrib = np.dot(cov, w)
                asset_risk_contrib = w * marginal_contrib
                
                # Compute risk per group
                group_rc = np.array([sum(asset_risk_contrib[i] for i in group_indices[g]) for g in groups])
                target_rc = (port_vol ** 2) / num_groups
                return np.sum((group_rc - target_rc) ** 2)

            obj_func = group_risk_budget_obj
        else:
            # Asset level ERC
            def asset_risk_budget_obj(w):
                port_var = np.dot(w.T, np.dot(cov, w))
                if port_var < 1e-10:
                    return 0.0
                marginal_contrib = np.dot(cov, w)
                asset_rc = w * marginal_contrib
                # Dimensionless relative risk error: (RC_i / port_var) - (1 / n)
                rel_error = (asset_rc / port_var) - (1.0 / n)
                return np.sum(rel_error ** 2)

            obj_func = asset_risk_budget_obj

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = tuple((0.001, 1.0) for _ in range(n))
        init_guess = np.ones(n) / n

        res = minimize(
            obj_func,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        weights = res.x if res.success else init_guess
        weights = np.maximum(weights, 0.0)
        weights = weights / np.sum(weights)

        ret, vol, sharpe = self.portfolio_performance(weights, cov_matrix=cov)

        return {
            "weights": weights,
            "expected_return": ret,
            "expected_volatility": vol,
            "sharpe_ratio": sharpe,
            "success": res.success,
            "method": f"Risk Parity ({mode})",
            "assets": self.assets,
        }

    def compute_black_litterman(
        self,
        market_weights: Optional[np.ndarray] = None,
        views_list: Optional[List[Dict[str, Any]]] = None,
        tau: float = 0.05,
        risk_aversion: float = 2.50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Black-Litterman posterior expected return and covariance matrix.
        
        Supports:
          - Relative views (e.g. NVDA outperforming 7203.T by 4%)
          - Absolute views (e.g. 1155.KL returns 8%)
          
        Returns:
            (pi_equilibrium, posterior_returns, posterior_covariance)
        """
        n = self.num_assets
        sigma = self.cov_matrix.values

        if market_weights is None:
            w_mkt = np.ones(n) / n
        else:
            w_mkt = np.array(market_weights) / np.sum(market_weights)

        # 1. Equilibrium Excess Returns: Pi = lambda * Sigma * w_mkt
        pi = risk_aversion * np.dot(sigma, w_mkt)

        if not views_list:
            return pi + self.rf, pi + self.rf, sigma

        # 2. Build P, Q, Omega
        valid_views = []
        for v in views_list:
            # Relative view
            if "asset_long" in v and "asset_short" in v:
                if v["asset_long"] in self.assets and v["asset_short"] in self.assets:
                    valid_views.append(v)
            # Absolute view
            elif "asset_long" in v or "asset" in v:
                asset_key = v.get("asset_long", v.get("asset"))
                if asset_key in self.assets:
                    valid_views.append(v)

        if not valid_views:
            return pi + self.rf, pi + self.rf, sigma

        k = len(valid_views)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)

        for i, v in enumerate(valid_views):
            conf = max(0.05, min(0.95, v.get("confidence", 0.60)))

            if "asset_short" in v and v["asset_short"]:
                # Relative view: long asset vs short asset
                idx_long = self.assets.index(v["asset_long"])
                idx_short = self.assets.index(v["asset_short"])
                P[i, idx_long] = 1.0
                P[i, idx_short] = -1.0
                Q[i] = v.get("relative_return", 0.04)
            else:
                # Absolute view
                asset_key = v.get("asset_long", v.get("asset"))
                idx = self.assets.index(asset_key)
                P[i, idx] = 1.0
                view_ret = v.get("view_return", 0.08)
                Q[i] = view_ret - self.rf

            # He-Litterman diagonal variance
            view_variance = np.dot(P[i, :], np.dot(tau * sigma, P[i, :].T))
            omega_diag[i] = view_variance * ((1.0 - conf) / conf)

        Omega = np.diag(omega_diag)

        # 3. Posterior Returns:
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
        posterior_cov = sigma + M

        return pi + self.rf, posterior_returns, posterior_cov

    def optimize_black_litterman(
        self,
        views_list: Optional[List[Dict[str, Any]]] = None,
        tau: float = 0.05,
        risk_aversion: float = 2.50,
        max_single_asset: float = 0.25,
    ) -> Dict[str, Any]:
        """Convenience method to compute Black-Litterman posterior and solve optimal allocation."""
        pi, post_ret, post_cov = self.compute_black_litterman(
            views_list=views_list, tau=tau, risk_aversion=risk_aversion
        )
        res = self.optimize_mean_variance(
            objective="max_sharpe",
            max_single_asset=max_single_asset,
            expected_returns=post_ret,
            cov_matrix=post_cov,
        )
        res["method"] = "Black-Litterman Allocation"
        res["prior_equilibrium"] = pi
        res["posterior_returns"] = post_ret
        return res

"""Multi-Model Quantitative Strategy Dispatcher.

Implements unified portfolio allocation strategies:
1. Maximum Sharpe Ratio (MVO)
2. Minimum Volatility
3. Risk Parity / Equal Risk Contribution (ERC)
4. Cross-Sectional Top-N Momentum
5. Inverse Volatility
6. Equal Weight (1/N)
"""

from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from data.aligner import load_universe_registry, get_tradable_tickers
from research.optimization import PortfolioOptimizer


STRATEGY_REGISTRY: Dict[str, str] = {
    "max_sharpe": "Maximum Sharpe Ratio (MVO)",
    "min_volatility": "Minimum Volatility",
    "risk_parity": "Risk Parity (ERC)",
    "top_n_momentum": "Cross-Sectional Top-N Momentum",
    "inverse_volatility": "Inverse Volatility",
    "equal_weight": "Equal Weight (1/N)",
}


class StrategyDispatcher:
    """Unified strategy dispatcher for multi-asset portfolio allocation."""

    @staticmethod
    def get_tradable_returns(returns_df: pd.DataFrame, registry: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Filter input DataFrame to tradable instruments (equities, proxies, commodities)."""
        tradable = get_tradable_tickers(registry)
        cols = [c for c in returns_df.columns if c in tradable]
        if len(cols) >= 2:
            return returns_df[cols].dropna()
        return returns_df.dropna()

    @classmethod
    def dispatch(
        cls,
        strategy_name: str,
        returns_df: pd.DataFrame,
        lookback_days: int = 252,
        risk_free_rate: float = 0.040,
        max_single_asset: float = 0.25,
        top_n: int = 5,
        prices_df: Optional[pd.DataFrame] = None,
        registry: Optional[Dict[str, Any]] = None,
        filter_tradable: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Dispatch strategy and return optimal asset weight vector.
        
        Args:
            strategy_name: Key in STRATEGY_REGISTRY (or title match).
            returns_df: DataFrame of daily percentage returns.
            lookback_days: Estimation window in trading days.
            risk_free_rate: Annualized risk-free rate proxy.
            max_single_asset: Maximum weight cap per asset in MVO.
            top_n: Count of momentum winners to hold.
            prices_df: Optional DataFrame of historical prices.
            registry: Optional universe metadata dictionary.
            filter_tradable: Whether to filter out benchmarks and indicators.
            **kwargs: Extra hyperparameters.
            
        Returns:
            Dict containing weights, assets, strategy metadata, and expected performance.
        """
        # Canonicalize strategy key
        key = strategy_name.lower().strip()
        for k, name in STRATEGY_REGISTRY.items():
            if key == k or key == name.lower().strip() or key in name.lower():
                key = k
                break

        # Filter tradable assets
        clean_rets = cls.get_tradable_returns(returns_df, registry) if filter_tradable else returns_df.dropna()
        if len(clean_rets) > lookback_days:
            sub_rets = clean_rets.iloc[-lookback_days:].copy()
        else:
            sub_rets = clean_rets.copy()

        assets = sub_rets.columns.tolist()
        n = len(assets)
        annual_days = 252

        # 1. Maximum Sharpe Ratio
        if key == "max_sharpe":
            opt = PortfolioOptimizer(sub_rets, risk_free_rate=risk_free_rate, filter_tradable=False)
            res = opt.optimize_mean_variance(objective="max_sharpe", max_single_asset=max_single_asset)
            res["strategy_key"] = "max_sharpe"
            res["strategy_name"] = STRATEGY_REGISTRY["max_sharpe"]
            return res

        # 2. Minimum Volatility
        elif key == "min_volatility":
            opt = PortfolioOptimizer(sub_rets, risk_free_rate=risk_free_rate, filter_tradable=False)
            res = opt.optimize_mean_variance(objective="min_volatility", max_single_asset=max_single_asset)
            res["strategy_key"] = "min_volatility"
            res["strategy_name"] = STRATEGY_REGISTRY["min_volatility"]
            return res

        # 3. Risk Parity (Equal Risk Contribution)
        elif key == "risk_parity":
            opt = PortfolioOptimizer(sub_rets, risk_free_rate=risk_free_rate, filter_tradable=False)
            res = opt.optimize_risk_parity(mode="asset_level")
            res["strategy_key"] = "risk_parity"
            res["strategy_name"] = STRATEGY_REGISTRY["risk_parity"]
            return res

        # 4. Cross-Sectional Top-N Momentum
        elif key == "top_n_momentum":
            cum_ret = (1.0 + sub_rets).prod() - 1.0
            ranked = cum_ret.sort_values(ascending=False)
            top_assets = ranked.iloc[: min(top_n, n)].index.tolist()
            
            w = np.zeros(n)
            w_each = 1.0 / len(top_assets)
            for i, a in enumerate(assets):
                if a in top_assets:
                    w[i] = w_each

            # Compute portfolio performance
            opt = PortfolioOptimizer(sub_rets, risk_free_rate=risk_free_rate, filter_tradable=False)
            ret, vol, sharpe = opt.portfolio_performance(w)
            return {
                "weights": w,
                "assets": assets,
                "expected_return": ret,
                "expected_volatility": vol,
                "sharpe_ratio": sharpe,
                "success": True,
                "strategy_key": "top_n_momentum",
                "strategy_name": STRATEGY_REGISTRY["top_n_momentum"],
                "top_assets": top_assets,
            }

        # 5. Inverse Volatility
        elif key == "inverse_volatility":
            ann_vols = sub_rets.std() * np.sqrt(annual_days)
            inv_vols = 1.0 / np.maximum(ann_vols.values, 1e-4)
            w = inv_vols / np.sum(inv_vols)

            opt = PortfolioOptimizer(sub_rets, risk_free_rate=risk_free_rate, filter_tradable=False)
            ret, vol, sharpe = opt.portfolio_performance(w)
            return {
                "weights": w,
                "assets": assets,
                "expected_return": ret,
                "expected_volatility": vol,
                "sharpe_ratio": sharpe,
                "success": True,
                "strategy_key": "inverse_volatility",
                "strategy_name": STRATEGY_REGISTRY["inverse_volatility"],
            }

        # 6. Equal Weight (1/N)
        else:
            w = np.ones(n) / n
            opt = PortfolioOptimizer(sub_rets, risk_free_rate=risk_free_rate, filter_tradable=False)
            ret, vol, sharpe = opt.portfolio_performance(w)
            return {
                "weights": w,
                "assets": assets,
                "expected_return": ret,
                "expected_volatility": vol,
                "sharpe_ratio": sharpe,
                "success": True,
                "strategy_key": "equal_weight",
                "strategy_name": STRATEGY_REGISTRY["equal_weight"],
            }

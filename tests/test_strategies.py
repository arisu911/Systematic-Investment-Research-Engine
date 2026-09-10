"""Unit tests for StrategyDispatcher and multi-strategy optimization models."""

import numpy as np
import pandas as pd
import pytest

from research.strategies import StrategyDispatcher, STRATEGY_REGISTRY
from research.risk import RiskEngine


@pytest.fixture
def synthetic_market_data():
    np.random.seed(42)
    n_days = 252
    assets = ["SPY", "QQQ", "TLT", "GLD", "EEM"]
    
    # Generate returns with different drifts and volatilities
    drifts = [0.0006, 0.0008, 0.0001, 0.0003, 0.0004]
    vols = [0.012, 0.016, 0.008, 0.010, 0.018]
    
    data = {}
    for a, d, v in zip(assets, drifts, vols):
        data[a] = np.random.normal(d, v, n_days)
        
    returns_df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=n_days, freq="B"))
    prices_df = (1.0 + returns_df).cumprod() * 100.0
    return returns_df, prices_df


def test_strategy_registry_completeness():
    expected_strategies = {
        "max_sharpe",
        "min_volatility",
        "risk_parity",
        "top_n_momentum",
        "inverse_volatility",
        "equal_weight",
    }
    assert expected_strategies.issubset(set(STRATEGY_REGISTRY.keys()))


def test_dispatch_all_strategies(synthetic_market_data):
    returns_df, prices_df = synthetic_market_data

    for strat_key in STRATEGY_REGISTRY.keys():
        res = StrategyDispatcher.dispatch(
            strategy_name=strat_key,
            returns_df=returns_df,
            prices_df=prices_df,
            risk_free_rate=0.04,
            filter_tradable=False,
        )

        assert "weights" in res
        assert "assets" in res
        assert "strategy_key" in res
        assert res["strategy_key"] == strat_key

        w = res["weights"]
        assert len(w) == len(res["assets"])
        # Weights must sum to 1.0 within tolerance
        assert np.isclose(np.sum(w), 1.0, atol=1e-3)
        # Weights must be non-negative (long-only)
        assert np.all(w >= -1e-5)


def test_risk_parity_balancing(synthetic_market_data):
    returns_df, _ = synthetic_market_data
    res = StrategyDispatcher.dispatch(
        strategy_name="risk_parity",
        returns_df=returns_df,
        filter_tradable=False,
    )
    w = res["weights"]
    cov = returns_df.cov().values * 252.0
    
    rc = RiskEngine.calculate_risk_contributions(w, cov)
    pcr = rc["pcr"]
    
    # In equal risk contribution, all PCRs should be approximately equal (1/N)
    target_pcr = 1.0 / len(w)
    for p in pcr:
        assert np.isclose(p, target_pcr, atol=0.05)


def test_top_n_momentum_logic():
    # Setup data where WINNER1 and WINNER2 have massive positive drift, others negative
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    prices_df = pd.DataFrame({
        "WINNER1": np.linspace(100, 200, 100),
        "WINNER2": np.linspace(100, 180, 100),
        "LOSER1": np.linspace(100, 50, 100),
        "LOSER2": np.linspace(100, 30, 100),
    }, index=dates)
    returns_df = prices_df.pct_change().dropna()

    res = StrategyDispatcher.dispatch(
        strategy_name="top_n_momentum",
        returns_df=returns_df,
        prices_df=prices_df,
        top_n=2,
        filter_tradable=False,
    )
    
    w = res["weights"]
    assets = res["assets"]
    w_dict = dict(zip(assets, w))
    
    # Top 2 winners should have weight, losers should have 0 weight
    assert w_dict["WINNER1"] > 0.40
    assert w_dict["WINNER2"] > 0.40
    assert w_dict["LOSER1"] == 0.0
    assert w_dict["LOSER2"] == 0.0

"""Unit tests for StressTestEngine macro scenario stress testing."""

import numpy as np
import pandas as pd
import pytest
from experiments.stress_test import StressTestEngine


@pytest.fixture
def macro_stress_data():
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=252, freq="B")
    
    # Asset returns
    ret_df = pd.DataFrame({
        "1155.KL": np.random.normal(0.0003, 0.010, len(dates)),
        "NVDA": np.random.normal(0.0010, 0.025, len(dates)),
        "7203.T": np.random.normal(0.0004, 0.012, len(dates)),
        "GC=F": np.random.normal(0.0002, 0.008, len(dates)),
    }, index=dates)

    # Macro factors
    macro_df = pd.DataFrame({
        "^VIX": 20.0 + np.cumsum(np.random.normal(0, 0.5, len(dates))),
        "^TNX": 4.0 + np.cumsum(np.random.normal(0, 0.02, len(dates))),
        "BZ=F": 80.0 + np.cumsum(np.random.normal(0, 1.0, len(dates))),
        "USDMYR=X": 4.4 + np.cumsum(np.random.normal(0, 0.01, len(dates))),
        "JPY=X": 150.0 + np.cumsum(np.random.normal(0, 0.3, len(dates))),
    }, index=dates)

    return ret_df, macro_df


def test_simulate_macro_shock(macro_stress_data):
    ret_df, macro_df = macro_stress_data
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    
    engine = StressTestEngine(ret_df, macro_df=macro_df)
    
    # VIX doubling shock (+100%)
    res_vix = engine.simulate_macro_shock(weights, factor_name="^VIX", shock_magnitude_pct=1.00)
    assert "portfolio_impact_pct" in res_vix
    assert isinstance(res_vix["portfolio_impact_pct"], float)
    assert len(res_vix["asset_breakdown"]) == 4

    # Oil shock (+50%)
    res_oil = engine.simulate_macro_shock(weights, factor_name="BZ=F", shock_magnitude_pct=0.50)
    assert "portfolio_impact_pct" in res_oil


def test_standard_scenarios_and_crises(macro_stress_data):
    ret_df, macro_df = macro_stress_data
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    
    engine = StressTestEngine(ret_df, macro_df=macro_df)
    
    scenarios = engine.run_standard_macro_scenarios(weights)
    assert len(scenarios) == 5
    for s in scenarios:
        assert "scenario_name" in s
        assert "portfolio_impact_pct" in s

    crises = engine.replay_historical_crises(weights)
    assert len(crises) == 4
    for c in crises:
        assert "Crisis" in c
        assert "Estimated_Max_Drawdown" in c
        assert c["Estimated_Max_Drawdown"] <= 0.0  # Drawdown is non-positive

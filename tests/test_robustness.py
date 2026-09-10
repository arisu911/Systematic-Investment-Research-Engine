"""Unit tests for parameter grid sweeps, cost sensitivity, and overfitting warnings."""

import numpy as np
import pandas as pd
from quant_engine.config.strategies import StrategyConfig
from quant_engine.signals.momentum import MovingAverageMomentum
from quant_engine.validation.parameter_sensitivity import ParameterSensitivityAnalyzer
from quant_engine.validation.robustness import RobustnessTester
from quant_engine.validation.overfitting import OverfittingDetector


def test_parameter_sweep_matrix():
    dates = pd.date_range("2021-01-01", periods=200, freq="B")
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.randn(200))
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 1.0,
            "Low": prices - 1.0,
            "Close": prices,
            "Volume": 100000,
        },
        index=dates,
    )

    base_cfg = StrategyConfig()
    analyzer = ParameterSensitivityAnalyzer()
    sharpe_df, cagr_df, stability = analyzer.sweep_2d(
        df=df,
        strategy_class=MovingAverageMomentum,
        base_config=base_cfg,
        param1_name="fast_period",
        param1_values=[5, 10],
        param2_name="slow_period",
        param2_values=[20, 40],
    )

    assert sharpe_df.shape == (2, 2)
    assert 0.0 <= stability <= 1.0


def test_overfitting_detector_flags_decay():
    # Severe decay: In-Sample 2.5, Out-of-Sample -0.2
    audit = OverfittingDetector.audit(
        in_sample_sharpe=2.5,
        out_of_sample_sharpe=-0.2,
        total_trades=8,
        annualized_turnover=35.0,
        parameter_stability_score=0.20,
    )

    assert audit.risk_level in ["HIGH", "SEVERE"]
    assert len(audit.warnings) >= 3
    assert not audit.is_acceptable_for_live_consideration

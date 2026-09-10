"""Unit tests for SQLite ResearchStore and automated FindingsGenerator."""

import pandas as pd
import numpy as np
from quant_engine.research.store import ResearchStore
from quant_engine.research.findings_generator import FindingsGenerator


def test_research_store_lifecycle(tmp_path):
    db_file = tmp_path / "test_research.db"
    store = ResearchStore(db_path=str(db_file))

    # 1. Test Run Metadata
    store.save_run(
        run_id="RUN-TEST-001",
        timestamp="2026-09-10T12:00:00Z",
        protocol_version="1.0.0",
        engine_version="0.1.0",
        status="COMPLETED",
        total_experiments=5,
        total_markets=2,
        notes="Automated test run",
    )
    latest_run = store.get_latest_run()
    assert latest_run is not None
    assert latest_run["run_id"] == "RUN-TEST-001"
    assert latest_run["total_experiments"] == 5

    # 2. Test Data Health Status
    data_rec = {
        "symbol": "1155.KL",
        "name": "Malayan Banking Berhad",
        "market": "MY",
        "provider": "TestProvider",
        "start_date": "2019-01-01",
        "end_date": "2024-12-31",
        "total_bars": 1400,
        "health_score": 98.5,
        "warnings": [],
        "is_demo": False,
        "updated_at": "2026-09-10T12:05:00Z",
    }
    store.save_data_status(data_rec)
    status_df = store.get_data_status()
    assert len(status_df) == 1
    assert status_df.iloc[0]["symbol"] == "1155.KL"
    assert status_df.iloc[0]["health_score"] == 98.5

    # 3. Test Strategy Results
    strat_rec_1 = {
        "experiment_id": "EXP-MY-1155-001",
        "run_id": "RUN-TEST-001",
        "market": "MY",
        "symbol": "1155.KL",
        "strategy_name": "Momentum Lookback 120d",
        "strategy_type": "TimeSeriesMomentum",
        "parameters": {"lookback_days": 120},
        "cagr": 0.14,
        "sharpe_ratio": 1.15,
        "sortino_ratio": 1.45,
        "calmar_ratio": 0.90,
        "max_drawdown": -0.15,
        "annualized_volatility": 0.16,
        "total_net_return": 0.95,
        "total_gross_return": 1.10,
        "total_fees_paid": 450.0,
        "total_slippage_paid": 320.0,
        "annualized_turnover": 4.5,
        "friction_survival_ratio": 0.86,
        "win_rate": 0.54,
        "profit_factor": 1.65,
        "total_trades": 38,
        "oos_sharpe": 0.85,
        "oos_cagr": 0.11,
        "oos_max_drawdown": -0.16,
        "walk_forward_efficiency": 0.74,
        "parameter_stability_score": 0.88,
        "breakeven_cost_bps": 45.0,
        "bull_sharpe": 1.40,
        "bear_sharpe": 0.20,
        "high_vol_sharpe": 0.45,
        "low_vol_sharpe": 1.25,
        "overfitting_risk_level": "LOW",
        "overfitting_score": 15.0,
    }

    strat_rec_2 = {
        "experiment_id": "EXP-US-SPY-002",
        "run_id": "RUN-TEST-001",
        "market": "US",
        "symbol": "SPY",
        "strategy_name": "Mean Reversion 20d",
        "strategy_type": "MeanReversion",
        "parameters": {"lookback_days": 20},
        "cagr": 0.08,
        "sharpe_ratio": 0.65,
        "sortino_ratio": 0.80,
        "calmar_ratio": 0.45,
        "max_drawdown": -0.18,
        "annualized_volatility": 0.18,
        "total_net_return": 0.05,
        "total_gross_return": 0.35,
        "total_fees_paid": 1200.0,
        "total_slippage_paid": 800.0,
        "annualized_turnover": 25.0,
        "friction_survival_ratio": 0.14,
        "win_rate": 0.48,
        "profit_factor": 1.05,
        "total_trades": 180,
        "oos_sharpe": -0.10,
        "oos_cagr": -0.02,
        "oos_max_drawdown": -0.22,
        "walk_forward_efficiency": -0.15,
        "parameter_stability_score": 0.40,
        "breakeven_cost_bps": 8.0,
        "bull_sharpe": 0.75,
        "bear_sharpe": -0.50,
        "high_vol_sharpe": -0.30,
        "low_vol_sharpe": 0.80,
        "overfitting_risk_level": "HIGH",
        "overfitting_score": 75.0,
    }

    store.save_strategy_result(strat_rec_1)
    store.save_strategy_result(strat_rec_2)

    res_df = store.get_strategy_results()
    assert len(res_df) == 2

    my_res = store.get_strategy_results(market="MY")
    assert len(my_res) == 1
    assert my_res.iloc[0]["symbol"] == "1155.KL"

    # 4. Test Equity Curve
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    curve_df = pd.DataFrame({
        "NAV": np.linspace(1.0, 1.2, 10),
        "Gross_NAV": np.linspace(1.0, 1.25, 10),
        "Benchmark_NAV": np.linspace(1.0, 1.1, 10),
        "Position": [1.0] * 10,
        "Drawdown": [0.0] * 10,
        "Turnover": [0.1] * 10,
    }, index=dates)
    store.save_equity_curve("EXP-MY-1155-001", curve_df)

    loaded_curve = store.get_equity_curve("EXP-MY-1155-001")
    assert len(loaded_curve) == 10
    assert "NAV" in loaded_curve.columns
    assert loaded_curve["NAV"].iloc[-1] == 1.2

    # 5. Test Findings Generation
    findings_gen = FindingsGenerator(store)
    findings = findings_gen.generate_all_findings(run_id="RUN-TEST-001")
    assert len(findings) >= 3  # Should produce OOS leader, friction decay, Bursa status
    store.save_findings(findings)

    saved_findings = store.get_findings(run_id="RUN-TEST-001")
    assert len(saved_findings) >= 3

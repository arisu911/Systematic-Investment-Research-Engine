"""Unit tests for experiment registry serialization and persistence."""

from quant_engine.registry.experiments import ExperimentRecord, ExperimentRegistry


def test_registry_save_and_retrieve(tmp_path):
    reg_file = tmp_path / "test_reg.json"
    registry = ExperimentRegistry(reg_file)

    rec = ExperimentRecord(
        strategy_name="Test Strategy",
        strategy_type="TimeSeriesMomentum",
        market="MY",
        symbol="1155.KL",
        start_date="2020-01-01",
        end_date="2022-12-31",
        parameters={"lookback_days": 120},
        transaction_costs={"brokerage_bps": 10.0},
        slippage_bps=5.0,
        metrics={"cagr": 0.12, "sharpe_ratio": 0.95, "max_drawdown": -0.15},
        notes="Test run",
    )

    exp_id = registry.save(rec)
    assert exp_id.startswith("EXP-")

    retrieved = registry.get(exp_id)
    assert retrieved is not None
    assert retrieved.strategy_name == "Test Strategy"
    assert retrieved.metrics["sharpe_ratio"] == 0.95

    df = registry.to_dataframe()
    assert len(df) == 1
    assert df.iloc[0]["ID"] == exp_id

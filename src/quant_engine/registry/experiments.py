"""Persistent Research Experiment Registry."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field
from quant_engine.config.settings import get_settings


class ExperimentRecord(BaseModel):
    """Complete snapshot of an empirical quantitative experiment."""

    experiment_id: str = Field(default_factory=lambda: f"EXP-{uuid.uuid4().hex[:8].upper()}")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    strategy_name: str
    strategy_type: str
    market: str
    symbol: str
    start_date: str
    end_date: str
    parameters: Dict[str, Any]
    transaction_costs: Dict[str, Any]
    slippage_bps: float
    metrics: Dict[str, Any]
    oos_metrics: Optional[Dict[str, Any]] = None
    overfitting_risk_level: str = "LOW"
    notes: Optional[str] = None


class ExperimentRegistry:
    """Manages persistent disk storage of experiments in JSON format."""

    def __init__(self, registry_file: Optional[Path] = None):
        self.file_path = registry_file or (get_settings().experiments_dir / "registry.json")
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self._save_all([])

    def _load_all(self) -> List[Dict[str, Any]]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_all(self, records: List[Dict[str, Any]]) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    def save(self, record: ExperimentRecord) -> str:
        records = self._load_all()
        records.append(record.model_dump())
        self._save_all(records)
        return record.experiment_id

    def list_all(self) -> List[ExperimentRecord]:
        raw = self._load_all()
        return [ExperimentRecord(**r) for r in raw]

    def get(self, experiment_id: str) -> Optional[ExperimentRecord]:
        for r in self.list_all():
            if r.experiment_id == experiment_id:
                return r
        return None

    def delete(self, experiment_id: str) -> bool:
        records = self._load_all()
        filtered = [r for r in records if r.get("experiment_id") != experiment_id]
        if len(filtered) != len(records):
            self._save_all(filtered)
            return True
        return False

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten stored experiments into a clean tabular DataFrame for Streamlit UI."""
        experiments = self.list_all()
        if not experiments:
            return pd.DataFrame()

        rows = []
        for e in experiments:
            row = {
                "ID": e.experiment_id,
                "Date": e.timestamp[:10],
                "Market": e.market,
                "Symbol": e.symbol,
                "Strategy": e.strategy_name,
                "CAGR (%)": round(e.metrics.get("cagr", 0.0) * 100.0, 2),
                "Sharpe": e.metrics.get("sharpe_ratio", 0.0),
                "Max DD (%)": round(abs(e.metrics.get("max_drawdown", 0.0)) * 100.0, 2),
                "OOS Sharpe": e.oos_metrics.get("oos_sharpe", "-") if e.oos_metrics else "-",
                "Overfitting": e.overfitting_risk_level,
                "Notes": e.notes or "",
            }
            rows.append(row)
        return pd.DataFrame(rows)

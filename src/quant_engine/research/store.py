"""Persistent Research Results Layer using SQLite.

High-performance storage and instant query interface for precomputed backtests,
walk-forward results, equity curves, data hygiene logs, and quantitative findings.
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from quant_engine.config.settings import get_settings


class ResearchStore:
    """Manages reading and writing to the precomputed quantitative research database."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_dir = get_settings().root_dir / "results" / "database"
            db_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = db_dir / "research.db"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create database tables and performance indexes if not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS research_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    protocol_version TEXT NOT NULL,
                    engine_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_experiments INTEGER DEFAULT 0,
                    total_markets INTEGER DEFAULT 0,
                    notes TEXT
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS data_status (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    market TEXT,
                    provider TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    total_bars INTEGER,
                    health_score REAL,
                    warnings TEXT,
                    is_demo INTEGER,
                    updated_at TEXT
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_results (
                    experiment_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    market TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    strategy_type TEXT NOT NULL,
                    parameters_json TEXT,
                    cagr REAL,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    calmar_ratio REAL,
                    max_drawdown REAL,
                    annualized_volatility REAL,
                    total_net_return REAL,
                    total_gross_return REAL,
                    total_fees_paid REAL,
                    total_slippage_paid REAL,
                    annualized_turnover REAL,
                    friction_survival_ratio REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    total_trades INTEGER,
                    oos_sharpe REAL,
                    oos_cagr REAL,
                    oos_max_drawdown REAL,
                    walk_forward_efficiency REAL,
                    parameter_stability_score REAL,
                    breakeven_cost_bps REAL,
                    bull_sharpe REAL,
                    bear_sharpe REAL,
                    high_vol_sharpe REAL,
                    low_vol_sharpe REAL,
                    overfitting_risk_level TEXT,
                    overfitting_score REAL
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS equity_curves (
                    experiment_id TEXT,
                    date TEXT,
                    nav REAL,
                    gross_nav REAL,
                    benchmark_nav REAL,
                    drawdown REAL,
                    position REAL,
                    turnover REAL,
                    PRIMARY KEY (experiment_id, date)
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS walk_forward_slices (
                    experiment_id TEXT,
                    slice_id INTEGER,
                    train_start TEXT,
                    train_end TEXT,
                    test_start TEXT,
                    test_end TEXT,
                    in_sample_sharpe REAL,
                    out_of_sample_sharpe REAL,
                    in_sample_cagr REAL,
                    out_of_sample_cagr REAL,
                    walk_forward_efficiency REAL,
                    PRIMARY KEY (experiment_id, slice_id)
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS research_findings (
                    finding_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    category TEXT,
                    headline TEXT,
                    narrative TEXT,
                    supporting_metric TEXT,
                    severity TEXT
                )
                """
            )

            # Performance indexes for instant Streamlit filtering
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_strat_market ON strategy_results(market)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_strat_type ON strategy_results(strategy_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_strat_symbol ON strategy_results(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_strat_sharpe ON strategy_results(sharpe_ratio)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_eq_exp ON equity_curves(experiment_id)")

            conn.commit()

    # ---------------- WRITE OPERATIONS ----------------

    def save_run(
        self,
        run_id: str,
        timestamp: str,
        protocol_version: str,
        engine_version: str,
        status: str = "COMPLETED",
        total_experiments: int = 0,
        total_markets: int = 0,
        notes: Optional[str] = None,
    ) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO research_runs
                (run_id, timestamp, protocol_version, engine_version, status, total_experiments, total_markets, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, timestamp, protocol_version, engine_version, status, total_experiments, total_markets, notes),
            )
            conn.commit()

    def save_data_status(self, record: Dict[str, Any]) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO data_status
                (symbol, name, market, provider, start_date, end_date, total_bars, health_score, warnings, is_demo, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["symbol"],
                    record.get("name", record["symbol"]),
                    record.get("market", "GLOBAL"),
                    record.get("provider", "Yahoo Finance"),
                    record.get("start_date", ""),
                    record.get("end_date", ""),
                    record.get("total_bars", 0),
                    record.get("health_score", 100.0),
                    json.dumps(record.get("warnings", [])),
                    1 if record.get("is_demo", False) else 0,
                    record.get("updated_at", ""),
                ),
            )
            conn.commit()

    def save_strategy_result(self, res: Dict[str, Any]) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO strategy_results
                (experiment_id, run_id, market, symbol, strategy_name, strategy_type, parameters_json,
                 cagr, sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown, annualized_volatility,
                 total_net_return, total_gross_return, total_fees_paid, total_slippage_paid,
                 annualized_turnover, friction_survival_ratio, win_rate, profit_factor, total_trades,
                 oos_sharpe, oos_cagr, oos_max_drawdown, walk_forward_efficiency,
                 parameter_stability_score, breakeven_cost_bps,
                 bull_sharpe, bear_sharpe, high_vol_sharpe, low_vol_sharpe,
                 overfitting_risk_level, overfitting_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    res["experiment_id"],
                    res.get("run_id", "RUN-DEFAULT"),
                    res["market"],
                    res["symbol"],
                    res["strategy_name"],
                    res["strategy_type"],
                    json.dumps(res.get("parameters", {})),
                    res.get("cagr", 0.0),
                    res.get("sharpe_ratio", 0.0),
                    res.get("sortino_ratio", 0.0),
                    res.get("calmar_ratio", 0.0),
                    res.get("max_drawdown", 0.0),
                    res.get("annualized_volatility", 0.0),
                    res.get("total_net_return", 0.0),
                    res.get("total_gross_return", 0.0),
                    res.get("total_fees_paid", 0.0),
                    res.get("total_slippage_paid", 0.0),
                    res.get("annualized_turnover", 0.0),
                    res.get("friction_survival_ratio", 0.0),
                    res.get("win_rate", 0.0),
                    res.get("profit_factor", 0.0),
                    res.get("total_trades", 0),
                    res.get("oos_sharpe", 0.0),
                    res.get("oos_cagr", 0.0),
                    res.get("oos_max_drawdown", 0.0),
                    res.get("walk_forward_efficiency", 0.0),
                    res.get("parameter_stability_score", 0.0),
                    res.get("breakeven_cost_bps", 0.0),
                    res.get("bull_sharpe", 0.0),
                    res.get("bear_sharpe", 0.0),
                    res.get("high_vol_sharpe", 0.0),
                    res.get("low_vol_sharpe", 0.0),
                    res.get("overfitting_risk_level", "LOW"),
                    res.get("overfitting_score", 0.0),
                ),
            )
            conn.commit()

    def save_equity_curve(self, experiment_id: str, curve_df: pd.DataFrame) -> None:
        """Batch insert daily time series for an experiment."""
        records = []
        for dt, row in curve_df.iterrows():
            date_str = str(dt.date()) if hasattr(dt, "date") else str(dt)[:10]
            records.append(
                (
                    experiment_id,
                    date_str,
                    float(row.get("NAV", 1.0)),
                    float(row.get("Gross_NAV", 1.0)),
                    float(row.get("Benchmark_NAV", 1.0)),
                    float(row.get("Drawdown", 0.0)),
                    float(row.get("Position", 0.0)),
                    float(row.get("Turnover", 0.0)),
                )
            )
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO equity_curves
                (experiment_id, date, nav, gross_nav, benchmark_nav, drawdown, position, turnover)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()

    def save_walk_forward_slices(self, experiment_id: str, slices: List[Dict[str, Any]]) -> None:
        records = [
            (
                experiment_id,
                s["slice_id"],
                s["train_start"],
                s["train_end"],
                s["test_start"],
                s["test_end"],
                s["in_sample_sharpe"],
                s["out_of_sample_sharpe"],
                s["in_sample_cagr"],
                s["out_of_sample_cagr"],
                s["walk_forward_efficiency"],
            )
            for s in slices
        ]
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO walk_forward_slices
                (experiment_id, slice_id, train_start, train_end, test_start, test_end,
                 in_sample_sharpe, out_of_sample_sharpe, in_sample_cagr, out_of_sample_cagr, walk_forward_efficiency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()

    def save_findings(self, findings: List[Dict[str, Any]]) -> None:
        records = [
            (
                f["finding_id"],
                f.get("run_id", "RUN-DEFAULT"),
                f.get("category", "General"),
                f.get("headline", ""),
                f.get("narrative", ""),
                json.dumps(f.get("supporting_metric", {})),
                f.get("severity", "INFO"),
            )
            for f in findings
        ]
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO research_findings
                (finding_id, run_id, category, headline, narrative, supporting_metric, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()

    # ---------------- QUERY OPERATIONS (FOR STREAMLIT) ----------------

    def get_latest_run_summary(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM research_runs ORDER BY timestamp DESC LIMIT 1").fetchone()
            return dict(row) if row else None

    def get_latest_run(self) -> Optional[Dict[str, Any]]:
        """Alias for get_latest_run_summary."""
        return self.get_latest_run_summary()

    def get_data_status_table(self) -> pd.DataFrame:
        with self._get_connection() as conn:
            df = pd.read_sql("SELECT * FROM data_status ORDER BY market, symbol", conn)
            return df

    def get_data_status(self) -> pd.DataFrame:
        """Alias for get_data_status_table."""
        return self.get_data_status_table()

    def get_strategy_results(
        self,
        market: Optional[str] = None,
        strategy_type: Optional[str] = None,
        symbol: Optional[str] = None,
        min_sharpe: Optional[float] = None,
    ) -> pd.DataFrame:
        query = "SELECT * FROM strategy_results WHERE 1=1"
        params = []
        if market and market != "ALL":
            query += " AND market = ?"
            params.append(market)
        if strategy_type and strategy_type != "ALL":
            query += " AND strategy_type = ?"
            params.append(strategy_type)
        if symbol and symbol != "ALL":
            query += " AND symbol = ?"
            params.append(symbol)
        if min_sharpe is not None:
            query += " AND sharpe_ratio >= ?"
            params.append(min_sharpe)

        query += " ORDER BY sharpe_ratio DESC"

        with self._get_connection() as conn:
            df = pd.read_sql(query, conn, params=params)
            return df

    def get_equity_curve(self, experiment_id: str) -> pd.DataFrame:
        with self._get_connection() as conn:
            df = pd.read_sql(
                "SELECT date, nav, gross_nav, benchmark_nav, drawdown, position, turnover FROM equity_curves WHERE experiment_id = ? ORDER BY date",
                conn,
                params=[experiment_id],
                index_col="date",
                parse_dates=["date"],
            )
            df.rename(
                columns={
                    "nav": "NAV",
                    "gross_nav": "Gross_NAV",
                    "benchmark_nav": "Benchmark_NAV",
                    "drawdown": "Drawdown",
                    "position": "Position",
                    "turnover": "Turnover",
                },
                inplace=True,
            )
            return df

    def get_walk_forward_slices(self, experiment_id: str) -> pd.DataFrame:
        with self._get_connection() as conn:
            df = pd.read_sql(
                "SELECT * FROM walk_forward_slices WHERE experiment_id = ? ORDER BY slice_id",
                conn,
                params=[experiment_id],
            )
            return df

    def get_all_findings(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM research_findings"
        params = []
        if category and category != "ALL":
            query += " WHERE category = ?"
            params.append(category)
        query += " ORDER BY rowid ASC"
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_findings(self, run_id: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve research findings with optional filtering by run_id or category."""
        query = "SELECT * FROM research_findings WHERE 1=1"
        params = []
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if category and category != "ALL":
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY rowid ASC"
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_distinct_markets(self) -> List[str]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT DISTINCT market FROM strategy_results ORDER BY market").fetchall()
            return [r[0] for r in rows]

    def get_distinct_strategies(self) -> List[str]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT DISTINCT strategy_name FROM strategy_results ORDER BY strategy_name").fetchall()
            return [r[0] for r in rows]

    def get_distinct_symbols(self, market: Optional[str] = None) -> List[str]:
        query = "SELECT DISTINCT symbol FROM strategy_results"
        params = []
        if market:
            query += " WHERE market = ?"
            params.append(market)
        query += " ORDER BY symbol"
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [r[0] for r in rows]

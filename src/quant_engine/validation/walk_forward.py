"""Walk-forward analysis: Rolling and expanding window in-sample/out-of-sample validation."""

from typing import List, Dict, Any, Type, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant_engine.config.strategies import StrategyConfig
from quant_engine.signals.base import SignalGenerator
from quant_engine.backtest.engine import BacktestEngine
from quant_engine.analytics.performance import calculate_sharpe_ratio, calculate_cagr


class WalkForwardSlice(BaseModel):
    """Execution metrics for a single walk-forward train/test slice."""

    slice_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    in_sample_cagr: float
    out_of_sample_cagr: float
    walk_forward_efficiency: float


class WalkForwardResult(BaseModel):
    """Aggregate walk-forward validation results."""

    slices: List[WalkForwardSlice]
    mean_is_sharpe: float
    mean_oos_sharpe: float
    overall_wfe: float
    oos_total_return: float
    oos_cagr: float
    oos_sharpe: float
    oos_max_drawdown: float


class WalkForwardEngine:
    """Executes chronological rolling or expanding walk-forward validation."""

    def __init__(
        self,
        train_window_bars: int = 504,   # ~2 trading years
        test_window_bars: int = 126,    # ~6 months
        step_bars: int = 126,
        expanding: bool = False,
    ):
        self.train_window_bars = train_window_bars
        self.test_window_bars = test_window_bars
        self.step_bars = step_bars
        self.expanding = expanding

    def run(
        self,
        df: pd.DataFrame,
        strategy_class: Type[SignalGenerator],
        strategy_config: StrategyConfig,
        backtest_engine: Optional[BacktestEngine] = None,
    ) -> tuple[WalkForwardResult, pd.Series]:
        """Execute walk-forward testing.
        
        Returns:
            result: WalkForwardResult summary object
            oos_equity_curve: Continuous Out-of-Sample equity curve series
        """
        assert len(df) >= self.train_window_bars + self.test_window_bars, (
            f"Insufficient data length ({len(df)}) for train={self.train_window_bars} and test={self.test_window_bars}"
        )

        engine = backtest_engine or BacktestEngine()
        total_bars = len(df)
        slice_records: List[WalkForwardSlice] = []
        oos_returns_list: List[pd.Series] = []

        start_train = 0
        slice_id = 1

        while True:
            end_train = start_train + self.train_window_bars if not self.expanding else (slice_id - 1) * self.step_bars + self.train_window_bars
            start_test = end_train
            end_test = start_test + self.test_window_bars

            if end_test > total_bars:
                break

            train_df = df.iloc[start_train:end_train]
            test_df = df.iloc[start_test:end_test]

            # 1. In-Sample Execution
            strat_is = strategy_class(strategy_config)
            is_signals = strat_is.generate_signals(train_df)
            is_res_df, _, _ = engine.run(train_df, is_signals)

            is_sharpe = calculate_sharpe_ratio(is_res_df["Net_Return"])
            is_cagr = calculate_cagr(is_res_df["NAV"])

            # 2. Out-of-Sample Execution
            strat_oos = strategy_class(strategy_config)
            # Strategy on test_df: Note that for technical indicators to warm up without leakage,
            # we provide the preceding window but strictly slice the test return window
            warmup_window = min(60, len(train_df))
            eval_window = pd.concat([train_df.iloc[-warmup_window:], test_df])
            all_signals = strat_oos.generate_signals(eval_window)
            oos_signals = all_signals.loc[test_df.index]

            oos_res_df, _, _ = engine.run(test_df, oos_signals)
            oos_sharpe = calculate_sharpe_ratio(oos_res_df["Net_Return"])
            oos_cagr = calculate_cagr(oos_res_df["NAV"])

            wfe = (oos_sharpe / is_sharpe) if is_sharpe > 0.1 else (1.0 if oos_sharpe > 0 else 0.0)

            slice_records.append(
                WalkForwardSlice(
                    slice_id=slice_id,
                    train_start=str(train_df.index[0].date()),
                    train_end=str(train_df.index[-1].date()),
                    test_start=str(test_df.index[0].date()),
                    test_end=str(test_df.index[-1].date()),
                    in_sample_sharpe=round(float(is_sharpe), 2),
                    out_of_sample_sharpe=round(float(oos_sharpe), 2),
                    in_sample_cagr=round(float(is_cagr), 4),
                    out_of_sample_cagr=round(float(oos_cagr), 4),
                    walk_forward_efficiency=round(float(wfe), 2),
                )
            )

            oos_returns_list.append(oos_res_df["Net_Return"])

            slice_id += 1
            start_train += self.step_bars

        # Assemble continuous out-of-sample returns
        if oos_returns_list:
            combined_oos_returns = pd.concat(oos_returns_list)
            combined_oos_returns = combined_oos_returns[~combined_oos_returns.index.duplicated(keep="first")]
            oos_equity = (1.0 + combined_oos_returns).cumprod()
        else:
            combined_oos_returns = pd.Series(dtype=float)
            oos_equity = pd.Series(dtype=float)

        mean_is = float(np.mean([s.in_sample_sharpe for s in slice_records])) if slice_records else 0.0
        mean_oos = float(np.mean([s.out_of_sample_sharpe for s in slice_records])) if slice_records else 0.0
        overall_wfe = (mean_oos / mean_is) if mean_is > 0.1 else 0.0

        overall_oos_sharpe = calculate_sharpe_ratio(combined_oos_returns) if not combined_oos_returns.empty else 0.0
        overall_oos_cagr = calculate_cagr(oos_equity) if not oos_equity.empty else 0.0
        peak = oos_equity.cummax() if not oos_equity.empty else pd.Series([1.0])
        dd = ((oos_equity - peak) / peak).min() if not oos_equity.empty else 0.0

        result = WalkForwardResult(
            slices=slice_records,
            mean_is_sharpe=round(mean_is, 2),
            mean_oos_sharpe=round(mean_oos, 2),
            overall_wfe=round(overall_wfe, 2),
            oos_total_return=round(float(oos_equity.iloc[-1] - 1.0) if not oos_equity.empty else 0.0, 4),
            oos_cagr=round(float(overall_oos_cagr), 4),
            oos_sharpe=round(float(overall_oos_sharpe), 2),
            oos_max_drawdown=round(float(dd), 4),
        )

        return result, oos_equity

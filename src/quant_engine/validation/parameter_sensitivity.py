"""Parameter sensitivity analysis, grid sweeps, and surface stability scoring."""

from typing import Dict, Any, List, Type, Tuple, Optional
import numpy as np
import pandas as pd
from quant_engine.config.strategies import StrategyConfig
from quant_engine.signals.base import SignalGenerator
from quant_engine.backtest.engine import BacktestEngine
from quant_engine.analytics.performance import calculate_sharpe_ratio, calculate_cagr


class ParameterSensitivityAnalyzer:
    """Evaluates strategy performance across multidimensional parameter grids."""

    def __init__(self, backtest_engine: Optional[BacktestEngine] = None):
        self.engine = backtest_engine or BacktestEngine()

    def sweep_2d(
        self,
        df: pd.DataFrame,
        strategy_class: Type[SignalGenerator],
        base_config: StrategyConfig,
        param1_name: str,
        param1_values: List[Any],
        param2_name: str,
        param2_values: List[Any],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """Perform 2D parameter grid sweep.
        
        Returns:
            sharpe_matrix: DataFrame of Sharpe ratios (rows=param1, cols=param2)
            cagr_matrix: DataFrame of CAGRs
            stability_score: 0.0 (brittle) to 1.0 (highly robust plateau)
        """
        sharpe_grid = np.zeros((len(param1_values), len(param2_values)))
        cagr_grid = np.zeros((len(param1_values), len(param2_values)))

        for i, val1 in enumerate(param1_values):
            for j, val2 in enumerate(param2_values):
                test_params = dict(base_config.parameters)
                test_params[param1_name] = val1
                test_params[param2_name] = val2

                cfg = StrategyConfig(
                    name=base_config.name,
                    strategy_type=base_config.strategy_type,
                    parameters=test_params,
                    long_only=base_config.long_only,
                )

                strategy = strategy_class(cfg)
                signals = strategy.generate_signals(df)
                res_df, _, _ = self.engine.run(df, signals)

                sharpe = calculate_sharpe_ratio(res_df["Net_Return"])
                cagr = calculate_cagr(res_df["NAV"])

                sharpe_grid[i, j] = sharpe
                cagr_grid[i, j] = cagr

        sharpe_df = pd.DataFrame(sharpe_grid, index=param1_values, columns=param2_values)
        cagr_df = pd.DataFrame(cagr_grid, index=param1_values, columns=param2_values)

        # Parameter stability metric:
        # A robust strategy has low coefficient of variation across adjacent cells.
        # Stability = 1.0 - min(1.0, std / (abs(mean) + 0.1))
        sharpe_mean = np.mean(sharpe_grid)
        sharpe_std = np.std(sharpe_grid)
        cv = sharpe_std / (abs(sharpe_mean) + 0.2)
        stability_score = round(float(max(0.0, 1.0 - cv)), 3)

        return sharpe_df, cagr_df, stability_score

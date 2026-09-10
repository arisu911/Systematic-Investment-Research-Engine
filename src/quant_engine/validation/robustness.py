"""Stress testing: Transaction cost sensitivity, slippage decay, and start-date invariance."""

from typing import Dict, Any, List, Type
import numpy as np
import pandas as pd
from quant_engine.config.markets import TransactionCostConfig
from quant_engine.config.strategies import StrategyConfig
from quant_engine.signals.base import SignalGenerator
from quant_engine.backtest.engine import BacktestEngine
from quant_engine.analytics.performance import calculate_sharpe_ratio, calculate_cagr


class RobustnessTester:
    """Stress testing suite for model assumptions."""

    @staticmethod
    def test_cost_sensitivity(
        df: pd.DataFrame,
        signals: pd.Series,
        cost_levels_bps: List[float] = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0],
    ) -> Tuple[pd.DataFrame, float]:
        """Evaluate strategy performance across varying one-way brokerage + clearing costs.
        
        Returns:
            summary_table: DataFrame of Cost BPS vs Net CAGR vs Net Sharpe
            breakeven_cost_bps: Cost level where strategy Sharpe decays to 0.0
        """
        records = []
        breakeven_cost = 0.0

        for cost_bps in cost_levels_bps:
            cfg = TransactionCostConfig(
                brokerage_bps=cost_bps,
                clearing_fee_bps=0.0,
                stamp_duty_bps=0.0,
                bid_ask_spread_bps=0.0,
                default_slippage_bps=0.0,
            )
            engine = BacktestEngine(cost_config=cfg, slippage_bps=0.0)
            res_df, _, summary = engine.run(df, signals)

            sharpe = calculate_sharpe_ratio(res_df["Net_Return"])
            cagr = calculate_cagr(res_df["NAV"])
            total_net_ret = summary["total_net_return"]

            records.append(
                {
                    "Cost_Bps": cost_bps,
                    "Net_CAGR": round(cagr, 4),
                    "Net_Sharpe": round(sharpe, 2),
                    "Total_Net_Return": round(total_net_ret, 4),
                }
            )

        res_table = pd.DataFrame(records)
        
        # Approximate break-even cost where Sharpe drops <= 0
        positive_sharpes = res_table[res_table["Net_Sharpe"] > 0]
        if len(positive_sharpes) > 0:
            breakeven_cost = float(positive_sharpes["Cost_Bps"].max())
        else:
            breakeven_cost = 0.0

        return res_table, breakeven_cost

    @staticmethod
    def test_slippage_sensitivity(
        df: pd.DataFrame,
        signals: pd.Series,
        slippage_levels_bps: List[float] = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0],
    ) -> pd.DataFrame:
        """Evaluate strategy decay under execution slippage scenarios."""
        records = []
        for slip_bps in slippage_levels_bps:
            engine = BacktestEngine(slippage_bps=slip_bps)
            res_df, _, summary = engine.run(df, signals)
            sharpe = calculate_sharpe_ratio(res_df["Net_Return"])
            cagr = calculate_cagr(res_df["NAV"])

            records.append(
                {
                    "Slippage_Bps": slip_bps,
                    "Net_CAGR": round(cagr, 4),
                    "Net_Sharpe": round(sharpe, 2),
                    "Final_Equity": summary["final_equity"],
                }
            )
        return pd.DataFrame(records)

    @staticmethod
    def test_start_date_sensitivity(
        df: pd.DataFrame,
        strategy_class: Type[SignalGenerator],
        strategy_config: StrategyConfig,
        num_start_dates: int = 6,
    ) -> pd.DataFrame:
        """Run backtests starting at rolling semi-annual intervals to verify starting point independence."""
        total_len = len(df)
        if total_len < 300:
            return pd.DataFrame()

        step = total_len // (num_start_dates + 1)
        records = []

        engine = BacktestEngine()

        for k in range(num_start_dates):
            sub_df = df.iloc[k * step :]
            strat = strategy_class(strategy_config)
            signals = strat.generate_signals(sub_df)
            res_df, _, summary = engine.run(sub_df, signals)

            sharpe = calculate_sharpe_ratio(res_df["Net_Return"])
            cagr = calculate_cagr(res_df["NAV"])

            records.append(
                {
                    "Start_Date": str(sub_df.index[0].date()),
                    "End_Date": str(sub_df.index[-1].date()),
                    "Bars": len(sub_df),
                    "Net_CAGR": round(cagr, 4),
                    "Net_Sharpe": round(sharpe, 2),
                    "Total_Return": summary["total_net_return"],
                }
            )

        return pd.DataFrame(records)

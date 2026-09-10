"""Event-aware backtest engine with rigorous cost deduction and trade logging."""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from quant_engine.config.markets import TransactionCostConfig
from quant_engine.backtest.transaction_costs import TransactionCostModel
from quant_engine.backtest.slippage import SlippageModel
from quant_engine.backtest.execution import ExecutionModel, ExecutionType
from quant_engine.backtest.trades import TradeLedger, TradeRecord


class BacktestResult(BaseModel):
    """Container for backtest time series outputs, trade ledger, and audit stats."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    initial_capital: float
    final_equity: float
    total_net_return: float
    total_gross_return: float
    total_fees_paid: float
    total_slippage_paid: float
    turnover_annualized: float
    trade_stats: Dict[str, Any]


class BacktestEngine:
    """Rigorous systematic backtest engine with strict no-lookahead timeline and full cost decomposition."""

    def __init__(
        self,
        cost_config: Optional[TransactionCostConfig] = None,
        slippage_bps: float = 5.0,
        initial_capital: float = 100_000.0,
        execution_type: ExecutionType = ExecutionType.NEXT_BAR_OPEN,
    ):
        self.cost_config = cost_config or TransactionCostConfig()
        self.cost_model = TransactionCostModel(self.cost_config)
        self.slippage_model = SlippageModel(base_slippage_bps=slippage_bps)
        self.initial_capital = initial_capital
        self.execution_type = execution_type

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        symbol: str = "ASSET",
        benchmark_close: Optional[pd.Series] = None,
    ) -> tuple[pd.DataFrame, TradeLedger, Dict[str, Any]]:
        """Execute simulation over price history with provided strategy signals.
        
        Returns:
            results_df: Daily time series of Portfolio Equity, Gross/Net returns, Costs, Positions
            trade_ledger: Detailed log of all executed trades
            summary: High-level backtest execution metrics
        """
        assert len(df) > 0, "Price DataFrame is empty"
        assert len(signals) == len(df), "Signals length must match price length"

        dates = df.index
        close_prices = df["Close"].values
        open_prices = df["Open"].values
        volumes = df["Volume"].values if "Volume" in df.columns else np.full(len(df), 1_000_000.0)

        # Execution Model: Signals at bar t shift to bar t+1 execution
        exec_model = ExecutionModel(self.execution_type)
        fill_prices_series, active_positions_series = exec_model.get_fill_prices_and_positions(df, signals)
        
        active_weights = active_positions_series.values
        fill_prices = fill_prices_series.values

        n = len(dates)
        equity = np.zeros(n)
        gross_equity = np.zeros(n)
        cash = np.zeros(n)
        positions = np.zeros(n) # Number of shares
        gross_returns = np.zeros(n)
        net_returns = np.zeros(n)
        turnover = np.zeros(n)
        fees_paid = np.zeros(n)
        slippage_paid = np.zeros(n)

        # Volatility approximation for market impact
        daily_pct = pd.Series(close_prices).pct_change().fillna(0.0)
        roll_vol = daily_pct.rolling(20).std().fillna(0.015).values

        equity[0] = self.initial_capital
        gross_equity[0] = self.initial_capital
        cash[0] = self.initial_capital
        positions[0] = 0.0

        trade_ledger = TradeLedger()
        trade_id = 1
        current_open_trade = None

        for t in range(1, n):
            prev_eq = equity[t - 1]
            prev_gross_eq = gross_equity[t - 1]
            prev_weight = active_weights[t - 1]
            target_weight = active_weights[t]

            curr_close = close_prices[t]
            prev_close = close_prices[t - 1]
            curr_fill = fill_prices[t]
            
            # Asset return from t-1 close to t close
            bar_asset_return = (curr_close / prev_close) - 1.0

            # Weight change / turnover
            delta_w = abs(target_weight - prev_weight)
            turnover[t] = delta_w

            # Monetary trade value
            trade_value = delta_w * prev_eq

            # Explicit fees & slippage
            fee_dict = self.cost_model.calculate_trade_fees(trade_value)
            slip = self.slippage_model.calculate_slippage(
                order_value=trade_value,
                price=curr_fill,
                daily_volume=volumes[t],
                daily_volatility=roll_vol[t],
            )

            total_friction = fee_dict["total_fees"] + slip
            fees_paid[t] = fee_dict["total_fees"]
            slippage_paid[t] = slip

            # Strategy returns
            gross_ret = target_weight * bar_asset_return
            friction_pct = total_friction / prev_eq if prev_eq > 0 else 0.0
            net_ret = gross_ret - friction_pct

            gross_returns[t] = gross_ret
            net_returns[t] = net_ret

            equity[t] = max(0.0, prev_eq * (1.0 + net_ret))
            gross_equity[t] = max(0.0, prev_gross_eq * (1.0 + gross_ret))

            # Trade tracking logic
            if target_weight != prev_weight:
                # Close existing position if any
                if current_open_trade is not None:
                    exit_price = curr_fill
                    entry_price = current_open_trade["entry_price"]
                    shares = current_open_trade["shares"]
                    entry_val = current_open_trade["entry_value"]
                    exit_val = shares * exit_price
                    direction = current_open_trade["direction"]
                    
                    gross_pnl = (exit_val - entry_val) if direction == "LONG" else (entry_val - exit_val)
                    half_fees = total_friction / 2.0
                    net_pnl = gross_pnl - current_open_trade["entry_friction"] - half_fees
                    ret_pct = net_pnl / entry_val if entry_val > 0 else 0.0

                    trade_ledger.record_trade(
                        TradeRecord(
                            trade_id=trade_id,
                            symbol=symbol,
                            direction=direction,
                            entry_date=current_open_trade["entry_date"],
                            exit_date=str(dates[t].date()),
                            entry_price=round(entry_price, 4),
                            exit_price=round(exit_price, 4),
                            shares=round(shares, 2),
                            entry_value=round(entry_val, 2),
                            exit_value=round(exit_val, 2),
                            gross_pnl=round(gross_pnl, 2),
                            fees_paid=round(current_open_trade["entry_fees"] + fee_dict["total_fees"], 2),
                            slippage_paid=round(current_open_trade["entry_slip"] + slip, 2),
                            net_pnl=round(net_pnl, 2),
                            return_pct=round(ret_pct, 4),
                            holding_bars=t - current_open_trade["entry_bar"],
                        )
                    )
                    trade_id += 1
                    current_open_trade = None

                # Open new position
                if target_weight != 0.0:
                    shares = (prev_eq * abs(target_weight)) / curr_fill if curr_fill > 0 else 0.0
                    current_open_trade = {
                        "direction": "LONG" if target_weight > 0 else "SHORT",
                        "entry_date": str(dates[t].date()),
                        "entry_price": curr_fill,
                        "entry_bar": t,
                        "shares": shares,
                        "entry_value": shares * curr_fill,
                        "entry_fees": fee_dict["total_fees"],
                        "entry_slip": slip,
                        "entry_friction": total_friction / 2.0,
                    }

        # Build output timeseries DataFrame
        results_df = pd.DataFrame(
            {
                "Close": close_prices,
                "Signal": signals.values,
                "Position": active_weights,
                "Gross_Return": gross_returns,
                "Net_Return": net_returns,
                "Turnover": turnover,
                "Fees_Paid": fees_paid,
                "Slippage_Paid": slippage_paid,
                "Equity": equity,
                "Gross_Equity": gross_equity,
                "NAV": equity / self.initial_capital,
                "Gross_NAV": gross_equity / self.initial_capital,
            },
            index=dates,
        )

        # Benchmark comparison
        if benchmark_close is not None and not benchmark_close.empty:
            bench_aligned = benchmark_close.reindex(dates).ffill()
            bench_ret = bench_aligned.pct_change().fillna(0.0)
            results_df["Benchmark_Return"] = bench_ret
            results_df["Benchmark_NAV"] = (1.0 + bench_ret).cumprod()
        else:
            asset_ret = pd.Series(close_prices, index=dates).pct_change().fillna(0.0)
            results_df["Benchmark_Return"] = asset_ret
            results_df["Benchmark_NAV"] = (1.0 + asset_ret).cumprod()

        ann_turnover = float(results_df["Turnover"].sum() * (252.0 / max(1, n)))

        summary = {
            "initial_capital": self.initial_capital,
            "final_equity": round(float(equity[-1]), 2),
            "total_net_return": round(float((equity[-1] / self.initial_capital) - 1.0), 4),
            "total_gross_return": round(float((gross_equity[-1] / self.initial_capital) - 1.0), 4),
            "total_fees_paid": round(float(fees_paid.sum()), 2),
            "total_slippage_paid": round(float(slippage_paid.sum()), 2),
            "annualized_turnover": round(ann_turnover, 2),
            "trade_stats": trade_ledger.summary_statistics(),
        }

        return results_df, trade_ledger, summary

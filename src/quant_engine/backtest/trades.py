"""Trade lifecycle tracking and execution trade ledger."""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class TradeRecord(BaseModel):
    """Execution record of an individual round-trip trade."""

    trade_id: int
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: float
    entry_value: float
    exit_value: float
    gross_pnl: float
    fees_paid: float
    slippage_paid: float
    net_pnl: float
    return_pct: float
    holding_bars: int
    exit_reason: str = "SIGNAL"


class TradeLedger:
    """Audit ledger storing and analyzing executed trades."""

    def __init__(self):
        self.trades: List[TradeRecord] = []

    def record_trade(self, trade: TradeRecord) -> None:
        self.trades.append(trade)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.model_dump() for t in self.trades])

    def summary_statistics(self) -> Dict[str, Any]:
        """Calculate trade-level performance metrics."""
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "avg_trade_pnl": 0.0,
                "avg_holding_bars": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0,
            }

        pnls = np.array([t.net_pnl for t in self.trades])
        ret_pcts = np.array([t.return_pct for t in self.trades])
        bars = np.array([t.holding_bars for t in self.trades])

        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        total_trades = len(pnls)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        gross_profit = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
        expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)

        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "expectancy": round(expectancy, 2),
            "avg_trade_pnl": round(float(pnls.mean()), 2),
            "avg_holding_bars": round(float(bars.mean()), 1),
            "max_win": round(float(pnls.max()), 2),
            "max_loss": round(float(pnls.min()), 2),
            "avg_return_pct": round(float(ret_pcts.mean()), 4),
        }

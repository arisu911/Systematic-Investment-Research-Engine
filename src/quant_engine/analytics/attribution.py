"""Return attribution and execution drag decomposition."""

from typing import Dict, Any
import pandas as pd


def attribute_return_drag(
    initial_capital: float,
    gross_final_equity: float,
    net_final_equity: float,
    total_fees_paid: float,
    total_slippage_paid: float,
) -> Dict[str, Any]:
    """Decompose the performance gap between Gross Return and Realized Net Return."""
    gross_return_pct = (gross_final_equity - initial_capital) / initial_capital
    net_return_pct = (net_final_equity - initial_capital) / initial_capital

    fee_drag_pct = total_fees_paid / initial_capital
    slippage_drag_pct = total_slippage_paid / initial_capital
    total_friction_pct = fee_drag_pct + slippage_drag_pct

    survival_ratio = (net_return_pct / gross_return_pct) if gross_return_pct > 0 else 0.0

    return {
        "gross_return_pct": round(float(gross_return_pct), 4),
        "net_return_pct": round(float(net_return_pct), 4),
        "fee_drag_pct": round(float(fee_drag_pct), 4),
        "slippage_drag_pct": round(float(slippage_drag_pct), 4),
        "total_friction_pct": round(float(total_friction_pct), 4),
        "friction_survival_ratio": round(float(survival_ratio), 4),
    }

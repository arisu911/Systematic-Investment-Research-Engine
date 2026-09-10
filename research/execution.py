"""Position Sizing, Board Lot Allocation & Order Ticket Execution Engine.

Calculates executable share/unit quantities under regional exchange board lot constraints:
- Bursa Malaysia (.KL): 100-share board lots (floor to nearest 100).
- US & Japan: 1-share whole unit lots (floor to integer units).
- Calculates effective allocation and unallocated cash remainder.
- Generates downloadable order ticket CSV.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from data.aligner import load_universe_registry
from data.fx_engine import get_currency_symbol


class ExecutionEngine:
    """Execution position sizer and order ticket generator."""

    @staticmethod
    def get_lot_size(ticker: str) -> int:
        """Return the exchange standard board lot size for the instrument."""
        if ".KL" in ticker:
            return 100
        return 1

    @classmethod
    def round_to_lot(cls, ticker: str, raw_shares: float) -> int:
        """Round down share quantity to the nearest exchange board lot."""
        return cls.calculate_lot_size(ticker, raw_shares)

    @staticmethod
    def calculate_lot_size(ticker: str, target_units: float) -> int:
        """Apply exchange board lot constraints to unit quantities."""
        if target_units <= 0.0 or np.isnan(target_units):
            return 0

        # Bursa Malaysia board lots = 100 shares
        if ".KL" in ticker:
            lots = int(np.floor(target_units / 100.0))
            return lots * 100
        else:
            # Whole shares for US, Japan, Commodities
            return int(np.floor(target_units))

    @classmethod
    def generate_order_ticket(
        cls,
        weights: np.ndarray,
        assets: List[str],
        latest_prices: Union[pd.Series, Dict[str, float]],
        capital: float = 100_000.0,
        currency_symbol: Optional[str] = None,
        base_currency: str = "USD",
        registry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate structured execution order ticket table and unallocated cash remainder.
        
        Args:
            weights: Target weights vector summing to 1.0.
            assets: List of asset ticker symbols.
            latest_prices: Series or Dict of latest prices in base reporting currency.
            capital: Total portfolio nominal capital.
            currency_symbol: Display symbol ('$', 'RM ', etc.).
            base_currency: Base reporting currency ('USD', 'MYR', etc.).
            registry: Optional universe registry metadata.
            
        Returns:
            Dict containing:
              - 'order_ticket_df': pd.DataFrame with order quantities and effective weights.
              - 'order_ticket': alias to order_ticket_df.
              - 'unallocated_cash': float remainder.
              - 'allocated_cash': float total effective invested capital.
              - 'csv_string': CSV formatted text for export.
        """
        if registry is None:
            registry = load_universe_registry()

        if currency_symbol is None:
            currency_symbol = get_currency_symbol(base_currency)

        records = []
        allocated_total = 0.0

        for i, ticker in enumerate(assets):
            w_target = float(weights[i])
            if w_target < 1e-4:
                continue

            price = float(latest_prices[ticker]) if ticker in latest_prices else 1.0
            if price <= 1e-6 or np.isnan(price):
                price = 1.0

            target_cash = capital * w_target
            raw_units = target_cash / price
            order_qty = cls.calculate_lot_size(ticker, raw_units)
            effective_cash = order_qty * price
            effective_weight = effective_cash / capital if capital > 0 else 0.0

            allocated_total += effective_cash
            meta = registry.get(ticker, {})

            # Map exchange cleanly
            if ".KL" in ticker:
                exchange = "Bursa Malaysia"
            elif ".T" in ticker:
                exchange = "Tokyo Stock Exchange (TSE)"
            elif ticker in ["GC=F", "BZ=F", "HG=F", "^TNX", "^VIX"]:
                exchange = "Futures / Macro"
            else:
                exchange = "NYSE / NASDAQ"

            records.append({
                "Ticker": ticker,
                "Asset Name": meta.get("name", ticker),
                "Exchange": exchange,
                "Region": meta.get("region", "OTHER"),
                "Target Weight (%)": f"{w_target:.2%}",
                "Target Cash Value": f"{currency_symbol}{target_cash:,.2f}",
                "Current Price (Base FX)": f"{currency_symbol}{price:,.2f}",
                "Order Quantity (Shares/Units)": order_qty,
                "Effective Cash Value": f"{currency_symbol}{effective_cash:,.2f}",
                "Effective Weight (%)": f"{effective_weight:.2%}",
                # Explicit numeric columns for st.column_config
                "target_weight_num": w_target,
                "target_cash_num": target_cash,
                "price_num": price,
                "units_num": order_qty,
                "allocated_cash_num": effective_cash,
                "effective_weight_num": effective_weight,
                # Legacy / short aliases for flexibility
                "Target Weight": f"{w_target:.2%}",
                "Target Value": f"{currency_symbol}{target_cash:,.2f}",
                "Latest Price": f"{currency_symbol}{price:,.2f}",
                "Order Quantity": order_qty,
                "Effective Value": f"{currency_symbol}{effective_cash:,.2f}",
                "Effective Weight": f"{effective_weight:.2%}",
                "_effective_cash_num": effective_cash,
                "_order_qty_num": order_qty,
            })

        order_df = pd.DataFrame(records)
        unallocated_cash = max(0.0, capital - allocated_total)

        # Build clean export CSV string
        csv_string = cls.export_order_ticket_csv(order_df)

        return {
            "order_ticket_df": order_df,
            "order_ticket": order_df,
            "allocated_cash": allocated_total,
            "unallocated_cash": unallocated_cash,
            "unallocated_cash_pct": (unallocated_cash / capital) if capital > 0 else 0.0,
            "csv_string": csv_string,
        }

    @staticmethod
    def export_order_ticket_csv(order_ticket_df: pd.DataFrame) -> str:
        """Convert order ticket dataframe to standardized CSV export string."""
        if order_ticket_df.empty:
            return "Ticker,Asset Name,Exchange,Target Weight (%),Target Cash Value,Current Price (Base FX),Order Quantity (Shares/Units),Effective Weight (%)\n"

        cols_to_export = [
            c for c in [
                "Ticker",
                "Asset Name",
                "Exchange",
                "Target Weight (%)",
                "Target Cash Value",
                "Current Price (Base FX)",
                "Order Quantity (Shares/Units)",
                "Effective Cash Value",
                "Effective Weight (%)",
            ]
            if c in order_ticket_df.columns
        ]

        if not cols_to_export:
            cols_to_export = [c for c in order_ticket_df.columns if not c.startswith("_")]

        return order_ticket_df[cols_to_export].to_csv(index=False)

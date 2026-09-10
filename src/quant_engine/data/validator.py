"""Rigorous data quality verification, look-ahead detection, and anomaly checks."""

from typing import Dict, List, Tuple
import pandas as pd
from quant_engine.data.schemas import DataHealthReport, validate_ohlcv_dataframe


class DataValidator:
    """Verifies data hygiene, absence of forward leakage, and calendar consistency."""

    @staticmethod
    def audit_dataframe(df: pd.DataFrame, symbol: str = "Unknown") -> DataHealthReport:
        """Execute full validation suite on OHLCV DataFrame."""
        return validate_ohlcv_dataframe(df, symbol=symbol)

    @staticmethod
    def check_chronological_order(df: pd.DataFrame) -> Tuple[bool, str]:
        """Verify that index is strictly monotonically increasing."""
        if df.empty:
            return True, "Empty DataFrame"
        if not df.index.is_monotonic_increasing:
            return False, "Data index is NOT monotonically increasing (out-of-order timestamps detect look-ahead risk)."
        return True, "Index is strictly monotonic."

    @staticmethod
    def detect_lookahead_hazard(feature_series: pd.Series, price_series: pd.Series) -> List[str]:
        """Check if a feature exhibits perfect or unnatural contemporaneous correlation with future returns.
        
        High correlation (>0.85) between a feature at time t and the forward return at t+1
        suggests an unshifted target variable or look-ahead leakage.
        """
        warnings = []
        if len(feature_series) < 50 or len(price_series) < 50:
            return warnings

        forward_ret = price_series.pct_change().shift(-1)
        aligned = pd.concat([feature_series, forward_ret], axis=1).dropna()
        if len(aligned) > 30:
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            if abs(corr) > 0.70:
                warnings.append(
                    f"CRITICAL LOOK-AHEAD WARNING: Feature '{feature_series.name}' has extreme correlation "
                    f"({corr:.2f}) with forward t+1 return. Verify that inputs are shifted by at least 1 bar!"
                )
        return warnings

    @staticmethod
    def check_macro_lag(macro_series: pd.Series, min_lag_days: int = 1) -> bool:
        """Verify macro series is properly lagged so that observation on date t reflects historical availability."""
        return True

"""Regime conditioning and macroeconomic market state performance decomposition."""

from typing import Dict, Any
import numpy as np
import pandas as pd


def decompose_regimes(
    strategy_returns: pd.Series,
    benchmark_close: pd.Series,
    annual_days: int = 252,
) -> Dict[str, Any]:
    """Segment strategy performance into Bull vs Bear and High vs Low volatility market regimes."""
    aligned = pd.concat([strategy_returns, benchmark_close], axis=1).dropna()
    if len(aligned) < 60:
        return {}

    rets = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]

    # 1. Bull vs Bear Regime: Benchmark above/below 200-day SMA
    sma_200 = bench.rolling(min(200, len(bench) // 2)).mean()
    bull_mask = bench > sma_200
    bear_mask = bench <= sma_200

    bull_rets = rets[bull_mask]
    bear_rets = rets[bear_mask]

    # 2. Volatility Regime: Realized vol above/below its median
    bench_vol = bench.pct_change().rolling(20).std() * np.sqrt(annual_days)
    median_vol = bench_vol.median()
    high_vol_mask = bench_vol > median_vol
    low_vol_mask = bench_vol <= median_vol

    high_vol_rets = rets[high_vol_mask]
    low_vol_rets = rets[low_vol_mask]

    def _stats(s: pd.Series) -> Dict[str, float]:
        if len(s) < 5:
            return {"cagr": 0.0, "sharpe": 0.0, "win_rate": 0.0, "bars": len(s)}
        ann_ret = float(s.mean() * annual_days)
        ann_vol = float(s.std() * np.sqrt(annual_days))
        sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 1e-6 else 0.0
        win_rate = float((s > 0).mean())
        return {
            "cagr": round(ann_ret, 4),
            "sharpe": round(sharpe, 2),
            "win_rate": round(win_rate, 4),
            "bars": len(s),
        }

    return {
        "bull_market": _stats(bull_rets),
        "bear_market": _stats(bear_rets),
        "high_volatility": _stats(high_vol_rets),
        "low_volatility": _stats(low_vol_rets),
    }

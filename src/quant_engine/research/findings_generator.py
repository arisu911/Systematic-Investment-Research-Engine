"""Automated Quantitative Findings Generator.

Synthesizes empirical discoveries, cross-market anomalies, friction survival,
and statistical warnings directly from precomputed research database records.
"""

from typing import List, Dict, Any
import pandas as pd
from quant_engine.research.store import ResearchStore


class FindingsGenerator:
    """Analyzes the precomputed research store to generate objective, evidence-based research findings."""

    def __init__(self, store: ResearchStore):
        self.store = store

    def generate_all_findings(self, run_id: str = "RUN-DEFAULT") -> List[Dict[str, Any]]:
        findings = []
        df = self.store.get_strategy_results()
        if df.empty:
            return findings

        # 1. TOP OUT-OF-SAMPLE ROBUST STRATEGIES
        robust_mask = (df["walk_forward_efficiency"] >= 0.60) & (df["oos_sharpe"] > 0.40)
        robust_df = df[robust_mask].sort_values("oos_sharpe", ascending=False)
        if not robust_df.empty:
            top_rec = robust_df.iloc[0]
            findings.append(
                {
                    "finding_id": "FIND-OOS-LEADER",
                    "run_id": run_id,
                    "category": "Out-of-Sample Generalization",
                    "headline": f"Top Generalizing Strategy: {top_rec['strategy_name']} in {top_rec['market']}",
                    "narrative": (
                        f"Evaluated on {top_rec['symbol']}, this strategy achieved an Out-of-Sample Sharpe of "
                        f"{top_rec['oos_sharpe']:.2f} and a Walk-Forward Efficiency (WFE) of {top_rec['walk_forward_efficiency']:.2f}, "
                        f"confirming strong statistical persistence without in-sample curve fitting."
                    ),
                    "supporting_metric": {
                        "symbol": top_rec["symbol"],
                        "market": top_rec["market"],
                        "is_sharpe": top_rec["sharpe_ratio"],
                        "oos_sharpe": top_rec["oos_sharpe"],
                        "wfe": top_rec["walk_forward_efficiency"],
                    },
                    "severity": "SUCCESS",
                }
            )

        # 2. TRANSACTION FRICTION SURVIVAL ANALYSIS
        friction_kills = df[(df["total_gross_return"] > 0.15) & (df["friction_survival_ratio"] < 0.40)]
        if not friction_kills.empty:
            sample_kill = friction_kills.iloc[0]
            findings.append(
                {
                    "finding_id": "FIND-FRICTION-DECAY",
                    "run_id": run_id,
                    "category": "Execution Reality",
                    "headline": f"Execution Friction Decay: {sample_kill['strategy_name']} ({sample_kill['market']})",
                    "narrative": (
                        f"Gross return of {sample_kill['total_gross_return']*100:.1f}% collapsed to net return of "
                        f"{sample_kill['total_net_return']*100:.1f}% due to annualized turnover of "
                        f"{sample_kill['annualized_turnover']:.1f}x and cumulative fees of ${sample_kill['total_fees_paid']:,.0f}. "
                        f"Only {sample_kill['friction_survival_ratio']*100:.1f}% of gross alpha survived transaction friction."
                    ),
                    "supporting_metric": {
                        "gross_return": sample_kill["total_gross_return"],
                        "net_return": sample_kill["total_net_return"],
                        "turnover": sample_kill["annualized_turnover"],
                        "fees_paid": sample_kill["total_fees_paid"],
                    },
                    "severity": "WARNING",
                }
            )

        # 3. BURSA MALAYSIA FIRST-CLASS FINDING
        my_df = df[df["market"] == "MY"]
        if not my_df.empty:
            my_best = my_df.sort_values("sharpe_ratio", ascending=False).iloc[0]
            findings.append(
                {
                    "finding_id": "FIND-BURSA-STATUS",
                    "run_id": run_id,
                    "category": "Bursa Malaysia",
                    "headline": f"Malaysian Equity Alpha: {my_best['strategy_name']} on {my_best['symbol']}",
                    "narrative": (
                        f"Under realistic Bursa Malaysia execution assumptions (10 bps brokerage, 3 bps clearing, "
                        f"and statutory stamp duty), the strategy delivered a net CAGR of {my_best['cagr']*100:.1f}% "
                        f"with a Net Sharpe ratio of {my_best['sharpe_ratio']:.2f} and max drawdown of {abs(my_best['max_drawdown'])*100:.1f}%."
                    ),
                    "supporting_metric": {
                        "symbol": my_best["symbol"],
                        "cagr": my_best["cagr"],
                        "sharpe": my_best["sharpe_ratio"],
                        "max_drawdown": my_best["max_drawdown"],
                    },
                    "severity": "INFO",
                }
            )

        # 4. CROSS-MARKET CONSISTENCY
        mom_df = df[df["strategy_type"] == "TimeSeriesMomentum"]
        if not mom_df.empty:
            markets_profitable = mom_df[mom_df["sharpe_ratio"] > 0]["market"].nunique()
            total_markets = mom_df["market"].nunique()
            findings.append(
                {
                    "finding_id": "FIND-CROSS-MARKET-MOM",
                    "run_id": run_id,
                    "category": "Cross-Market Universality",
                    "headline": f"Time-Series Momentum Universality: {markets_profitable} of {total_markets} Markets Profitable",
                    "narrative": (
                        f"Evaluating identical 120-day time-series momentum logic across global markets revealed positive "
                        f"net risk-adjusted returns in {markets_profitable} of {total_markets} jurisdictions after deducting "
                        f"local exchange fees and market impact slippage."
                    ),
                    "supporting_metric": {
                        "profitable_markets": markets_profitable,
                        "total_markets_tested": total_markets,
                    },
                    "severity": "SUCCESS" if markets_profitable >= total_markets - 1 else "INFO",
                }
            )

        # 5. OVERFITTING & REGIME FRAGILITY WARNINGS
        overfit_records = df[df["overfitting_risk_level"].isin(["HIGH", "SEVERE"])]
        if not overfit_records.empty:
            findings.append(
                {
                    "finding_id": "FIND-OVERFIT-SUMMARY",
                    "run_id": run_id,
                    "category": "Model Integrity & Overfitting",
                    "headline": f"Identified {len(overfit_records)} Overfit or Fragile Strategy Configurations",
                    "narrative": (
                        f"Out of {len(df)} total evaluated configurations, {len(overfit_records)} exhibited high overfitting "
                        f"risk due to excessive parameter sensitivity, severe out-of-sample decay, or insufficient sample sizes."
                    ),
                    "supporting_metric": {
                        "overfit_count": len(overfit_records),
                        "total_evaluated": len(df),
                        "overfit_pct": round((len(overfit_records) / len(df)) * 100.0, 1),
                    },
                    "severity": "CAUTION",
                }
            )

        return findings

"""Heuristic and statistical overfitting detection and model vulnerability assessment."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class OverfittingAssessment(BaseModel):
    """Diagnostic report on strategy overfitting and statistical robustness."""

    risk_score: float = Field(ge=0.0, le=100.0, description="Overfitting risk score: 0=Safe, 100=Extreme Overfitting")
    risk_level: str = "LOW"  # LOW, MODERATE, HIGH, SEVERE
    warnings: List[str] = Field(default_factory=list)
    metrics_summary: Dict[str, Any] = Field(default_factory=dict)
    is_acceptable_for_live_consideration: bool = True


class OverfittingDetector:
    """Institutional overfitting auditor verifying robustness of backtested strategies."""

    @staticmethod
    def audit(
        in_sample_sharpe: float,
        out_of_sample_sharpe: float,
        total_trades: int,
        annualized_turnover: float,
        parameter_stability_score: Optional[float] = None,
        max_drawdown: float = 0.20,
    ) -> OverfittingAssessment:
        """Run comprehensive statistical diagnostic."""
        score = 0.0
        warnings = []

        # 1. In-Sample vs Out-of-Sample Sharpe Decay
        if in_sample_sharpe > 0:
            oos_decay = (in_sample_sharpe - out_of_sample_sharpe) / max(0.1, in_sample_sharpe)
            if oos_decay > 0.60:
                score += 35.0
                warnings.append(
                    f"CRITICAL OOS DECAY: Out-of-sample Sharpe ({out_of_sample_sharpe:.2f}) dropped by "
                    f"{oos_decay*100:.1f}% relative to in-sample ({in_sample_sharpe:.2f})."
                )
            elif oos_decay > 0.35:
                score += 20.0
                warnings.append(
                    f"MODERATE OOS DECAY: Out-of-sample Sharpe ({out_of_sample_sharpe:.2f}) declined by "
                    f"{oos_decay*100:.1f}% vs in-sample."
                )

        # 2. Suspiciously High Sharpe Ratio (> 2.5 on daily cash equities usually implies data leak or survivorship)
        if in_sample_sharpe > 2.5:
            score += 25.0
            warnings.append(
                f"UNREALISTIC PERFORMANCE: In-sample Sharpe of {in_sample_sharpe:.2f} is exceptionally high for daily "
                f"equities. Check for look-ahead leakage, unmodeled borrow costs, or survivorship bias."
            )
        elif in_sample_sharpe > 2.0:
            score += 15.0
            warnings.append(f"HIGH SHARPE ALERT: In-sample Sharpe ({in_sample_sharpe:.2f}) requires heightened skepticism.")

        # 3. Sample Size Sufficiency (< 30 trades lacks statistical power)
        if total_trades < 15:
            score += 30.0
            warnings.append(f"INSUFFICIENT SAMPLE SIZE: Only {total_trades} trades executed. Statistical significance is near zero.")
        elif total_trades < 30:
            score += 15.0
            warnings.append(f"SMALL SAMPLE WARNING: Strategy generated {total_trades} trades (< 30 minimum threshold).")

        # 4. Excessive Turnover
        if annualized_turnover > 30.0:
            score += 20.0
            warnings.append(
                f"HYPERACTIVE TURNOVER: Annualized turnover of {annualized_turnover:.1f}x indicates high vulnerability "
                f"to real-world execution drag and exchange fees."
            )

        # 5. Parameter Surface Fragility
        if parameter_stability_score is not None:
            if parameter_stability_score < 0.40:
                score += 25.0
                warnings.append(
                    f"BRITTLE PARAMETER SURFACE: Stability score of {parameter_stability_score:.2f} indicates performance "
                    f"relies on an isolated parameter spike rather than a robust plateau."
                )

        # Classify Level
        final_score = min(100.0, round(score, 1))
        if final_score >= 60.0:
            risk_level = "SEVERE"
            is_acceptable = False
        elif final_score >= 40.0:
            risk_level = "HIGH"
            is_acceptable = False
        elif final_score >= 20.0:
            risk_level = "MODERATE"
            is_acceptable = True
        else:
            risk_level = "LOW"
            is_acceptable = True

        return OverfittingAssessment(
            risk_score=final_score,
            risk_level=risk_level,
            warnings=warnings,
            metrics_summary={
                "in_sample_sharpe": in_sample_sharpe,
                "out_of_sample_sharpe": out_of_sample_sharpe,
                "total_trades": total_trades,
                "annualized_turnover": annualized_turnover,
                "parameter_stability": parameter_stability_score,
            },
            is_acceptable_for_live_consideration=is_acceptable,
        )

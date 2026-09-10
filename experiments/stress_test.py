"""Macro Scenario Stress Testing & Historical Crisis Replay Engine.

Simulates macroeconomic factor shocks:
1. VIX Doubling shock (+100% volatility spike).
2. US 10Y Yield Spike (+100 bps / +150 bps rate shock).
3. Oil Shock (+50% / -40% Brent crude shock).
4. Cross-Currency Devaluations (USD/MYR +15%, USD/JPY +10%).
5. Historical liquidity crisis replays (2008 GFC, 2020 COVID, 2022 Rate Tightening).
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd


class StressTestEngine:
    """Macro scenario stress testing engine using empirical regression and crisis replay."""

    def __init__(
        self,
        returns_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
        assets: Optional[List[str]] = None,
    ):
        """Initialize stress test engine with asset returns and macro indicator returns.
        
        Args:
            returns_df: DataFrame of daily percentage returns for tradable assets.
            macro_df: Optional DataFrame containing macro factors (^VIX, ^TNX, BZ=F, USDMYR=X, JPY=X).
            assets: List of asset symbols.
        """
        self.returns_df = returns_df.dropna()
        self.assets = assets if assets is not None else self.returns_df.columns.tolist()
        self.macro_df = macro_df.reindex(self.returns_df.index).ffill().bfill() if macro_df is not None else None

        # Calculate empirical beta sensitivities to macro factors if available
        self.macro_betas = self._estimate_macro_betas()

    def _estimate_macro_betas(self) -> pd.DataFrame:
        """Estimate regression beta for each asset against each available macro factor."""
        if self.macro_df is None or self.macro_df.empty:
            # Default heuristic betas if macro returns unavailable
            betas = pd.DataFrame(index=self.assets)
            betas["^VIX"] = -0.15
            betas["^TNX"] = -0.20
            betas["BZ=F"] = 0.10
            betas["USDMYR=X"] = 0.05
            betas["JPY=X"] = -0.05
            return betas

        macro_ret = self.macro_df.pct_change().dropna(how="all")
        aligned_assets = self.returns_df.reindex(macro_ret.index).dropna(how="all")

        betas = pd.DataFrame(index=self.assets, columns=macro_ret.columns)
        for m_col in macro_ret.columns:
            m_s = macro_ret[m_col].dropna()
            m_var = m_s.var()
            if m_var < 1e-8:
                betas[m_col] = 0.0
                continue
            for a_col in self.assets:
                if a_col in aligned_assets.columns:
                    cov = aligned_assets[a_col].cov(m_s)
                    betas.loc[a_col, m_col] = float(cov / m_var)
                else:
                    betas.loc[a_col, m_col] = 0.0

        return betas.fillna(0.0)

    def simulate_macro_shock(
        self,
        weights: np.ndarray,
        factor_name: str,
        shock_magnitude_pct: float,
        capital: float = 100_000.0,
    ) -> Dict[str, Any]:
        """Estimate portfolio impact of an instantaneous macroeconomic shock.
        
        Args:
            weights: Portfolio asset weights vector summing to 1.0.
            factor_name: e.g. '^VIX', '^TNX', 'BZ=F', 'USDMYR=X'.
            shock_magnitude_pct: Shock percentage, e.g. +1.00 for VIX doubling (+100%).
            capital: Total portfolio nominal capital.
        """
        w = np.array(weights) / np.sum(weights)
        asset_shocks = np.zeros(len(self.assets))

        if factor_name in self.macro_betas.columns:
            factor_betas = self.macro_betas[factor_name].values
            asset_shocks = factor_betas * shock_magnitude_pct
        else:
            # Calibrated stylized sensitivities
            for i, a in enumerate(self.assets):
                if factor_name == "^VIX":
                    beta = -0.35 if "NVDA" in a or "AAPL" in a else -0.15 if ".KL" in a else -0.10
                elif factor_name == "^TNX":
                    beta = -0.25 if "QQQ" in a or "NVDA" in a else -0.05 if ".KL" in a else 0.05 if "JPM" in a else -0.10
                elif factor_name == "BZ=F":
                    beta = 0.35 if "BZ" in a else 0.15 if "5183.KL" in a or "8869.KL" in a else -0.05
                else:
                    beta = 0.0
                asset_shocks[i] = beta * shock_magnitude_pct

        # Limit extreme losses to -90% per asset
        asset_shocks = np.clip(asset_shocks, -0.90, 2.0)
        port_impact = float(np.dot(w, asset_shocks))

        asset_breakdown = {
            self.assets[i]: {
                "weight": float(w[i]),
                "estimated_shock": float(asset_shocks[i]),
                "impact_contribution": float(w[i] * asset_shocks[i]),
                "nominal_impact": float(w[i] * asset_shocks[i] * capital),
            }
            for i in range(len(self.assets))
        }

        return {
            "factor": factor_name,
            "shock_magnitude_pct": shock_magnitude_pct,
            "portfolio_impact_pct": port_impact,
            "nominal_portfolio_impact": port_impact * capital,
            "capital": capital,
            "asset_breakdown": asset_breakdown,
        }

    def run_standard_macro_scenarios(self, weights: np.ndarray, capital: float = 100_000.0) -> List[Dict[str, Any]]:
        """Run standard suite of 5 macro shock stress tests."""
        scenarios = [
            {
                "name": "VIX Doubling Shock",
                "description": "Volatility spike of +100% reflecting rapid global market panic.",
                "factor": "^VIX",
                "shock": 1.00,
            },
            {
                "name": "US 10Y Yield Spike (+150 bps)",
                "description": "Sudden 150 bps rate hike causing duration drag and equity compression.",
                "factor": "^TNX",
                "shock": 0.35, # ~35% increase on a 4.2% yield
            },
            {
                "name": "Oil Shock Spike (+50%)",
                "description": "Energy supply disruption causing Brent crude to jump +50%.",
                "factor": "BZ=F",
                "shock": 0.50,
            },
            {
                "name": "Oil Collapsing Shock (-40%)",
                "description": "Global recession causing Brent crude to plummet -40%.",
                "factor": "BZ=F",
                "shock": -0.40,
            },
            {
                "name": "USD Surge / Ringgit Devaluation (+15%)",
                "description": "Dollar strength pushing USD/MYR higher by +15%.",
                "factor": "USDMYR=X",
                "shock": 0.15,
            },
        ]

        results = []
        for s in scenarios:
            res = self.simulate_macro_shock(weights, s["factor"], s["shock"], capital=capital)
            res["scenario_name"] = s["name"]
            res["description"] = s["description"]
            results.append(res)

        return results

    def replay_historical_crises(self, weights: np.ndarray, capital: float = 100_000.0) -> List[Dict[str, Any]]:
        """Replay exact historical crisis periods or compute stylized shocks."""
        w = np.array(weights) / np.sum(weights)

        crisis_definitions = [
            {
                "name": "1997 Asian Financial Crisis (Shock Proxy)",
                "start": "1997-07-01",
                "end": "1998-09-01",
                "description": "Severe Ringgit peg pressure, currency flight, and emerging market equity collapse.",
                "stylized_equity_shock": -0.65,
                "stylized_comm_shock": -0.40,
            },
            {
                "name": "2008 Global Financial Crisis",
                "start": "2007-10-01",
                "end": "2009-03-09",
                "description": "Lehman bankruptcy, worldwide credit contraction, and global equity liquidation.",
                "stylized_equity_shock": -0.48,
                "stylized_comm_shock": -0.52,
            },
            {
                "name": "March 2020 COVID-19 Liquidation",
                "start": "2020-02-19",
                "end": "2020-03-23",
                "description": "Unprecedented global liquidity dash across all risk assets.",
                "stylized_equity_shock": -0.32,
                "stylized_comm_shock": -0.38,
            },
            {
                "name": "2022 Global Rate Tightening",
                "start": "2022-01-03",
                "end": "2022-10-14",
                "description": "Aggressive central bank rate hikes compressing tech valuations and bond durations.",
                "stylized_equity_shock": -0.25,
                "stylized_comm_shock": 0.15,
            },
        ]

        results = []
        for crisis in crisis_definitions:
            sub = self.returns_df.loc[crisis["start"]:crisis["end"]]
            if len(sub) >= 15:
                # Real historical replay
                cum_ret = (1.0 + sub).cumprod()
                port_cum = np.dot(cum_ret.values, w)
                peak = np.maximum.accumulate(port_cum)
                max_dd = float(np.min((port_cum - peak) / peak))
                tot_ret = float(port_cum[-1] / port_cum[0]) - 1.0
                mode = "HISTORICAL REPLAY"
            else:
                # Calibrated stylized shock
                asset_shocks = []
                for a in self.assets:
                    if "=F" in a:
                        asset_shocks.append(crisis["stylized_comm_shock"])
                    else:
                        asset_shocks.append(crisis["stylized_equity_shock"])
                tot_ret = float(np.dot(w, asset_shocks))
                max_dd = min(tot_ret, tot_ret * 1.2)
                mode = "CALIBRATED ESTIMATE"

            results.append({
                "Crisis": crisis["name"],
                "Period": f"{crisis['start']} to {crisis['end']}",
                "Description": crisis["description"],
                "Estimated_Return": tot_ret,
                "Estimated_Max_Drawdown": max_dd,
                "Nominal_Loss": tot_ret * capital,
                "Nominal_Max_Drawdown": max_dd * capital,
                "Evaluation_Mode": mode,
            })

        return results

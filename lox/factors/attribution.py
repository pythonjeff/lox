"""
P&L factor attribution — decompose trailing returns into factor contributions.

Standard Brinson-style factor attribution: contribution = beta x factor_return.
Alpha residual = total_return - sum(factor_contributions).

Author: Lox Capital Research
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lox.factors.loadings import FACTOR_NAMES, FACTOR_DF_COLS
from lox.factors.portfolio import PortfolioFactorProfile


@dataclass(frozen=True)
class FactorContribution:
    """Single factor's contribution to portfolio return."""

    factor: str
    contribution_pct: float  # percentage points
    beta: float
    factor_return_pct: float  # factor's own return over the period


@dataclass(frozen=True)
class FactorAttribution:
    """Full P&L decomposition for a period."""

    period_days: int
    total_return_pct: float
    factor_contributions: list[FactorContribution]
    alpha_residual_pct: float
    explained_pct: float  # % of total explained by factors

    def verdict(self) -> str:
        """Generate a one-line PM verdict."""
        if abs(self.total_return_pct) < 0.05:
            return "Flat period — not enough return to decompose meaningfully"

        if abs(self.total_return_pct) > 0.01 and abs(self.explained_pct) > 200:
            return "Factor contributions offset each other — alpha is the residual"

        alpha_share = abs(self.alpha_residual_pct) / max(abs(self.total_return_pct), 0.01) * 100

        # Find dominant factor
        sorted_fc = sorted(self.factor_contributions, key=lambda c: abs(c.contribution_pct), reverse=True)
        dominant = sorted_fc[0] if sorted_fc else None

        if alpha_share > 50:
            return "Majority of return is alpha — your active bets are driving P&L"
        elif alpha_share > 25:
            sign = "positive" if self.alpha_residual_pct > 0 else "negative"
            return f"Meaningful {sign} alpha alongside factor exposure"

        if dominant and abs(dominant.contribution_pct) > abs(self.total_return_pct) * 0.6:
            direction = "gain" if dominant.contribution_pct > 0 else "drag"
            return f"P&L dominated by {dominant.factor} — {direction} from factor exposure, limited alpha"

        return "Returns spread across multiple factors with modest alpha"

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_days": self.period_days,
            "total_return_pct": round(self.total_return_pct, 4),
            "factor_contributions": [
                {
                    "factor": fc.factor,
                    "contribution_pct": round(fc.contribution_pct, 4),
                    "beta": round(fc.beta, 4),
                    "factor_return_pct": round(fc.factor_return_pct, 4),
                }
                for fc in self.factor_contributions
            ],
            "alpha_residual_pct": round(self.alpha_residual_pct, 4),
            "explained_pct": round(self.explained_pct, 2),
            "verdict": self.verdict(),
        }


def compute_attribution(
    profile: PortfolioFactorProfile,
    factor_df: pd.DataFrame,
    position_returns: pd.DataFrame | None = None,
    period_days: int = 20,
) -> FactorAttribution:
    """Decompose trailing P&L into factor contributions.

    Args:
        profile: Portfolio factor profile with betas
        factor_df: Full factor DataFrame (from fetch_french_factors)
        position_returns: Optional — weighted portfolio daily returns.
            If None, estimates total return from factor model.
        period_days: Trailing period in trading days
    """
    # Get trailing factor returns
    recent_factors = factor_df.tail(period_days)
    if len(recent_factors) < 5:
        return FactorAttribution(
            period_days=period_days,
            total_return_pct=0.0,
            factor_contributions=[],
            alpha_residual_pct=0.0,
            explained_pct=0.0,
        )

    # Cumulative factor returns over the period
    cum_factor_returns: dict[str, float] = {}
    for factor_name, df_col in zip(FACTOR_NAMES, FACTOR_DF_COLS):
        cum_ret = float(np.prod(1 + recent_factors[df_col].values) - 1)
        cum_factor_returns[factor_name] = cum_ret

    # Factor contributions = beta x factor_return
    contributions: list[FactorContribution] = []
    total_factor_contribution = 0.0

    for factor in FACTOR_NAMES:
        beta = profile.portfolio_betas.get(factor, 0.0)
        factor_ret = cum_factor_returns[factor]
        contribution = beta * factor_ret

        contributions.append(FactorContribution(
            factor=factor,
            contribution_pct=contribution * 100,
            beta=beta,
            factor_return_pct=factor_ret * 100,
        ))
        total_factor_contribution += contribution

    # Total portfolio return: use position_returns if available, else estimate
    if position_returns is not None and len(position_returns) >= period_days:
        recent_port = position_returns.tail(period_days)
        total_ret = float(np.prod(1 + recent_port.values) - 1)
    else:
        # Estimate from factor model + recent RF
        rf_cum = float(np.prod(1 + recent_factors["RF"].values) - 1)
        total_ret = total_factor_contribution + rf_cum

    total_ret_pct = total_ret * 100
    total_fc_pct = total_factor_contribution * 100
    alpha_pct = total_ret_pct - total_fc_pct

    explained = (total_fc_pct / total_ret_pct * 100) if abs(total_ret_pct) > 0.01 else 0.0

    # Sort by absolute contribution
    contributions.sort(key=lambda c: abs(c.contribution_pct), reverse=True)

    return FactorAttribution(
        period_days=period_days,
        total_return_pct=total_ret_pct,
        factor_contributions=contributions,
        alpha_residual_pct=alpha_pct,
        explained_pct=explained,
    )

"""
Portfolio-level factor aggregation.

Weight-averages position loadings into a portfolio factor profile
with concentration analysis and human-readable tilt description.

Author: Lox Capital Research
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lox.factors.loadings import PositionLoadings, FACTOR_NAMES


@dataclass(frozen=True)
class PortfolioFactorProfile:
    """Aggregated portfolio factor exposure."""

    portfolio_betas: dict[str, float]
    factor_concentration: dict[str, float]  # % of total absolute exposure
    dominant_factor: str
    dominant_factor_pct: float
    concentration_warning: str
    tilt_description: str
    portfolio_r_squared: float
    position_loadings: list[PositionLoadings]
    n_data_warnings: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio_betas": {k: round(v, 4) for k, v in self.portfolio_betas.items()},
            "factor_concentration": {k: round(v, 1) for k, v in self.factor_concentration.items()},
            "dominant_factor": self.dominant_factor,
            "dominant_factor_pct": round(self.dominant_factor_pct, 1),
            "concentration_warning": self.concentration_warning,
            "tilt_description": self.tilt_description,
            "portfolio_r_squared": round(self.portfolio_r_squared, 4),
            "n_positions": len(self.position_loadings),
            "n_data_warnings": self.n_data_warnings,
            "position_loadings": [p.to_dict() for p in self.position_loadings],
        }


def _build_tilt_description(betas: dict[str, float]) -> str:
    """Generate PM-readable description of factor tilts."""
    parts: list[str] = []

    mkt = betas.get("Mkt", 0)
    if mkt > 1.15:
        parts.append("above-market beta (aggressive)")
    elif mkt > 0.85:
        parts.append("near-market beta")
    elif mkt > 0.5:
        parts.append("below-market beta (defensive)")
    elif mkt > 0:
        parts.append("low beta")
    else:
        parts.append("negative beta (hedged/short)")

    hml = betas.get("HML", 0)
    if hml < -0.15:
        parts.append("growth-tilted (negative value)")
    elif hml > 0.15:
        parts.append("value-tilted")

    mom = betas.get("Mom", 0)
    if mom > 0.15:
        parts.append("momentum loading")
    elif mom < -0.15:
        parts.append("contrarian (anti-momentum)")

    smb = betas.get("SMB", 0)
    if smb > 0.15:
        parts.append("small-cap tilt")
    elif smb < -0.15:
        parts.append("large-cap tilt")

    rmw = betas.get("RMW", 0)
    if rmw > 0.15:
        parts.append("quality tilt")
    elif rmw < -0.15:
        parts.append("junk/speculative tilt")

    cma = betas.get("CMA", 0)
    if cma > 0.15:
        parts.append("conservative-investment tilt")
    elif cma < -0.15:
        parts.append("aggressive-investment tilt")

    if not parts:
        return "Neutral factor profile"
    return ", ".join(parts[:4])  # cap at 4 descriptors


def build_portfolio_profile(
    loadings: list[PositionLoadings],
) -> PortfolioFactorProfile:
    """Aggregate position-level loadings into portfolio profile."""
    if not loadings:
        return PortfolioFactorProfile(
            portfolio_betas={f: 0.0 for f in FACTOR_NAMES},
            factor_concentration={f: 0.0 for f in FACTOR_NAMES},
            dominant_factor="N/A",
            dominant_factor_pct=0.0,
            concentration_warning="No positions with factor data",
            tilt_description="N/A",
            portfolio_r_squared=0.0,
            position_loadings=[],
            n_data_warnings=0,
        )

    # Weighted average betas (weight is signed — shorts contribute negatively)
    total_abs_weight = sum(abs(p.weight_pct) for p in loadings)
    if total_abs_weight == 0:
        total_abs_weight = 1.0

    portfolio_betas: dict[str, float] = {}
    for factor in FACTOR_NAMES:
        weighted_sum = sum(
            (p.weight_pct / 100.0) * p.betas_dict()[factor]
            for p in loadings
        )
        portfolio_betas[factor] = weighted_sum

    # Factor concentration: % of total absolute factor exposure
    abs_exposures = {f: abs(b) for f, b in portfolio_betas.items()}
    total_abs_exposure = sum(abs_exposures.values())
    if total_abs_exposure == 0:
        total_abs_exposure = 1.0

    factor_concentration = {
        f: abs_exposures[f] / total_abs_exposure * 100
        for f in FACTOR_NAMES
    }

    # Dominant factor
    dominant = max(factor_concentration, key=lambda f: factor_concentration[f])
    dominant_pct = factor_concentration[dominant]

    warning = ""
    if dominant_pct > 80:
        warning = f"{dominant_pct:.0f}% of factor exposure from {dominant} — very concentrated, limited diversification"
    elif dominant_pct > 70:
        warning = f"{dominant_pct:.0f}% of factor exposure from {dominant} — concentrated"

    # Weighted R-squared
    total_r2_weight = 0.0
    weighted_r2 = 0.0
    for p in loadings:
        w = abs(p.weight_pct)
        weighted_r2 += p.r_squared * w
        total_r2_weight += w
    portfolio_r2 = weighted_r2 / total_r2_weight if total_r2_weight > 0 else 0.0

    n_warnings = sum(1 for p in loadings if p.data_warning)

    return PortfolioFactorProfile(
        portfolio_betas=portfolio_betas,
        factor_concentration=factor_concentration,
        dominant_factor=dominant,
        dominant_factor_pct=dominant_pct,
        concentration_warning=warning,
        tilt_description=_build_tilt_description(portfolio_betas),
        portfolio_r_squared=portfolio_r2,
        position_loadings=loadings,
        n_data_warnings=n_warnings,
    )

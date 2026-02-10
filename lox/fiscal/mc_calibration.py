"""
Calibrated Monte Carlo adjustments from fiscal regime pressure.

Maps the Fiscal Pressure Index (FPI) and sub-scores to continuous MC
parameter adjustments (term premium, equity crowding-out, jump risk, etc.).

All functions return incremental adjustments — callers add these to
whatever base MC parameters exist.

Reference ranges are calibrated to historical post-GFC / COVID fiscal episodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lox.fiscal.scoring import FiscalScorecard


@dataclass(frozen=True)
class FiscalMCParams:
    """Fiscal-driven Monte Carlo parameter adjustments."""
    term_premium_bps: float        # Added term premium in basis points
    equity_crowding_out: float     # Annual equity drift penalty (e.g. -0.02 = -2%)
    jump_prob_multiplier: float    # Multiplier on tail-event probability (1.0 = no change)
    jump_size_multiplier: float    # Multiplier on tail-event magnitude
    spread_adj_bps: float          # Credit spread adjustment in basis points
    rate_drift_adj: float          # Additional annual rate drift (decimal)
    description: str               # Human-readable summary


def _smooth_ramp(x: float, low: float, high: float, min_val: float, max_val: float) -> float:
    """
    Smooth linear ramp between [low, high] mapping to [min_val, max_val].
    Clamps at boundaries.
    """
    if x <= low:
        return min_val
    if x >= high:
        return max_val
    # Linear interpolation
    t = (x - low) / (high - low)
    return min_val + t * (max_val - min_val)


def calibrate_fiscal_mc(scorecard: FiscalScorecard) -> FiscalMCParams:
    """
    Convert a FiscalScorecard into calibrated Monte Carlo adjustments.

    Calibration logic (CFA-aligned reasoning):

    1. **Term Premium** (supply-yield elasticity):
       FPI 0-25: 0bp (benign)
       FPI 25-45: 0-15bp (elevated supply → modest term premium)
       FPI 45-65: 15-40bp (stress building → significant term premium)
       FPI 65-80: 40-75bp (fiscal stress → large term premium)
       FPI 80-100: 75-120bp (dominance risk → extreme term premium)

    2. **Equity Crowding-Out** (fiscal pressure displacing private investment):
       Driven by FPI. Range: 0% to -4% annual drift.

    3. **Jump Risk** (auction failure / confidence event):
       Primarily driven by Auction Absorption sub-score.
       Range: 1.0x to 3.0x probability multiplier, 1.0x to 1.5x size multiplier.

    4. **Credit Spreads** (flight-from-quality / crowding):
       Driven by FPI + Demand Structure sub-score.
       Range: 0bp to +80bp.

    5. **Rate Drift** (supply-driven yield increase):
       Driven by Deficit Intensity + Issuance Duration sub-scores.
       Range: 0 to +1.5% annual drift.
    """
    fpi = scorecard.fpi

    # ── 1. Term Premium ──────────────────────────────────────────────────
    term_premium = _smooth_ramp(fpi, 25, 100, 0.0, 120.0)

    # ── 2. Equity Crowding-Out ───────────────────────────────────────────
    # Negative drift = headwind. Scale from 0% at FPI=20 to -4% at FPI=85.
    equity_crowd = _smooth_ramp(fpi, 20, 85, 0.0, -0.04)

    # ── 3. Jump Risk (auction-driven) ────────────────────────────────────
    auction_score = _get_sub_score(scorecard, "Auction Absorption")
    demand_score = _get_sub_score(scorecard, "Demand Structure")

    # Base jump probability from auction stress
    jump_prob = _smooth_ramp(auction_score, 30, 90, 1.0, 3.0)
    # Demand deterioration adds a modest bump
    if demand_score > 60:
        jump_prob *= 1.0 + _smooth_ramp(demand_score, 60, 90, 0.0, 0.3)

    # Jump size: modest amplification under acute stress
    jump_size = _smooth_ramp(auction_score, 50, 90, 1.0, 1.5)

    # ── 4. Credit Spreads ────────────────────────────────────────────────
    spread_adj = _smooth_ramp(fpi, 30, 80, 0.0, 60.0)
    # Demand structure adds incremental widening
    spread_adj += _smooth_ramp(demand_score, 50, 90, 0.0, 20.0)

    # ── 5. Rate Drift (supply-driven yield rise) ─────────────────────────
    deficit_score = _get_sub_score(scorecard, "Deficit Intensity")
    duration_score = _get_sub_score(scorecard, "Issuance Duration")

    # Blend: heavier weight to deficit level, lighter to duration tilt
    blended_supply = 0.6 * deficit_score + 0.4 * duration_score
    rate_drift = _smooth_ramp(blended_supply, 30, 85, 0.0, 0.015)  # 0 to 150bp/year

    # ── Description ──────────────────────────────────────────────────────
    desc_parts = [f"FPI {fpi:.0f}/100"]
    if term_premium > 5:
        desc_parts.append(f"term premium +{term_premium:.0f}bp")
    if equity_crowd < -0.005:
        desc_parts.append(f"equity drag {equity_crowd*100:+.1f}%")
    if jump_prob > 1.1:
        desc_parts.append(f"jump prob {jump_prob:.1f}x")
    if spread_adj > 5:
        desc_parts.append(f"spread +{spread_adj:.0f}bp")
    if rate_drift > 0.001:
        desc_parts.append(f"rate drift +{rate_drift*100:.0f}bp/yr")
    desc = "; ".join(desc_parts)

    return FiscalMCParams(
        term_premium_bps=round(term_premium, 1),
        equity_crowding_out=round(equity_crowd, 4),
        jump_prob_multiplier=round(jump_prob, 2),
        jump_size_multiplier=round(jump_size, 2),
        spread_adj_bps=round(spread_adj, 1),
        rate_drift_adj=round(rate_drift, 5),
        description=desc,
    )


def _get_sub_score(scorecard: FiscalScorecard, name: str) -> float:
    """Get a sub-score by name, returning 50.0 (neutral) if missing."""
    for s in scorecard.sub_scores:
        if s.name == name:
            return s.score
    return 50.0

"""
Fiscal Pressure Index (FPI): continuous 0-100 scoring engine.

CFA-aligned approach: six sub-score pillars, each scored via historical
percentile rank (10-year window), then combined into a weighted composite.

Regime labels are derived from FPI thresholds + pattern overrides.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from lox.fiscal.models import FiscalInputs


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FiscalSubScore:
    """One of six pillars in the fiscal scorecard."""
    name: str
    score: float          # 0-100 (higher = more fiscal stress)
    percentile: float     # historical percentile (0-100)
    components: dict      # raw values that fed the score
    weight: float         # contribution weight (sums to 1.0)


@dataclass(frozen=True)
class FiscalScorecard:
    """Full fiscal regime scorecard: composite + breakdown."""
    fpi: float                                  # Fiscal Pressure Index 0-100
    sub_scores: list[FiscalSubScore]            # 6 pillars
    regime_label: str                           # e.g. "Stress Building"
    regime_name: str                            # e.g. "stress_building"
    regime_description: str
    percentile_overall: float | None = None     # where FPI sits in history


# ─────────────────────────────────────────────────────────────────────────────
# Pillar definitions
# ─────────────────────────────────────────────────────────────────────────────
# Each pillar: name, input fields (from FiscalInputs), weight, and a
# scoring function that maps raw values → 0-100.

def _pctile_score(*vals: float | None) -> float:
    """
    Convert z-scored values to a 0-100 score using the standard normal CDF.

    Positive z → higher score (more stress). Missing values are skipped.
    If all values are missing, returns 50 (neutral).
    """
    from scipy.stats import norm
    valid = [v for v in vals if v is not None and np.isfinite(v)]
    if not valid:
        return 50.0
    avg_z = float(np.mean(valid))
    # CDF maps z=0 → 0.5, z=+2 → ~0.977, z=-2 → ~0.023
    return float(norm.cdf(avg_z) * 100.0)


def _deficit_intensity_score(inputs: FiscalInputs) -> FiscalSubScore:
    """How large is the deficit relative to GDP and tax receipts."""
    # Use z_deficit_12m as primary signal; deficit_pct_receipts as augmentation
    vals = [inputs.z_deficit_12m]
    # deficit_pct_receipts: higher ratio = more stress, but no z-score yet.
    # Use a simple heuristic: 0.3 = neutral, above → positive z-like signal.
    if inputs.deficit_pct_receipts is not None:
        # Typical range ~0.05 (5%) to 0.50 (50%). Map to z-like scale.
        dpr_z = (inputs.deficit_pct_receipts - 0.20) / 0.10  # center at 20%, 10% = 1 sigma
        vals.append(dpr_z)

    score = _pctile_score(*vals)
    return FiscalSubScore(
        name="Deficit Intensity",
        score=score,
        percentile=score,
        components={
            "z_deficit_12m": inputs.z_deficit_12m,
            "deficit_pct_receipts": inputs.deficit_pct_receipts,
        },
        weight=0.25,
    )


def _deficit_momentum_score(inputs: FiscalInputs) -> FiscalSubScore:
    """Is the deficit getting worse or better."""
    vals = []
    # z_deficit_12m captures level; for momentum, we want the trend slope
    if inputs.deficit_trend_slope is not None:
        # Slope is in deficit-units per month; positive = deteriorating.
        # Normalize: assume std ~ 5000 (millions), so divide by 5000 for z-like scale.
        slope_z = inputs.deficit_trend_slope / 5000.0
        vals.append(slope_z)

    # interest_expense_yoy_accel: positive = accelerating cost of debt
    if inputs.interest_expense_yoy_accel is not None:
        vals.append(inputs.interest_expense_yoy_accel / 5.0)  # 5pp as ~1 sigma

    score = _pctile_score(*vals)
    return FiscalSubScore(
        name="Deficit Momentum",
        score=score,
        percentile=score,
        components={
            "deficit_trend_slope": inputs.deficit_trend_slope,
            "interest_expense_yoy_accel": inputs.interest_expense_yoy_accel,
        },
        weight=0.15,
    )


def _issuance_duration_score(inputs: FiscalInputs) -> FiscalSubScore:
    """Duration absorption risk from issuance mix."""
    vals = [inputs.z_long_duration_issuance_share]
    if inputs.wam_chg_12m is not None:
        # WAM change: positive = lengthening duration, more stress.
        vals.append(inputs.wam_chg_12m / 0.3)  # 0.3y change = ~1 sigma
    score = _pctile_score(*vals)
    return FiscalSubScore(
        name="Issuance Duration",
        score=score,
        percentile=score,
        components={
            "z_long_duration_issuance_share": inputs.z_long_duration_issuance_share,
            "wam_chg_12m": inputs.wam_chg_12m,
        },
        weight=0.15,
    )


def _auction_absorption_score(inputs: FiscalInputs) -> FiscalSubScore:
    """Is the market digesting Treasury supply?"""
    vals = [inputs.z_auction_tail_bps, inputs.z_dealer_take_pct]
    # Bid-to-cover: lower = worse absorption. Invert so higher score = more stress.
    if inputs.bid_to_cover_avg is not None:
        # Typical BTC ~2.3-2.8. Below 2.0 is weak.
        btc_z = -(inputs.bid_to_cover_avg - 2.5) / 0.3  # inverted: lower BTC → higher z
        vals.append(btc_z)
    score = _pctile_score(*vals)
    return FiscalSubScore(
        name="Auction Absorption",
        score=score,
        percentile=score,
        components={
            "z_auction_tail_bps": inputs.z_auction_tail_bps,
            "z_dealer_take_pct": inputs.z_dealer_take_pct,
            "bid_to_cover_avg": inputs.bid_to_cover_avg,
        },
        weight=0.20,
    )


def _demand_structure_score(inputs: FiscalInputs) -> FiscalSubScore:
    """Are marginal buyers retreating."""
    vals = []
    # Foreign holdings declining = stress. Negative change → positive z.
    if inputs.foreign_holdings_chg_6m is not None:
        # Normalize: typical 6m change ~±50B. Declining = stress.
        fh_z = -inputs.foreign_holdings_chg_6m / 50_000.0  # millions → inverted z-like
        vals.append(fh_z)
    if inputs.custody_holdings_chg_4w is not None:
        # Custody declining = foreign CBs selling. Typical 4w change ~±10B.
        cust_z = -inputs.custody_holdings_chg_4w / 10_000.0  # millions → inverted z-like
        vals.append(cust_z)
    score = _pctile_score(*vals)
    return FiscalSubScore(
        name="Demand Structure",
        score=score,
        percentile=score,
        components={
            "foreign_holdings_chg_6m": inputs.foreign_holdings_chg_6m,
            "custody_holdings_chg_4w": inputs.custody_holdings_chg_4w,
        },
        weight=0.10,
    )


def _bond_market_stress_score(inputs: FiscalInputs) -> FiscalSubScore:
    """Bond market pricing of fiscal risk."""
    vals = []
    if inputs.move_index_z is not None:
        vals.append(inputs.move_index_z)
    if inputs.z_tga_chg_28d is not None:
        # TGA up sharply (tightening) → stress. z already captures direction.
        vals.append(inputs.z_tga_chg_28d)
    score = _pctile_score(*vals)
    return FiscalSubScore(
        name="Bond Market Stress",
        score=score,
        percentile=score,
        components={
            "move_index_z": inputs.move_index_z,
            "z_tga_chg_28d": inputs.z_tga_chg_28d,
        },
        weight=0.15,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Regime label derivation
# ─────────────────────────────────────────────────────────────────────────────
# FPI thresholds → labels. Auction absorption override escalates by one notch.

_FPI_LABELS = [
    (0, 25, "benign_funding", "Benign"),
    (25, 45, "elevated_funding", "Elevated Funding"),
    (45, 65, "stress_building", "Stress Building"),
    (65, 80, "fiscal_stress", "Fiscal Stress"),
    (80, 101, "fiscal_dominance_risk", "Fiscal Dominance Risk"),
]


def _label_from_fpi(fpi: float, auction_score: float) -> tuple[str, str]:
    """Return (regime_name, regime_label) from FPI, with auction override."""
    idx = 0
    for i, (lo, hi, name, label) in enumerate(_FPI_LABELS):
        if lo <= fpi < hi:
            idx = i
            break
    else:
        idx = len(_FPI_LABELS) - 1

    # Auction absorption override: if auction sub-score > 80 and we're not
    # already at max, escalate by one notch.
    if auction_score > 80 and idx < len(_FPI_LABELS) - 1:
        idx += 1

    return _FPI_LABELS[idx][2], _FPI_LABELS[idx][3]


# ─────────────────────────────────────────────────────────────────────────────
# Main scoring function
# ─────────────────────────────────────────────────────────────────────────────

def score_fiscal_regime(inputs: FiscalInputs) -> FiscalScorecard:
    """
    Compute the Fiscal Pressure Index (FPI) and full scorecard.

    Returns a FiscalScorecard with:
    - fpi: 0-100 composite score (higher = more fiscal stress)
    - sub_scores: 6 pillar breakdowns
    - regime_label / regime_name: derived from FPI + overrides
    - regime_description: contextual explanation
    """
    # Compute all 6 sub-scores
    pillars = [
        _deficit_intensity_score(inputs),
        _deficit_momentum_score(inputs),
        _issuance_duration_score(inputs),
        _auction_absorption_score(inputs),
        _demand_structure_score(inputs),
        _bond_market_stress_score(inputs),
    ]

    # Weighted composite
    total_weight = sum(p.weight for p in pillars)
    fpi = sum(p.score * p.weight for p in pillars) / total_weight if total_weight > 0 else 50.0

    # Derive regime label
    auction_score = next((p.score for p in pillars if p.name == "Auction Absorption"), 50.0)
    regime_name, regime_label = _label_from_fpi(fpi, auction_score)

    # Build description
    high_pillars = sorted(
        [p for p in pillars if p.score >= 60],
        key=lambda p: -p.score,
    )
    if high_pillars:
        drivers = ", ".join(f"{p.name} ({p.score:.0f})" for p in high_pillars[:3])
        desc = f"FPI {fpi:.0f}/100. Key drivers: {drivers}."
    else:
        desc = f"FPI {fpi:.0f}/100. No individual pillar above 60 — broad-based low pressure."

    return FiscalScorecard(
        fpi=fpi,
        sub_scores=pillars,
        regime_label=regime_label,
        regime_name=regime_name,
        regime_description=desc,
        percentile_overall=None,  # requires historical FPI series; future enhancement
    )

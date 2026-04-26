"""
Fiscal Pressure Index (FPI): continuous 0-100 scoring engine.

MMT sectoral balance orientation: higher score = more contractionary for
the private sector (less NFA support). A *smaller* deficit scores higher
because the private sector receives fewer net financial assets.

Six sub-score pillars, each scored via historical percentile rank
(10-year window), then combined into a weighted composite.

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
    sub_scores: list[FiscalSubScore]            # 6 weighted pillars + display-only sub-scores
    regime_label: str                           # e.g. "Fiscal Contraction"
    regime_name: str                            # e.g. "fiscal_contraction"
    regime_description: str
    percentile_overall: float | None = None     # where FPI sits in history
    # Divergence flags fired by the auction split (clearing vs demand-quality).
    # Populated by score_fiscal_regime when the gap exceeds the threshold.
    divergence_flags: dict[str, bool] = field(default_factory=dict)


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
    """MMT: how much NFA support does the private sector receive from the deficit?

    Inverted from conventional framing: a *smaller* deficit scores higher
    (more contractionary for private sector = more stress for risk assets).
    """
    vals = []
    # Negate z_deficit_12m: positive z (large deficit) → low score (supportive);
    # negative z (small deficit) → high score (contractionary).
    if inputs.z_deficit_12m is not None:
        vals.append(-inputs.z_deficit_12m)
    # deficit_pct_receipts: higher ratio = more govt spending relative to taxes = more NFA.
    # Invert: low ratio → high score (private sector squeeze).
    if inputs.deficit_pct_receipts is not None:
        dpr_z = -(inputs.deficit_pct_receipts - 0.20) / 0.10
        vals.append(dpr_z)

    score = _pctile_score(*vals)
    return FiscalSubScore(
        name="NFA Squeeze",
        score=score,
        percentile=score,
        components={
            "z_deficit_12m": inputs.z_deficit_12m,
            "deficit_pct_receipts": inputs.deficit_pct_receipts,
        },
        weight=0.25,
    )


def _deficit_momentum_score(inputs: FiscalInputs) -> FiscalSubScore:
    """MMT: is the fiscal impulse contracting or expanding?

    Inverted: a *declining* deficit trajectory (negative slope) scores higher
    because the private sector's NFA inflow is shrinking.
    """
    vals = []
    if inputs.deficit_trend_slope is not None:
        # Negate: negative slope (shrinking deficit) → positive z-like → high score (drag).
        slope_z = -inputs.deficit_trend_slope / 5000.0
        vals.append(slope_z)

    # Interest expense acceleration remains same direction: higher = more stress.
    if inputs.interest_expense_yoy_accel is not None:
        vals.append(inputs.interest_expense_yoy_accel / 5.0)

    score = _pctile_score(*vals)
    return FiscalSubScore(
        name="Fiscal Impulse Drag",
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
        btc_z = -(inputs.bid_to_cover_avg - 2.5) / 0.3
        vals.append(btc_z)
    score = _pctile_score(*vals)

    # Floor: absolute tail/dealer levels override if z-scores understate stress.
    # Blended z-scores can be diluted when some tenors are fine and others stressed.
    floor = 0.0
    if inputs.auction_tail_bps is not None:
        if inputs.auction_tail_bps >= 5.0:
            floor = max(floor, 60.0)
        elif inputs.auction_tail_bps >= 3.0:
            floor = max(floor, 45.0)
    if inputs.dealer_take_pct is not None:
        if inputs.dealer_take_pct >= 35.0:
            floor = max(floor, 60.0)
        elif inputs.dealer_take_pct >= 25.0:
            floor = max(floor, 45.0)
    score = max(score, floor)

    return FiscalSubScore(
        name="Auction Absorption",
        score=score,
        percentile=score,
        components={
            "z_auction_tail_bps": inputs.z_auction_tail_bps,
            "z_dealer_take_pct": inputs.z_dealer_take_pct,
            "bid_to_cover_avg": inputs.bid_to_cover_avg,
            "auction_tail_bps_raw": inputs.auction_tail_bps,
            "dealer_take_pct_raw": inputs.dealer_take_pct,
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
# Auction split: clearing vs demand quality (display-only, weight=0.0)
# ─────────────────────────────────────────────────────────────────────────────
# These two sub-scores decompose what the existing "Auction Absorption" pillar
# combines into one number. Both are surfaced (not weighted into FPI yet) so
# downstream consumers can flag a *divergence*: a "clean" tail and bid-to-cover
# masking a structural decay in the bidder mix is the canonical demand-quality
# canary. Adding the two scores at weight=0.0 keeps FPI numerically identical
# to the prior 6-pillar composite while exposing the new signal.

# Healthy post-GFC baselines for coupon issuance (rough but workable):
_BASELINE_INDIRECT_PCT = 65.0
_BASELINE_DEALER_PCT = 20.0
# 1-sigma scaling (pp). Tuned to make ±10pp shifts move ~1 z.
_DEMAND_QUALITY_SIGMA = 10.0


def _auction_clearing_score(inputs: FiscalInputs) -> FiscalSubScore:
    """
    How well did the most recent auctions *clear*?
    Inputs: tail (z and raw) + bid-to-cover. Higher score = worse clearing.

    This overlaps the legacy Auction Absorption sub-score by design — it isolates
    the price-discovery half (tail/BTC) from the bidder-mix half (Demand Quality).
    """
    vals: list[float] = []
    if inputs.z_auction_tail_bps is not None:
        vals.append(inputs.z_auction_tail_bps)
    if inputs.bid_to_cover_avg is not None:
        # Lower BTC = worse clearing. Center 2.5, sigma 0.3.
        btc_z = -(inputs.bid_to_cover_avg - 2.5) / 0.3
        vals.append(btc_z)

    score = _pctile_score(*vals)

    # Floor on raw tail magnitude (matches the legacy absorption-floor pattern).
    floor = 0.0
    if inputs.auction_tail_bps is not None:
        if inputs.auction_tail_bps >= 5.0:
            floor = max(floor, 60.0)
        elif inputs.auction_tail_bps >= 3.0:
            floor = max(floor, 45.0)
    score = max(score, floor)

    return FiscalSubScore(
        name="Auction Clearing",
        score=score,
        percentile=score,
        components={
            "z_auction_tail_bps": inputs.z_auction_tail_bps,
            "auction_tail_bps_raw": inputs.auction_tail_bps,
            "bid_to_cover_avg": inputs.bid_to_cover_avg,
        },
        weight=0.0,  # display-only for now; rebalance in a later phase
    )


def _auction_demand_quality_score(inputs: FiscalInputs) -> FiscalSubScore:
    """
    Who actually showed up to the auction?
    Inputs: indirect / direct / dealer bidder shares (last 6 coupon auctions).

    Construction: stress-z = (dealer - 20)/10  -  (indirect - 65)/10
      → 0 at baseline (dealer 20%, indirect 65%)
      → positive when dealers absorb more *or* indirects retreat
      → score = norm.cdf(stress-z) * 100  (higher = worse demand quality)

    Direct bidder share is recorded but not yet penalized — direct can absorb
    a healthy chunk without signalling stress, unlike the dealer wall.

    Returns a neutral 50 if no bidder data is available.
    """
    parts: list[float] = []
    if inputs.dealer_take_pct is not None:
        parts.append((inputs.dealer_take_pct - _BASELINE_DEALER_PCT) / _DEMAND_QUALITY_SIGMA)
    if inputs.indirect_bid_share is not None:
        parts.append(-(inputs.indirect_bid_share - _BASELINE_INDIRECT_PCT) / _DEMAND_QUALITY_SIGMA)

    if not parts:
        score = 50.0
    else:
        # Sum of component z-stresses (each is signed).
        stress_z = float(np.sum(parts))
        score = _pctile_score(stress_z)

    return FiscalSubScore(
        name="Auction Demand Quality",
        score=score,
        percentile=score,
        components={
            "indirect_bid_share": inputs.indirect_bid_share,
            "direct_bid_share": inputs.direct_bid_share,
            "primary_dealer_pct": inputs.dealer_take_pct,
            "auction_demand_window": inputs.auction_demand_window,
            "baseline_indirect": _BASELINE_INDIRECT_PCT,
            "baseline_dealer": _BASELINE_DEALER_PCT,
        },
        weight=0.0,  # display-only for now; rebalance in a later phase
    )


# Threshold at which clearing vs demand-quality divergence is flagged.
_AUCTION_DIVERGENCE_THRESHOLD = 40.0


# ─────────────────────────────────────────────────────────────────────────────
# Regime label derivation
# ─────────────────────────────────────────────────────────────────────────────
# FPI thresholds → labels. Auction absorption override escalates by one notch.

_FPI_LABELS = [
    (0, 25, "strong_fiscal_stimulus", "Strong Fiscal Stimulus"),
    (25, 45, "moderate_fiscal_support", "Moderate Fiscal Support"),
    (45, 65, "fiscal_drag", "Fiscal Drag"),
    (65, 80, "fiscal_contraction", "Fiscal Contraction"),
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

    # Auction absorption override: if auction sub-score > 60 and we're not
    # already at max, escalate by one notch. Auctions are the most
    # market-visible stress signal and should escalate the label early.
    if auction_score > 60 and idx < len(_FPI_LABELS) - 1:
        idx += 1

    return _FPI_LABELS[idx][2], _FPI_LABELS[idx][3]


# ─────────────────────────────────────────────────────────────────────────────
# Main scoring function
# ─────────────────────────────────────────────────────────────────────────────

def score_fiscal_regime(inputs: FiscalInputs) -> FiscalScorecard:
    """
    Compute the Fiscal Pressure Index (FPI) and full scorecard.

    MMT orientation: higher FPI = more contractionary for the private sector
    (less NFA support, fiscal drag, or market absorption stress).

    Returns a FiscalScorecard with:
    - fpi: 0-100 composite score (higher = more private sector squeeze)
    - sub_scores: 6 pillar breakdowns
    - regime_label / regime_name: derived from FPI + overrides
    - regime_description: contextual explanation
    """
    # Compute all 6 weighted sub-scores
    pillars = [
        _deficit_intensity_score(inputs),
        _deficit_momentum_score(inputs),
        _issuance_duration_score(inputs),
        _auction_absorption_score(inputs),
        _demand_structure_score(inputs),
        _bond_market_stress_score(inputs),
    ]

    # Weighted composite (uses the 6 weighted pillars only)
    total_weight = sum(p.weight for p in pillars)
    fpi = sum(p.score * p.weight for p in pillars) / total_weight if total_weight > 0 else 50.0

    # Derive regime label
    auction_score = next((p.score for p in pillars if p.name == "Auction Absorption"), 50.0)
    regime_name, regime_label = _label_from_fpi(fpi, auction_score)

    # Display-only sub-scores: split the auction signal into clearing vs
    # demand-quality so divergence between them can be surfaced. Weight=0.0
    # → no impact on FPI. Order matters: clearing first, then quality.
    clearing = _auction_clearing_score(inputs)
    quality = _auction_demand_quality_score(inputs)
    pillars.append(clearing)
    pillars.append(quality)

    # Divergence flag: clearing looks fine while bidder mix is decaying
    # (or vice versa). Only meaningful if both scores have real inputs —
    # if demand-quality is the neutral 50 default, suppress the flag.
    divergence_flags: dict[str, bool] = {}
    has_quality_inputs = (
        inputs.indirect_bid_share is not None or inputs.dealer_take_pct is not None
    )
    if has_quality_inputs:
        divergence_flags["auction_clearing_vs_quality"] = (
            abs(clearing.score - quality.score) > _AUCTION_DIVERGENCE_THRESHOLD
        )

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
        divergence_flags=divergence_flags,
    )

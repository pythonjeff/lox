"""
SPY options flow regime classifier.

3 orthogonal signals from a single options chain fetch:
  1. GEX ($bn) — dealer gamma mechanics
  2. Put/Call Ratio (OI-weighted) — hedging demand
  3. 25-delta Skew (vol pts) — smart money fear gauge

Score 0 = extreme complacency → 100 = panic.
Higher = more defensive / vol-amplifying environment.

Author: Lox Capital Research
"""
from __future__ import annotations

from lox.regimes.base import RegimeResult


# ─────────────────────────────────────────────────────────────────────────────
# Sub-score mapping functions (0-100, higher = more stress)
# ─────────────────────────────────────────────────────────────────────────────

def _gex_subscore(gex_bn: float) -> float:
    """Map GEX ($bn) to 0-100.

    Positive GEX = dealers long gamma → market stabilizing (low score).
    Negative GEX = dealers short gamma → vol amplifying (high score).
    """
    if gex_bn < -5.0:
        return 90
    if gex_bn < -2.0:
        return 75
    if gex_bn < -0.5:
        return 62
    if gex_bn < 0.5:
        return 50
    if gex_bn < 2.0:
        return 38
    if gex_bn < 5.0:
        return 25
    return 12


def _pcr_subscore(pcr: float) -> float:
    """Map put/call ratio to 0-100.

    >1.0 = fearful put buying, <0.65 = complacent call buying.
    """
    if pcr > 1.5:
        return 95
    if pcr > 1.2:
        return 80
    if pcr > 1.0:
        return 65
    if pcr > 0.8:
        return 50
    if pcr > 0.65:
        return 35
    if pcr > 0.5:
        return 20
    return 10


def _skew_subscore(skew_25d: float) -> float:
    """Map 25-delta risk reversal (vol pts) to 0-100.

    Positive = puts richer than calls (downside fear).
    Negative/zero = calls richer (complacency).
    """
    if skew_25d > 10:
        return 95
    if skew_25d > 8:
        return 82
    if skew_25d > 5:
        return 65
    if skew_25d > 2:
        return 48
    if skew_25d > 0:
        return 35
    return 15


# ─────────────────────────────────────────────────────────────────────────────
# Weights
# ─────────────────────────────────────────────────────────────────────────────

W_GEX = 0.40
W_PCR = 0.30
W_SKEW = 0.30


# ─────────────────────────────────────────────────────────────────────────────
# Regime labels (5 states)
# ─────────────────────────────────────────────────────────────────────────────

REGIMES = [
    # (min_score, name, label, description)
    (80, "dealer_short_gamma_panic",
     "Dealer Short Gamma / Panic",
     "Dealers short gamma, heavy put buying, steep skew — expect violent moves, "
     "mechanical selling accelerates drawdowns. Size down, widen stops."),
    (60, "hedging_demand",
     "Elevated Hedging Demand",
     "Put protection building, GEX declining — wider daily ranges likely. "
     "Directional trades need wider risk budgets."),
    (40, "balanced_flow",
     "Balanced Flow",
     "Neutral options positioning — no strong mechanical signal from flow. "
     "Fundamentals and macro dominate price action."),
    (20, "pinned_complacent",
     "Pinned / Complacent",
     "Dealers long gamma, calls dominate, flat skew — vol suppressed, "
     "mean-reversion favored. Good for premium selling."),
    (0, "extreme_complacency",
     "Extreme Complacency",
     "Max dealer gamma, call-heavy flow, inverted skew — contrarian warning. "
     "Historically precedes vol spikes. Watch for catalyst."),
]


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

def classify_spy_regime(
    gex_bn: float | None,
    pcr: float | None,
    skew_25d: float | None,
    spot: float | None = None,
    flip_level: float | None = None,
) -> RegimeResult:
    """Classify SPY options flow regime from 3 signals.

    Returns a standard RegimeResult with domain="spy".
    Handles missing data gracefully — skips unavailable sub-scores.
    """
    # Compute available sub-scores
    scores: list[tuple[float, float]] = []  # (weight, score)
    if gex_bn is not None:
        scores.append((W_GEX, _gex_subscore(gex_bn)))
    if pcr is not None:
        scores.append((W_PCR, _pcr_subscore(pcr)))
    if skew_25d is not None:
        scores.append((W_SKEW, _skew_subscore(skew_25d)))

    if not scores:
        return RegimeResult(
            name="insufficient_data",
            label="Insufficient Data",
            description="Not enough options chain data to classify regime.",
            score=50,
            domain="spy",
            tags=["no_data"],
        )

    # Weighted average (re-normalize weights if some signals missing)
    total_w = sum(w for w, _ in scores)
    composite = sum(w * s for w, s in scores) / total_w

    # Layer 2: cross-signal amplifier
    if (gex_bn is not None and gex_bn < 0
            and pcr is not None and pcr > 1.0
            and skew_25d is not None and skew_25d > 5):
        composite += 5  # fear confirmation across all 3 signals
    elif (gex_bn is not None and gex_bn > 3
            and pcr is not None and pcr < 0.65
            and skew_25d is not None and skew_25d < 1):
        composite -= 5  # complacency trap confirmation

    composite = max(0, min(100, composite))

    # Map to regime label
    for min_score, name, label, description in REGIMES:
        if composite >= min_score:
            break
    else:
        name, label, description = REGIMES[-1][1], REGIMES[-1][2], REGIMES[-1][3]

    # Tags
    tags = ["spy", "options_flow"]
    if gex_bn is not None:
        tags.append("short_gamma" if gex_bn < 0 else "long_gamma")
    if composite >= 65:
        tags.append("fear")
    elif composite <= 25:
        tags.append("complacent")

    return RegimeResult(
        name=name,
        label=label,
        description=description,
        score=round(composite, 1),
        domain="spy",
        tags=tags,
        metrics={
            "gex_bn": gex_bn,
            "pcr": pcr,
            "skew_25d": skew_25d,
            "spot": spot,
            "flip_level": flip_level,
        },
    )

"""
Credit regime classifier — two-layer approach.

Layer 1 (Base Score): Spread level readings with interpolation.
    - HY OAS absolute level (primary signal)
    - BBB OAS level (IG benchmark)
    - HY percentile vs history (distributional context)

Layer 2 (Amplifiers): Momentum, dispersion, cross-market confirmation.
    - Velocity: 5d, 10d, 30d widening speed (crisis acceleration)
    - Dispersion: HY widening faster than IG = quality deterioration
    - Cross-market: VIX correlation (confirms systemic vs idiosyncratic)
    - HY-IG curve steepness

Credit leads equity vol — this is one of the most important regimes.
Score 0 = euphoria / tight spreads → 100 = credit stress / wide spreads.
"""
from __future__ import annotations

from lox.regimes.base import RegimeResult


def _interp(value: float, low: float, high: float, score_low: float, score_high: float) -> float:
    """Linear interpolation of score between two thresholds."""
    if high == low:
        return score_high
    ratio = max(0.0, min(1.0, (value - low) / (high - low)))
    return score_low + ratio * (score_high - score_low)


def classify_credit(
    hy_oas: float | None,
    bbb_oas: float | None,
    aaa_oas: float | None,
    hy_oas_30d_chg: float | None,
    hy_oas_90d_percentile: float | None,
    *,
    # New Layer 1 inputs
    ig_oas: float | None = None,
    hy_oas_1y_percentile: float | None = None,
    # New Layer 2 inputs (velocity)
    hy_oas_5d_chg: float | None = None,
    hy_oas_10d_chg: float | None = None,
    bbb_oas_30d_chg: float | None = None,
    # New Layer 2 inputs (cross-market)
    vix: float | None = None,
) -> RegimeResult:
    """Classify the Credit regime using two-layer approach.

    Layer 1 — Base score from spread levels.
    Layer 2 — Amplifiers from velocity, dispersion, cross-market.

    Args:
        hy_oas: Current HY OAS in bps (typical 300-800, crisis 1000+).
        bbb_oas: Current BBB OAS in bps.
        aaa_oas: Current AAA OAS in bps.
        hy_oas_30d_chg: Change in HY OAS over last 30 days (positive = widening).
        hy_oas_90d_percentile: Where current HY OAS sits in last 90 days (0-100).
        ig_oas: Current IG corporate OAS in bps.
        hy_oas_1y_percentile: HY OAS percentile over last 1 year (0-100).
        hy_oas_5d_chg: 5-day change in HY OAS (bps).
        hy_oas_10d_chg: 10-day change in HY OAS (bps).
        bbb_oas_30d_chg: 30-day change in BBB OAS (bps).
        vix: Current VIX level (for cross-market confirmation).
    """

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 1: Base score from spread levels
    # ═════════════════════════════════════════════════════════════════════════

    def _hy_score(val: float | None) -> float | None:
        """Map HY OAS (bps) to 0-100. Wider spreads = higher risk."""
        if val is None:
            return None
        if val <= 250:
            return _interp(val, 150, 250, 8, 18)
        if val <= 300:
            return _interp(val, 250, 300, 18, 28)
        if val <= 350:
            return _interp(val, 300, 350, 28, 38)
        if val <= 400:
            return _interp(val, 350, 400, 38, 50)
        if val <= 500:
            return _interp(val, 400, 500, 50, 65)
        if val <= 700:
            return _interp(val, 500, 700, 65, 80)
        if val <= 1000:
            return _interp(val, 700, 1000, 80, 92)
        return min(98.0, 92 + (val - 1000) / 100)

    def _bbb_score(val: float | None) -> float | None:
        """Map BBB OAS to 0-100."""
        if val is None:
            return None
        if val <= 100:
            return 15.0
        if val <= 150:
            return _interp(val, 100, 150, 15, 35)
        if val <= 200:
            return _interp(val, 150, 200, 35, 50)
        if val <= 300:
            return _interp(val, 200, 300, 50, 70)
        if val <= 500:
            return _interp(val, 300, 500, 70, 88)
        return 95.0

    measures: list[tuple[float | None, float]] = [
        (_hy_score(hy_oas), 0.50),        # HY is primary credit signal
        (_bbb_score(bbb_oas), 0.30),       # BBB confirms breadth
    ]

    # Percentile adds distributional context
    if hy_oas_1y_percentile is not None:
        measures.append((hy_oas_1y_percentile, 0.20))
    elif hy_oas_90d_percentile is not None:
        measures.append((hy_oas_90d_percentile, 0.20))

    total_weight = 0.0
    weighted_sum = 0.0
    for sub_score, weight in measures:
        if sub_score is not None:
            weighted_sum += sub_score * weight
            total_weight += weight

    if total_weight > 0:
        base_score = weighted_sum / total_weight
    else:
        base_score = 50.0

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 2: Amplifiers
    # ═════════════════════════════════════════════════════════════════════════

    amplifier = 0.0
    tags: list[str] = ["credit", "spreads"]

    # ── Velocity: speed of widening matters enormously ────────────────────
    # Sharp widening on multiple timeframes = crisis acceleration
    if hy_oas_5d_chg is not None:
        if hy_oas_5d_chg > 50:
            amplifier += 6
            tags.append("rapid_widening")
        elif hy_oas_5d_chg > 25:
            amplifier += 3
        elif hy_oas_5d_chg < -30:
            amplifier -= 3
            tags.append("rapid_tightening")

    if hy_oas_30d_chg is not None:
        if hy_oas_30d_chg > 100:
            amplifier += 8
            tags.append("crisis_acceleration")
        elif hy_oas_30d_chg > 50:
            amplifier += 4
        elif hy_oas_30d_chg < -50:
            amplifier -= 4

    # Multi-timeframe confirmation: 5d AND 30d both widening sharply
    if (hy_oas_5d_chg is not None and hy_oas_5d_chg > 25 and
            hy_oas_30d_chg is not None and hy_oas_30d_chg > 50):
        amplifier += 3  # momentum confirmation

    # ── Quality Dispersion: HY widening faster than IG ────────────────────
    # Lower quality deteriorating first = classic pre-crisis signal
    if hy_oas_30d_chg is not None and bbb_oas_30d_chg is not None:
        dispersion = hy_oas_30d_chg - bbb_oas_30d_chg
        if dispersion > 30:
            amplifier += 4
            tags.append("quality_deterioration")
        elif dispersion < -20:
            amplifier -= 2  # HY tightening faster than IG = risk-on

    # ── HY-IG Curve ───────────────────────────────────────────────────────
    hy_ig_spread: float | None = None
    if hy_oas is not None and bbb_oas is not None:
        hy_ig_spread = hy_oas - bbb_oas
        if hy_ig_spread > 400:
            amplifier += 3
        elif hy_ig_spread < 150:
            amplifier -= 2

    # ── Cross-Market Confirmation: VIX ────────────────────────────────────
    # If spreads widening AND VIX elevated → systemic stress confirmed
    # If spreads widening but VIX flat → may be idiosyncratic
    if vix is not None and hy_oas_30d_chg is not None:
        if hy_oas_30d_chg > 30 and vix > 25:
            amplifier += 4
            tags.append("cross_market_stress")
        elif hy_oas_30d_chg > 30 and vix < 15:
            amplifier -= 2  # spreads widening without vol → likely idiosyncratic
            tags.append("idiosyncratic")

    # ═════════════════════════════════════════════════════════════════════════
    # COMBINE AND LABEL
    # ═════════════════════════════════════════════════════════════════════════

    score = max(0, min(100, base_score + amplifier))

    if score >= 80:
        label = "Credit Crisis"
    elif score >= 65:
        label = "Credit Stress"
    elif score >= 55:
        label = "Credit Widening"
    elif score >= 40:
        label = "Credit Neutral"
    elif score >= 25:
        label = "Credit Tight"
    else:
        label = "Credit Euphoria"

    parts: list[str] = []
    if hy_oas is not None:
        pctl_str = ""
        if hy_oas_1y_percentile is not None:
            pctl_str = f" ({hy_oas_1y_percentile:.0f}th pctl 1Y)"
        elif hy_oas_90d_percentile is not None:
            pctl_str = f" ({hy_oas_90d_percentile:.0f}th pctl 90d)"
        parts.append(f"HY OAS: {hy_oas:.0f}bp{pctl_str}")
    if hy_oas_30d_chg is not None:
        parts.append(f"30d \u0394: {hy_oas_30d_chg:+.0f}bp")
    if bbb_oas is not None:
        parts.append(f"BBB: {bbb_oas:.0f}bp")
    if aaa_oas is not None:
        parts.append(f"AAA: {aaa_oas:.0f}bp")
    if hy_ig_spread is not None:
        parts.append(f"HY-IG: {hy_ig_spread:.0f}bp")

    return RegimeResult(
        name="credit",
        label=label,
        description=" | ".join(parts) if parts else "Insufficient data",
        score=score,
        domain="credit",
        tags=tags,
        metrics={
            "hy_oas": hy_oas,
            "bbb_oas": bbb_oas,
            "aaa_oas": aaa_oas,
            "hy_oas_30d_chg": hy_oas_30d_chg,
            "hy_ig_spread": hy_ig_spread,
            "hy_oas_5d_chg": hy_oas_5d_chg,
        },
    )

"""
Credit regime classifier — three-layer approach.

Layer 1 (Base Score): Spread level readings with interpolation.
    - HY OAS absolute level (primary signal)
    - BBB OAS level (IG benchmark)
    - HY percentile vs history (distributional context)

Layer 2 (Amplifiers): Momentum, dispersion, cross-market confirmation.
    - Velocity: 5d, 10d, 30d widening speed (crisis acceleration)
    - Dispersion: HY widening faster than IG = quality deterioration
    - Cross-market: VIX correlation (confirms systemic vs idiosyncratic)
    - HY-IG curve steepness

Layer 3 (Shadow Credit): Hidden stress in private credit / non-public markets.
    - CCC-BB quality spread: When weakest public credits widen vs BB,
      stress is too big for even private credit to absorb.
      When it's TIGHT + other signals flash, stress is hiding in private credit.
    - Credit card delinquency rate: Consumer stress leads corporate credit stress.
      The same households whose debt ends up in Ares/Apollo portfolios.
    - SLOOS lending standards: Banks tightening C&I lending pushes borrowers
      to private credit — the exact mechanism that creates hidden stress.

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
    # Layer 3: Shadow credit stress (private credit / hidden risk)
    ccc_oas: float | None = None,
    bb_oas: float | None = None,
    single_b_oas: float | None = None,
    cc_delinquency_rate: float | None = None,
    sloos_tightening: float | None = None,
) -> RegimeResult:
    """Classify the Credit regime using three-layer approach.

    Layer 1 — Base score from spread levels.
    Layer 2 — Amplifiers from velocity, dispersion, cross-market.
    Layer 3 — Shadow credit stress from quality tiers, consumer delinquency,
              and bank lending standards.

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
        ccc_oas: CCC & Lower OAS in bps (ICE BofA BAMLH0A3HYC).
        bb_oas: BB OAS in bps (ICE BofA BAMLH0A1HYBB).
        single_b_oas: Single-B OAS in bps (ICE BofA BAMLH0A2HYB).
        cc_delinquency_rate: Credit card delinquency rate % (FRED DRCCLACBS).
        sloos_tightening: Net % of banks tightening C&I lending (FRED DRTSCLCC).
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
    # LAYER 3: Shadow Credit Stress (private credit / hidden risk)
    # ═════════════════════════════════════════════════════════════════════════
    #
    # The weakest borrowers that historically blew out HY indices now sit in
    # Ares, Apollo, Owl Rock portfolios — marked quarterly by the lender.
    # This layer detects stress hiding in those structures.

    shadow_amp = 0.0
    shadow_signals = 0  # count how many shadow signals are flashing

    # ── CCC-BB Quality Spread ──────────────────────────────────────────
    # When CCC widens vs BB, stress is too big for even private credit
    # to absorb. When TIGHT + other signals, stress is hiding.
    ccc_bb_spread: float | None = None
    if ccc_oas is not None and bb_oas is not None:
        ccc_bb_spread = ccc_oas - bb_oas
        if ccc_bb_spread > 1200:
            # Distress: even public markets can't hide it
            shadow_amp += 8
            shadow_signals += 1
            tags.append("ccc_distress")
        elif ccc_bb_spread > 900:
            # Stress emerging in public quality tiers
            shadow_amp += 5
            shadow_signals += 1
            tags.append("quality_stress")
        elif ccc_bb_spread > 600:
            # Normal discrimination
            shadow_amp += 1
        elif ccc_bb_spread < 400:
            # VERY tight — either all is well, or worst credits migrated
            # to private credit. Ambiguous alone, but combined with other
            # shadow signals = hidden risk.
            shadow_amp -= 1
            tags.append("quality_compressed")

    # ── B-BB Spread (granular quality discrimination) ──────────────────
    b_bb_spread: float | None = None
    if single_b_oas is not None and bb_oas is not None:
        b_bb_spread = single_b_oas - bb_oas
        if b_bb_spread > 250:
            shadow_amp += 2
            shadow_signals += 1
        elif b_bb_spread < 80:
            # Extremely compressed — B credits trading like BB
            tags.append("b_bb_compressed")

    # ── Credit Card Delinquency Rate ───────────────────────────────────
    # Consumer stress leads corporate credit stress. The same households
    # whose debt ends up in private credit portfolios.
    if cc_delinquency_rate is not None:
        if cc_delinquency_rate > 5.0:
            shadow_amp += 6
            shadow_signals += 1
            tags.append("consumer_distress")
        elif cc_delinquency_rate > 3.5:
            shadow_amp += 3
            shadow_signals += 1
            tags.append("consumer_stress")
        elif cc_delinquency_rate > 2.5:
            shadow_amp += 1
        # Below 2.5% = healthy, no adjustment

    # ── SLOOS Lending Standards ────────────────────────────────────────
    # Banks tightening C&I lending pushes borrowers to private credit.
    # This is THE mechanism that creates hidden stress.
    if sloos_tightening is not None:
        if sloos_tightening > 40:
            # Crisis-level tightening: private credit is the only game in town
            shadow_amp += 6
            shadow_signals += 1
            tags.append("lending_crisis")
        elif sloos_tightening > 20:
            # Significant tightening: borrowers being pushed to shadow lenders
            shadow_amp += 3
            shadow_signals += 1
            tags.append("lending_tight")
        elif sloos_tightening > 0:
            shadow_amp += 1
        elif sloos_tightening < -10:
            # Easing: risk appetite returning
            shadow_amp -= 2

    # ── Multi-signal confirmation (the key insight) ────────────────────
    # Any single signal is ambiguous. Multiple signals = hidden stress.
    if shadow_signals >= 3:
        shadow_amp += 5
        tags.append("shadow_credit_stress")
    elif shadow_signals >= 2:
        shadow_amp += 2
        tags.append("shadow_credit_warning")

    # ── Paradox detection: tight HY + shadow stress = worst combo ──────
    # If public spreads are tight but shadow signals are flashing,
    # that's the Ares/Apollo/Owl Rock scenario: stress is HIDDEN.
    if hy_oas is not None and hy_oas < 350 and shadow_signals >= 2:
        shadow_amp += 4
        tags.append("hidden_stress")  # public calm, private cracks

    amplifier += shadow_amp

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
        label = "Credit Calm"
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
        parts.append(f"30d Δ: {hy_oas_30d_chg:+.0f}bp")
    if bbb_oas is not None:
        parts.append(f"BBB: {bbb_oas:.0f}bp")
    if aaa_oas is not None:
        parts.append(f"AAA: {aaa_oas:.0f}bp")
    if hy_ig_spread is not None:
        parts.append(f"HY-IG: {hy_ig_spread:.0f}bp")
    # Shadow credit metrics
    if ccc_bb_spread is not None:
        parts.append(f"CCC-BB: {ccc_bb_spread:.0f}bp")
    if cc_delinquency_rate is not None:
        parts.append(f"CC Delinq: {cc_delinquency_rate:.1f}%")
    if sloos_tightening is not None:
        parts.append(f"SLOOS: {sloos_tightening:+.0f}%")

    return RegimeResult(
        name="credit",
        label=label,
        description=" | ".join(parts) if parts else "Insufficient data",
        score=score,
        domain="credit",
        tags=tags,
        metrics={
            "HY OAS": f"{hy_oas:.0f}bp" if hy_oas is not None else None,
            "BBB OAS": f"{bbb_oas:.0f}bp" if bbb_oas is not None else None,
            "AAA OAS": f"{aaa_oas:.0f}bp" if aaa_oas is not None else None,
            "HY 30d Chg": f"{hy_oas_30d_chg:+.0f}bp" if hy_oas_30d_chg is not None else None,
            "HY-IG": f"{hy_ig_spread:.0f}bp" if hy_ig_spread is not None else None,
            "HY 5d Chg": f"{hy_oas_5d_chg:+.0f}bp" if hy_oas_5d_chg is not None else None,
            "CCC OAS": f"{ccc_oas:.0f}bp" if ccc_oas is not None else None,
            "BB OAS": f"{bb_oas:.0f}bp" if bb_oas is not None else None,
            "B OAS": f"{single_b_oas:.0f}bp" if single_b_oas is not None else None,
            "CCC-BB": f"{ccc_bb_spread:.0f}bp" if ccc_bb_spread is not None else None,
            "B-BB": f"{b_bb_spread:.0f}bp" if b_bb_spread is not None else None,
            "CC Delinq": f"{cc_delinquency_rate:.1f}%" if cc_delinquency_rate is not None else None,
            "SLOOS": f"{sloos_tightening:+.0f}%" if sloos_tightening is not None else None,
            "Shadow Sigs": f"{shadow_signals}" if shadow_signals is not None else None,
        },
    )

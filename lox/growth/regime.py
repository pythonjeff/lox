"""
Growth regime classifier — two-layer approach.

Layer 1 (Base Score): Activity level readings with interpolation.
    - Payrolls, ISM Manufacturing, Initial Claims, Industrial Production
    - Weighted average mapped to 0-100 through calibrated thresholds.

Layer 2 (Amplifiers): Momentum, labor breadth, leading indicators.
    - Payroll momentum: 1m vs 3m trend (deteriorating/improving)
    - Unemployment: level + direction (Sahm-rule inspired)
    - Claims momentum: 4wk vs 13wk avg (early warning)
    - Leading indicators: LEI, ISM New Orders when available

Score 0 = boom / strong risk-on → 100 = contraction / risk-off.
"""
from __future__ import annotations

from lox.regimes.base import RegimeResult


def _interp(value: float, low: float, high: float, score_low: float, score_high: float) -> float:
    """Linear interpolation of score between two thresholds."""
    if high == low:
        return score_high
    ratio = max(0.0, min(1.0, (value - low) / (high - low)))
    return score_low + ratio * (score_high - score_low)


def classify_growth(
    payrolls_3m_ann: float | None,
    ism: float | None,
    claims_4wk: float | None,
    indpro_yoy: float | None,
    *,
    # New Layer 2 inputs (momentum / breadth)
    payrolls_mom: float | None = None,
    unemployment_rate: float | None = None,
    claims_13wk: float | None = None,
    # New Layer 2 inputs (leading indicators)
    lei_yoy: float | None = None,
    ism_new_orders: float | None = None,
) -> RegimeResult:
    """Classify the Growth regime using two-layer approach.

    Layer 1 — Base score from activity level readings.
    Layer 2 — Amplifiers from momentum, labor breadth, leading indicators.

    Args:
        payrolls_3m_ann: Nonfarm payrolls 3-month annualized level change
                         (thousands of jobs per month, e.g. +200 means 200K/mo).
        ism: ISM Manufacturing PMI (50 = expansion/contraction threshold).
        claims_4wk: Initial jobless claims 4-week average (absolute, e.g. 220_000).
        indpro_yoy: Industrial Production Index year-over-year % change.
        payrolls_mom: Payrolls month-over-month % change.
        unemployment_rate: Unemployment rate (UNRATE, %).
        claims_13wk: Initial claims 13-week average (for momentum).
        lei_yoy: Conference Board LEI YoY % change.
        ism_new_orders: ISM New Orders sub-index (leads headline by 2-3 months).
    """

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 1: Base score from activity readings
    # ═════════════════════════════════════════════════════════════════════════

    def _payrolls_score(val: float | None) -> float | None:
        """Map payrolls 3m ann (K/mo) to 0-100. Higher payrolls → lower risk score."""
        if val is None:
            return None
        if val <= -100:
            return 95.0
        if val <= 0:
            return _interp(val, -100, 0, 95, 75)
        if val <= 50:
            return _interp(val, 0, 50, 75, 60)
        if val <= 150:
            return _interp(val, 50, 150, 60, 40)
        if val <= 250:
            return _interp(val, 150, 250, 40, 25)
        if val <= 400:
            return _interp(val, 250, 400, 25, 10)
        return 5.0

    def _ism_score(val: float | None) -> float | None:
        """Map ISM PMI to 0-100. Below 50 = contraction → higher risk."""
        if val is None:
            return None
        if val <= 42:
            return 95.0
        if val <= 47:
            return _interp(val, 42, 47, 95, 78)
        if val <= 50:
            return _interp(val, 47, 50, 78, 60)
        if val <= 53:
            return _interp(val, 50, 53, 60, 45)
        if val <= 57:
            return _interp(val, 53, 57, 45, 30)
        if val <= 62:
            return _interp(val, 57, 62, 30, 15)
        return 10.0

    def _claims_score(val: float | None) -> float | None:
        """Map initial claims 4wk avg to 0-100. Higher claims → higher risk."""
        if val is None:
            return None
        if val <= 180_000:
            return 15.0
        if val <= 220_000:
            return _interp(val, 180_000, 220_000, 15, 35)
        if val <= 260_000:
            return _interp(val, 220_000, 260_000, 35, 50)
        if val <= 350_000:
            return _interp(val, 260_000, 350_000, 50, 70)
        if val <= 500_000:
            return _interp(val, 350_000, 500_000, 70, 90)
        return 95.0

    def _indpro_score(val: float | None) -> float | None:
        """Map industrial production YoY % to 0-100."""
        if val is None:
            return None
        if val <= -5:
            return 92.0
        if val <= -2:
            return _interp(val, -5, -2, 92, 72)
        if val <= 0:
            return _interp(val, -2, 0, 72, 55)
        if val <= 2:
            return _interp(val, 0, 2, 55, 40)
        if val <= 5:
            return _interp(val, 2, 5, 40, 25)
        return 15.0

    measures: list[tuple[float | None, float]] = [
        (_payrolls_score(payrolls_3m_ann), 0.35),  # labor market is primary
        (_ism_score(ism), 0.25),                    # manufacturing + survey
        (_claims_score(claims_4wk), 0.25),           # high frequency early warning
        (_indpro_score(indpro_yoy), 0.15),            # hard data confirmation
    ]

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
    tags: list[str] = ["growth", "cycle"]

    # ── Payroll Momentum: MoM trend vs 3m ─────────────────────────────────
    # If monthly payrolls are deteriorating vs 3m trend → early warning
    if payrolls_mom is not None and payrolls_3m_ann is not None:
        # payrolls_mom is a pct change; approximate monthly K/mo
        # If MoM is significantly weaker than 3m average → deteriorating
        if payrolls_mom < 0 and payrolls_3m_ann > 0:
            amplifier += 5  # monthly went negative while 3m still positive
            tags.append("payrolls_deteriorating")
        elif payrolls_mom > 0 and payrolls_3m_ann < 0:
            amplifier -= 4  # monthly inflecting positive
            tags.append("payrolls_inflecting")

    # ── Unemployment Level + Direction (Sahm-rule inspired) ───────────────
    # Sahm: recession signal when 3m avg unemployment rises 0.5pp from 12m low.
    # We simplify: level + context.
    if unemployment_rate is not None:
        if unemployment_rate > 5.0:
            amplifier += 8
            tags.append("unemployment_elevated")
        elif unemployment_rate > 4.5:
            amplifier += 5
        elif unemployment_rate > 4.0:
            amplifier += 2
        elif unemployment_rate < 3.5:
            amplifier -= 3  # very tight labor market

    # ── Claims Momentum: 4wk vs 13wk ─────────────────────────────────────
    # Rising claims above trend = early recession signal
    if claims_4wk is not None and claims_13wk is not None:
        claims_pct_above = (claims_4wk / claims_13wk - 1.0) * 100 if claims_13wk > 0 else 0
        if claims_pct_above > 10:
            amplifier += 6
            tags.append("claims_spiking")
        elif claims_pct_above > 5:
            amplifier += 3
            tags.append("claims_rising")
        elif claims_pct_above < -5:
            amplifier -= 3
            tags.append("claims_improving")

    # ── Leading Indicator: Conference Board LEI ───────────────────────────
    if lei_yoy is not None:
        if lei_yoy < -5:
            amplifier += 6
            tags.append("lei_recessionary")
        elif lei_yoy < -2:
            amplifier += 3
            tags.append("lei_weakening")
        elif lei_yoy > 3:
            amplifier -= 4
            tags.append("lei_expanding")

    # ── ISM New Orders (most forward-looking component) ───────────────────
    if ism_new_orders is not None:
        if ism_new_orders < 45:
            amplifier += 5
            tags.append("new_orders_contracting")
        elif ism_new_orders < 48:
            amplifier += 2
        elif ism_new_orders > 57:
            amplifier -= 4
            tags.append("new_orders_booming")
        elif ism_new_orders > 53:
            amplifier -= 2

    # ═════════════════════════════════════════════════════════════════════════
    # COMBINE AND LABEL
    # ═════════════════════════════════════════════════════════════════════════

    score = max(0, min(100, base_score + amplifier))

    if score >= 75:
        label = "Contraction"
    elif score >= 60:
        label = "Slowing"
    elif score >= 45:
        label = "Stable Growth"
    elif score >= 30:
        label = "Accelerating"
    else:
        label = "Boom"

    parts: list[str] = []
    if payrolls_3m_ann is not None:
        parts.append(f"Payrolls 3m ann: {payrolls_3m_ann:+,.0f}K")
    if ism is not None:
        parts.append(f"ISM: {ism:.1f}")
    if claims_4wk is not None:
        parts.append(f"Claims 4wk: {claims_4wk:,.0f}")
    if indpro_yoy is not None:
        parts.append(f"IndProd YoY: {indpro_yoy:+.1f}%")
    if unemployment_rate is not None:
        parts.append(f"UNRATE: {unemployment_rate:.1f}%")

    return RegimeResult(
        name="growth",
        label=label,
        description=" | ".join(parts) if parts else "Insufficient data",
        score=score,
        domain="growth",
        tags=tags,
        metrics={
            "payrolls_3m_ann": payrolls_3m_ann,
            "ism": ism,
            "claims_4wk": claims_4wk,
            "indpro_yoy": indpro_yoy,
            "unemployment_rate": unemployment_rate,
            "lei_yoy": lei_yoy,
        },
    )

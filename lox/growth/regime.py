"""
Growth regime classifier — split from the old Macro regime.

Focuses on real-economy activity: payrolls, ISM, claims, industrial production.
Score 0 = boom / strong risk-on → 100 = contraction / risk-off.
"""
from __future__ import annotations

from lox.regimes.base import RegimeResult


def classify_growth(
    payrolls_3m_ann: float | None,
    ism: float | None,
    claims_4wk: float | None,
    indpro_yoy: float | None,
) -> RegimeResult:
    """Classify the Growth regime from real-economy indicators.

    Args:
        payrolls_3m_ann: Nonfarm payrolls 3-month annualized *level* change
                         (thousands of jobs per month, e.g. +200 means 200K/mo).
        ism: ISM Manufacturing PMI (50 = expansion/contraction threshold).
        claims_4wk: Initial jobless claims 4-week average (absolute, e.g. 220_000).
        indpro_yoy: Industrial Production Index year-over-year % change.
    """
    score = 50  # neutral baseline

    # ── Payrolls ──────────────────────────────────────────────────────────
    if payrolls_3m_ann is not None:
        if payrolls_3m_ann > 200:
            score -= 15  # strong growth → lower risk score
        elif payrolls_3m_ann < 0:
            score += 20  # contraction → higher risk score
        elif payrolls_3m_ann < 50:
            score += 10  # tepid

    # ── ISM Manufacturing PMI ─────────────────────────────────────────────
    if ism is not None:
        if ism > 55:
            score -= 10
        elif ism < 48:
            score += 15
        elif ism < 50:
            score += 5

    # ── Initial Claims ────────────────────────────────────────────────────
    if claims_4wk is not None:
        if claims_4wk > 300_000:
            score += 10
        elif claims_4wk < 200_000:
            score -= 5

    # ── Industrial Production YoY ─────────────────────────────────────────
    if indpro_yoy is not None:
        if indpro_yoy < -2:
            score += 10
        elif indpro_yoy > 3:
            score -= 5

    score = max(0, min(100, score))

    if score >= 70:
        label = "Contraction"
    elif score >= 55:
        label = "Slowing"
    elif score >= 45:
        label = "Stable Growth"
    elif score >= 30:
        label = "Accelerating"
    else:
        label = "Boom"

    # Build description from available inputs
    parts: list[str] = []
    if payrolls_3m_ann is not None:
        parts.append(f"Payrolls 3m ann: {payrolls_3m_ann:+,.0f}K")
    if ism is not None:
        parts.append(f"ISM: {ism:.1f}")
    if claims_4wk is not None:
        parts.append(f"Claims 4wk: {claims_4wk:,.0f}")
    if indpro_yoy is not None:
        parts.append(f"IndProd YoY: {indpro_yoy:+.1f}%")

    return RegimeResult(
        name="growth",
        label=label,
        description=" | ".join(parts) if parts else "Insufficient data",
        score=score,
        domain="growth",
        tags=["growth", "cycle"],
        metrics={
            "payrolls_3m_ann": payrolls_3m_ann,
            "ism": ism,
            "claims_4wk": claims_4wk,
            "indpro_yoy": indpro_yoy,
        },
    )

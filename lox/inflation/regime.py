"""
Inflation regime classifier — split from the old Macro regime.

Focuses on price-level dynamics: CPI, Core PCE, breakevens, PPI.
Score 0 = deflationary / risk-on → 100 = hot inflation / risk-off.
"""
from __future__ import annotations

from lox.regimes.base import RegimeResult


def classify_inflation(
    cpi_yoy: float | None,
    core_pce_yoy: float | None,
    breakeven_5y: float | None,
    ppi_yoy: float | None,
) -> RegimeResult:
    """Classify the Inflation regime.

    Args:
        cpi_yoy: CPI All Urban YoY % change.
        core_pce_yoy: Core PCE YoY % change (Fed's preferred).
        breakeven_5y: 5-year breakeven inflation rate (T5YIE).
        ppi_yoy: PPI Final Demand YoY % change.
    """
    score = 50

    # ── CPI YoY ───────────────────────────────────────────────────────────
    if cpi_yoy is not None:
        if cpi_yoy > 4.0:
            score += 20
        elif cpi_yoy > 3.0:
            score += 10
        elif cpi_yoy < 1.5:
            score -= 15
        elif cpi_yoy < 2.0:
            score -= 5

    # ── Core PCE (Fed target = 2%) ────────────────────────────────────────
    if core_pce_yoy is not None:
        if core_pce_yoy > 3.0:
            score += 10
        elif core_pce_yoy < 1.5:
            score -= 10

    # ── 5Y Breakevens (market pricing) ────────────────────────────────────
    if breakeven_5y is not None:
        if breakeven_5y > 3.0:
            score += 10
        elif breakeven_5y < 1.5:
            score -= 10

    # ── PPI (leading indicator for CPI) ───────────────────────────────────
    if ppi_yoy is not None:
        if ppi_yoy > 5.0:
            score += 5
        elif ppi_yoy < 0:
            score -= 5

    score = max(0, min(100, score))

    if score >= 75:
        label = "Hot Inflation"
    elif score >= 60:
        label = "Above Target"
    elif score >= 40:
        label = "At Target"
    elif score >= 25:
        label = "Below Target"
    else:
        label = "Deflationary"

    parts: list[str] = []
    if cpi_yoy is not None:
        parts.append(f"CPI YoY: {cpi_yoy:.1f}%")
    if core_pce_yoy is not None:
        parts.append(f"Core PCE: {core_pce_yoy:.1f}%")
    if breakeven_5y is not None:
        parts.append(f"5Y BE: {breakeven_5y:.2f}%")
    if ppi_yoy is not None:
        parts.append(f"PPI YoY: {ppi_yoy:+.1f}%")

    return RegimeResult(
        name="inflation",
        label=label,
        description=" | ".join(parts) if parts else "Insufficient data",
        score=score,
        domain="inflation",
        tags=["inflation", "prices"],
        metrics={
            "cpi_yoy": cpi_yoy,
            "core_pce_yoy": core_pce_yoy,
            "breakeven_5y": breakeven_5y,
            "ppi_yoy": ppi_yoy,
        },
    )

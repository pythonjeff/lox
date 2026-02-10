"""
Credit regime classifier — NEW domain.

Monitors corporate credit spreads (HY OAS, IG OAS) and rate of change.
Credit leads equity vol, making this one of the most important regimes.
Score 0 = euphoria / tight spreads → 100 = credit stress / wide spreads.
"""
from __future__ import annotations

from lox.regimes.base import RegimeResult


def classify_credit(
    hy_oas: float | None,
    bbb_oas: float | None,
    aaa_oas: float | None,
    hy_oas_30d_chg: float | None,
    hy_oas_90d_percentile: float | None,
) -> RegimeResult:
    """Classify the Credit regime from corporate spread data.

    Args:
        hy_oas: Current HY OAS in bps (typical 300-800, crisis 1000+).
        bbb_oas: Current BBB OAS in bps.
        aaa_oas: Current AAA OAS in bps.
        hy_oas_30d_chg: Change in HY OAS over last 30 days (positive = widening).
        hy_oas_90d_percentile: Where current HY OAS sits in last 90 days (0-100).
    """
    score = 50

    # ── Absolute level of HY spreads ──────────────────────────────────────
    if hy_oas is not None:
        if hy_oas > 700:
            score += 25  # crisis territory
        elif hy_oas > 500:
            score += 15  # stressed
        elif hy_oas > 400:
            score += 5   # elevated
        elif hy_oas < 300:
            score -= 15  # very tight (complacency risk)
        elif hy_oas < 350:
            score -= 5   # tight

    # ── Rate of change ────────────────────────────────────────────────────
    if hy_oas_30d_chg is not None:
        if hy_oas_30d_chg > 100:
            score += 20  # rapid widening
        elif hy_oas_30d_chg > 50:
            score += 10  # meaningful widening
        elif hy_oas_30d_chg < -50:
            score -= 10  # rapid tightening

    # ── Credit curve: HY - IG spread ──────────────────────────────────────
    hy_ig_spread: float | None = None
    if hy_oas is not None and bbb_oas is not None:
        hy_ig_spread = hy_oas - bbb_oas
        if hy_ig_spread > 400:
            score += 5
        elif hy_ig_spread < 200:
            score -= 5

    score = max(0, min(100, score))

    if score >= 75:
        label = "Credit Stress"
    elif score >= 60:
        label = "Credit Widening"
    elif score >= 40:
        label = "Credit Neutral"
    elif score >= 25:
        label = "Credit Tight"
    else:
        label = "Credit Euphoria"

    parts: list[str] = []
    if hy_oas is not None:
        pctl_str = f" ({hy_oas_90d_percentile:.0f}th pctl)" if hy_oas_90d_percentile is not None else ""
        parts.append(f"HY OAS: {hy_oas:.0f}bp{pctl_str}")
    if hy_oas_30d_chg is not None:
        parts.append(f"30d Δ: {hy_oas_30d_chg:+.0f}bp")
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
        tags=["credit", "spreads"],
        metrics={
            "hy_oas": hy_oas,
            "bbb_oas": bbb_oas,
            "aaa_oas": aaa_oas,
            "hy_oas_30d_chg": hy_oas_30d_chg,
            "hy_ig_spread": hy_ig_spread,
        },
    )

"""
Positioning regime classifier — NEW domain.

Monitors market positioning: VIX term structure, put/call ratio, sentiment surveys.
Score 0 = extreme complacency → 100 = panic positioning.
"""
from __future__ import annotations

from lox.regimes.base import RegimeResult


def classify_positioning(
    vix_term_slope: float | None,
    put_call_ratio: float | None,
    aaii_bull_pct: float | None = None,
) -> RegimeResult:
    """Classify the Positioning regime.

    Args:
        vix_term_slope: VIX3M / VIX (>1 = contango/normal, <1 = backwardation/panic).
        put_call_ratio: Equity put/call ratio (typical 0.6-1.0).
        aaii_bull_pct: AAII % bullish (typical 25-45%).
    """
    score = 50

    # ── VIX Term Structure (THE most important positioning signal) ────────
    if vix_term_slope is not None:
        if vix_term_slope < 0.85:
            score += 25  # deep backwardation = panic
        elif vix_term_slope < 0.95:
            score += 10  # mild backwardation = caution
        elif vix_term_slope > 1.15:
            score -= 10  # steep contango = complacency

    # ── Put/Call Ratio ────────────────────────────────────────────────────
    if put_call_ratio is not None:
        if put_call_ratio > 1.1:
            score += 10  # heavy put buying = fear
        elif put_call_ratio < 0.5:
            score -= 10  # low put buying = complacency

    # ── AAII Sentiment ────────────────────────────────────────────────────
    contrarian_tag: str | None = None
    if aaii_bull_pct is not None:
        if aaii_bull_pct > 55:
            score -= 10  # extreme bullishness = contrarian warning
            contrarian_tag = "crowded_long"
        elif aaii_bull_pct < 20:
            score += 10  # extreme bearishness = current fear
            contrarian_tag = "crowded_short"

    score = max(0, min(100, score))

    tags = ["positioning", "sentiment"]
    if contrarian_tag:
        tags.append(contrarian_tag)

    if score >= 70:
        label = "Panic Positioning"
    elif score >= 55:
        label = "Defensive Positioning"
    elif score >= 45:
        label = "Neutral Positioning"
    elif score >= 30:
        label = "Complacent"
    else:
        label = "Extreme Complacency"

    parts: list[str] = []
    if vix_term_slope is not None:
        shape = "backwardation ⚠️" if vix_term_slope < 1 else "contango"
        parts.append(f"VIX Term: {vix_term_slope:.2f}x ({shape})")
    if put_call_ratio is not None:
        parts.append(f"P/C Ratio: {put_call_ratio:.2f}")
    if aaii_bull_pct is not None:
        parts.append(f"AAII Bull: {aaii_bull_pct:.0f}%")

    return RegimeResult(
        name="positioning",
        label=label,
        description=" | ".join(parts) if parts else "Insufficient data",
        score=score,
        domain="positioning",
        tags=tags,
        metrics={
            "vix_term_slope": vix_term_slope,
            "put_call_ratio": put_call_ratio,
            "aaii_bull_pct": aaii_bull_pct,
        },
    )

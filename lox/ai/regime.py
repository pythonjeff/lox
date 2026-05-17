"""
AI bubble regime classifier (v0).

Score interpretation:
  0   — no AI trade / unwind already done
  50  — strong AI trade, no froth
  75+ — blow-off / melt-up
  100 — blow-off WITH visible cracks (breadth narrowing while leaders hold)

The interesting regime we want to flag is "high score + breadth deterioration":
late-stage bubble where the index keeps printing highs on a shrinking number
of names.  That's the setup that historically precedes nasty resets.
"""
from __future__ import annotations

from lox.regimes.base import RegimeResult


def classify_ai(
    basket_ytd_excess: float | None,        # AI basket YTD return minus SPY YTD (pp)
    basket_3m_excess: float | None,         # 3-month excess return vs SPY (pp)
    pct_above_50dma: float | None,          # % of basket above its 50-day MA (0-100)
    pct_above_200dma: float | None,         # % of basket above its 200-day MA (0-100)
    avg_drawdown_from_52w: float | None,    # avg drawdown from 52w high across basket (pp, negative)
    chip_vs_power_spread: float | None,     # chips 3M return minus power 3M return (pp)
    vol_ratio_vs_spy: float | None,         # basket 20d realized vol / SPY 20d realized vol
    hyperscaler_capex_yoy: float | None = None,   # aggregate TTM capex YoY % across the 5 spenders
    capex_to_ocf_pct: float | None = None,        # aggregate capex / operating cash flow % (TTM)
) -> RegimeResult:
    """Classify the AI bubble regime from price-only inputs."""
    score = 30  # baseline: AI trade exists at all

    # ── Excess return: bubble fuel ───────────────────────────────────────
    if basket_ytd_excess is not None:
        if basket_ytd_excess > 40:
            score += 25  # blow-off territory
        elif basket_ytd_excess > 20:
            score += 15
        elif basket_ytd_excess > 10:
            score += 8
        elif basket_ytd_excess < -10:
            score -= 15  # AI trade unwinding

    if basket_3m_excess is not None:
        if basket_3m_excess > 15:
            score += 10  # short-term melt-up
        elif basket_3m_excess < -10:
            score -= 10

    # ── Vol ratio: overheating ──────────────────────────────────────────
    if vol_ratio_vs_spy is not None:
        if vol_ratio_vs_spy > 2.5:
            score += 10  # basket vol blowing out vs market
        elif vol_ratio_vs_spy > 1.8:
            score += 5

    # ── Cracks: breadth deterioration ───────────────────────────────────
    cracks_score = 0
    cracks: list[str] = []
    if pct_above_50dma is not None and pct_above_50dma < 50:
        cracks_score += 8
        cracks.append(f"only {pct_above_50dma:.0f}% above 50dma")
    if pct_above_200dma is not None and pct_above_200dma < 60:
        cracks_score += 10
        cracks.append(f"only {pct_above_200dma:.0f}% above 200dma")
    if avg_drawdown_from_52w is not None and avg_drawdown_from_52w < -12:
        cracks_score += 8
        cracks.append(f"avg DD {avg_drawdown_from_52w:.0f}% from 52w high")

    # Chip-vs-power divergence: power rolls before chips
    if chip_vs_power_spread is not None and chip_vs_power_spread > 20:
        cracks_score += 8
        cracks.append(f"power lagging chips by {chip_vs_power_spread:.0f}pp 3M")

    # ── Hyperscaler capex (v1 hard data) ────────────────────────────────
    # Extreme capex growth is bubble fuel; capex eating cash flow is a crack.
    if hyperscaler_capex_yoy is not None:
        if hyperscaler_capex_yoy > 60:
            score += 12   # parabolic capex — late-cycle
        elif hyperscaler_capex_yoy > 35:
            score += 7
        elif hyperscaler_capex_yoy < 0:
            score -= 8    # capex contracting = trade rolling

    if capex_to_ocf_pct is not None:
        if capex_to_ocf_pct > 95:
            cracks_score += 12
            cracks.append(f"capex consuming {capex_to_ocf_pct:.0f}% of OCF — debt-funded")
        elif capex_to_ocf_pct > 75:
            cracks_score += 6
            cracks.append(f"capex at {capex_to_ocf_pct:.0f}% of OCF — stretched")

    score += cracks_score

    # Late-stage bubble = high price action AND cracks emerging
    blowoff = basket_ytd_excess is not None and basket_ytd_excess > 20
    has_cracks = cracks_score >= 16

    tags = ["ai"]
    if blowoff:
        tags.append("blowoff")
    if has_cracks:
        tags.append("cracks")

    score = max(0, min(100, score))

    if blowoff and has_cracks:
        label = "Blow-off with Cracks"
    elif score >= 75:
        label = "Late-stage Blow-off"
    elif score >= 60:
        label = "Strong AI Bid"
    elif score >= 45:
        label = "AI Trade Active"
    elif score >= 30:
        label = "AI Cooling"
    else:
        label = "AI Trade Off"

    parts: list[str] = []
    if basket_ytd_excess is not None:
        parts.append(f"YTD vs SPY: {basket_ytd_excess:+.0f}pp")
    if pct_above_200dma is not None:
        parts.append(f"breadth(200d): {pct_above_200dma:.0f}%")
    if avg_drawdown_from_52w is not None:
        parts.append(f"avg DD: {avg_drawdown_from_52w:.0f}%")
    if cracks:
        parts.append("cracks: " + ", ".join(cracks))

    return RegimeResult(
        name="ai",
        label=label,
        description=" | ".join(parts) if parts else "Insufficient data",
        score=score,
        domain="ai",
        tags=tags,
        metrics={
            "YTD vs SPY": f"{basket_ytd_excess:+.1f}pp" if basket_ytd_excess is not None else None,
            "3M vs SPY": f"{basket_3m_excess:+.1f}pp" if basket_3m_excess is not None else None,
            "% > 50dma": f"{pct_above_50dma:.0f}%" if pct_above_50dma is not None else None,
            "% > 200dma": f"{pct_above_200dma:.0f}%" if pct_above_200dma is not None else None,
            "Avg DD": f"{avg_drawdown_from_52w:.1f}%" if avg_drawdown_from_52w is not None else None,
            "Vol ratio": f"{vol_ratio_vs_spy:.2f}x" if vol_ratio_vs_spy is not None else None,
        },
    )

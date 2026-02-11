"""
Inflation regime classifier — two-layer approach.

Layer 1 (Base Score): Actual inflation readings across measures.
    - CPI YoY, Core CPI, Core PCE, Trimmed Mean PCE, Median CPI
    - Weighted average with interpolation between thresholds.

Layer 2 (Amplifiers): Momentum, breadth, expectations, supply pipeline.
    - Momentum: CPI 3m ann vs 6m ann vs YoY (accelerating / decelerating)
    - Breadth: Median CPI vs headline (broad-based or concentrated)
    - Expectations: 5y5y forward anchoring, breakeven slope
    - Supply pipeline: PPI > CPI gap, oil momentum

Score 0 = deflationary / risk-on → 100 = hot inflation / risk-off.
"""
from __future__ import annotations

from lox.regimes.base import RegimeResult


def _interp(value: float, low: float, high: float, score_low: float, score_high: float) -> float:
    """Linear interpolation of score between two thresholds."""
    if high == low:
        return score_high
    ratio = max(0.0, min(1.0, (value - low) / (high - low)))
    return score_low + ratio * (score_high - score_low)


def classify_inflation(
    cpi_yoy: float | None,
    core_pce_yoy: float | None,
    breakeven_5y: float | None,
    ppi_yoy: float | None,
    *,
    # New Layer 1 inputs (breadth measures)
    core_cpi_yoy: float | None = None,
    trimmed_mean_pce_yoy: float | None = None,
    median_cpi_yoy: float | None = None,
    # New Layer 2 inputs (momentum)
    cpi_3m_ann: float | None = None,
    cpi_6m_ann: float | None = None,
    # New Layer 2 inputs (expectations)
    breakeven_5y5y: float | None = None,
    breakeven_10y: float | None = None,
    # New Layer 2 inputs (supply pipeline)
    oil_price_yoy_pct: float | None = None,
) -> RegimeResult:
    """Classify the Inflation regime using two-layer approach.

    Layer 1 — Base score from actual inflation readings.
    Layer 2 — Amplifiers from momentum, breadth, expectations, supply.

    Args:
        cpi_yoy: CPI All Urban YoY % change.
        core_pce_yoy: Core PCE YoY % change (Fed's preferred).
        breakeven_5y: 5-year breakeven inflation rate (T5YIE).
        ppi_yoy: PPI Final Demand YoY % change.
        core_cpi_yoy: Core CPI (ex food & energy) YoY %.
        trimmed_mean_pce_yoy: Dallas Fed Trimmed Mean PCE YoY %.
        median_cpi_yoy: Cleveland Fed Median CPI YoY %.
        cpi_3m_ann: CPI 3-month annualized rate.
        cpi_6m_ann: CPI 6-month annualized rate.
        breakeven_5y5y: 5y5y forward inflation expectation rate.
        breakeven_10y: 10-year breakeven inflation rate.
        oil_price_yoy_pct: WTI crude oil YoY % change.
    """

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 1: Base score from actual inflation readings
    # ═════════════════════════════════════════════════════════════════════════
    #
    # Each measure maps to a sub-score (0-100). We take a weighted average.
    # Thresholds: <1.0% → 10, 2.0% → 40 (target), 3.0% → 60, 4.0% → 75, >5.5% → 95

    def _level_score(value: float | None) -> float | None:
        """Map an inflation YoY reading to a 0-100 sub-score."""
        if value is None:
            return None
        if value <= 0.0:
            return 5.0
        if value <= 1.0:
            return _interp(value, 0.0, 1.0, 5, 15)
        if value <= 2.0:
            return _interp(value, 1.0, 2.0, 15, 40)
        if value <= 3.0:
            return _interp(value, 2.0, 3.0, 40, 60)
        if value <= 4.0:
            return _interp(value, 3.0, 4.0, 60, 75)
        if value <= 5.5:
            return _interp(value, 4.0, 5.5, 75, 92)
        return min(98.0, 92 + (value - 5.5) * 2)

    # Weighted contributions (weights sum to 1.0)
    measures: list[tuple[float | None, float]] = [
        (_level_score(cpi_yoy), 0.25),          # headline CPI
        (_level_score(core_pce_yoy), 0.25),      # Fed's preferred
        (_level_score(core_cpi_yoy), 0.15),       # core CPI
        (_level_score(trimmed_mean_pce_yoy), 0.15),  # trimmed mean (outlier-robust)
        (_level_score(median_cpi_yoy), 0.10),     # median CPI (stickiness)
        (_level_score(ppi_yoy), 0.10),            # PPI (leading / pipeline)
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
        base_score = 50.0  # no data fallback

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 2: Amplifiers (shift base score up/down)
    # ═════════════════════════════════════════════════════════════════════════

    amplifier = 0.0
    tags: list[str] = ["inflation", "prices"]

    # ── Momentum: CPI 3m vs 6m vs YoY ─────────────────────────────────────
    # If short-term inflation > longer-term → re-accelerating
    # If short-term inflation < longer-term → decelerating
    if cpi_3m_ann is not None and cpi_6m_ann is not None and cpi_yoy is not None:
        # 3m vs YoY spread: positive = accelerating
        accel_spread = cpi_3m_ann - cpi_yoy
        if accel_spread > 1.5:
            amplifier += 8   # strongly re-accelerating
            tags.append("re_accelerating")
        elif accel_spread > 0.5:
            amplifier += 4
            tags.append("accelerating")
        elif accel_spread < -1.5:
            amplifier -= 8   # strongly decelerating
            tags.append("decelerating_fast")
        elif accel_spread < -0.5:
            amplifier -= 4
            tags.append("decelerating")

        # Additional: 3m vs 6m (very near-term trend)
        near_spread = cpi_3m_ann - cpi_6m_ann
        if near_spread > 1.0:
            amplifier += 3   # inflecting higher
        elif near_spread < -1.0:
            amplifier -= 3   # inflecting lower

    # ── Breadth: Median CPI vs Headline ────────────────────────────────────
    # If median > headline → inflation is broad-based (worse)
    # If median << headline → concentrated in outlier categories (better)
    if median_cpi_yoy is not None and cpi_yoy is not None:
        breadth_gap = median_cpi_yoy - cpi_yoy
        if breadth_gap > 0.5:
            amplifier += 4
            tags.append("broad_based")
        elif breadth_gap < -1.0:
            amplifier -= 3
            tags.append("concentrated")

    # ── Expectations Anchoring ─────────────────────────────────────────────
    # 5y5y forward: the market's expectation of inflation 5-10 years from now.
    # Well-anchored = near 2.0-2.3%. Unanchoring = >2.5% or rising.
    if breakeven_5y5y is not None:
        if breakeven_5y5y > 2.8:
            amplifier += 6
            tags.append("expectations_unanchored")
        elif breakeven_5y5y > 2.5:
            amplifier += 3
            tags.append("expectations_drifting")
        elif breakeven_5y5y < 1.8:
            amplifier -= 4
            tags.append("expectations_deflationary")

    # Breakeven slope: 10Y BE - 5Y BE.
    # Positive = market expects inflation to persist/rise longer-term.
    # Negative = market expects inflation to moderate.
    if breakeven_10y is not None and breakeven_5y is not None:
        be_slope = breakeven_10y - breakeven_5y
        if be_slope > 0.3:
            amplifier += 3   # market expects persistence
        elif be_slope < -0.3:
            amplifier -= 3   # market expects moderation

    # ── Supply Pipeline: PPI-CPI gap + oil ─────────────────────────────────
    # PPI running above CPI = cost pressure building in the pipeline.
    if ppi_yoy is not None and cpi_yoy is not None:
        pipeline_gap = ppi_yoy - cpi_yoy
        if pipeline_gap > 2.0:
            amplifier += 4
            tags.append("pipeline_pressure")
        elif pipeline_gap < -2.0:
            amplifier -= 3
            tags.append("pipeline_easing")

    # Oil price momentum (energy cost-push)
    if oil_price_yoy_pct is not None:
        if oil_price_yoy_pct > 30:
            amplifier += 4
            tags.append("energy_cost_push")
        elif oil_price_yoy_pct > 15:
            amplifier += 2
        elif oil_price_yoy_pct < -20:
            amplifier -= 3
            tags.append("energy_disinflation")

    # ═════════════════════════════════════════════════════════════════════════
    # COMBINE AND LABEL
    # ═════════════════════════════════════════════════════════════════════════

    score = max(0, min(100, base_score + amplifier))

    if score >= 80:
        label = "Hot Inflation"
    elif score >= 65:
        label = "Above Target"
    elif score >= 50:
        label = "Elevated"
    elif score >= 35:
        label = "At Target"
    elif score >= 20:
        label = "Below Target"
    else:
        label = "Deflationary"

    # ── Description ──────────────────────────────────────────────────────────
    parts: list[str] = []
    if cpi_yoy is not None:
        parts.append(f"CPI YoY: {cpi_yoy:.1f}%")
    if core_pce_yoy is not None:
        parts.append(f"Core PCE: {core_pce_yoy:.1f}%")
    if breakeven_5y is not None:
        parts.append(f"5Y BE: {breakeven_5y:.2f}%")
    if ppi_yoy is not None:
        parts.append(f"PPI YoY: {ppi_yoy:+.1f}%")

    # Add momentum signal to description
    if cpi_3m_ann is not None and cpi_yoy is not None:
        accel = cpi_3m_ann - cpi_yoy
        if abs(accel) > 0.3:
            direction = "accelerating" if accel > 0 else "decelerating"
            parts.append(f"3m trend: {direction}")

    return RegimeResult(
        name="inflation",
        label=label,
        description=" | ".join(parts) if parts else "Insufficient data",
        score=score,
        domain="inflation",
        tags=tags,
        metrics={
            "cpi_yoy": cpi_yoy,
            "core_pce_yoy": core_pce_yoy,
            "core_cpi_yoy": core_cpi_yoy,
            "trimmed_mean_pce_yoy": trimmed_mean_pce_yoy,
            "median_cpi_yoy": median_cpi_yoy,
            "breakeven_5y": breakeven_5y,
            "breakeven_5y5y": breakeven_5y5y,
            "ppi_yoy": ppi_yoy,
            "cpi_3m_ann": cpi_3m_ann,
            "cpi_6m_ann": cpi_6m_ann,
        },
    )

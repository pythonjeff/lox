"""
Policy / geopolitical uncertainty regime classifier.

3-layer scoring architecture:
  Layer 1 — Base Score: weighted EPU index, news intensity, import price pressure
  Layer 2 — Amplifiers: EPU momentum, news spike, Hormuz, VIX, USD safe-haven
  Layer 3 — Cross-signal: trade war cascade, complacency, de-escalation

Score 0 = policy calm → 100 = policy crisis.
Higher = more uncertainty/stress.

Author: Lox Capital Research
"""
from __future__ import annotations

from lox.regimes.base import RegimeResult


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Sub-score mapping functions
# ─────────────────────────────────────────────────────────────────────────────

def _epu_subscore(epu_level: float, epu_1y_percentile: float | None) -> float:
    """Map EPU level to 0-100 sub-score.

    Historical range: ~50 (extreme calm) to 600+ (crisis).
    All-time highs: ~900 (COVID Mar 2020), ~700 (trade war 2019).
    Median ~110. Mean ~130.

    Prefer percentile if available (more robust); fall back to absolute level.
    """
    if epu_1y_percentile is not None:
        return min(100.0, epu_1y_percentile)
    # Absolute fallback
    if epu_level > 400:
        return 95
    if epu_level > 300:
        return 82
    if epu_level > 200:
        return 65
    if epu_level > 150:
        return 50
    if epu_level > 100:
        return 35
    if epu_level > 70:
        return 20
    return 10


def _news_intensity_subscore(count_7d: int) -> float:
    """Map 7d policy article count to 0-100 sub-score.

    Based on Alpaca+FMP combined output with policy keywords.
    Normal: 2-8 articles/week. Elevated: 15+. Crisis: 30+.
    """
    if count_7d >= 30:
        return 95
    if count_7d >= 20:
        return 80
    if count_7d >= 15:
        return 65
    if count_7d >= 10:
        return 50
    if count_7d >= 5:
        return 30
    if count_7d >= 2:
        return 15
    return 5


def _import_pressure_subscore(
    import_price_yoy: float,
    import_price_mom_accel: float | None,
) -> float:
    """Map import price momentum to 0-100 sub-score.

    Rising import prices = tariff/trade friction passing through.
    Acceleration (2nd derivative) matters more than level.
    """
    if import_price_yoy > 10:
        base = 85.0
    elif import_price_yoy > 5:
        base = 70.0
    elif import_price_yoy > 2:
        base = 55.0
    elif import_price_yoy > 0:
        base = 40.0
    elif import_price_yoy > -2:
        base = 30.0
    elif import_price_yoy > -5:
        base = 20.0
    else:
        base = 10.0

    # Acceleration bonus
    if import_price_mom_accel is not None:
        if import_price_mom_accel > 1.0:
            base += 8
        elif import_price_mom_accel > 0.5:
            base += 4
        elif import_price_mom_accel < -1.0:
            base -= 5

    return max(0.0, min(100.0, base))


# ─────────────────────────────────────────────────────────────────────────────
# Weights
# ─────────────────────────────────────────────────────────────────────────────

_W_EPU = 0.45
_W_NEWS = 0.30
_W_IMPORT = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Main classifier
# ─────────────────────────────────────────────────────────────────────────────

def classify_policy_regime(
    epu_level: float | None,
    epu_1y_percentile: float | None,
    epu_30d_change: float | None,
    news_article_count_7d: int | None,
    news_article_count_30d: int | None,
    import_price_yoy: float | None,
    import_price_mom_accel: float | None,
    *,
    # Layer 2 amplifier inputs
    vix_level: float | None = None,
    dxy_20d_chg: float | None = None,
    oil_disruption_score: float | None = None,
    # Layer 3 cross-signal inputs (from UnifiedRegimeState)
    inflation_score: float | None = None,
    commodities_score: float | None = None,
    volatility_score: float | None = None,
) -> RegimeResult:
    """Classify the Policy regime.

    Returns RegimeResult with score 0-100 (higher = more uncertainty/stress).
    Handles missing data gracefully — sub-scores with None are excluded
    and weights re-normalized.
    """

    # ── LAYER 1: Base Score ──────────────────────────────────────────────
    measures: list[tuple[float | None, float]] = [
        (_epu_subscore(epu_level, epu_1y_percentile) if epu_level is not None else None, _W_EPU),
        (_news_intensity_subscore(news_article_count_7d) if news_article_count_7d is not None else None, _W_NEWS),
        (_import_pressure_subscore(import_price_yoy, import_price_mom_accel) if import_price_yoy is not None else None, _W_IMPORT),
    ]

    total_weight = 0.0
    weighted_sum = 0.0
    for sub_score, weight in measures:
        if sub_score is not None:
            weighted_sum += sub_score * weight
            total_weight += weight

    base_score = weighted_sum / total_weight if total_weight > 0 else 50.0

    # ── LAYER 2: Amplifiers ──────────────────────────────────────────────
    amplifier = 0.0
    tags: list[str] = ["policy"]

    # EPU momentum (30d change)
    if epu_30d_change is not None:
        if epu_30d_change > 100:
            amplifier += 8
            tags.append("rapid_escalation")
        elif epu_30d_change > 50:
            amplifier += 5
            tags.append("escalating")
        elif epu_30d_change > 25:
            amplifier += 3
        elif epu_30d_change < -100:
            amplifier -= 5
            tags.append("rapid_de_escalation")
        elif epu_30d_change < -50:
            amplifier -= 3
            tags.append("de_escalating")

    # News spike detection (7d vs 30d weekly average)
    if news_article_count_7d is not None and news_article_count_30d is not None:
        weekly_avg_30d = news_article_count_30d / 4.28  # ~4.28 weeks in 30 days
        if weekly_avg_30d > 0:
            if news_article_count_7d > weekly_avg_30d * 2.0:
                amplifier += 6
                tags.append("news_spike")
            elif news_article_count_7d > weekly_avg_30d * 1.5:
                amplifier += 4
                tags.append("news_elevated")

    # Oil supply disruption cross-signal (composite: Hormuz, Bab el-Mandeb, Suez, Malacca, Bosporus)
    if oil_disruption_score is not None:
        if oil_disruption_score >= 30:
            amplifier += 5
            tags.append("supply_disruption")
        elif oil_disruption_score >= 15:
            amplifier += 3

    # Import price tag
    if import_price_yoy is not None:
        if import_price_yoy > 5:
            tags.append("import_cost_push")
        if import_price_mom_accel is not None and import_price_mom_accel > 0.5:
            tags.append("tariff_pass_through")

    # VIX confirmation (only if policy already elevated)
    interim_score = base_score + amplifier
    if vix_level is not None and interim_score >= 50:
        if vix_level > 25:
            amplifier += 4
            tags.append("cross_market_confirmed")
        elif vix_level > 20:
            amplifier += 2

    # USD safe-haven signal
    if dxy_20d_chg is not None and interim_score >= 50:
        if dxy_20d_chg > 2.0:
            amplifier += 3
            tags.append("safe_haven_demand")
        elif dxy_20d_chg > 1.0:
            amplifier += 2

    # ── LAYER 3: Cross-Signal Confirmation ────────────────────────────────
    score_pre_l3 = base_score + amplifier

    # Trade war cascade: policy + inflation + commodities all elevated
    if (inflation_score is not None and commodities_score is not None
            and inflation_score >= 50 and commodities_score >= 50
            and score_pre_l3 >= 55):
        amplifier += 5
        tags.append("trade_war_cascade")

    # Complacency: policy elevated but vol subdued — market ignoring risk
    if (volatility_score is not None
            and score_pre_l3 >= 55 and volatility_score < 35):
        amplifier -= 3
        tags.append("complacency_divergence")

    # De-escalation: EPU falling AND news volume declining
    if (epu_30d_change is not None and epu_30d_change < -25
            and news_article_count_7d is not None
            and news_article_count_30d is not None):
        weekly_avg = news_article_count_30d / 4.28
        if weekly_avg > 0 and news_article_count_7d < weekly_avg * 0.7:
            amplifier -= 3
            tags.append("de_escalation")

    # ── Final score + label ──────────────────────────────────────────────
    score = max(0.0, min(100.0, base_score + amplifier))

    if score >= 80:
        label = "Policy Crisis"
    elif score >= 65:
        label = "Policy Stress"
    elif score >= 50:
        label = "Elevated Uncertainty"
    elif score >= 35:
        label = "Moderate Uncertainty"
    elif score >= 20:
        label = "Low Uncertainty"
    else:
        label = "Policy Calm"

    # ── Description ──────────────────────────────────────────────────────
    parts: list[str] = []
    if epu_level is not None:
        pctl_str = f" ({epu_1y_percentile:.0f}th pctl)" if epu_1y_percentile is not None else ""
        parts.append(f"EPU: {epu_level:.0f}{pctl_str}")
    if epu_30d_change is not None:
        parts.append(f"30d Δ: {epu_30d_change:+.0f}")
    if news_article_count_7d is not None:
        parts.append(f"News 7d: {news_article_count_7d}")
    if import_price_yoy is not None:
        parts.append(f"Import Px: {import_price_yoy:+.1f}% YoY")

    description = " | ".join(parts) if parts else "Insufficient data"

    # ── Metrics dict ─────────────────────────────────────────────────────
    metrics: dict[str, str | None] = {
        "EPU": f"{epu_level:.0f}" if epu_level is not None else None,
        "EPU %ile": f"{epu_1y_percentile:.0f}" if epu_1y_percentile is not None else None,
        "EPU 30d Chg": f"{epu_30d_change:+.0f}" if epu_30d_change is not None else None,
        "News 7d": f"{news_article_count_7d}" if news_article_count_7d is not None else None,
        "News 30d": f"{news_article_count_30d}" if news_article_count_30d is not None else None,
        "Import Px YoY": f"{import_price_yoy:+.1f}%" if import_price_yoy is not None else None,
        "Import Accel": f"{import_price_mom_accel:+.2f}" if import_price_mom_accel is not None else None,
        "VIX": f"{vix_level:.1f}" if vix_level is not None else None,
        "DXY 20d Chg": f"{dxy_20d_chg:+.1f}%" if dxy_20d_chg is not None else None,
    }

    return RegimeResult(
        name="policy",
        label=label,
        description=description,
        score=score,
        domain="policy",
        tags=tags,
        metrics=metrics,
    )

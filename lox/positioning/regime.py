"""
Positioning regime classifier.

3-layer scoring architecture:
  Layer 1 — Base Score: VIX term, P/C ratio, AAII sentiment, COT, GEX, short interest
  Layer 2 — Amplifiers: cross-signal positioning extremes (panic/complacency traps)
  Layer 3 — Cross-signal: regime-aware confirmation (vol, credit)

Score 0 = extreme complacency → 100 = capitulation.
Higher = more defensive/panic positioning.

Author: Lox Capital Research
"""
from __future__ import annotations

from lox.regimes.base import RegimeResult


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Sub-score mapping functions
# ─────────────────────────────────────────────────────────────────────────────

def _vix_term_subscore(slope: float) -> float:
    """Map VIX term slope (VIX3M/VIX) to 0-100 sub-score.

    >1 = contango (normal, complacent), <1 = backwardation (panic).
    Deep backwardation (<0.85) is historically rare and signals panic.
    """
    if slope < 0.80:
        return 95
    if slope < 0.85:
        return 85
    if slope < 0.90:
        return 72
    if slope < 0.95:
        return 60
    if slope < 1.00:
        return 52
    if slope < 1.05:
        return 40
    if slope < 1.10:
        return 30
    if slope < 1.15:
        return 20
    return 10  # steep contango = max complacency


def _pcr_subscore(pcr: float) -> float:
    """Map put/call ratio to 0-100 sub-score.

    Typical range: 0.5 (complacent) to 1.2 (fearful).
    Extreme: <0.4 = max complacency, >1.5 = capitulation.
    """
    if pcr > 1.5:
        return 95
    if pcr > 1.2:
        return 80
    if pcr > 1.0:
        return 65
    if pcr > 0.8:
        return 50
    if pcr > 0.65:
        return 35
    if pcr > 0.5:
        return 20
    return 10  # extremely low put buying


def _aaii_subscore(bull_pct: float) -> float:
    """Map AAII bullish % to 0-100 sub-score.

    AAII is a contrarian indicator:
    - High bullish % = everyone is long = complacent (LOW score)
    - Low bullish % = everyone is bearish = fear (HIGH score)
    Typical range: 20-50%. Extremes: <15% or >60%.
    """
    if bull_pct > 60:
        return 10  # extreme bullishness = contrarian warning for complacency
    if bull_pct > 50:
        return 22
    if bull_pct > 40:
        return 38
    if bull_pct > 30:
        return 52
    if bull_pct > 20:
        return 68
    if bull_pct > 15:
        return 82
    return 95  # extreme bearishness = contrarian signal for panic


def _cot_subscore(z: float) -> float:
    """Map COT net speculative z-score to 0-100 sub-score.

    Positive z = crowded long (complacent). Negative z = crowded short (fear).
    |z| > 2 is historically extreme.
    """
    if z < -2.0:
        return 92
    if z < -1.5:
        return 78
    if z < -1.0:
        return 65
    if z < -0.5:
        return 55
    if z < 0.5:
        return 45
    if z < 1.0:
        return 35
    if z < 1.5:
        return 25
    if z < 2.0:
        return 15
    return 8  # extreme long = max complacency


def _gex_subscore(gex_bn: float) -> float:
    """Map GEX ($bn) to 0-100 sub-score.

    Positive GEX = dealers long gamma → market stabilizing (low score).
    Negative GEX = dealers short gamma → vol amplifying (high score).
    Typical SPY GEX range: -$5bn to +$10bn.
    """
    if gex_bn < -5.0:
        return 90
    if gex_bn < -2.0:
        return 75
    if gex_bn < -0.5:
        return 62
    if gex_bn < 0.5:
        return 50
    if gex_bn < 2.0:
        return 38
    if gex_bn < 5.0:
        return 25
    return 12  # very positive GEX = strong pinning, complacent


def _si_subscore(avg_si_pct: float) -> float:
    """Map average short interest (% float) to 0-100 sub-score.

    High SI = bears positioned for downside (directional). Not a pure fear/complacency
    signal — could indicate smart money hedging or crowded shorts.
    """
    if avg_si_pct > 15:
        return 75
    if avg_si_pct > 10:
        return 62
    if avg_si_pct > 5:
        return 50
    if avg_si_pct > 2:
        return 38
    return 25  # low SI = nobody betting against


# ─────────────────────────────────────────────────────────────────────────────
# Weights
# ─────────────────────────────────────────────────────────────────────────────

_W_VIX_TERM = 0.25
_W_COT = 0.20
_W_GEX = 0.20
_W_PCR = 0.15
_W_AAII = 0.10
_W_SI = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Main classifier
# ─────────────────────────────────────────────────────────────────────────────

def classify_positioning(
    vix_term_slope: float | None = None,
    put_call_ratio: float | None = None,
    aaii_bull_pct: float | None = None,
    cot_net_spec: dict[str, float] | None = None,
    cot_z_score: dict[str, float] | None = None,
    gex_total: float | None = None,
    gex_flip_level: float | None = None,
    skew_25d: float | None = None,
    short_interest_pct: dict[str, float] | None = None,
    *,
    # Layer 2/3 cross-regime inputs
    vol_score: float | None = None,
    credit_score: float | None = None,
) -> RegimeResult:
    """Classify the Positioning regime.

    Returns RegimeResult with score 0-100 (higher = more defensive/panic positioning).
    Handles missing data gracefully — sub-scores with None are excluded
    and weights re-normalized.
    """

    # ── LAYER 1: Base Score ──────────────────────────────────────────────
    measures: list[tuple[float | None, float]] = []

    # VIX Term Structure
    if vix_term_slope is not None:
        measures.append((_vix_term_subscore(vix_term_slope), _W_VIX_TERM))

    # Put/Call Ratio
    if put_call_ratio is not None:
        measures.append((_pcr_subscore(put_call_ratio), _W_PCR))

    # AAII Sentiment
    if aaii_bull_pct is not None:
        measures.append((_aaii_subscore(aaii_bull_pct), _W_AAII))

    # COT Net Speculative (use ES z-score as primary signal)
    cot_z_primary: float | None = None
    if cot_z_score:
        # Prefer ES (S&P futures), fallback to any available
        cot_z_primary = cot_z_score.get("ES") or next(iter(cot_z_score.values()), None)
    if cot_z_primary is not None:
        measures.append((_cot_subscore(cot_z_primary), _W_COT))

    # GEX
    if gex_total is not None:
        measures.append((_gex_subscore(gex_total), _W_GEX))

    # Short Interest
    avg_si: float | None = None
    if short_interest_pct:
        vals = [v for v in short_interest_pct.values() if v is not None]
        if vals:
            avg_si = sum(vals) / len(vals)
    if avg_si is not None:
        measures.append((_si_subscore(avg_si), _W_SI))

    # Weighted average with re-normalization for missing inputs
    total_weight = sum(w for _, w in measures)
    if total_weight > 0:
        base_score = sum(s * w for s, w in measures) / total_weight
    else:
        base_score = 50.0

    # ── LAYER 2: Amplifiers ──────────────────────────────────────────────
    amplifier = 0.0
    tags: list[str] = ["positioning"]

    # COT extreme + VIX backwardation → positioning panic
    if (cot_z_primary is not None and cot_z_primary < -1.5
            and vix_term_slope is not None and vix_term_slope < 0.95):
        amplifier += 5
        tags.append("panic_positioning")

    # GEX negative + high P/C → mechanical selling pressure
    if (gex_total is not None and gex_total < 0
            and put_call_ratio is not None and put_call_ratio > 1.0):
        amplifier += 5
        tags.append("mechanical_selling")

    # AAII extreme bullish + GEX positive + low P/C → complacency trap
    if (aaii_bull_pct is not None and aaii_bull_pct > 55
            and gex_total is not None and gex_total > 2.0
            and put_call_ratio is not None and put_call_ratio < 0.6):
        amplifier -= 5
        tags.append("complacency_trap")

    # COT crowded long + AAII euphoric → contrarian short signal
    if (cot_z_primary is not None and cot_z_primary > 1.5
            and aaii_bull_pct is not None and aaii_bull_pct > 50):
        amplifier -= 8
        tags.append("crowded_long")

    # COT crowded short + AAII bearish → contrarian long signal
    if (cot_z_primary is not None and cot_z_primary < -1.5
            and aaii_bull_pct is not None and aaii_bull_pct < 25):
        amplifier += 5
        tags.append("crowded_short")

    # Skew amplifier (elevated puts = fear)
    if skew_25d is not None:
        if skew_25d > 8:  # >8 vol pts = very steep put skew
            amplifier += 4
            tags.append("steep_skew")
        elif skew_25d > 5:
            amplifier += 2
        elif skew_25d < 0:  # inverted skew = extreme complacency
            amplifier -= 3
            tags.append("inverted_skew")

    # ── LAYER 3: Cross-Signal Confirmation ────────────────────────────────
    score_pre_l3 = base_score + amplifier

    # Vol regime confirms positioning panic
    if vol_score is not None:
        if score_pre_l3 >= 60 and vol_score >= 60:
            amplifier += 4
            tags.append("vol_confirmed")
        elif score_pre_l3 >= 55 and vol_score < 30:
            # Positioning fear but vol calm → possible overreaction
            amplifier -= 3
            tags.append("vol_divergence")

    # Credit regime confirms positioning stress
    if credit_score is not None:
        if score_pre_l3 >= 60 and credit_score >= 55:
            amplifier += 3
            tags.append("credit_confirmed")

    # ── Final score + label ──────────────────────────────────────────────
    score = max(0.0, min(100.0, base_score + amplifier))

    if score >= 80:
        label = "Capitulation"
    elif score >= 65:
        label = "Panic Positioning"
    elif score >= 50:
        label = "Defensive Positioning"
    elif score >= 35:
        label = "Neutral Positioning"
    elif score >= 20:
        label = "Complacent"
    else:
        label = "Extreme Complacency"

    # ── Description ──────────────────────────────────────────────────────
    parts: list[str] = []
    if vix_term_slope is not None:
        shape = "backwardation" if vix_term_slope < 1 else "contango"
        parts.append(f"VIX Term: {vix_term_slope:.2f}x ({shape})")
    if put_call_ratio is not None:
        parts.append(f"P/C: {put_call_ratio:.2f}")
    if cot_z_primary is not None:
        parts.append(f"COT z: {cot_z_primary:+.1f}")
    if gex_total is not None:
        parts.append(f"GEX: {gex_total:+.1f}bn")
    if aaii_bull_pct is not None:
        parts.append(f"AAII: {aaii_bull_pct:.0f}%")

    description = " | ".join(parts) if parts else "Insufficient data"

    # ── Metrics dict ─────────────────────────────────────────────────────
    metrics: dict[str, object] = {
        "vix_term_slope": vix_term_slope,
        "put_call_ratio": put_call_ratio,
        "aaii_bull_pct": aaii_bull_pct,
        "cot_z_primary": cot_z_primary,
        "gex_total_bn": gex_total,
        "gex_flip_level": gex_flip_level,
        "skew_25d": skew_25d,
        "avg_short_interest": avg_si,
    }

    return RegimeResult(
        name="positioning",
        label=label,
        description=description,
        score=score,
        domain="positioning",
        tags=tags,
        metrics=metrics,
    )

"""
Earnings regime classifier — market-wide S&P 500 earnings health.

3-layer scoring architecture (matches credit/funding pattern):
  Layer 1 — Base Score: weighted combination of beat rate, surprise %, revision ratio
  Layer 2 — Sector Divergence: narrow leadership, sector stress, broad strength
  Layer 3 — Cross-signal Confirmation: synchronized deterioration, lowered bar, trough forming

Score 0 = earnings boom → 100 = earnings recession.

Signals (all derived from bulk FMP endpoints, ~4 API calls total):
  Primary:   Beat rate, avg surprise %, net revision ratio
  Sector:    Sector dispersion, worst/best sector beat rates
  Context:   Reporting density, sectors beating count

Author: Lox Capital Research
"""
from __future__ import annotations

from lox.regimes.base import RegimeResult


# ─────────────────────────────────────────────────────────────────────
# Layer 1 — Sub-score mapping tables
# ─────────────────────────────────────────────────────────────────────
# Each maps an input to a 0-100 sub-score (higher = more stress).

def _beat_rate_subscore(beat_rate: float) -> float:
    """Map beat rate (0-100%) to stress sub-score (0-100).

    Historical S&P 500 norms:
      >80% = blowout (only 2009 recovery, 2021)
      70-80% = healthy / above-average
      60-70% = weakening
      <60% = recessionary (2001, 2008, 2020)
    """
    if beat_rate > 80:
        return 5    # blowout
    if beat_rate > 75:
        return 15   # very strong
    if beat_rate > 70:
        return 30   # healthy, above-average
    if beat_rate > 65:
        return 45   # normal range
    if beat_rate > 60:
        return 60   # weakening
    if beat_rate > 55:
        return 75   # clearly deteriorating
    return 95       # earnings recession territory


def _surprise_subscore(avg_surprise_pct: float) -> float:
    """Map average surprise % to stress sub-score (0-100).

    Typical S&P 500 avg surprise is ~5%.
    Negative avg = net misses (extremely rare, extremely bad).
    """
    if avg_surprise_pct > 10:
        return 5    # exceptional beats
    if avg_surprise_pct > 7:
        return 18   # strong beats
    if avg_surprise_pct > 4:
        return 35   # normal, healthy
    if avg_surprise_pct > 1:
        return 55   # below-average beats
    if avg_surprise_pct > 0:
        return 72   # barely beating
    return 92       # aggregate misses


def _revision_subscore(net_revision_ratio: float) -> float:
    """Map net revision ratio (-1 to +1) to stress sub-score (0-100).

    Positive = more upgrades than downgrades.
    This is the most forward-looking signal — leads earnings by ~6 weeks.
    """
    if net_revision_ratio > 0.20:
        return 8    # broad upgrade cycle
    if net_revision_ratio > 0.10:
        return 22   # solidly positive
    if net_revision_ratio > 0:
        return 38   # mildly positive
    if net_revision_ratio > -0.10:
        return 58   # mildly negative
    if net_revision_ratio > -0.20:
        return 75   # net negative
    return 92       # broad downgrade cycle


# ─────────────────────────────────────────────────────────────────────
# Layer 1 weights
# ─────────────────────────────────────────────────────────────────────

_WEIGHT_BEAT_RATE = 0.45   # primary signal
_WEIGHT_SURPRISE  = 0.30   # secondary
_WEIGHT_REVISION  = 0.25   # forward-looking amplifier


# ─────────────────────────────────────────────────────────────────────
# Main classifier
# ─────────────────────────────────────────────────────────────────────

def classify_earnings_regime(
    beat_rate: float | None,
    avg_surprise_pct: float | None,
    net_revision_ratio: float | None,
    reporting_density: int | None,
    *,
    sector_dispersion: float | None = None,
    worst_sector_beat_rate: float | None = None,
    sectors_beating: int | None = None,
    total_sectors_rated: int | None = None,
    best_sector: str | None = None,
    worst_sector: str | None = None,
) -> RegimeResult:
    """Classify the Earnings regime from market-wide S&P 500 data.

    3-layer architecture:
      Layer 1 — Base Score: weighted beat_rate / surprise / revision sub-scores
      Layer 2 — Sector Divergence: amplifies score based on cross-sector spread
      Layer 3 — Cross-signal Confirmation: paradox/sync patterns

    Args:
        beat_rate: % of S&P 500 companies that beat EPS estimates in the
            most recent reporting window (0-100).  Historical norm ~70%.
        avg_surprise_pct: Mean EPS surprise % across reporting S&P 500
            companies.  Typical range 0-10%.
        net_revision_ratio: (analyst upgrades - downgrades) / total rated
            names.  Range roughly -1 to +1.  Positive = improving revisions.
        reporting_density: Number of S&P 500 names that reported earnings
            in the trailing 30 days.  >200 = peak earnings season.
        sector_dispersion: Spread (pp) between best and worst sector beat
            rates.  High = narrow leadership, earnings quality risk.
        worst_sector_beat_rate: Beat rate of the weakest GICS sector.
        sectors_beating: Number of sectors with beat rate > 65%.
        total_sectors_rated: Total sectors with enough data to rate.
        best_sector: Name of the top-beating sector.
        worst_sector: Name of the worst-beating sector.

    Returns:
        RegimeResult with domain="earnings", score 0-100.
    """
    tags: list[str] = ["earnings"]

    # ═══════════════════════════════════════════════════════════════════
    # LAYER 1 — Base Score (weighted sub-scores)
    # ═══════════════════════════════════════════════════════════════════
    subscores: list[tuple[float, float]] = []  # (weight, sub_score)

    if beat_rate is not None:
        subscores.append((_WEIGHT_BEAT_RATE, _beat_rate_subscore(beat_rate)))
        if beat_rate > 80:
            tags.append("earnings_blowout")
        elif beat_rate < 55:
            tags.append("earnings_recession")

    if avg_surprise_pct is not None:
        subscores.append((_WEIGHT_SURPRISE, _surprise_subscore(avg_surprise_pct)))
        if avg_surprise_pct < 0:
            tags.append("net_misses")

    if net_revision_ratio is not None:
        subscores.append((_WEIGHT_REVISION, _revision_subscore(net_revision_ratio)))
        if net_revision_ratio > 0.10:
            tags.append("revision_positive")
        elif net_revision_ratio < -0.10:
            tags.append("revision_negative")

    # Weighted average (re-normalize if any input is missing)
    if subscores:
        total_weight = sum(w for w, _ in subscores)
        score = sum(w * s for w, s in subscores) / total_weight
    else:
        score = 50.0  # no data → neutral

    # ═══════════════════════════════════════════════════════════════════
    # LAYER 2 — Sector Divergence Amplifiers
    # ═══════════════════════════════════════════════════════════════════
    # High dispersion = narrow leadership = fragile earnings picture

    if sector_dispersion is not None:
        if sector_dispersion > 25:
            score += 5
            tags.append("narrow_leadership")
        elif sector_dispersion > 15:
            score += 3

    if worst_sector_beat_rate is not None and worst_sector_beat_rate < 50:
        score += 4  # at least one sector in recession
        tags.append("sector_stress")

    if (
        sectors_beating is not None
        and total_sectors_rated is not None
        and total_sectors_rated > 0
    ):
        pct_beating = sectors_beating / total_sectors_rated
        if pct_beating >= 0.80:
            score -= 5  # >80% of sectors healthy = broad-based strength
            tags.append("broad_strength")
        elif pct_beating <= 0.30:
            score += 3  # <30% of sectors healthy = concentrated weakness
            tags.append("concentrated_risk")

    # Sector rotation detection: best sector very strong but worst very weak
    if (
        worst_sector_beat_rate is not None
        and beat_rate is not None
        and worst_sector_beat_rate < 55
        and beat_rate > 70
    ):
        tags.append("sector_rotation")

    # ═══════════════════════════════════════════════════════════════════
    # LAYER 3 — Cross-signal Confirmation & Paradox
    # ═══════════════════════════════════════════════════════════════════

    # Synchronized deterioration: all three primary signals weakening
    if (
        beat_rate is not None
        and avg_surprise_pct is not None
        and net_revision_ratio is not None
        and beat_rate < 65
        and avg_surprise_pct < 3
        and net_revision_ratio < 0
    ):
        score += 5
        tags.append("synchronized_deterioration")

    # Lowered-bar paradox: companies beat EPS but forward guidance negative
    # Classic late-cycle pattern — beats are misleading
    if (
        beat_rate is not None
        and net_revision_ratio is not None
        and beat_rate > 70
        and net_revision_ratio < -0.10
    ):
        score += 5
        tags.append("lowered_bar")

    # Trough-forming signal: beat rate weak but revisions turning positive
    # Early recovery pattern — worst is likely behind
    if (
        beat_rate is not None
        and net_revision_ratio is not None
        and beat_rate < 60
        and net_revision_ratio > 0.10
    ):
        score -= 3
        tags.append("trough_forming")

    # Concentrated risk: wide sector dispersion + negative revisions
    if (
        sector_dispersion is not None
        and net_revision_ratio is not None
        and sector_dispersion > 20
        and net_revision_ratio < -0.10
    ):
        score += 3
        if "concentrated_risk" not in tags:
            tags.append("concentrated_risk")

    # ═══════════════════════════════════════════════════════════════════
    # Reporting density (CONTEXT, not scored)
    # ═══════════════════════════════════════════════════════════════════
    if reporting_density is not None and reporting_density > 200:
        tags.append("peak_season")

    # ═══════════════════════════════════════════════════════════════════
    # Clamp, label, describe
    # ═══════════════════════════════════════════════════════════════════
    score = max(0.0, min(100.0, score))

    if score <= 20:
        label = "Earnings Boom"
    elif score <= 35:
        label = "Earnings Expansion"
    elif score <= 50:
        label = "Earnings Steady"
    elif score <= 65:
        label = "Earnings Slowdown"
    elif score <= 80:
        label = "Earnings Contraction"
    else:
        label = "Earnings Recession"

    # ── Description string (top-line summary for panel) ───────────────
    parts: list[str] = []
    if beat_rate is not None:
        parts.append(f"Beat Rate: {beat_rate:.0f}%")
    if avg_surprise_pct is not None:
        parts.append(f"Avg Surprise: {avg_surprise_pct:+.1f}%")
    if net_revision_ratio is not None:
        parts.append(f"Net Revisions: {net_revision_ratio:+.2f}")
    if sector_dispersion is not None:
        parts.append(f"Dispersion: {sector_dispersion:.0f}pp")

    return RegimeResult(
        name="earnings",
        label=label,
        description=" | ".join(parts) if parts else "Insufficient data",
        score=round(score, 1),
        domain="earnings",
        tags=tags,
        metrics={
            "Beat Rate": f"{beat_rate:.0f}%" if beat_rate is not None else None,
            "Avg Surprise": f"{avg_surprise_pct:+.1f}%" if avg_surprise_pct is not None else None,
            "Net Revisions": f"{net_revision_ratio:+.2f}" if net_revision_ratio is not None else None,
            "Reporting": f"{reporting_density}" if reporting_density is not None else None,
            "Dispersion": f"{sector_dispersion:.0f}pp" if sector_dispersion is not None else None,
            "Weakest Sector": f"{worst_sector} ({worst_sector_beat_rate:.0f}%)" if worst_sector and worst_sector_beat_rate is not None else worst_sector,
            "Sectors Beating": f"{sectors_beating}/{total_sectors_rated}" if sectors_beating is not None and total_sectors_rated is not None else None,
            "Strongest Sector": best_sector,
        },
    )

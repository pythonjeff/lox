"""
Regime trend & momentum computation engine.

Computes directional momentum, velocity, persistence, and z-scored trend
signals from the regime score time series stored in regime_history.

Output: RegimeTrend dataclass per domain — used by display layer and LLM.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Trend direction labels ────────────────────────────────────────────────
TREND_DETERIORATING = "DETERIORATING"
TREND_WEAKENING = "WEAKENING"
TREND_STABLE = "STABLE"
TREND_IMPROVING = "IMPROVING"
TREND_STRENGTHENING = "STRENGTHENING"

# Arrow/symbol mapping for compact display
TREND_ARROWS = {
    TREND_DETERIORATING: "▼▼",
    TREND_WEAKENING: "▼",
    TREND_STABLE: "—",
    TREND_IMPROVING: "▲",
    TREND_STRENGTHENING: "▲▲",
}

TREND_COLORS = {
    TREND_DETERIORATING: "red",
    TREND_WEAKENING: "yellow",
    TREND_STABLE: "dim",
    TREND_IMPROVING: "green",
    TREND_STRENGTHENING: "bright_green",
}


@dataclass(frozen=True)
class RegimeTrend:
    """
    Quantitative trend/momentum state for a single regime domain.

    All score deltas are in raw score units (0-100 scale).
    Positive delta = score went UP = more stress (worse, for most pillars).
    """
    domain: str

    # Current vs previous
    current_score: float
    current_label: str
    prev_score: Optional[float] = None
    prev_label: Optional[str] = None
    prev_date: Optional[str] = None

    # Score deltas over lookback windows
    score_chg_1d: Optional[float] = None   # 1-day change
    score_chg_7d: Optional[float] = None   # 7-day change
    score_chg_30d: Optional[float] = None  # 30-day change

    # Momentum (z-scored 7d change relative to historical 7d changes)
    momentum_z: Optional[float] = None

    # Trend classification
    trend_direction: str = TREND_STABLE
    trend_arrow: str = "—"
    trend_color: str = "dim"

    # Persistence: how many consecutive days in the current regime label
    days_in_regime: int = 0

    # Volatility of the score itself (std of last 30 scores)
    score_volatility: Optional[float] = None

    # Regime transition count in last 30 days
    transitions_30d: int = 0

    # Score range (min/max) over last 30 days
    score_30d_low: Optional[float] = None
    score_30d_high: Optional[float] = None

    # Velocity: rate of change per day over last 7 days (slope of linear fit)
    velocity_7d: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to serializable dict for LLM context injection."""
        d = {
            "domain": self.domain,
            "current_score": self.current_score,
            "current_label": self.current_label,
            "trend_direction": self.trend_direction,
            "days_in_regime": self.days_in_regime,
        }
        if self.prev_label and self.prev_label != self.current_label:
            d["prev_label"] = self.prev_label
            d["prev_date"] = self.prev_date
        if self.score_chg_7d is not None:
            d["score_chg_7d"] = round(self.score_chg_7d, 1)
        if self.score_chg_30d is not None:
            d["score_chg_30d"] = round(self.score_chg_30d, 1)
        if self.momentum_z is not None:
            d["momentum_z"] = round(self.momentum_z, 2)
        if self.velocity_7d is not None:
            d["velocity_7d"] = round(self.velocity_7d, 2)
        if self.transitions_30d > 0:
            d["transitions_30d"] = self.transitions_30d
        return d


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _linear_slope(values: list[float]) -> float:
    """OLS slope of values against integer index (0, 1, 2, ...)."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0


def compute_regime_trend(
    domain: str,
    current_score: float,
    current_label: str,
    series: list[dict],
) -> RegimeTrend:
    """
    Compute trend/momentum for a single domain from its score series.

    Args:
        domain: Regime domain name
        current_score: Today's score
        current_label: Today's regime label
        series: List of {"date", "score", "label", "name"} sorted ascending by date.
                This is the historical series from regime_history.

    Returns:
        RegimeTrend with all computed fields.
    """
    if not series:
        return RegimeTrend(
            domain=domain,
            current_score=current_score,
            current_label=current_label,
        )

    today = datetime.now()
    scores = [float(e["score"]) for e in series]
    dates = [_parse_date(e["date"]) for e in series]
    labels = [e.get("label", "") for e in series]

    # ── Previous score/label ──────────────────────────────────────────────
    prev_score = None
    prev_label = None
    prev_date = None
    if len(series) >= 2:
        prev_score = float(series[-2]["score"])
        prev_label = series[-2].get("label")
        prev_date = series[-2].get("date")
    elif len(series) == 1:
        prev_score = float(series[-1]["score"])
        prev_label = series[-1].get("label")
        prev_date = series[-1].get("date")

    # ── Score deltas at lookback windows ──────────────────────────────────
    def _score_at_lookback(days: int) -> Optional[float]:
        cutoff = today - timedelta(days=days)
        # Find the entry closest to (but not after) the cutoff
        candidates = [(d, s) for d, s in zip(dates, scores) if d <= cutoff]
        if not candidates:
            # If no entry before cutoff, use earliest available if within 2x window
            earliest = min(zip(dates, scores), key=lambda x: x[0])
            if (today - earliest[0]).days <= days * 2:
                return earliest[1]
            return None
        return candidates[-1][1]

    score_1d = _score_at_lookback(1)
    score_7d = _score_at_lookback(7)
    score_30d = _score_at_lookback(30)

    score_chg_1d = (current_score - score_1d) if score_1d is not None else None
    score_chg_7d = (current_score - score_7d) if score_7d is not None else None
    score_chg_30d = (current_score - score_30d) if score_30d is not None else None

    # ── Momentum z-score (7d change vs distribution of historical 7d changes)
    momentum_z = None
    if len(scores) >= 8:
        chg_7ds = [scores[i] - scores[i - 7] for i in range(7, len(scores))]
        if score_chg_7d is not None and len(chg_7ds) >= 3:
            # Append current 7d change
            chg_7ds.append(score_chg_7d)
            mu = sum(chg_7ds) / len(chg_7ds)
            std = math.sqrt(sum((c - mu) ** 2 for c in chg_7ds) / len(chg_7ds))
            if std > 0.5:  # Avoid division by near-zero
                momentum_z = (score_chg_7d - mu) / std

    # ── Trend direction classification ────────────────────────────────────
    # Uses 7d change primarily, falls back to 1d
    ref_chg = score_chg_7d if score_chg_7d is not None else score_chg_1d
    # NOTE: positive change = score up = MORE stress = DETERIORATING
    if ref_chg is None:
        trend_direction = TREND_STABLE
    elif ref_chg > 8:
        trend_direction = TREND_DETERIORATING
    elif ref_chg > 3:
        trend_direction = TREND_WEAKENING
    elif ref_chg < -8:
        trend_direction = TREND_STRENGTHENING
    elif ref_chg < -3:
        trend_direction = TREND_IMPROVING
    else:
        trend_direction = TREND_STABLE

    # ── Days in current regime label ──────────────────────────────────────
    days_in_regime = 0
    for i in range(len(labels) - 1, -1, -1):
        if labels[i] == current_label:
            days_in_regime = (today - dates[i]).days
            # Keep counting backwards as long as label matches
        else:
            break
    # At minimum 0 (brand new today)
    days_in_regime = max(0, days_in_regime)

    # ── Score volatility (std of last 30 scores) ─────────────────────────
    score_volatility = None
    recent_scores = scores[-30:] if len(scores) >= 3 else scores
    if len(recent_scores) >= 3:
        mu_s = sum(recent_scores) / len(recent_scores)
        score_volatility = math.sqrt(
            sum((s - mu_s) ** 2 for s in recent_scores) / len(recent_scores)
        )

    # ── Transition count (last 30 days) ───────────────────────────────────
    transitions_30d = 0
    cutoff_30d = today - timedelta(days=30)
    for i in range(1, len(labels)):
        if dates[i] >= cutoff_30d and labels[i] != labels[i - 1]:
            transitions_30d += 1

    # ── Score range (30d) ─────────────────────────────────────────────────
    recent_30d = [s for d, s in zip(dates, scores) if d >= cutoff_30d]
    # Include current score
    recent_30d.append(current_score)
    score_30d_low = min(recent_30d) if recent_30d else None
    score_30d_high = max(recent_30d) if recent_30d else None

    # ── Velocity: daily slope over last 7 entries ─────────────────────────
    velocity_7d = None
    tail = scores[-7:] if len(scores) >= 2 else scores
    if len(tail) >= 2:
        velocity_7d = _linear_slope(tail)

    return RegimeTrend(
        domain=domain,
        current_score=current_score,
        current_label=current_label,
        prev_score=prev_score,
        prev_label=prev_label,
        prev_date=prev_date,
        score_chg_1d=score_chg_1d,
        score_chg_7d=score_chg_7d,
        score_chg_30d=score_chg_30d,
        momentum_z=momentum_z,
        trend_direction=trend_direction,
        trend_arrow=TREND_ARROWS[trend_direction],
        trend_color=TREND_COLORS[trend_direction],
        days_in_regime=days_in_regime,
        score_volatility=score_volatility,
        transitions_30d=transitions_30d,
        score_30d_low=score_30d_low,
        score_30d_high=score_30d_high,
        velocity_7d=velocity_7d,
    )


def compute_all_trends(
    state,  # UnifiedRegimeState
    all_series: dict[str, list[dict]],
) -> dict[str, RegimeTrend]:
    """
    Compute RegimeTrend for every domain in the unified state.

    Args:
        state: UnifiedRegimeState (uses domain attributes for current scores)
        all_series: {domain: [score_series entries]} from regime_history

    Returns:
        {domain: RegimeTrend}
    """
    from lox.regimes.features import ALL_DOMAINS

    trends = {}
    for domain in ALL_DOMAINS:
        regime = getattr(state, domain, None)
        if regime is None:
            continue
        series = all_series.get(domain, [])
        trends[domain] = compute_regime_trend(
            domain=domain,
            current_score=regime.score,
            current_label=regime.label,
            series=series,
        )
    return trends


def get_domain_trend(domain: str, current_score: float, current_label: str) -> RegimeTrend:
    """Convenience: compute trend for a single domain from persisted history."""
    from lox.data.regime_history import get_score_series

    series = get_score_series(domain)
    return compute_regime_trend(domain, current_score, current_label, series)

"""
Signal Pillar 3: Regime Alignment.

Maps each ticker to factor exposures (ETFs from FACTOR_EXPOSURES,
stocks via SECTOR_DEFAULTS), then scores against current regime
pillar directions and trend velocity.

Higher score = stronger macro tailwind for this ticker.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from lox.config import Settings

logger = logging.getLogger(__name__)

# Factor → regime domain mapping
_FACTOR_TO_DOMAIN: dict[str, str] = {
    "equity_beta": "growth",
    "duration": "rates",
    "credit": "credit",
    "commodities": "commodities",
    "gold": "commodities",
    "vol": "volatility",
    "inflation": "inflation",
}


@dataclass
class RegimeAlignmentSignal:
    ticker: str
    alignment: str  # TAILWIND, HEADWIND, NEUTRAL
    primary_factor: str
    regime_context: str  # e.g. "growth improving, vol low"
    sub_score: float  # 0-100


def _get_exposures(
    ticker: str,
    ticker_sectors: dict[str, str],
) -> dict[str, float] | None:
    """Get factor exposures: explicit table first, then sector defaults."""
    from lox.cli_commands.shared.book_impact import (
        FACTOR_EXPOSURES, SECTOR_DEFAULTS,
    )

    if ticker in FACTOR_EXPOSURES:
        return FACTOR_EXPOSURES[ticker]

    sector = ticker_sectors.get(ticker, "").lower()
    if sector:
        return SECTOR_DEFAULTS.get(sector)

    return None


def _regime_factor_directions(regime_state: Any) -> dict[str, float]:
    """Extract aggregate factor directions from current regime state.

    Looks at each regime pillar score and converts to factor direction:
    - Score < 35 (green): favorable for risk-on factors
    - Score > 65 (red): unfavorable for risk-on factors
    - Score 35-64 (yellow): neutral

    Also incorporates trend velocity for recency weighting.
    """
    from lox.cli_commands.shared.book_impact import _classify_regime_direction

    factor_dirs: dict[str, float] = {}
    factor_counts: dict[str, int] = {}

    for domain in ("growth", "rates", "inflation", "volatility",
                    "credit", "commodities", "liquidity"):
        regime = getattr(regime_state, domain, None)
        if regime is None:
            continue

        directions = _classify_regime_direction(domain, regime.label)

        # Weight by velocity if trend data available
        trends = getattr(regime_state, "trends", {}) or {}
        trend = trends.get(domain)
        velocity_boost = 1.0
        if trend:
            vel = abs(getattr(trend, "velocity_7d", 0.0) or 0.0)
            if vel > 0.5:
                velocity_boost = 1.3  # recent momentum amplifies signal
            elif vel > 0.2:
                velocity_boost = 1.1

        for factor, direction in directions.items():
            weighted = direction * velocity_boost
            factor_dirs[factor] = factor_dirs.get(factor, 0.0) + weighted
            factor_counts[factor] = factor_counts.get(factor, 0) + 1

    # Average across contributing domains
    for f in factor_dirs:
        cnt = factor_counts.get(f, 1)
        if cnt > 1:
            factor_dirs[f] /= cnt

    return factor_dirs


def score_regime_alignment(
    *,
    regime_state: Any,
    tickers: list[str],
    ticker_sectors: dict[str, str],
) -> dict[str, RegimeAlignmentSignal]:
    """Score each ticker's alignment with the current macro regime.

    Args:
        regime_state: UnifiedRegimeState with pillar scores and trends.
        ticker_sectors: ticker -> GICS sector string (from FMP profiles).
    """
    if regime_state is None:
        return {}

    factor_dirs = _regime_factor_directions(regime_state)
    if not factor_dirs:
        return {}

    # Build regime context string
    context_parts = []
    for domain in ("growth", "volatility", "rates", "credit"):
        regime = getattr(regime_state, domain, None)
        if regime and regime.label:
            score = regime.score
            if score < 35 or score > 65:
                context_parts.append(f"{domain} {regime.label.lower()}")
    regime_context = ", ".join(context_parts[:3]) if context_parts else "mixed"

    out: dict[str, RegimeAlignmentSignal] = {}
    for ticker in tickers:
        exposures = _get_exposures(ticker, ticker_sectors)
        if not exposures:
            out[ticker] = RegimeAlignmentSignal(
                ticker=ticker,
                alignment="NEUTRAL",
                primary_factor="unknown",
                regime_context=regime_context,
                sub_score=50.0,
            )
            continue

        # Dot-product: factor exposure × regime direction
        alignment_score = 0.0
        max_contribution = 0.0
        primary_factor = "equity_beta"

        for factor, loading in exposures.items():
            if abs(loading) < 0.1:
                continue
            direction = factor_dirs.get(factor, 0.0)
            contribution = loading * direction
            alignment_score += contribution
            if abs(contribution) > abs(max_contribution):
                max_contribution = contribution
                primary_factor = factor

        # Normalize to 0-100: alignment_score typically ranges [-3, +3]
        # Map: -2 = 0, 0 = 50, +2 = 100
        normalized = max(0.0, min(100.0, 50.0 + alignment_score * 25.0))

        if normalized >= 65:
            alignment = "TAILWIND"
        elif normalized <= 35:
            alignment = "HEADWIND"
        else:
            alignment = "NEUTRAL"

        out[ticker] = RegimeAlignmentSignal(
            ticker=ticker,
            alignment=alignment,
            primary_factor=primary_factor,
            regime_context=regime_context,
            sub_score=round(normalized, 1),
        )

    return out

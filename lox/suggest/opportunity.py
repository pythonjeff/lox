"""
Composite scoring for the opportunity scanner.

Combines 4 signal pillars with regime-conditional weights.
Applies anti-staleness adjustments and classifies each candidate
by its dominant signal type.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Regime-conditional pillar weights — different regimes prioritize different signals
PILLAR_WEIGHTS: dict[str, dict[str, float]] = {
    "RISK_ON":     {"momentum": 0.30, "flow": 0.30, "regime": 0.20, "catalyst": 0.20},
    "REFLATION":   {"momentum": 0.25, "flow": 0.25, "regime": 0.30, "catalyst": 0.20},
    "STAGFLATION": {"momentum": 0.20, "flow": 0.25, "regime": 0.35, "catalyst": 0.20},
    "RISK_OFF":    {"momentum": 0.20, "flow": 0.30, "regime": 0.30, "catalyst": 0.20},
    "TRANSITION":  {"momentum": 0.25, "flow": 0.30, "regime": 0.25, "catalyst": 0.20},
}

_SIGNAL_TYPE_MAP = {
    "regime": "REGIME_TAILWIND",
    "flow": "FLOW_ACCELERATION",
    "momentum": "REVERSION_SETUP",
    "catalyst": "CATALYST_DRIVEN",
}


@dataclass
class ScoredOpportunity:
    ticker: str
    name: str
    direction: str  # LONG or SHORT
    composite_score: float  # 0-100
    conviction: str  # HIGH, MEDIUM, LOW

    # Signal pillar sub-scores
    momentum_score: float
    flow_score: float
    regime_score: float
    catalyst_score: float

    # Key metrics
    price: float
    change_pct: float
    volume_surge: float
    zscore_20d: float

    # Classification
    signal_type: str  # REGIME_TAILWIND, FLOW_ACCELERATION, REVERSION_SETUP, CATALYST_DRIVEN
    sector: str
    is_etf: bool
    thesis: str

    # Anti-staleness
    rotation_penalty: float = 0.0
    days_since_last_rec: int | None = None

    # Track record fields (populated after logging)
    price_at_rec: float = 0.0


def _build_scenario_map(regime_state: Any) -> dict[str, str]:
    """Extract active scenario trade expressions → {ticker: direction}.

    If an active scenario says LONG GLD, returns {"GLD": "LONG"}.
    This lets us detect conflicts (e.g. shorting oil during geopolitical stress).
    """
    scenario_trades: dict[str, str] = {}
    if regime_state is None:
        return scenario_trades

    try:
        from lox.regimes.scenarios import evaluate_scenarios, SCENARIOS
        active = evaluate_scenarios(regime_state, SCENARIOS)
        for s in active:
            for t in s.trades:
                ticker = t.ticker.upper()
                direction = t.direction.upper()
                if direction in ("LONG", "SHORT"):
                    # If multiple scenarios disagree, last wins (rare)
                    scenario_trades[ticker] = direction
    except Exception:
        pass

    return scenario_trades


def compute_opportunity_scores(
    *,
    tickers: list[str],
    momentum_signals: dict[str, Any],
    flow_signals: dict[str, Any],
    regime_signals: dict[str, Any],
    catalyst_signals: dict[str, Any],
    quote_data: dict[str, dict[str, Any]],
    ticker_names: dict[str, str],
    ticker_sectors: dict[str, str],
    etf_tickers: set[str],
    composite_regime: str,
    rotation_penalties: dict[str, float],
    weight_adjustments: dict[str, float] | None = None,
    regime_state: Any = None,
) -> list[ScoredOpportunity]:
    """Score all candidates and return sorted by composite score.

    Args:
        weight_adjustments: from track record self-evaluation.
            Maps pillar name -> multiplier (e.g. {"flow": 1.1, "catalyst": 0.8}).
        regime_state: UnifiedRegimeState for scenario integration.
    """
    # Get base weights for current regime
    regime_key = composite_regime.upper().replace(" ", "_")
    # Match to closest known regime
    if "RISK_ON" in regime_key or "GOLDILOCKS" in regime_key:
        base_weights = PILLAR_WEIGHTS["RISK_ON"]
    elif "REFLATION" in regime_key:
        base_weights = PILLAR_WEIGHTS["REFLATION"]
    elif "STAGFLATION" in regime_key:
        base_weights = PILLAR_WEIGHTS["STAGFLATION"]
    elif "RISK_OFF" in regime_key or "DEFLAT" in regime_key:
        base_weights = PILLAR_WEIGHTS["RISK_OFF"]
    else:
        base_weights = PILLAR_WEIGHTS["TRANSITION"]

    # Apply track record weight adjustments
    weights = dict(base_weights)
    if weight_adjustments:
        for pillar, multiplier in weight_adjustments.items():
            if pillar in weights:
                weights[pillar] *= multiplier
        # Re-normalize to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

    # Build scenario trade map for conflict detection
    scenario_trades = _build_scenario_map(regime_state)

    scored: list[ScoredOpportunity] = []
    for ticker in tickers:
        mom = momentum_signals.get(ticker)
        flw = flow_signals.get(ticker)
        reg = regime_signals.get(ticker)
        cat = catalyst_signals.get(ticker)
        q = quote_data.get(ticker, {})

        mom_score = mom.sub_score if mom else 0.0
        flow_score = flw.sub_score if flw else 0.0
        raw_regime_score = reg.sub_score if reg else 50.0
        cat_score = cat.sub_score if cat else 0.0

        # ── Direction inference (multi-signal, not just momentum) ──
        # Use a voting system: each signal contributes a directional vote.
        # This prevents momentum from blindly overriding regime/flow signals.
        long_votes = 0.0
        short_votes = 0.0

        if mom:
            if mom.signal == "EXTENDED_UP":
                short_votes += 3.0  # strong: mean-reversion short
            elif mom.signal == "EXTENDED_DOWN":
                long_votes += 3.0  # strong: mean-reversion long
            elif mom.signal == "BREAKOUT":
                long_votes += 2.0  # momentum continuation
            elif mom.signal == "OVERSOLD_BOUNCE":
                long_votes += 2.5  # dip buy in uptrend
            elif mom.signal == "TRENDING_UP":
                long_votes += 1.0
            elif mom.signal == "TRENDING_DOWN":
                short_votes += 1.0

            # Trend quality modifies momentum direction
            trend = getattr(mom, "trend_quality", "RANGE_BOUND")
            if trend == "PULLBACK_IN_UPTREND":
                long_votes += 1.5  # buy the dip
            elif trend == "BREAKDOWN":
                short_votes += 1.0
            elif trend == "STRONG_DOWN":
                short_votes += 0.5

        if flw:
            if flw.flow_direction == "ACCUMULATION":
                long_votes += 2.0
            elif flw.flow_direction == "DISTRIBUTION":
                short_votes += 2.0

        if reg:
            if reg.alignment == "TAILWIND":
                long_votes += 1.5
            elif reg.alignment == "HEADWIND":
                short_votes += 1.5

        # Scenario override: strong signal
        scenario_dir = scenario_trades.get(ticker)
        if scenario_dir == "LONG":
            long_votes += 2.0
        elif scenario_dir == "SHORT":
            short_votes += 2.0

        direction = "SHORT" if short_votes > long_votes else "LONG"

        # ── Direction-aware regime score ──
        # The raw regime score measures "does this ticker have macro tailwinds?"
        # If we're going SHORT a ticker with tailwinds, we're fighting the regime.
        # Invert the score for SHORT positions: high tailwind → low effective score.
        if direction == "SHORT" and raw_regime_score > 50:
            # Shorting into a tailwind: invert (82 → 18, 65 → 35)
            regime_score = 100.0 - raw_regime_score
        elif direction == "LONG" and raw_regime_score < 50:
            # Going long into a headwind: keep as-is (low score = bad)
            regime_score = raw_regime_score
        else:
            regime_score = raw_regime_score

        # ── Scenario conflict penalty ──
        # If active scenarios explicitly recommend the opposite direction, penalize
        scenario_penalty = 0.0
        scenario_dir = scenario_trades.get(ticker)
        if scenario_dir and scenario_dir != direction:
            # Scenario says LONG but we say SHORT (or vice versa) → big penalty
            scenario_penalty = 20.0
        elif scenario_dir and scenario_dir == direction:
            # Scenario confirms our direction → bonus
            scenario_penalty = -10.0  # negative penalty = bonus

        # Weighted composite
        base = (
            weights["momentum"] * mom_score
            + weights["flow"] * flow_score
            + weights["regime"] * regime_score
            + weights["catalyst"] * cat_score
        )

        # Rotation penalty
        penalty = rotation_penalties.get(ticker, 0.0)

        # Novelty bonus for never-recommended tickers
        novelty = 5.0 if ticker not in rotation_penalties else 0.0

        composite = max(0.0, min(100.0, base - penalty + novelty - scenario_penalty))

        # Conviction
        if composite >= 70:
            conviction = "HIGH"
        elif composite >= 45:
            conviction = "MEDIUM"
        else:
            conviction = "LOW"

        # Signal type = dominant pillar
        pillar_scores = {
            "momentum": mom_score,
            "flow": flow_score,
            "regime": regime_score,
            "catalyst": cat_score,
        }
        dominant = max(pillar_scores, key=pillar_scores.get)
        signal_type = _SIGNAL_TYPE_MAP.get(dominant, "REVERSION_SETUP")

        # Build thesis (include scenario conflict warning)
        thesis = _build_thesis(
            ticker, direction, mom, flw, reg, cat, ticker_names,
            scenario_conflict=(scenario_dir and scenario_dir != direction),
        )

        price = 0.0
        change_pct = 0.0
        try:
            price = float(q.get("price", 0))
            change_pct = float(q.get("changesPercentage", 0))
        except (ValueError, TypeError):
            pass

        scored.append(ScoredOpportunity(
            ticker=ticker,
            name=ticker_names.get(ticker, ticker),
            direction=direction,
            composite_score=round(composite, 1),
            conviction=conviction,
            momentum_score=round(mom_score, 1),
            flow_score=round(flow_score, 1),
            regime_score=round(regime_score, 1),
            catalyst_score=round(cat_score, 1),
            price=price,
            change_pct=round(change_pct, 2),
            volume_surge=round(flw.volume_surge if flw else 1.0, 2),
            zscore_20d=round(mom.zscore_20d if mom else 0.0, 2),
            signal_type=signal_type,
            sector=ticker_sectors.get(ticker, ""),
            is_etf=ticker in etf_tickers,
            thesis=thesis,
            rotation_penalty=penalty,
            days_since_last_rec=None,
            price_at_rec=price,
        ))

    scored.sort(key=lambda x: x.composite_score, reverse=True)
    return scored


def _build_thesis(
    ticker: str,
    direction: str,
    mom: Any | None,
    flw: Any | None,
    reg: Any | None,
    cat: Any | None,
    ticker_names: dict[str, str],
    scenario_conflict: bool = False,
) -> str:
    """Generate a concise 1-line thesis from signal data."""
    from lox.suggest.cross_asset import TICKER_DESC

    parts = []

    # Lead with the setup type (what kind of trade is this?)
    if mom:
        trend = getattr(mom, "trend_quality", "RANGE_BOUND")
        signal = mom.signal
        if signal == "BREAKOUT":
            parts.append("breakout near 52w high")
        elif signal == "OVERSOLD_BOUNCE":
            parts.append("pullback in uptrend")
        elif signal == "EXTENDED_UP":
            parts.append(f"overbought {mom.zscore_20d:+.1f}σ")
        elif signal == "EXTENDED_DOWN":
            parts.append(f"oversold {mom.zscore_20d:+.1f}σ")
        elif trend == "PULLBACK_IN_UPTREND" and direction == "LONG":
            parts.append("dip buy opportunity")
        elif trend == "BREAKDOWN":
            parts.append("trend breakdown")
        elif trend == "STRONG_UP" and direction == "LONG":
            parts.append("uptrend continuation")

    # Scenario conflict warning
    if scenario_conflict:
        parts.append("⚠ scenario conflict")

    # Regime context — be specific, not "mixed"
    if reg and reg.alignment != "NEUTRAL" and reg.regime_context != "mixed":
        if direction == "SHORT" and reg.alignment == "TAILWIND":
            parts.append("regime tailwind (caution)")
        elif direction == "LONG" and reg.alignment == "HEADWIND":
            parts.append("regime headwind (caution)")
        else:
            parts.append(reg.regime_context)

    # Flow signal (only when meaningful)
    if flw and flw.volume_surge > 2.0:
        flow_dir = "buying" if flw.flow_direction == "ACCUMULATION" else "selling" if flw.flow_direction == "DISTRIBUTION" else "vol"
        parts.append(f"{flw.volume_surge:.1f}x {flow_dir}")
    if flw and flw.short_interest_pct and flw.short_interest_pct > 15:
        parts.append(f"{flw.short_interest_pct:.0f}% SI")

    # Relative strength (from momentum)
    if mom and hasattr(mom, "rel_strength_20d"):
        rs = mom.rel_strength_20d
        if rs > 0.06:
            parts.append("strong vs SPY")
        elif rs < -0.06:
            parts.append("weak vs SPY")

    # Catalyst
    if cat:
        if cat.days_to_earnings is not None and cat.days_to_earnings <= 14:
            parts.append(f"earnings in {cat.days_to_earnings}d")
        elif getattr(cat, "recent_gap_pct", 0) > 0.03:
            parts.append(f"{cat.recent_gap_pct:.0%} gap")

    # Fallback: add ticker description if we have nothing interesting
    if not parts:
        desc = TICKER_DESC.get(ticker) or ticker_names.get(ticker, ticker)
        parts.append(desc)

    return "; ".join(parts[:5])

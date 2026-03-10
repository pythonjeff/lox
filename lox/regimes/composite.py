"""
Composite regime identification — hedge-fund-style macro regime classification.

Compresses 12 pillar scores into 5 named macro regimes via distance-based
prototype matching.  Produces:

- Headline regime + confidence score
- Transition outlook (where we're heading next month)
- Swing factors (which pillars are closest to flipping the regime)
- Canonical playbook (positioning guidance per regime)
- Playbook deviation (where the book disagrees with the playbook)

Usage:
    from lox.regimes.composite import classify_composite_regime
    result = classify_composite_regime(unified_state)
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lox.regimes.features import UnifiedRegimeState
    from lox.regimes.trend import RegimeTrend

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

COMPOSITE_REGIMES = [
    "RISK_ON",
    "REFLATION",
    "STAGFLATION",
    "RISK_OFF",
    "TRANSITION",
]

COMPOSITE_LABELS: dict[str, str] = {
    "RISK_ON":     "RISK-ON / GOLDILOCKS",
    "REFLATION":   "REFLATION",
    "STAGFLATION": "STAGFLATION",
    "RISK_OFF":    "RISK-OFF / DEFLATIONARY",
    "TRANSITION":  "TRANSITION / MIXED",
}

COMPOSITE_DESCRIPTIONS: dict[str, str] = {
    "RISK_ON": (
        "Benign macro: low vol, tight credit, stable growth, contained "
        "inflation. Deploy risk and harvest carry."
    ),
    "REFLATION": (
        "Growth strong with rising inflation and commodity strength. "
        "Favor real assets, cyclicals, short duration."
    ),
    "STAGFLATION": (
        "Growth weakening while inflation stays sticky. Worst of both "
        "worlds — bonds and equities both under pressure."
    ),
    "RISK_OFF": (
        "Growth declining, vol spiking, credit widening. "
        "De-risk, raise cash, add tail hedges."
    ),
    "TRANSITION": (
        "Mixed signals across pillars. High dispersion, low conviction. "
        "Reduce gross, wait for clarity."
    ),
}

# Pillars used for classification (ordered)
CLASSIFICATION_PILLARS = [
    "growth", "inflation", "volatility", "credit",
    "rates", "liquidity", "consumer", "commodities",
]

# Pillar weights — higher = more influence on classification
PILLAR_WEIGHTS: dict[str, float] = {
    "growth":      2.0,
    "inflation":   1.5,
    "volatility":  1.5,
    "credit":      1.5,
    "rates":       1.0,
    "liquidity":   0.8,
    "consumer":    0.7,
    "commodities": 0.8,
}

# Prototype score vectors — ideal scores (0-100, higher=stress) per regime
REGIME_PROTOTYPES: dict[str, dict[str, float]] = {
    "RISK_ON": {
        "growth": 25, "inflation": 30, "volatility": 20,
        "credit": 20, "rates": 35, "liquidity": 25,
        "consumer": 25, "commodities": 35,
    },
    "REFLATION": {
        "growth": 25, "inflation": 65, "volatility": 35,
        "credit": 35, "rates": 60, "liquidity": 35,
        "consumer": 35, "commodities": 70,
    },
    "STAGFLATION": {
        "growth": 70, "inflation": 70, "volatility": 55,
        "credit": 55, "rates": 55, "liquidity": 50,
        "consumer": 65, "commodities": 60,
    },
    "RISK_OFF": {
        "growth": 75, "inflation": 30, "volatility": 75,
        "credit": 75, "rates": 40, "liquidity": 65,
        "consumer": 70, "commodities": 30,
    },
    "TRANSITION": {
        "growth": 50, "inflation": 50, "volatility": 50,
        "credit": 50, "rates": 50, "liquidity": 50,
        "consumer": 50, "commodities": 50,
    },
}

# Softmax temperature — lower = more decisive, higher = more uniform
_TEMPERATURE = 15.0

# TRANSITION override thresholds
_DISPERSION_THRESHOLD = 18.0
_CONFIDENCE_THRESHOLD = 0.40


# ═════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SwingFactor:
    """A pillar close to causing a regime change."""

    pillar: str
    current_score: float
    target_regime: str
    target_score: float
    distance_to_flip: float
    direction: str                        # "UP" or "DOWN"
    velocity_7d: Optional[float] = None
    days_to_flip: Optional[float] = None  # Estimated days at current velocity


@dataclass(frozen=True)
class RegimePlaybook:
    """Canonical positioning for a composite regime."""

    regime: str
    equity_stance: str       # "OVERWEIGHT" | "NEUTRAL" | "UNDERWEIGHT"
    duration_stance: str     # "LONG" | "NEUTRAL" | "SHORT"
    credit_stance: str       # "OVERWEIGHT" | "NEUTRAL" | "UNDERWEIGHT"
    commodity_stance: str    # "OVERWEIGHT" | "NEUTRAL" | "UNDERWEIGHT"
    vol_stance: str          # "SELL" | "NEUTRAL" | "BUY"
    cash_target_pct: float
    gross_exposure: str      # "FULL" | "REDUCED" | "MINIMAL"
    key_expressions: tuple[tuple[str, str, str], ...]
    # Each tuple: (direction, ticker, rationale)


@dataclass(frozen=True)
class CompositeRegimeResult:
    """Full result of composite regime classification."""

    # Primary
    regime: str
    label: str
    description: str
    confidence: float                          # 0-1

    # Distances & probabilities
    distances: dict[str, float]
    regime_probabilities: dict[str, float]

    # Current score vector used for classification
    score_vector: dict[str, float]

    # Transition outlook (next month)
    transition_outlook: dict[str, float]

    # Swing factors — pillars closest to flipping regime
    swing_factors: list[SwingFactor]

    # Score dispersion across pillars
    pillar_dispersion: float

    # Playbook
    playbook: RegimePlaybook

    def to_dict(self) -> dict:
        """Serialize for JSON output and LLM context."""
        return {
            "regime": self.regime,
            "label": self.label,
            "description": self.description,
            "confidence": round(self.confidence, 2),
            "regime_probabilities": {
                k: round(v, 3) for k, v in self.regime_probabilities.items()
            },
            "transition_outlook": {
                k: round(v, 3) for k, v in self.transition_outlook.items()
            },
            "swing_factors": [
                {
                    "pillar": sf.pillar,
                    "current_score": round(sf.current_score, 1),
                    "target_regime": sf.target_regime,
                    "target_score": round(sf.target_score, 1),
                    "distance": round(sf.distance_to_flip, 1),
                    "direction": sf.direction,
                    "velocity_7d": round(sf.velocity_7d, 2) if sf.velocity_7d else None,
                    "days_to_flip": round(sf.days_to_flip, 0) if sf.days_to_flip else None,
                }
                for sf in self.swing_factors[:5]
            ],
            "pillar_dispersion": round(self.pillar_dispersion, 1),
            "score_vector": {k: round(v, 1) for k, v in self.score_vector.items()},
            "playbook": {
                "equity": self.playbook.equity_stance,
                "duration": self.playbook.duration_stance,
                "credit": self.playbook.credit_stance,
                "commodity": self.playbook.commodity_stance,
                "vol": self.playbook.vol_stance,
                "cash_pct": self.playbook.cash_target_pct,
                "gross": self.playbook.gross_exposure,
                "key_expressions": [
                    {"direction": d, "ticker": t, "rationale": r}
                    for d, t, r in self.playbook.key_expressions
                ],
            },
        }


# ═════════════════════════════════════════════════════════════════════════════
# Playbook Definitions
# ═════════════════════════════════════════════════════════════════════════════

PLAYBOOKS: dict[str, RegimePlaybook] = {
    "RISK_ON": RegimePlaybook(
        regime="RISK_ON",
        equity_stance="OVERWEIGHT",
        duration_stance="SHORT",
        credit_stance="OVERWEIGHT",
        commodity_stance="NEUTRAL",
        vol_stance="SELL",
        cash_target_pct=5.0,
        gross_exposure="FULL",
        key_expressions=(
            ("LONG", "SPY", "Broad equity — ride the momentum"),
            ("LONG", "QQQ", "Growth/tech outperforms in low-rate, low-vol"),
            ("LONG", "HYG", "Harvest carry — spreads tight and tightening"),
            ("SHORT", "TLT", "Duration risk — rates drift higher in growth"),
            ("SHORT", "VXX", "Sell vol — premium decay in calm markets"),
        ),
    ),
    "REFLATION": RegimePlaybook(
        regime="REFLATION",
        equity_stance="OVERWEIGHT",
        duration_stance="SHORT",
        credit_stance="NEUTRAL",
        commodity_stance="OVERWEIGHT",
        vol_stance="NEUTRAL",
        cash_target_pct=10.0,
        gross_exposure="FULL",
        key_expressions=(
            ("LONG", "XLE", "Energy producers — direct commodity exposure"),
            ("LONG", "DBC", "Broad commodities — inflation pass-through"),
            ("LONG", "TIP", "TIPS — breakevens widen in reflation"),
            ("SHORT", "TLT", "Short duration — rates reprice for growth + inflation"),
            ("LONG", "XLI", "Cyclicals/industrials — capex beneficiaries"),
        ),
    ),
    "STAGFLATION": RegimePlaybook(
        regime="STAGFLATION",
        equity_stance="UNDERWEIGHT",
        duration_stance="SHORT",
        credit_stance="UNDERWEIGHT",
        commodity_stance="OVERWEIGHT",
        vol_stance="BUY",
        cash_target_pct=25.0,
        gross_exposure="REDUCED",
        key_expressions=(
            ("LONG", "GLD", "Gold — inflation hedge + uncertainty premium"),
            ("SHORT", "XLY", "Short discretionary — consumer squeezed"),
            ("LONG", "XLE", "Energy — inflation pass-through"),
            ("SHORT", "TLT", "Short duration — real yields rising"),
            ("LONG", "TIP", "TIPS over nominals — breakevens widen"),
        ),
    ),
    "RISK_OFF": RegimePlaybook(
        regime="RISK_OFF",
        equity_stance="UNDERWEIGHT",
        duration_stance="LONG",
        credit_stance="UNDERWEIGHT",
        commodity_stance="UNDERWEIGHT",
        vol_stance="BUY",
        cash_target_pct=30.0,
        gross_exposure="MINIMAL",
        key_expressions=(
            ("LONG", "TLT", "Flight to safety — rate cuts being priced"),
            ("LONG", "GLD", "Safe haven demand in multi-asset stress"),
            ("SHORT", "HYG", "Credit spreads blow out — buy protection"),
            ("SHORT", "SPY", "Equity downside protection"),
            ("LONG", "UVXY", "Long vol — convexity in tail events"),
        ),
    ),
    "TRANSITION": RegimePlaybook(
        regime="TRANSITION",
        equity_stance="NEUTRAL",
        duration_stance="NEUTRAL",
        credit_stance="NEUTRAL",
        commodity_stance="NEUTRAL",
        vol_stance="BUY",
        cash_target_pct=20.0,
        gross_exposure="REDUCED",
        key_expressions=(
            ("LONG", "GLD", "Optionality on regime resolution"),
            ("LONG", "BIL", "Park cash at short-duration safety"),
            ("LONG", "UVXY", "Buy vol — regime transitions spike vol"),
            ("REDUCE", "SPY", "Trim equity — wait for clarity"),
        ),
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# Classification Engine
# ═════════════════════════════════════════════════════════════════════════════

def _extract_score_vector(state: UnifiedRegimeState) -> dict[str, float]:
    """Extract score vector from state for classification pillars."""
    scores: dict[str, float] = {}
    for pillar in CLASSIFICATION_PILLARS:
        regime = getattr(state, pillar, None)
        scores[pillar] = regime.score if regime is not None else 50.0
    return scores


def _weighted_euclidean_distance(
    scores: dict[str, float],
    prototype: dict[str, float],
) -> float:
    """Weighted Euclidean distance between score vector and regime prototype."""
    dist_sq = 0.0
    for pillar in CLASSIFICATION_PILLARS:
        diff = scores.get(pillar, 50.0) - prototype.get(pillar, 50.0)
        weight = PILLAR_WEIGHTS.get(pillar, 1.0)
        dist_sq += weight * (diff ** 2)
    return math.sqrt(dist_sq)


def _compute_dispersion(scores: dict[str, float]) -> float:
    """Standard deviation of pillar scores — high = mixed signals."""
    values = [scores.get(p, 50.0) for p in CLASSIFICATION_PILLARS]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _softmax_probabilities(distances: dict[str, float]) -> dict[str, float]:
    """Convert distances to probabilities via softmax (closer = higher prob)."""
    min_dist = min(distances.values())
    exp_scores = {
        r: math.exp(-(d - min_dist) / _TEMPERATURE)
        for r, d in distances.items()
    }
    total = sum(exp_scores.values())
    return {r: v / total for r, v in exp_scores.items()}


def _compute_swing_factors(
    scores: dict[str, float],
    current_regime: str,
    trends: dict[str, Any],
) -> list[SwingFactor]:
    """
    Find which pillars are closest to flipping the regime classification.

    For each non-current regime, identify the pillar whose movement would
    most reduce distance to that regime.  Returns top 8 sorted by
    distance_to_flip ascending.
    """
    all_factors: list[SwingFactor] = []

    for target_regime, prototype in REGIME_PROTOTYPES.items():
        if target_regime == current_regime or target_regime == "TRANSITION":
            continue

        for pillar in CLASSIFICATION_PILLARS:
            current = scores.get(pillar, 50.0)
            target = prototype.get(pillar, 50.0)
            diff = abs(current - target)
            if diff < 3.0:
                continue  # Already aligned, not a swing factor

            weight = PILLAR_WEIGHTS.get(pillar, 1.0)
            weighted_diff = diff * math.sqrt(weight)
            direction = "UP" if target > current else "DOWN"

            # Estimate ETA using pillar velocity
            trend = trends.get(pillar)
            velocity = getattr(trend, "velocity_7d", None) if trend else None
            days_to_flip = None
            if velocity is not None and abs(velocity) > 0.1:
                # Velocity moving in the flip direction?
                if (direction == "UP" and velocity > 0) or \
                   (direction == "DOWN" and velocity < 0):
                    days_to_flip = diff / abs(velocity)

            all_factors.append(SwingFactor(
                pillar=pillar,
                current_score=current,
                target_regime=target_regime,
                target_score=target,
                distance_to_flip=weighted_diff,
                direction=direction,
                velocity_7d=velocity,
                days_to_flip=days_to_flip,
            ))

    # Sort: active velocity toward target gets priority (0.5x distance)
    all_factors.sort(key=lambda sf: (
        sf.distance_to_flip * 0.5 if sf.days_to_flip is not None
        else sf.distance_to_flip
    ))

    # Deduplicate: keep only the closest target per pillar
    seen_pillars: set[str] = set()
    deduped: list[SwingFactor] = []
    for sf in all_factors:
        if sf.pillar not in seen_pillars:
            seen_pillars.add(sf.pillar)
            deduped.append(sf)

    return deduped[:8]


def _compute_transition_outlook(
    scores: dict[str, float],
    current_probs: dict[str, float],
    trends: dict[str, Any],
) -> dict[str, float]:
    """
    Estimate probability of being in each regime next month.

    Projects scores forward 21 days using velocity_7d, then reclassifies
    on the projected vector.  Blends 70/30 with current probs for stability.
    """
    # Project scores forward 21 trading days
    projected: dict[str, float] = {}
    for pillar in CLASSIFICATION_PILLARS:
        current = scores.get(pillar, 50.0)
        trend = trends.get(pillar)
        velocity = getattr(trend, "velocity_7d", None) if trend else None
        if velocity is not None:
            projected[pillar] = max(0.0, min(100.0, current + velocity * 21))
        else:
            projected[pillar] = current

    # Distances from projected scores
    proj_distances = {
        regime: _weighted_euclidean_distance(projected, proto)
        for regime, proto in REGIME_PROTOTYPES.items()
    }
    proj_probs = _softmax_probabilities(proj_distances)

    # Blend: 70% projected, 30% current
    blended = {
        r: 0.7 * proj_probs.get(r, 0.2) + 0.3 * current_probs.get(r, 0.2)
        for r in COMPOSITE_REGIMES
    }
    total = sum(blended.values())
    return {r: v / total for r, v in blended.items()}


def classify_composite_regime(
    state: UnifiedRegimeState,
) -> CompositeRegimeResult:
    """
    Classify current state into one of 5 composite macro regimes.

    Algorithm:
    1. Extract score vector from 8 classification pillars
    2. Compute weighted Euclidean distance to each regime prototype
    3. Convert distances to probabilities via softmax
    4. Apply TRANSITION override if dispersion high + no clear winner
    5. Compute swing factors (pillars closest to flipping regime)
    6. Estimate transition outlook via velocity projection
    """
    scores = _extract_score_vector(state)
    dispersion = _compute_dispersion(scores)

    # Distances to all prototypes
    distances = {
        regime: _weighted_euclidean_distance(scores, proto)
        for regime, proto in REGIME_PROTOTYPES.items()
    }

    # Probabilities via softmax
    probabilities = _softmax_probabilities(distances)

    # Primary regime
    primary = max(probabilities, key=lambda k: probabilities[k])
    primary_prob = probabilities[primary]

    # TRANSITION override: high dispersion + no clear winner
    if dispersion > _DISPERSION_THRESHOLD and primary_prob < _CONFIDENCE_THRESHOLD:
        primary = "TRANSITION"
        probabilities["TRANSITION"] = max(
            probabilities["TRANSITION"], primary_prob + 0.05
        )
        total = sum(probabilities.values())
        probabilities = {r: v / total for r, v in probabilities.items()}

    # Confidence: gap between top and second-highest, scaled to 0-1
    sorted_probs = sorted(probabilities.values(), reverse=True)
    gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
    confidence = min(1.0, gap * 2.5)

    trends = state.trends or {}

    swing_factors = _compute_swing_factors(scores, primary, trends)
    transition_outlook = _compute_transition_outlook(
        scores, probabilities, trends,
    )

    return CompositeRegimeResult(
        regime=primary,
        label=COMPOSITE_LABELS[primary],
        description=COMPOSITE_DESCRIPTIONS[primary],
        confidence=confidence,
        distances=distances,
        regime_probabilities=probabilities,
        score_vector=scores,
        transition_outlook=transition_outlook,
        swing_factors=swing_factors,
        pillar_dispersion=dispersion,
        playbook=PLAYBOOKS[primary],
    )


# ═════════════════════════════════════════════════════════════════════════════
# LLM Formatter
# ═════════════════════════════════════════════════════════════════════════════

def format_composite_for_llm(result: CompositeRegimeResult) -> str:
    """Format composite regime as markdown for LLM system prompt injection."""
    lines = [
        "## Composite Regime",
        "",
        f"**Current Regime: {result.label}** (confidence: {result.confidence:.0%})",
        f"{result.description}",
        "",
        "### Regime Probabilities",
    ]
    for r in COMPOSITE_REGIMES:
        prob = result.regime_probabilities.get(r, 0)
        marker = " <-- current" if r == result.regime else ""
        lines.append(f"  - {COMPOSITE_LABELS[r]}: {prob:.0%}{marker}")

    lines.append("")
    lines.append("### Transition Outlook (next 30 days)")
    for r in COMPOSITE_REGIMES:
        prob = result.transition_outlook.get(r, 0)
        lines.append(f"  - {COMPOSITE_LABELS[r]}: {prob:.0%}")

    if result.swing_factors:
        lines.append("")
        lines.append("### Swing Factors (closest to flipping regime)")
        for sf in result.swing_factors[:5]:
            eta = (
                f"{sf.days_to_flip:.0f}d at current velocity"
                if sf.days_to_flip else "velocity not aligned"
            )
            lines.append(
                f"  - {sf.pillar.upper()} ({sf.current_score:.0f} -> "
                f"{sf.target_score:.0f} for {COMPOSITE_LABELS[sf.target_regime]}): "
                f"{sf.distance_to_flip:.1f} weighted pts {sf.direction}, {eta}"
            )

    lines.append("")
    lines.append(f"### Canonical Playbook: {result.label}")
    pb = result.playbook
    lines.append(
        f"  Equity: {pb.equity_stance} | Duration: {pb.duration_stance} | "
        f"Credit: {pb.credit_stance}"
    )
    lines.append(
        f"  Commodities: {pb.commodity_stance} | Vol: {pb.vol_stance} | "
        f"Cash: {pb.cash_target_pct:.0f}% | Gross: {pb.gross_exposure}"
    )
    lines.append("")
    lines.append("### Key Expressions")
    for direction, ticker, rationale in pb.key_expressions:
        lines.append(f"  - {direction} {ticker}: {rationale}")

    return "\n".join(lines)

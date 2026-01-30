"""
Silver regime classification logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import SilverInputs


@dataclass(frozen=True)
class SilverRegime:
    """Silver market regime classification."""

    name: str
    label: str
    description: str
    tags: tuple[str, ...] = ()

    def __str__(self) -> str:
        return self.label


# Valid regime names for validation
SILVER_REGIME_CHOICES = (
    "silver_rally",
    "silver_breakdown",
    "silver_squeeze",
    "silver_consolidation",
    "silver_capitulation",
    "silver_recovery",
    "silver_neutral",
)


def classify_silver_regime(inputs: SilverInputs) -> SilverRegime:
    """
    Classify current silver market regime based on inputs.

    Regime hierarchy (checked in order):
    1. silver_capitulation - High volume selloff, extreme weakness
    2. silver_squeeze - Rapid rally, often short covering
    3. silver_breakdown - Breaking down, bearish structure
    4. silver_rally - Strong uptrend, bullish structure
    5. silver_recovery - Bouncing from lows
    6. silver_consolidation - Range-bound, low volatility
    7. silver_neutral - Default
    """

    # Extract key metrics with defaults
    ret_5d = inputs.slv_ret_5d_pct or 0
    ret_20d = inputs.slv_ret_20d_pct or 0
    ret_60d = inputs.slv_ret_60d_pct or 0
    zscore_20d = inputs.slv_zscore_20d or 0
    zscore_60d = inputs.slv_zscore_60d or 0
    vol_zscore = inputs.slv_vol_zscore or 0
    volume_ratio = inputs.slv_volume_ratio or 1.0
    above_50ma = inputs.slv_above_50ma
    above_200ma = inputs.slv_above_200ma
    golden_cross = inputs.slv_50ma_above_200ma
    trend_score = inputs.trend_score or 0
    momentum_score = inputs.momentum_score or 0
    gsr_expanding = inputs.gsr_expanding

    # 1. CAPITULATION: Extreme selloff with high volume
    if (
        ret_5d < -8
        and ret_20d < -15
        and volume_ratio > 1.5
        and zscore_20d < -2
    ):
        return SilverRegime(
            name="silver_capitulation",
            label="Capitulation",
            description="High-volume selloff, potential washout",
            tags=("bearish", "extreme", "high_vol"),
        )

    # 2. SQUEEZE: Rapid rally, often short covering
    if (
        ret_5d > 8
        and ret_20d > 10
        and (volume_ratio > 1.3 or zscore_20d > 2)
    ):
        return SilverRegime(
            name="silver_squeeze",
            label="Squeeze",
            description="Rapid rally, possible short covering",
            tags=("bullish", "extreme", "momentum"),
        )

    # 3. BREAKDOWN: Breaking support, bearish structure
    if (
        above_50ma is False
        and above_200ma is False
        and ret_20d < -5
        and (golden_cross is False or trend_score < -30)
    ):
        return SilverRegime(
            name="silver_breakdown",
            label="Breakdown",
            description="Below key MAs, bearish structure",
            tags=("bearish", "trend"),
        )

    # 4. RALLY: Strong uptrend, bullish structure
    if (
        above_50ma is True
        and above_200ma is True
        and golden_cross is True
        and ret_20d > 3
        and trend_score > 30
    ):
        return SilverRegime(
            name="silver_rally",
            label="Rally",
            description="Above key MAs, bullish structure",
            tags=("bullish", "trend"),
        )

    # 5. RECOVERY: Bouncing from lows, early signs of stabilization
    if (
        above_50ma is True
        and above_200ma is False
        and ret_5d > 2
        and ret_20d > 0
    ):
        return SilverRegime(
            name="silver_recovery",
            label="Recovery",
            description="Bouncing from lows, testing resistance",
            tags=("neutral", "transition"),
        )

    # 6. CONSOLIDATION: Range-bound, low volatility
    if (
        abs(ret_20d) < 5
        and abs(ret_60d) < 8
        and vol_zscore < 0.5
        and abs(trend_score) < 20
    ):
        return SilverRegime(
            name="silver_consolidation",
            label="Consolidation",
            description="Range-bound, building energy",
            tags=("neutral", "low_vol"),
        )

    # 7. DEFAULT: Neutral
    return SilverRegime(
        name="silver_neutral",
        label="Neutral",
        description="Mixed signals, no clear regime",
        tags=("neutral",),
    )


def get_regime_color(regime: SilverRegime) -> str:
    """Get display color for regime."""
    if "bearish" in regime.tags:
        return "red"
    elif "bullish" in regime.tags:
        return "green"
    else:
        return "yellow"


def get_put_outlook(regime: SilverRegime, inputs: SilverInputs) -> dict:
    """
    Get outlook for SLV puts based on current regime.
    
    Returns dict with:
    - bias: "favorable", "unfavorable", "neutral"
    - confidence: 0-100
    - notes: list of observations
    """
    notes = []
    bias = "neutral"
    confidence = 50

    # Regime-based bias
    if regime.name == "silver_breakdown":
        bias = "favorable"
        confidence = 70
        notes.append("Bearish structure supports put thesis")
    elif regime.name == "silver_capitulation":
        bias = "neutral"  # Could bounce
        confidence = 40
        notes.append("Extreme weakness, but washout may trigger bounce")
    elif regime.name == "silver_rally":
        bias = "unfavorable"
        confidence = 70
        notes.append("Strong uptrend works against puts")
    elif regime.name == "silver_squeeze":
        bias = "unfavorable"
        confidence = 80
        notes.append("Rapid rally, puts under pressure")
    elif regime.name == "silver_recovery":
        bias = "unfavorable"
        confidence = 55
        notes.append("Bounce in progress, timing risk for puts")
    elif regime.name == "silver_consolidation":
        bias = "neutral"
        confidence = 50
        notes.append("Range-bound, wait for directional break")

    # GSR context
    if inputs.gsr_expanding is True:
        confidence += 5
        notes.append("GSR expanding (silver weakening vs gold)")
    elif inputs.gsr_expanding is False:
        confidence -= 5
        notes.append("GSR contracting (silver strengthening vs gold)")

    # Trend context
    if inputs.trend_score is not None:
        if inputs.trend_score < -50:
            confidence += 10
            notes.append("Strong downtrend supports puts")
        elif inputs.trend_score > 50:
            confidence -= 10
            notes.append("Strong uptrend works against puts")

    # MA context
    if inputs.slv_above_200ma is False:
        notes.append("Below 200-day MA (bearish)")
    else:
        notes.append("Above 200-day MA (bullish)")

    confidence = max(0, min(100, confidence))

    return {
        "bias": bias,
        "confidence": confidence,
        "notes": notes,
    }

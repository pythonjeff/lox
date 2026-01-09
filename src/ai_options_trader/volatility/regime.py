from __future__ import annotations

from dataclasses import dataclass

from ai_options_trader.volatility.models import VolatilityInputs


@dataclass(frozen=True)
class VolatilityRegime:
    name: str
    label: str | None = None
    description: str = ""
    tags: tuple[str, ...] = ()


def classify_volatility_regime(inputs: VolatilityInputs) -> VolatilityRegime:
    """
    Volatility regime classifier (MVP).

    This is intentionally simple and model-friendly:
    - z-scores are vs recent history (not a claim of absolute rarity)
    - term structure inversion and 5d momentum help flag "spike risk"
    """
    z_vix = inputs.z_vix
    z_mom = inputs.z_vix_chg_5d
    z_term = inputs.z_vix_term
    persist = inputs.persist_20d
    score = inputs.vol_pressure_score

    # Shock / stress: high level or sharp rise + persistence
    if (z_vix is not None and z_vix >= 2.0) or (z_mom is not None and z_mom >= 2.0) or (
        persist is not None and persist >= 0.35
    ):
        return VolatilityRegime(
            name="vol_shock",
            label="Vol shock / stress (hedging bid)",
            description="VIX is high and/or rising sharply vs recent history; conditions consistent with risk-off volatility demand.",
            tags=("volatility", "stress"),
        )

    # Elevated: risk appetite more fragile
    if (z_vix is not None and z_vix >= 1.0) or (score is not None and score >= 1.0) or (
        z_term is not None and z_term >= 1.0
    ):
        return VolatilityRegime(
            name="elevated_vol",
            label="Elevated volatility (fragile risk)",
            description="Volatility is elevated vs recent history and/or term structure is less healthy (more backwardation-ish).",
            tags=("volatility", "risk_off"),
        )

    return VolatilityRegime(
        name="normal_vol",
        label="Normal volatility (baseline)",
        description="No strong signal of elevated or spiking volatility vs recent history.",
        tags=("volatility",),
    )



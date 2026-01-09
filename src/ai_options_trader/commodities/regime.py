from __future__ import annotations

from dataclasses import dataclass

from ai_options_trader.commodities.models import CommoditiesInputs


@dataclass(frozen=True)
class CommoditiesRegime:
    name: str
    label: str | None = None
    description: str = ""
    tags: tuple[str, ...] = ()


def classify_commodities_regime(inputs: CommoditiesInputs) -> CommoditiesRegime:
    """
    Commodities regime classifier (MVP).

    Intuition:
    - Commodities reflation / inflation pressure often shows up via energy (WTI) + broad index strength.
    - Energy shocks can behave like a tax on the consumer and a risk-off catalyst for equities.
    """
    score = inputs.commodity_pressure_score

    if bool(inputs.energy_shock):
        return CommoditiesRegime(
            name="energy_shock",
            label="Energy shock (inflation impulse)",
            description="WTI is spiking vs recent history; watch inflation expectations, margins, and rates vol.",
            tags=("commodities", "energy", "inflation"),
        )

    if score is not None and score >= 1.25:
        return CommoditiesRegime(
            name="commodity_reflation",
            label="Commodity reflation (inflation pressure)",
            description="Broad commodities are strengthening vs recent history; consistent with inflation pressure / reflation impulse.",
            tags=("commodities", "inflation"),
        )

    if score is not None and score <= -1.25:
        return CommoditiesRegime(
            name="commodity_disinflation",
            label="Commodity disinflation (pressure easing)",
            description="Commodities are weak vs recent history; consistent with easing inflation impulse / demand softness.",
            tags=("commodities", "disinflation"),
        )

    return CommoditiesRegime(
        name="neutral",
        label="Neutral commodities backdrop",
        description="No strong *recent-momentum* signal of commodity-driven inflation impulse (score uses ~20–60 trading-day returns z-scored vs history).",
        tags=("commodities",),
    )



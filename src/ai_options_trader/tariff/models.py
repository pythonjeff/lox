from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Dict, Optional, List


class TariffInputs(BaseModel):
    # Component scores (z-scored unless otherwise stated)
    z_cost_pressure: Optional[float] = None
    equity_denial_beta: Optional[float] = None  # beta of rel returns vs cost changes (rolling)
    z_earnings_fragility: Optional[float] = None

    # Composite
    tariff_regime_score: Optional[float] = None
    is_tariff_regime: Optional[bool] = None

    # Debug / transparency
    components: Dict[str, Optional[float]] = Field(default_factory=dict)


class TariffRegimeState(BaseModel):
    asof: str
    start_date: str
    basket: str | None = None
    universe: List[str]
    benchmark: str | None = None
    inputs: TariffInputs
    notes: str = ""

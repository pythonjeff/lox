from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel


class CommoditiesInputs(BaseModel):
    # Levels
    wti: Optional[float] = None
    gold: Optional[float] = None
    copper: Optional[float] = None
    broad_index: Optional[float] = None

    # Returns (percent)
    wti_ret_20d_pct: Optional[float] = None
    gold_ret_20d_pct: Optional[float] = None
    copper_ret_60d_pct: Optional[float] = None
    broad_ret_60d_pct: Optional[float] = None

    # Context (z-scores vs recent history)
    z_wti_ret_20d: Optional[float] = None
    z_gold_ret_20d: Optional[float] = None
    z_copper_ret_60d: Optional[float] = None
    z_broad_ret_60d: Optional[float] = None

    # Composite
    commodity_pressure_score: Optional[float] = None

    # Shock flag (energy-driven)
    energy_shock: Optional[bool] = None
    # Industrial metals impulse flag (growth/inflation cross-signal)
    metals_impulse: Optional[bool] = None

    components: Optional[Dict[str, float]] = None


class CommoditiesState(BaseModel):
    asof: str
    start_date: str
    inputs: CommoditiesInputs
    notes: str | None = None



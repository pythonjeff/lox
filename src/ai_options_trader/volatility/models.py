from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel


class VolatilityInputs(BaseModel):
    # Levels
    vix: Optional[float] = None
    vix9d: Optional[float] = None
    vix3m: Optional[float] = None

    # Momentum / shock
    vix_chg_1d_pct: Optional[float] = None
    vix_chg_5d_pct: Optional[float] = None

    # Term structure (spot - 3m); positive implies backwardation-ish / stress
    vix_term_spread: Optional[float] = None

    # Context (z-scores vs recent history)
    z_vix: Optional[float] = None
    z_vix_chg_5d: Optional[float] = None
    z_vix_term: Optional[float] = None

    # Spike / persistence indicators (like funding regime, but for VIX)
    spike_20d_pct: Optional[float] = None
    persist_20d: Optional[float] = None
    threshold_vix: Optional[float] = None

    # Composite (positive => more volatility pressure)
    vol_pressure_score: Optional[float] = None

    components: Optional[Dict[str, float]] = None


class VolatilityState(BaseModel):
    asof: str
    start_date: str
    inputs: VolatilityInputs
    notes: str | None = None



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
    # Source for the 3m anchor used in term structure:
    # - "fred:VIX3M" when available
    # - "fmp:VXV" as a best-effort proxy (3-month VIX index level) when FRED lacks VIX3M
    vix_term_source: Optional[str] = None

    # Context (z-scores vs recent history)
    z_vix: Optional[float] = None
    z_vix_chg_5d: Optional[float] = None
    z_vix_term: Optional[float] = None

    # Spike / persistence indicators (like funding regime, but for VIX)
    spike_20d_pct: Optional[float] = None
    persist_20d: Optional[float] = None
    threshold_vix: Optional[float] = None

    # VX futures term structure (from CBOE settlement CSV)
    vx_m1: Optional[float] = None          # front-month VX settle price
    vx_m2: Optional[float] = None          # second-month VX settle price
    vx_contango_pct: Optional[float] = None  # (M2/M1 - 1)*100; negative = backwardation
    spot_minus_m1: Optional[float] = None    # VIX spot - M1 settle; positive = spot premium (panic)
    spot_basis_pct: Optional[float] = None   # (spot/M1 - 1)*100; positive = backwardation signal
    vix9d_vix_ratio: Optional[float] = None  # VIX9D/VIX; >1 = near-term fear exceeds 30d

    # Composite (positive => more volatility pressure)
    vol_pressure_score: Optional[float] = None

    components: Optional[Dict[str, float]] = None


class VolatilityState(BaseModel):
    asof: str
    start_date: str
    inputs: VolatilityInputs
    notes: str | None = None



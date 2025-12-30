from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class LiquidityInputs(BaseModel):
    # Core levels
    ust_10y: Optional[float] = None
    hy_oas: Optional[float] = None
    ig_oas: Optional[float] = None
    baa10ym: Optional[float] = None  # Baa corporate yield - 10Y treasury (if available)

    # Dynamics
    ust_10y_chg_20d_bps: Optional[float] = None
    ust_10y_chg_60d_bps: Optional[float] = None

    # Standardized readings
    z_hy_oas: Optional[float] = None
    z_ig_oas: Optional[float] = None
    z_ust_10y_chg_20d: Optional[float] = None

    # Composite
    liquidity_tightness_score: Optional[float] = None
    is_liquidity_tight: Optional[bool] = None

    # Debug / transparency
    components: Dict[str, Optional[float]] = Field(default_factory=dict)


class LiquidityState(BaseModel):
    asof: str
    start_date: str
    inputs: LiquidityInputs
    notes: str = ""



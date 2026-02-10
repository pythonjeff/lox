from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class UsdInputs(BaseModel):
    # Core level (trade-weighted broad dollar index)
    usd_index_broad: Optional[float] = None

    # Dynamics (%)
    usd_chg_20d_pct: Optional[float] = None
    usd_chg_60d_pct: Optional[float] = None

    # Standardized readings (z)
    z_usd_level: Optional[float] = None
    z_usd_chg_60d: Optional[float] = None

    # Composite (positive = stronger USD regime)
    usd_strength_score: Optional[float] = None
    is_usd_strong: Optional[bool] = None
    is_usd_weak: Optional[bool] = None

    # Debug / transparency
    components: Dict[str, Optional[float]] = Field(default_factory=dict)


class UsdState(BaseModel):
    asof: str
    start_date: str
    inputs: UsdInputs
    notes: str = ""



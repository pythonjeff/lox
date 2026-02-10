from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class MonetaryInputs(BaseModel):
    # Core MVP series (levels)
    effr: Optional[float] = None  # Effective Fed Funds Rate (%)
    total_reserves: Optional[float] = None  # Total reserves (FRED units; typically USD millions)
    fed_assets: Optional[float] = None  # Fed balance sheet total assets (FRED units; typically USD millions)
    on_rrp: Optional[float] = None  # ON RRP usage (FRED units; often USD billions or millions depending on series)

    # Derived (simple)
    reserves_chg_13w: Optional[float] = None
    fed_assets_chg_13w: Optional[float] = None
    on_rrp_chg_13w: Optional[float] = None

    # Standardized context (best-effort)
    z_total_reserves: Optional[float] = None
    z_on_rrp: Optional[float] = None
    z_fed_assets_chg_13w: Optional[float] = None

    # Debug / transparency
    components: Dict[str, Optional[float]] = Field(default_factory=dict)


class MonetaryState(BaseModel):
    asof: str
    start_date: str
    inputs: MonetaryInputs
    notes: str = ""



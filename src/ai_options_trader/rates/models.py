from __future__ import annotations

from typing import Optional, Dict

from pydantic import BaseModel


class RatesInputs(BaseModel):
    # Levels (percent)
    ust_2y: Optional[float] = None
    ust_10y: Optional[float] = None
    ust_3m: Optional[float] = None

    # Curve (percent, 10y - 2y, 10y - 3m)
    curve_2s10s: Optional[float] = None
    curve_3m10y: Optional[float] = None

    # Changes (percent)
    ust_10y_chg_20d: Optional[float] = None
    curve_2s10s_chg_20d: Optional[float] = None

    # Context (z, vs recent history)
    z_ust_10y: Optional[float] = None
    z_ust_10y_chg_20d: Optional[float] = None
    z_curve_2s10s: Optional[float] = None
    z_curve_2s10s_chg_20d: Optional[float] = None

    components: Optional[Dict[str, float]] = None


class RatesState(BaseModel):
    asof: str
    start_date: str
    inputs: RatesInputs
    notes: str | None = None



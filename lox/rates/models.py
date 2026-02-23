from __future__ import annotations

from typing import Optional, Dict

from pydantic import BaseModel


class RatesInputs(BaseModel):
    # Levels (percent)
    ust_2y: Optional[float] = None
    ust_5y: Optional[float] = None
    ust_10y: Optional[float] = None
    ust_30y: Optional[float] = None
    ust_3m: Optional[float] = None

    # Curve slopes (percent)
    curve_2s10s: Optional[float] = None
    curve_2s30s: Optional[float] = None
    curve_5s30s: Optional[float] = None
    curve_3m10y: Optional[float] = None

    # Per-tenor momentum (20d changes, percent)
    ust_2y_chg_20d: Optional[float] = None
    ust_10y_chg_20d: Optional[float] = None
    ust_30y_chg_20d: Optional[float] = None
    curve_2s10s_chg_20d: Optional[float] = None
    curve_2s30s_chg_20d: Optional[float] = None

    # Real yield decomposition (percent)
    real_yield_10y: Optional[float] = None
    real_yield_5y: Optional[float] = None
    breakeven_10y: Optional[float] = None
    breakeven_5y: Optional[float] = None
    real_yield_10y_chg_20d: Optional[float] = None

    # Z-scores (vs 3y rolling history)
    z_ust_10y: Optional[float] = None
    z_ust_10y_chg_20d: Optional[float] = None
    z_ust_2y_chg_20d: Optional[float] = None
    z_ust_30y_chg_20d: Optional[float] = None
    z_curve_2s10s: Optional[float] = None
    z_curve_2s10s_chg_20d: Optional[float] = None
    z_curve_2s30s: Optional[float] = None
    z_curve_2s30s_chg_20d: Optional[float] = None

    components: Optional[Dict[str, float]] = None


class RatesState(BaseModel):
    asof: str
    start_date: str
    inputs: RatesInputs
    notes: str | None = None



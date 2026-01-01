from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class TickerSnapshot(BaseModel):
    ticker: str
    asof: str

    # Levels
    last_close: Optional[float] = None

    # Returns (%)
    ret_1m_pct: Optional[float] = None
    ret_3m_pct: Optional[float] = None
    ret_6m_pct: Optional[float] = None
    ret_12m_pct: Optional[float] = None

    # Risk
    vol_20d_ann_pct: Optional[float] = None
    vol_60d_ann_pct: Optional[float] = None
    max_drawdown_12m_pct: Optional[float] = None

    # Relative strength vs benchmark (defaults to SPY)
    benchmark: str = "SPY"
    rel_ret_3m_pct: Optional[float] = None
    rel_ret_12m_pct: Optional[float] = None

    # Debug / transparency
    components: Dict[str, Optional[float]] = Field(default_factory=dict)



from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class FiscalInputs(BaseModel):
    # Core (MVP)
    deficit_12m: Optional[float] = None  # rolling 12m deficit (positive = larger deficit), $ (series units)
    tga_level: Optional[float] = None  # Treasury General Account level
    tga_chg_28d: Optional[float] = None  # 28d change in TGA level
    interest_expense_yoy: Optional[float] = None  # % YoY growth
    interest_expense_yoy_accel: Optional[float] = None  # YoY change vs 1y ago (pp)

    # Issuance (optional; may be NaN until data source wired)
    net_issuance_bills: Optional[float] = None
    net_issuance_coupons: Optional[float] = None
    net_issuance_long: Optional[float] = None  # e.g. >=10y bucket
    long_duration_issuance_share: Optional[float] = None  # 0..1

    # Auctions (optional)
    auction_tail_bps: Optional[float] = None
    dealer_take_pct: Optional[float] = None  # 0..100
    bid_to_cover_avg: Optional[float] = None  # recent auction avg bid-to-cover ratio

    # Demand / sustainability (quant upgrade)
    deficit_pct_receipts: Optional[float] = None  # deficit / federal tax receipts
    foreign_holdings_pct: Optional[float] = None  # foreign share of marketable debt
    foreign_holdings_chg_6m: Optional[float] = None  # 6m change in foreign holdings
    custody_holdings_chg_4w: Optional[float] = None  # 4w change in Fed custody (weekly)
    wam_years: Optional[float] = None  # weighted avg maturity of outstanding debt
    wam_chg_12m: Optional[float] = None  # 12m change in WAM
    deficit_trend_slope: Optional[float] = None  # 12m OLS slope of deficit trajectory

    # Bond market stress
    move_index: Optional[float] = None  # MOVE index level
    move_index_z: Optional[float] = None  # MOVE z-score (3yr window)

    # Standardized readings (best-effort)
    z_deficit_12m: Optional[float] = None
    z_tga_chg_28d: Optional[float] = None
    z_interest_expense_yoy: Optional[float] = None
    z_long_duration_issuance_share: Optional[float] = None
    z_auction_tail_bps: Optional[float] = None
    z_dealer_take_pct: Optional[float] = None

    # Composite
    fiscal_pressure_score: Optional[float] = None  # higher = more fiscal pressure / funding stress

    # Debug / transparency
    components: Dict[str, Optional[float]] = Field(default_factory=dict)


class FiscalState(BaseModel):
    asof: str
    start_date: str
    inputs: FiscalInputs
    notes: str = ""



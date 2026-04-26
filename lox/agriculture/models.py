from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel


class AgricultureInputs(BaseModel):
    # ── Crop prices (ETF proxies) ──────────────────────────────────────
    corn: Optional[float] = None
    wheat: Optional[float] = None
    soybeans: Optional[float] = None
    ag_broad: Optional[float] = None  # DBA broad agriculture

    # ── Input costs ────────────────────────────────────────────────────
    natgas: Optional[float] = None  # Henry Hub — primary fertilizer input
    diesel: Optional[float] = None  # GASDESW — farm equipment fuel
    ppi_fertilizer: Optional[float] = None  # PPI ag chemicals & fertilizers

    # ── Fertilizer equity basket (proxy for fert spot prices) ──────────
    fert_basket_level: Optional[float] = None  # equal-weight CF+NTR+MOS index

    # ── Returns (percent) ──────────────────────────────────────────────
    corn_ret_20d_pct: Optional[float] = None
    corn_ret_60d_pct: Optional[float] = None
    wheat_ret_20d_pct: Optional[float] = None
    natgas_ret_20d_pct: Optional[float] = None
    natgas_ret_60d_pct: Optional[float] = None
    diesel_ret_60d_pct: Optional[float] = None
    fert_basket_ret_20d_pct: Optional[float] = None
    fert_basket_ret_60d_pct: Optional[float] = None

    # ── Z-scores (vs 3yr rolling history) ──────────────────────────────
    z_corn_ret_20d: Optional[float] = None
    z_corn_ret_60d: Optional[float] = None
    z_wheat_ret_20d: Optional[float] = None
    z_natgas_ret_20d: Optional[float] = None
    z_natgas_ret_60d: Optional[float] = None
    z_diesel_ret_60d: Optional[float] = None
    z_fert_basket_ret_20d: Optional[float] = None
    z_fert_basket_ret_60d: Optional[float] = None
    z_ppi_fert_60d: Optional[float] = None

    # ── Seasonal z-scores (vs same calendar period in prior years) ─────
    sz_corn_ret_20d: Optional[float] = None
    sz_wheat_ret_20d: Optional[float] = None
    sz_natgas_ret_20d: Optional[float] = None

    # ── Composite scores ───────────────────────────────────────────────
    input_cost_score: Optional[float] = None  # weighted z of all input costs
    crop_momentum_score: Optional[float] = None  # weighted z of crop prices
    food_inflation_score: Optional[float] = None  # overall ag inflation pressure

    # ── CPI Food Components (YoY %) ──────────────────────────────────
    cpi_food_yoy: Optional[float] = None
    cpi_food_home_yoy: Optional[float] = None
    cpi_food_away_yoy: Optional[float] = None
    cpi_cereals_yoy: Optional[float] = None
    cpi_meats_yoy: Optional[float] = None
    cpi_dairy_yoy: Optional[float] = None
    cpi_fruits_veg_yoy: Optional[float] = None

    # ── CPI Food Momentum ─────────────────────────────────────────────
    cpi_food_3m_ann: Optional[float] = None
    cpi_food_6m_ann: Optional[float] = None
    cpi_food_accel: Optional[float] = None  # 3m ann - yoy (positive = accelerating)

    # ── PPI Food (factory gate) ───────────────────────────────────────
    ppi_food_mfg_yoy: Optional[float] = None

    # ── Farm-to-Retail Pipeline ───────────────────────────────────────
    farm_to_retail_spread: Optional[float] = None  # CPI food home yoy - PPI food mfg yoy
    grocery_restaurant_gap: Optional[float] = None  # CPI food home yoy - CPI food away yoy

    # ── Breadth ───────────────────────────────────────────────────────
    food_breadth_count: Optional[int] = None  # 0-5 sub-categories above median
    food_breadth_pct: Optional[float] = None  # 0-100

    # ── Protein Complex ───────────────────────────────────────────────
    beef_price: Optional[float] = None
    chicken_price: Optional[float] = None
    egg_price: Optional[float] = None
    protein_yoy_avg: Optional[float] = None
    protein_z: Optional[float] = None

    # ── Soft Commodities ──────────────────────────────────────────────
    sugar_price: Optional[float] = None
    coffee_price: Optional[float] = None
    cocoa_price: Optional[float] = None
    soft_ret_20d: Optional[float] = None  # composite 20d return
    soft_z: Optional[float] = None

    # ── Divergence / cross-signals ─────────────────────────────────────
    fert_corn_divergence: Optional[float] = None  # fert leading corn (positive = bullish corn)
    natgas_corn_ratio_z: Optional[float] = None  # z-score of natgas/corn ratio

    # ── Flags ──────────────────────────────────────────────────────────
    input_shock: Optional[bool] = None  # broad input cost spike
    crop_surge: Optional[bool] = None  # crop prices spiking
    cost_pass_through_lag: Optional[bool] = None  # inputs up, crops haven't followed
    broad_food_inflation: Optional[bool] = None  # breadth>=4 AND cpi_food_yoy>4%
    food_accel_flag: Optional[bool] = None  # 3m ann > yoy by >1.5pp
    grocery_shock: Optional[bool] = None  # food-home yoy > food-away by >3pp
    protein_spike: Optional[bool] = None  # protein_z > 2.0

    # ── Historical analog (2022 peak comparison) ───────────────────────
    corn_pct_of_2022_peak: Optional[float] = None
    natgas_pct_of_2022_peak: Optional[float] = None
    fert_pct_of_2022_peak: Optional[float] = None

    # ── CFTC COT positioning (managed money / non-commercial) ──────────
    cot_corn_net: Optional[float] = None          # net speculative contracts
    cot_corn_z: Optional[float] = None            # z-score of net spec
    cot_wheat_net: Optional[float] = None
    cot_wheat_z: Optional[float] = None
    cot_soybeans_net: Optional[float] = None
    cot_soybeans_z: Optional[float] = None
    cot_date: Optional[str] = None                # most recent COT report date

    # ── USDA WASDE supply/demand ───────────────────────────────────────
    wasde_corn_stu_pct: Optional[float] = None    # stocks-to-use ratio (%)
    wasde_wheat_stu_pct: Optional[float] = None
    wasde_soy_stu_pct: Optional[float] = None
    wasde_corn_ending_stocks: Optional[float] = None  # 1000 MT or million bushels
    wasde_wheat_ending_stocks: Optional[float] = None
    wasde_soy_ending_stocks: Optional[float] = None
    wasde_market_year: Optional[int] = None

    # ── USDA NASS crop reports (Prospective Plantings, Progress, Condition)
    corn_planted_acres_m: Optional[float] = None       # million acres
    corn_planted_yoy_pct: Optional[float] = None       # YoY change %
    soy_planted_acres_m: Optional[float] = None
    soy_planted_yoy_pct: Optional[float] = None
    wheat_planted_acres_m: Optional[float] = None
    wheat_planted_yoy_pct: Optional[float] = None
    corn_pct_planted: Optional[float] = None           # weekly crop progress
    corn_pct_planted_vs_avg: Optional[float] = None    # vs 5yr avg (pp diff)
    corn_condition_ge: Optional[float] = None          # % good + excellent
    corn_condition_ge_vs_avg: Optional[float] = None   # vs 5yr avg (pp diff)
    crop_report_week: Optional[str] = None             # latest report week ending

    # ── Soybean progress & condition ─────────────────────────────────
    soy_pct_planted: Optional[float] = None
    soy_pct_planted_vs_avg: Optional[float] = None
    soy_pct_emerged: Optional[float] = None
    soy_condition_ge: Optional[float] = None
    soy_condition_ge_vs_avg: Optional[float] = None

    # ── Wheat progress & condition ───────────────────────────────────
    wheat_pct_planted: Optional[float] = None          # winter wheat: % planted in fall
    wheat_pct_planted_vs_avg: Optional[float] = None
    wheat_condition_ge: Optional[float] = None         # winter wheat condition
    wheat_condition_ge_vs_avg: Optional[float] = None

    # ── Corn growth stages ───────────────────────────────────────────
    corn_pct_emerged: Optional[float] = None
    corn_pct_emerged_vs_avg: Optional[float] = None
    corn_pct_silking: Optional[float] = None           # mid-summer pollination stress

    # ── Aggregate crop health ────────────────────────────────────────
    crop_condition_composite: Optional[float] = None   # avg G/E across available crops
    planting_delay_count: Optional[int] = None         # crops with pct_planted > 5pp behind avg

    components: Optional[Dict[str, float]] = None


class AgricultureState(BaseModel):
    asof: str
    start_date: str
    inputs: AgricultureInputs
    notes: str | None = None

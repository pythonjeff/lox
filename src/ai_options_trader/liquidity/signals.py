from __future__ import annotations

from typing import Dict

import pandas as pd

from ai_options_trader.config import Settings
from ai_options_trader.data.fred import FredClient
from ai_options_trader.liquidity.models import LiquidityInputs, LiquidityState
from ai_options_trader.macro.transforms import merge_series_daily, zscore


# FRED series IDs (widely used / liquid proxies for "liquidity" conditions)
LIQUIDITY_FRED_SERIES: Dict[str, str] = {
    # Government bond market
    "DGS10": "DGS10",  # 10-Year Treasury Constant Maturity Rate (%)
    # Corporate credit liquidity / stress
    "HY_OAS": "BAMLH0A0HYM2",  # ICE BofA US High Yield Index Option-Adjusted Spread (%)
    "IG_OAS": "BAMLC0A0CM",  # ICE BofA US Corporate Index Option-Adjusted Spread (%)
    # Spread-like yield measure
    "BAA10YM": "BAA10YM",  # Moody's Seasoned Baa Corporate Bond Yield Relative to 10Y Treasury (%)
}


def build_liquidity_dataset(settings: Settings, start_date: str = "2011-01-01", refresh: bool = False) -> pd.DataFrame:
    if not settings.FRED_API_KEY:
        raise RuntimeError("Missing FRED_API_KEY in environment / .env")

    fred = FredClient(api_key=settings.FRED_API_KEY)

    series_frames: Dict[str, pd.DataFrame] = {}
    for name, sid in LIQUIDITY_FRED_SERIES.items():
        df = fred.fetch_series(series_id=sid, start_date=start_date, refresh=refresh)
        df = df.rename(columns={"value": name}).sort_values("date")
        series_frames[name] = df

    # Build daily grid and forward-fill (FRED series can have missing weekends/holidays)
    max_date = max(df["date"].max() for df in series_frames.values())
    base = pd.DataFrame({"date": pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(max_date), freq="D")})
    merged = merge_series_daily(base, series_frames, ffill=True)

    # 10Y yield dynamics (bps)
    merged["DGS10_CHG_20D_BPS"] = (merged["DGS10"] - merged["DGS10"].shift(20)) * 100.0
    merged["DGS10_CHG_60D_BPS"] = (merged["DGS10"] - merged["DGS10"].shift(60)) * 100.0

    # Standardize key components
    merged["Z_HY_OAS"] = zscore(merged["HY_OAS"], window=252)
    merged["Z_IG_OAS"] = zscore(merged["IG_OAS"], window=252)
    merged["Z_DGS10_CHG_20D"] = zscore(merged["DGS10_CHG_20D_BPS"], window=252)

    # Composite tightness score (positive = tighter liquidity)
    merged["LIQ_TIGHTNESS_SCORE"] = (
        0.45 * merged["Z_HY_OAS"] + 0.35 * merged["Z_IG_OAS"] + 0.20 * merged["Z_DGS10_CHG_20D"]
    )

    return merged


def build_liquidity_state(settings: Settings, start_date: str = "2011-01-01", refresh: bool = False) -> LiquidityState:
    df = build_liquidity_dataset(settings=settings, start_date=start_date, refresh=refresh)
    last = df.dropna(subset=["DGS10", "HY_OAS", "IG_OAS"]).iloc[-1]

    score = float(last["LIQ_TIGHTNESS_SCORE"]) if pd.notna(last["LIQ_TIGHTNESS_SCORE"]) else None
    inputs = LiquidityInputs(
        ust_10y=float(last["DGS10"]) if pd.notna(last["DGS10"]) else None,
        hy_oas=float(last["HY_OAS"]) if pd.notna(last["HY_OAS"]) else None,
        ig_oas=float(last["IG_OAS"]) if pd.notna(last["IG_OAS"]) else None,
        baa10ym=float(last["BAA10YM"]) if "BAA10YM" in last and pd.notna(last["BAA10YM"]) else None,
        ust_10y_chg_20d_bps=float(last["DGS10_CHG_20D_BPS"]) if pd.notna(last["DGS10_CHG_20D_BPS"]) else None,
        ust_10y_chg_60d_bps=float(last["DGS10_CHG_60D_BPS"]) if pd.notna(last["DGS10_CHG_60D_BPS"]) else None,
        z_hy_oas=float(last["Z_HY_OAS"]) if pd.notna(last["Z_HY_OAS"]) else None,
        z_ig_oas=float(last["Z_IG_OAS"]) if pd.notna(last["Z_IG_OAS"]) else None,
        z_ust_10y_chg_20d=float(last["Z_DGS10_CHG_20D"]) if pd.notna(last["Z_DGS10_CHG_20D"]) else None,
        liquidity_tightness_score=score,
        is_liquidity_tight=bool(score is not None and score > 1.0),
        components={
            "w_z_hy_oas": 0.45,
            "w_z_ig_oas": 0.35,
            "w_z_ust_10y_chg_20d": 0.20,
        },
    )

    return LiquidityState(
        asof=str(pd.to_datetime(last["date"]).date()),
        start_date=start_date,
        inputs=inputs,
        notes="Liquidity tightness: combines HY/IG OAS z-scores + 20d 10Y yield change (bps) z-score. Positive = tighter.",
    )



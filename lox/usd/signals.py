from __future__ import annotations

from typing import Dict

import pandas as pd

from lox.config import Settings
from lox.data.fred import FredClient
from lox.macro.transforms import merge_series_daily, zscore
from lox.usd.models import UsdInputs, UsdState


# Trade Weighted U.S. Dollar Index: Broad (goods only), daily.
# FRED series id: DTWEXBGS
USD_FRED_SERIES: Dict[str, str] = {"USD_BROAD": "DTWEXBGS"}


def build_usd_dataset(settings: Settings, start_date: str = "2011-01-01", refresh: bool = False) -> pd.DataFrame:
    if not settings.FRED_API_KEY:
        raise RuntimeError("Missing FRED_API_KEY in environment / .env")

    fred = FredClient(api_key=settings.FRED_API_KEY)
    frames: Dict[str, pd.DataFrame] = {}

    for name, sid in USD_FRED_SERIES.items():
        df = fred.fetch_series(series_id=sid, start_date=start_date, refresh=refresh)
        df = df.rename(columns={"value": name}).sort_values("date")
        frames[name] = df

    max_date = max(df["date"].max() for df in frames.values())
    base = pd.DataFrame({"date": pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(max_date), freq="D")})
    merged = merge_series_daily(base, frames, ffill=True)

    # Changes in %
    merged["USD_CHG_20D_PCT"] = (merged["USD_BROAD"] / merged["USD_BROAD"].shift(20) - 1.0) * 100.0
    merged["USD_CHG_60D_PCT"] = (merged["USD_BROAD"] / merged["USD_BROAD"].shift(60) - 1.0) * 100.0

    # Standardize (3-year window so sustained moves aren't normalized away)
    merged["Z_USD_LEVEL"] = zscore(merged["USD_BROAD"], window=756)
    merged["Z_USD_CHG_60D"] = zscore(merged["USD_CHG_60D_PCT"], window=756)

    # Composite score (positive = stronger USD regime)
    merged["USD_STRENGTH_SCORE"] = 0.60 * merged["Z_USD_LEVEL"] + 0.40 * merged["Z_USD_CHG_60D"]

    # Extended metrics
    merged["USD_YOY_CHG_PCT"] = (merged["USD_BROAD"] / merged["USD_BROAD"].shift(252) - 1.0) * 100.0
    ma200 = merged["USD_BROAD"].rolling(200).mean()
    merged["USD_200D_MA_DIST_PCT"] = (merged["USD_BROAD"] / ma200 - 1.0) * 100.0
    merged["USD_90D_RVOL"] = merged["USD_BROAD"].pct_change().rolling(63).std() * (252 ** 0.5) * 100.0

    return merged


def build_usd_state(settings: Settings, start_date: str = "2011-01-01", refresh: bool = False) -> UsdState:
    df = build_usd_dataset(settings=settings, start_date=start_date, refresh=refresh)
    last = df.dropna(subset=["USD_BROAD"]).iloc[-1]

    score = float(last["USD_STRENGTH_SCORE"]) if pd.notna(last["USD_STRENGTH_SCORE"]) else None
    inputs = UsdInputs(
        usd_index_broad=float(last["USD_BROAD"]) if pd.notna(last["USD_BROAD"]) else None,
        usd_chg_20d_pct=float(last["USD_CHG_20D_PCT"]) if pd.notna(last["USD_CHG_20D_PCT"]) else None,
        usd_chg_60d_pct=float(last["USD_CHG_60D_PCT"]) if pd.notna(last["USD_CHG_60D_PCT"]) else None,
        z_usd_level=float(last["Z_USD_LEVEL"]) if pd.notna(last["Z_USD_LEVEL"]) else None,
        z_usd_chg_60d=float(last["Z_USD_CHG_60D"]) if pd.notna(last["Z_USD_CHG_60D"]) else None,
        usd_strength_score=score,
        is_usd_strong=bool(score is not None and score > 1.0),
        is_usd_weak=bool(score is not None and score < -1.0),
        usd_yoy_chg_pct=float(last["USD_YOY_CHG_PCT"]) if pd.notna(last.get("USD_YOY_CHG_PCT")) else None,
        usd_200d_ma_dist_pct=float(last["USD_200D_MA_DIST_PCT"]) if pd.notna(last.get("USD_200D_MA_DIST_PCT")) else None,
        usd_90d_rvol=float(last["USD_90D_RVOL"]) if pd.notna(last.get("USD_90D_RVOL")) else None,
        components={"w_z_usd_level": 0.60, "w_z_usd_chg_60d": 0.40},
    )

    return UsdState(
        asof=str(pd.to_datetime(last["date"]).date()),
        start_date=start_date,
        inputs=inputs,
        notes="USD strength regime uses DTWEXBGS (broad trade-weighted USD). Positive score = stronger USD vs history.",
    )



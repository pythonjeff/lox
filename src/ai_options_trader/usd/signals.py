from __future__ import annotations

from typing import Dict

import pandas as pd

from ai_options_trader.config import Settings
from ai_options_trader.data.fred import FredClient
from ai_options_trader.macro.transforms import merge_series_daily, zscore
from ai_options_trader.usd.models import UsdInputs, UsdState


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

    # Standardize
    merged["Z_USD_LEVEL"] = zscore(merged["USD_BROAD"], window=252)
    merged["Z_USD_CHG_60D"] = zscore(merged["USD_CHG_60D_PCT"], window=252)

    # Composite score (positive = stronger USD regime)
    merged["USD_STRENGTH_SCORE"] = 0.60 * merged["Z_USD_LEVEL"] + 0.40 * merged["Z_USD_CHG_60D"]

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
        components={"w_z_usd_level": 0.60, "w_z_usd_chg_60d": 0.40},
    )

    return UsdState(
        asof=str(pd.to_datetime(last["date"]).date()),
        start_date=start_date,
        inputs=inputs,
        notes="USD strength regime uses DTWEXBGS (broad trade-weighted USD). Positive score = stronger USD vs history.",
    )



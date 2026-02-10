from __future__ import annotations

from typing import Dict

import pandas as pd

from lox.config import Settings
from lox.data.fred import FredClient
from lox.macro.transforms import merge_series_daily, zscore
from lox.rates.models import RatesInputs, RatesState


RATES_FRED_SERIES: Dict[str, str] = {
    "UST_2Y": "DGS2",
    "UST_10Y": "DGS10",
    # 3M is a common curve anchor; not in DEFAULT_SERIES but FRED supports it.
    "UST_3M": "DGS3MO",
}


def build_rates_dataset(settings: Settings, start_date: str = "2011-01-01", refresh: bool = False) -> pd.DataFrame:
    if not settings.FRED_API_KEY:
        raise RuntimeError("Missing FRED_API_KEY in environment / .env")

    fred = FredClient(api_key=settings.FRED_API_KEY)

    frames: Dict[str, pd.DataFrame] = {}
    for name, sid in RATES_FRED_SERIES.items():
        try:
            df = fred.fetch_series(series_id=sid, start_date=start_date, refresh=refresh)
        except Exception:
            # Best-effort: if 3M fails, still build off 2y/10y.
            if name == "UST_3M":
                continue
            raise
        if df is None or df.empty:
            if name == "UST_3M":
                continue
            raise RuntimeError(f"Failed to load rates series {name} ({sid})")
        frames[name] = df.sort_values("date").reset_index(drop=True)

    max_date = max(df["date"].max() for df in frames.values())
    base = pd.DataFrame(
        {"date": pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(max_date), freq="D")}
    )
    merged = merge_series_daily(base, frames, ffill=True)

    # Derived curve measures (percent)
    if "UST_10Y" in merged.columns and "UST_2Y" in merged.columns:
        merged["CURVE_2S10S"] = pd.to_numeric(merged["UST_10Y"], errors="coerce") - pd.to_numeric(
            merged["UST_2Y"], errors="coerce"
        )
    if "UST_10Y" in merged.columns and "UST_3M" in merged.columns:
        merged["CURVE_3M10Y"] = pd.to_numeric(merged["UST_10Y"], errors="coerce") - pd.to_numeric(
            merged["UST_3M"], errors="coerce"
        )

    # 20d changes (percent)
    chg_days = 20
    merged["UST_10Y_CHG_20D"] = pd.to_numeric(merged["UST_10Y"], errors="coerce") - pd.to_numeric(
        merged["UST_10Y"], errors="coerce"
    ).shift(chg_days)
    if "CURVE_2S10S" in merged.columns:
        merged["CURVE_2S10S_CHG_20D"] = pd.to_numeric(merged["CURVE_2S10S"], errors="coerce") - pd.to_numeric(
            merged["CURVE_2S10S"], errors="coerce"
        ).shift(chg_days)

    # Z-scores (vs recent history)
    win = 252 * 3
    merged["Z_UST_10Y"] = zscore(pd.to_numeric(merged["UST_10Y"], errors="coerce"), window=win)
    merged["Z_UST_10Y_CHG_20D"] = zscore(pd.to_numeric(merged["UST_10Y_CHG_20D"], errors="coerce"), window=win)
    if "CURVE_2S10S" in merged.columns:
        merged["Z_CURVE_2S10S"] = zscore(pd.to_numeric(merged["CURVE_2S10S"], errors="coerce"), window=win)
        merged["Z_CURVE_2S10S_CHG_20D"] = zscore(
            pd.to_numeric(merged["CURVE_2S10S_CHG_20D"], errors="coerce"), window=win
        )

    return merged


def build_rates_state(settings: Settings, start_date: str = "2011-01-01", refresh: bool = False) -> RatesState:
    df = build_rates_dataset(settings=settings, start_date=start_date, refresh=refresh)
    # Require the 2y/10y core.
    last = df.dropna(subset=["UST_2Y", "UST_10Y"]).iloc[-1]

    inp = RatesInputs(
        ust_2y=float(last["UST_2Y"]) if pd.notna(last.get("UST_2Y")) else None,
        ust_10y=float(last["UST_10Y"]) if pd.notna(last.get("UST_10Y")) else None,
        ust_3m=float(last["UST_3M"]) if "UST_3M" in last and pd.notna(last.get("UST_3M")) else None,
        curve_2s10s=float(last["CURVE_2S10S"]) if "CURVE_2S10S" in last and pd.notna(last.get("CURVE_2S10S")) else None,
        curve_3m10y=float(last["CURVE_3M10Y"]) if "CURVE_3M10Y" in last and pd.notna(last.get("CURVE_3M10Y")) else None,
        ust_10y_chg_20d=float(last["UST_10Y_CHG_20D"]) if pd.notna(last.get("UST_10Y_CHG_20D")) else None,
        curve_2s10s_chg_20d=float(last["CURVE_2S10S_CHG_20D"])
        if "CURVE_2S10S_CHG_20D" in last and pd.notna(last.get("CURVE_2S10S_CHG_20D"))
        else None,
        z_ust_10y=float(last["Z_UST_10Y"]) if pd.notna(last.get("Z_UST_10Y")) else None,
        z_ust_10y_chg_20d=float(last["Z_UST_10Y_CHG_20D"]) if pd.notna(last.get("Z_UST_10Y_CHG_20D")) else None,
        z_curve_2s10s=float(last["Z_CURVE_2S10S"]) if "Z_CURVE_2S10S" in last and pd.notna(last.get("Z_CURVE_2S10S")) else None,
        z_curve_2s10s_chg_20d=float(last["Z_CURVE_2S10S_CHG_20D"])
        if "Z_CURVE_2S10S_CHG_20D" in last and pd.notna(last.get("Z_CURVE_2S10S_CHG_20D"))
        else None,
        components={"window_days": float(252 * 3), "chg_20d_days": float(20)},
    )

    return RatesState(
        asof=str(pd.to_datetime(last["date"]).date()),
        start_date=start_date,
        inputs=inp,
        notes="Rates/Yield curve MVP: 2y/10y/3m Treasury yields, curve slopes, and 20d rate shocks (with z-context).",
    )



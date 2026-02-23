from __future__ import annotations

from typing import Dict

import pandas as pd

from lox.config import Settings
from lox.data.fred import FredClient
from lox.macro.transforms import merge_series_daily, zscore
from lox.rates.models import RatesInputs, RatesState


ZSCORE_WINDOW_DAYS = 252 * 3  # 3-year rolling window for z-scores
MOMENTUM_WINDOW_DAYS = 20     # 20 trading days for rate changes

RATES_FRED_SERIES: Dict[str, str] = {
    "UST_2Y": "DGS2",
    "UST_5Y": "DGS5",
    "UST_10Y": "DGS10",
    "UST_30Y": "DGS30",
    "UST_3M": "DGS3MO",
    "TIPS_10Y": "DFII10",
    "TIPS_5Y": "DFII5",
}

_OPTIONAL_SERIES = {"UST_3M", "UST_5Y", "UST_30Y", "TIPS_10Y", "TIPS_5Y"}


def _num(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col, errors="coerce")


def build_rates_dataset(settings: Settings, start_date: str = "2011-01-01", refresh: bool = False) -> pd.DataFrame:
    if not settings.FRED_API_KEY:
        raise RuntimeError("Missing FRED_API_KEY in environment / .env")

    fred = FredClient(api_key=settings.FRED_API_KEY)

    frames: Dict[str, pd.DataFrame] = {}
    for name, sid in RATES_FRED_SERIES.items():
        try:
            df = fred.fetch_series(series_id=sid, start_date=start_date, refresh=refresh)
        except Exception:
            if name in _OPTIONAL_SERIES:
                continue
            raise
        if df is None or df.empty:
            if name in _OPTIONAL_SERIES:
                continue
            raise RuntimeError(f"Failed to load rates series {name} ({sid})")
        frames[name] = df.sort_values("date").reset_index(drop=True)

    max_date = max(df["date"].max() for df in frames.values())
    base = pd.DataFrame(
        {"date": pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(max_date), freq="D")}
    )
    merged = merge_series_daily(base, frames, ffill=True)

    # ── Curve slopes (percent) ───────────────────────────────────────────
    _2y = _num(merged["UST_2Y"]) if "UST_2Y" in merged.columns else None
    _5y = _num(merged["UST_5Y"]) if "UST_5Y" in merged.columns else None
    _10y = _num(merged["UST_10Y"]) if "UST_10Y" in merged.columns else None
    _30y = _num(merged["UST_30Y"]) if "UST_30Y" in merged.columns else None
    _3m = _num(merged["UST_3M"]) if "UST_3M" in merged.columns else None

    if _10y is not None and _2y is not None:
        merged["CURVE_2S10S"] = _10y - _2y
    if _30y is not None and _2y is not None:
        merged["CURVE_2S30S"] = _30y - _2y
    if _30y is not None and _5y is not None:
        merged["CURVE_5S30S"] = _30y - _5y
    if _10y is not None and _3m is not None:
        merged["CURVE_3M10Y"] = _10y - _3m

    # ── Real yield decomposition ─────────────────────────────────────────
    _tips10 = _num(merged["TIPS_10Y"]) if "TIPS_10Y" in merged.columns else None
    _tips5 = _num(merged["TIPS_5Y"]) if "TIPS_5Y" in merged.columns else None

    if _tips10 is not None:
        merged["REAL_YIELD_10Y"] = _tips10
        if _10y is not None:
            merged["BREAKEVEN_10Y"] = _10y - _tips10
    if _tips5 is not None:
        merged["REAL_YIELD_5Y"] = _tips5
        if _5y is not None:
            merged["BREAKEVEN_5Y"] = _5y - _tips5

    # ── Per-tenor momentum changes (percent) ─────────────────────────────
    chg_days = MOMENTUM_WINDOW_DAYS

    if _2y is not None:
        merged["UST_2Y_CHG_20D"] = _2y - _2y.shift(chg_days)
    if _10y is not None:
        merged["UST_10Y_CHG_20D"] = _10y - _10y.shift(chg_days)
    if _30y is not None:
        merged["UST_30Y_CHG_20D"] = _30y - _30y.shift(chg_days)
    if "CURVE_2S10S" in merged.columns:
        merged["CURVE_2S10S_CHG_20D"] = _num(merged["CURVE_2S10S"]) - _num(merged["CURVE_2S10S"]).shift(chg_days)
    if "CURVE_2S30S" in merged.columns:
        merged["CURVE_2S30S_CHG_20D"] = _num(merged["CURVE_2S30S"]) - _num(merged["CURVE_2S30S"]).shift(chg_days)
    if _tips10 is not None:
        merged["REAL_YIELD_10Y_CHG_20D"] = _tips10 - _tips10.shift(chg_days)

    # ── Z-scores (rolling window) ────────────────────────────────────────
    win = ZSCORE_WINDOW_DAYS

    if _10y is not None:
        merged["Z_UST_10Y"] = zscore(_10y, window=win)
    if "UST_2Y_CHG_20D" in merged.columns:
        merged["Z_UST_2Y_CHG_20D"] = zscore(_num(merged["UST_2Y_CHG_20D"]), window=win)
    if "UST_10Y_CHG_20D" in merged.columns:
        merged["Z_UST_10Y_CHG_20D"] = zscore(_num(merged["UST_10Y_CHG_20D"]), window=win)
    if "UST_30Y_CHG_20D" in merged.columns:
        merged["Z_UST_30Y_CHG_20D"] = zscore(_num(merged["UST_30Y_CHG_20D"]), window=win)
    if "CURVE_2S10S" in merged.columns:
        merged["Z_CURVE_2S10S"] = zscore(_num(merged["CURVE_2S10S"]), window=win)
    if "CURVE_2S10S_CHG_20D" in merged.columns:
        merged["Z_CURVE_2S10S_CHG_20D"] = zscore(_num(merged["CURVE_2S10S_CHG_20D"]), window=win)
    if "CURVE_2S30S" in merged.columns:
        merged["Z_CURVE_2S30S"] = zscore(_num(merged["CURVE_2S30S"]), window=win)
    if "CURVE_2S30S_CHG_20D" in merged.columns:
        merged["Z_CURVE_2S30S_CHG_20D"] = zscore(_num(merged["CURVE_2S30S_CHG_20D"]), window=win)

    return merged


def _get(last: pd.Series, col: str) -> float | None:
    if col in last.index and pd.notna(last.get(col)):
        return float(last[col])
    return None


def build_rates_state(settings: Settings, start_date: str = "2011-01-01", refresh: bool = False) -> RatesState:
    df = build_rates_dataset(settings=settings, start_date=start_date, refresh=refresh)
    last = df.dropna(subset=["UST_2Y", "UST_10Y"]).iloc[-1]

    inp = RatesInputs(
        ust_2y=_get(last, "UST_2Y"),
        ust_5y=_get(last, "UST_5Y"),
        ust_10y=_get(last, "UST_10Y"),
        ust_30y=_get(last, "UST_30Y"),
        ust_3m=_get(last, "UST_3M"),
        curve_2s10s=_get(last, "CURVE_2S10S"),
        curve_2s30s=_get(last, "CURVE_2S30S"),
        curve_5s30s=_get(last, "CURVE_5S30S"),
        curve_3m10y=_get(last, "CURVE_3M10Y"),
        ust_2y_chg_20d=_get(last, "UST_2Y_CHG_20D"),
        ust_10y_chg_20d=_get(last, "UST_10Y_CHG_20D"),
        ust_30y_chg_20d=_get(last, "UST_30Y_CHG_20D"),
        curve_2s10s_chg_20d=_get(last, "CURVE_2S10S_CHG_20D"),
        curve_2s30s_chg_20d=_get(last, "CURVE_2S30S_CHG_20D"),
        real_yield_10y=_get(last, "REAL_YIELD_10Y"),
        real_yield_5y=_get(last, "REAL_YIELD_5Y"),
        breakeven_10y=_get(last, "BREAKEVEN_10Y"),
        breakeven_5y=_get(last, "BREAKEVEN_5Y"),
        real_yield_10y_chg_20d=_get(last, "REAL_YIELD_10Y_CHG_20D"),
        z_ust_10y=_get(last, "Z_UST_10Y"),
        z_ust_10y_chg_20d=_get(last, "Z_UST_10Y_CHG_20D"),
        z_ust_2y_chg_20d=_get(last, "Z_UST_2Y_CHG_20D"),
        z_ust_30y_chg_20d=_get(last, "Z_UST_30Y_CHG_20D"),
        z_curve_2s10s=_get(last, "Z_CURVE_2S10S"),
        z_curve_2s10s_chg_20d=_get(last, "Z_CURVE_2S10S_CHG_20D"),
        z_curve_2s30s=_get(last, "Z_CURVE_2S30S"),
        z_curve_2s30s_chg_20d=_get(last, "Z_CURVE_2S30S_CHG_20D"),
        components={"window_days": float(ZSCORE_WINDOW_DAYS), "chg_20d_days": float(MOMENTUM_WINDOW_DAYS)},
    )

    return RatesState(
        asof=str(pd.to_datetime(last["date"]).date()),
        start_date=start_date,
        inputs=inp,
        notes=(
            "Rates/Yield curve: full curve (3M/2Y/5Y/10Y/30Y), slopes (2s10s/2s30s/5s30s), "
            "per-tenor momentum, real yields (TIPS), breakevens, and steepener/flattener dynamics."
        ),
    )

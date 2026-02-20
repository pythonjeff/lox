from __future__ import annotations

from typing import Dict

import pandas as pd

from lox.config import Settings
from lox.data.fred import FredClient
from lox.macro.transforms import merge_series_daily, zscore
from lox.monetary.models import MonetaryInputs, MonetaryState


MONETARY_FRED_SERIES: Dict[str, str] = {
    # 1) Policy rate
    "EFFR": "DFF",
    # 2) Reserve balances at the Fed (weekly Wed, USD millions) — H.4.1 release
    "RESERVES": "WRESBAL",
    # 3) Fed balance sheet total assets (weekly, USD millions)
    "FED_ASSETS": "WALCL",
    # 4) ON RRP usage (daily, USD)
    "ON_RRP": "RRPONTSYD",
}


def _gdp_asof(*, fred: FredClient, asof: pd.Timestamp, start_date: str, refresh: bool) -> dict[str, object | None]:
    """
    Best-effort GDP as-of (quarterly) for normalization.

    Prefer nominal GDP (series_id="GDP"). Fall back to real GDP ("GDPC1") if needed.
    Returns GDP in USD *millions* (FRED GDP is typically USD billions SAAR).
    """
    for sid in ("GDP", "GDPC1"):
        try:
            g = fred.fetch_series(series_id=sid, start_date=start_date, refresh=refresh)
        except Exception:
            continue
        if g is None or g.empty:
            continue
        g = g.sort_values("date").reset_index(drop=True)
        g["date"] = pd.to_datetime(g["date"], errors="coerce")
        g["value"] = pd.to_numeric(g["value"], errors="coerce")
        g = g.dropna(subset=["date", "value"])
        if g.empty:
            continue
        sub = g[g["date"] <= asof]
        if sub.empty:
            continue
        last = sub.iloc[-1]
        return {
            "series": sid,
            "asof": str(pd.to_datetime(last["date"]).date()),
            "gdp_millions": float(last["value"]) * 1000.0,
        }
    return {"series": None, "asof": None, "gdp_millions": None}


def build_monetary_dataset(settings: Settings, start_date: str = "2011-01-01", refresh: bool = False) -> pd.DataFrame:
    if not settings.FRED_API_KEY:
        raise RuntimeError("Missing FRED_API_KEY in environment / .env")

    fred = FredClient(api_key=settings.FRED_API_KEY)

    series_frames: Dict[str, pd.DataFrame] = {}
    for name, sid in MONETARY_FRED_SERIES.items():
        df = fred.fetch_series(series_id=sid, start_date=start_date, refresh=refresh)
        if df is None or df.empty:
            raise RuntimeError(f"Failed to load monetary series {name} ({sid})")
        series_frames[name] = df.sort_values("date").reset_index(drop=True)

    max_date = max(df["date"].max() for df in series_frames.values())
    base = pd.DataFrame({"date": pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(max_date), freq="D")})
    merged = merge_series_daily(base, series_frames, ffill=True)

    # ---------------------------------------------------------------------
    # Unit normalization (for consistent formatting + deltas):
    # - WRESBAL is already in $ millions on FRED — no conversion needed.
    # - RRPONTSYD is reported in $ billions on FRED; convert to $ millions to match WALCL.
    # ---------------------------------------------------------------------
    if "RESERVES" in merged.columns:
        merged["RESERVES"] = pd.to_numeric(merged["RESERVES"], errors="coerce")
    if "ON_RRP" in merged.columns:
        merged["ON_RRP"] = pd.to_numeric(merged["ON_RRP"], errors="coerce") * 1000.0

    # 13-week changes for weekly-ish plumbing series (approx 91 days on daily grid)
    days_13w = 91
    if "RESERVES" in merged.columns:
        merged["RESERVES_CHG_13W"] = merged["RESERVES"] - merged["RESERVES"].shift(days_13w)
    if "FED_ASSETS" in merged.columns:
        merged["FED_ASSETS_CHG_13W"] = merged["FED_ASSETS"] - merged["FED_ASSETS"].shift(days_13w)
    if "ON_RRP" in merged.columns:
        merged["ON_RRP_CHG_13W"] = merged["ON_RRP"] - merged["ON_RRP"].shift(days_13w)

    # Standardized context (long windows to reduce step-artifacts from weekly series)
    win = 252 * 3
    merged["Z_RESERVES"] = zscore(pd.to_numeric(merged["RESERVES"], errors="coerce"), window=win)
    merged["Z_ON_RRP"] = zscore(pd.to_numeric(merged["ON_RRP"], errors="coerce"), window=win)
    merged["Z_FED_ASSETS_CHG_13W"] = zscore(pd.to_numeric(merged.get("FED_ASSETS_CHG_13W"), errors="coerce"), window=win)

    return merged


def build_monetary_state(settings: Settings, start_date: str = "2011-01-01", refresh: bool = False) -> MonetaryState:
    df = build_monetary_dataset(settings=settings, start_date=start_date, refresh=refresh)
    last = df.dropna(subset=["EFFR", "RESERVES", "FED_ASSETS", "ON_RRP"]).iloc[-1]

    inputs = MonetaryInputs(
        effr=float(last["EFFR"]) if pd.notna(last["EFFR"]) else None,
        total_reserves=float(last["RESERVES"]) if pd.notna(last["RESERVES"]) else None,
        fed_assets=float(last["FED_ASSETS"]) if pd.notna(last["FED_ASSETS"]) else None,
        on_rrp=float(last["ON_RRP"]) if pd.notna(last["ON_RRP"]) else None,
        reserves_chg_13w=float(last["RESERVES_CHG_13W"]) if pd.notna(last.get("RESERVES_CHG_13W")) else None,
        fed_assets_chg_13w=float(last["FED_ASSETS_CHG_13W"]) if pd.notna(last.get("FED_ASSETS_CHG_13W")) else None,
        on_rrp_chg_13w=float(last["ON_RRP_CHG_13W"]) if pd.notna(last.get("ON_RRP_CHG_13W")) else None,
        z_total_reserves=float(last["Z_RESERVES"]) if pd.notna(last.get("Z_RESERVES")) else None,
        z_on_rrp=float(last["Z_ON_RRP"]) if pd.notna(last.get("Z_ON_RRP")) else None,
        z_fed_assets_chg_13w=float(last["Z_FED_ASSETS_CHG_13W"]) if pd.notna(last.get("Z_FED_ASSETS_CHG_13W")) else None,
        components={
            "window_days": float(252 * 3),
            "chg_13w_days": float(91),
        },
    )
    return MonetaryState(
        asof=str(pd.to_datetime(last["date"]).date()),
        start_date=start_date,
        inputs=inputs,
        notes="Monetary MVP: EFFR (DFF), reserve balances (WRESBAL, weekly), Fed assets (WALCL), ON RRP (RRPONTSYD).",
    )


def build_monetary_page_data(
    settings: Settings,
    lookback_years: int = 5,
    refresh: bool = False,
) -> dict[str, object]:
    """
    Lean monetary snapshot payload for CLI:
    - last EFFR, reserves, Fed assets, ON RRP
    - 13-week deltas + simple z-context for reserves/RRP/QT pace
    """
    start_date = str((pd.Timestamp.today().normalize() - pd.DateOffset(years=int(lookback_years))).date())
    state = build_monetary_state(settings=settings, start_date=start_date, refresh=refresh)

    inp = state.inputs
    # Normalization anchor: reserves as % of GDP (prevents "low vs 5y" from sounding like "scarce")
    fred = FredClient(api_key=settings.FRED_API_KEY)  # type: ignore[arg-type]
    asof_ts = pd.to_datetime(state.asof)
    gdp = _gdp_asof(fred=fred, asof=asof_ts, start_date=start_date, refresh=refresh)
    reserves_pct_gdp = None
    gdp_m = gdp.get("gdp_millions")
    if isinstance(inp.total_reserves, (int, float)) and isinstance(gdp_m, (int, float)) and float(gdp_m) != 0.0:
        reserves_pct_gdp = 100.0 * float(inp.total_reserves) / float(gdp_m)

    return {
        "asof": state.asof,
        "lookback_years": int(lookback_years),
        "effr": inp.effr,
        "gdp": gdp,
        "reserves": {
            "level": inp.total_reserves,
            "chg_13w": inp.reserves_chg_13w,
            "z_level": inp.z_total_reserves,
            "pct_gdp": reserves_pct_gdp,
        },
        "fed_assets": {
            "level": inp.fed_assets,
            "chg_13w": inp.fed_assets_chg_13w,
            "z_chg_13w": inp.z_fed_assets_chg_13w,
        },
        "on_rrp": {
            "level": inp.on_rrp,
            "chg_13w": inp.on_rrp_chg_13w,
            "z_level": inp.z_on_rrp,
        },
        "series_used": sorted(
            set(MONETARY_FRED_SERIES.values())
            | ({str(gdp.get("series"))} if gdp.get("series") else set())
        ),
        "notes": state.notes,
    }



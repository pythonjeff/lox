from __future__ import annotations

from typing import Dict

import pandas as pd

from lox.commodities.models import CommoditiesInputs, CommoditiesState
from lox.config import Settings
from lox.data.fred import FredClient
from lox.macro.transforms import merge_series_daily, zscore


COMMODITY_FRED_SERIES: Dict[str, str] = {
    # Daily
    "WTI": "DCOILWTICO",
    # Industrial metal (often daily)
    "COPPER": "PCOPPUSDM",
    # Broad (often lower frequency, ffilled)
    "BROAD": "PALLFNFINDEXQ",
}

GOLD_SERIES_CANDIDATES: tuple[str, ...] = (
    # Some environments/accounts see intermittent 400s on the AM fix; try PM as a fallback.
    "GOLDAMGBD228NLBM",
    "GOLDPMGBD228NLBM",
)

GOLD_PROXY_TICKERS: tuple[str, ...] = ("GLDM", "GLD")
SILVER_PROXY_TICKERS: tuple[str, ...] = ("SLV", "SIVR")
COPPER_PROXY_TICKERS: tuple[str, ...] = ("CPER",)
BROAD_PROXY_TICKERS: tuple[str, ...] = ("DBC",)


def build_commodities_dataset(settings: Settings, start_date: str = "2011-01-01", refresh: bool = False) -> pd.DataFrame:
    if not settings.FRED_API_KEY:
        raise RuntimeError("Missing FRED_API_KEY in environment / .env")

    fred = FredClient(api_key=settings.FRED_API_KEY)
    frames: Dict[str, pd.DataFrame] = {}

    # Gold (best-effort; try multiple FRED ids)
    gold_loaded = False
    for sid in GOLD_SERIES_CANDIDATES:
        try:
            df = fred.fetch_series(series_id=sid, start_date=start_date, refresh=refresh)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        frames["GOLD"] = df.sort_values("date").reset_index(drop=True)
        gold_loaded = True
        break

    # If FRED gold fails, fall back to a tradable proxy (Alpaca daily closes).
    # This keeps the commodities regime usable and aligns with how we actually express trades (GLDM options).
    if not gold_loaded and (settings.alpaca_data_key or settings.alpaca_api_key):
        try:
            from lox.data.market import fetch_equity_daily_closes

            px = fetch_equity_daily_closes(settings=settings, symbols=list(GOLD_PROXY_TICKERS), start=start_date, refresh=refresh)
            # Prefer GLDM; otherwise GLD.
            chosen = None
            for t in GOLD_PROXY_TICKERS:
                if t in px.columns and px[t].dropna().shape[0] > 10:
                    chosen = t
                    break
            if chosen:
                s = pd.to_numeric(px[chosen], errors="coerce").dropna()
                gdf = pd.DataFrame({"date": s.index, "value": s.values})
                frames["GOLD"] = gdf.sort_values("date").reset_index(drop=True)
                gold_loaded = True
        except Exception:
            pass

    # Silver via ETF proxy (no reliable daily FRED series)
    if not frames.get("SILVER") and (settings.alpaca_data_key or settings.alpaca_api_key):
        try:
            from lox.data.market import fetch_equity_daily_closes

            px = fetch_equity_daily_closes(settings=settings, symbols=list(SILVER_PROXY_TICKERS), start=start_date, refresh=refresh)
            chosen = None
            for t in SILVER_PROXY_TICKERS:
                if t in px.columns and px[t].dropna().shape[0] > 10:
                    chosen = t
                    break
            if chosen:
                s = pd.to_numeric(px[chosen], errors="coerce").dropna()
                sdf = pd.DataFrame({"date": s.index, "value": s.values})
                frames["SILVER"] = sdf.sort_values("date").reset_index(drop=True)
        except Exception:
            pass

    # Prefer daily tradable proxies for copper/broad (avoid low-frequency/ffill artifacts in some FRED series).
    if settings.alpaca_data_key or settings.alpaca_api_key:
        try:
            from lox.data.market import fetch_equity_daily_closes

            px = fetch_equity_daily_closes(settings=settings, symbols=list(COPPER_PROXY_TICKERS + BROAD_PROXY_TICKERS), start=start_date, refresh=refresh)
            px = px.sort_index().ffill()
            if "CPER" in px.columns and px["CPER"].dropna().shape[0] > 30:
                s = pd.to_numeric(px["CPER"], errors="coerce").dropna()
                frames["COPPER_PROXY"] = pd.DataFrame({"date": s.index, "value": s.values}).sort_values("date").reset_index(drop=True)
            if "DBC" in px.columns and px["DBC"].dropna().shape[0] > 30:
                s = pd.to_numeric(px["DBC"], errors="coerce").dropna()
                frames["BROAD_PROXY"] = pd.DataFrame({"date": s.index, "value": s.values}).sort_values("date").reset_index(drop=True)
        except Exception:
            pass

    for name, sid in COMMODITY_FRED_SERIES.items():
        try:
            df = fred.fetch_series(series_id=sid, start_date=start_date, refresh=refresh)
        except Exception:
            # Optional series: availability varies; keep daily WTI+gold as the core.
            if name in {"BROAD", "COPPER"}:
                continue
            raise
        if df is None or df.empty:
            if name in {"BROAD", "COPPER"}:
                continue
            raise RuntimeError(f"Failed to load commodity series {name} ({sid})")
        frames[name] = df.sort_values("date").reset_index(drop=True)

    max_date = max(df["date"].max() for df in frames.values())
    base = pd.DataFrame({"date": pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(max_date), freq="D")})
    merged = merge_series_daily(base, frames, ffill=True)

    # Use business-day shifts for "20d/60d" to match trading-day intuition (and proxy trend panel).
    idx = merged.set_index("date")
    b = idx.asfreq("B").ffill()

    # Level series (prefer proxies where available)
    if "COPPER_PROXY" in b.columns:
        b["COPPER"] = pd.to_numeric(b["COPPER_PROXY"], errors="coerce")
    if "BROAD_PROXY" in b.columns:
        b["BROAD"] = pd.to_numeric(b["BROAD_PROXY"], errors="coerce")

    wti_b = pd.to_numeric(b["WTI"], errors="coerce") if "WTI" in b.columns else None
    gold_b = pd.to_numeric(b["GOLD"], errors="coerce") if "GOLD" in b.columns else None
    copper_b = pd.to_numeric(b["COPPER"], errors="coerce") if "COPPER" in b.columns else None
    broad_b = pd.to_numeric(b["BROAD"], errors="coerce") if "BROAD" in b.columns else None

    if wti_b is not None:
        b["WTI_RET_20D_PCT"] = (wti_b / wti_b.shift(20) - 1.0) * 100.0
    if gold_b is not None:
        b["GOLD_RET_20D_PCT"] = (gold_b / gold_b.shift(20) - 1.0) * 100.0
    if copper_b is not None:
        b["COPPER_RET_60D_PCT"] = (copper_b / copper_b.shift(60) - 1.0) * 100.0
    if broad_b is not None:
        b["BROAD_RET_60D_PCT"] = (broad_b / broad_b.shift(60) - 1.0) * 100.0

    # Re-merge derived columns back onto the daily grid (ffill across non-business days)
    for c in ["WTI_RET_20D_PCT", "GOLD_RET_20D_PCT", "COPPER_RET_60D_PCT", "BROAD_RET_60D_PCT", "COPPER", "BROAD"]:
        if c in b.columns:
            idx[c] = b[c].reindex(idx.index, method="ffill")
    merged = idx.reset_index()

    win = 252 * 3
    merged["Z_WTI_RET_20D"] = zscore(pd.to_numeric(merged["WTI_RET_20D_PCT"], errors="coerce"), window=win)
    if "GOLD_RET_20D_PCT" in merged.columns:
        merged["Z_GOLD_RET_20D"] = zscore(pd.to_numeric(merged["GOLD_RET_20D_PCT"], errors="coerce"), window=win)
    if "COPPER_RET_60D_PCT" in merged.columns:
        merged["Z_COPPER_RET_60D"] = zscore(pd.to_numeric(merged["COPPER_RET_60D_PCT"], errors="coerce"), window=win)
    if "BROAD_RET_60D_PCT" in merged.columns:
        merged["Z_BROAD_RET_60D"] = zscore(pd.to_numeric(merged["BROAD_RET_60D_PCT"], errors="coerce"), window=win)

    # Composite (2026-aligned): energy + industrial metals + broad complex, with gold as policy/real-rate hedge.
    # If broad/copper are missing, gracefully fall back to oil+gold.
    if "Z_GOLD_RET_20D" in merged.columns:
        score = 0.65 * merged["Z_WTI_RET_20D"] + 0.35 * merged["Z_GOLD_RET_20D"]
    else:
        score = 1.0 * merged["Z_WTI_RET_20D"]
    if "Z_COPPER_RET_60D" in merged.columns:
        if "Z_GOLD_RET_20D" in merged.columns:
            score = 0.55 * merged["Z_WTI_RET_20D"] + 0.25 * merged["Z_COPPER_RET_60D"] + 0.20 * merged["Z_GOLD_RET_20D"]
        else:
            score = 0.70 * merged["Z_WTI_RET_20D"] + 0.30 * merged["Z_COPPER_RET_60D"]
    if "Z_BROAD_RET_60D" in merged.columns and "Z_COPPER_RET_60D" in merged.columns:
        if "Z_GOLD_RET_20D" in merged.columns:
            score = (
                0.45 * merged["Z_WTI_RET_20D"]
                + 0.25 * merged["Z_COPPER_RET_60D"]
                + 0.20 * merged["Z_BROAD_RET_60D"]
                + 0.10 * merged["Z_GOLD_RET_20D"]
            )
        else:
            score = 0.55 * merged["Z_WTI_RET_20D"] + 0.25 * merged["Z_COPPER_RET_60D"] + 0.20 * merged["Z_BROAD_RET_60D"]
    elif "Z_BROAD_RET_60D" in merged.columns:
        if "Z_GOLD_RET_20D" in merged.columns:
            score = 0.55 * merged["Z_WTI_RET_20D"] + 0.30 * merged["Z_BROAD_RET_60D"] + 0.15 * merged["Z_GOLD_RET_20D"]
        else:
            score = 0.65 * merged["Z_WTI_RET_20D"] + 0.35 * merged["Z_BROAD_RET_60D"]
    merged["COMMODITY_PRESSURE_SCORE"] = score

    # Energy shock flag: big oil impulse
    merged["ENERGY_SHOCK"] = (merged["WTI_RET_20D_PCT"] > 20.0) & (merged["Z_WTI_RET_20D"] > 2.0)
    # Metals impulse: copper strong (often growth impulse / China impulse proxy)
    if "COPPER_RET_60D_PCT" in merged.columns and "Z_COPPER_RET_60D" in merged.columns:
        merged["METALS_IMPULSE"] = (merged["COPPER_RET_60D_PCT"] > 10.0) & (merged["Z_COPPER_RET_60D"] > 1.5)

    return merged


def build_commodities_state(settings: Settings, start_date: str = "2011-01-01", refresh: bool = False) -> CommoditiesState:
    df = build_commodities_dataset(settings=settings, start_date=start_date, refresh=refresh)
    # Require core WTI; other series are best-effort.
    last = df.dropna(subset=["WTI"]).iloc[-1]

    inp = CommoditiesInputs(
        wti=float(last["WTI"]) if pd.notna(last.get("WTI")) else None,
        gold=float(last["GOLD"]) if "GOLD" in last and pd.notna(last.get("GOLD")) else None,
        silver=float(last["SILVER"]) if "SILVER" in last and pd.notna(last.get("SILVER")) else None,
        # Copper/Broad are now often proxy ETF levels (CPER/DBC) to avoid ffill artifacts.
        copper=float(last["COPPER"]) if "COPPER" in last and pd.notna(last.get("COPPER")) else None,
        broad_index=float(last["BROAD"]) if "BROAD" in last and pd.notna(last.get("BROAD")) else None,
        wti_ret_20d_pct=float(last["WTI_RET_20D_PCT"]) if pd.notna(last.get("WTI_RET_20D_PCT")) else None,
        gold_ret_20d_pct=float(last["GOLD_RET_20D_PCT"])
        if "GOLD_RET_20D_PCT" in last and pd.notna(last.get("GOLD_RET_20D_PCT"))
        else None,
        copper_ret_60d_pct=float(last["COPPER_RET_60D_PCT"])
        if "COPPER_RET_60D_PCT" in last and pd.notna(last.get("COPPER_RET_60D_PCT"))
        else None,
        broad_ret_60d_pct=float(last["BROAD_RET_60D_PCT"])
        if "BROAD_RET_60D_PCT" in last and pd.notna(last.get("BROAD_RET_60D_PCT"))
        else None,
        z_wti_ret_20d=float(last["Z_WTI_RET_20D"]) if pd.notna(last.get("Z_WTI_RET_20D")) else None,
        z_gold_ret_20d=float(last["Z_GOLD_RET_20D"])
        if "Z_GOLD_RET_20D" in last and pd.notna(last.get("Z_GOLD_RET_20D"))
        else None,
        z_copper_ret_60d=float(last["Z_COPPER_RET_60D"])
        if "Z_COPPER_RET_60D" in last and pd.notna(last.get("Z_COPPER_RET_60D"))
        else None,
        z_broad_ret_60d=float(last["Z_BROAD_RET_60D"]) if "Z_BROAD_RET_60D" in last and pd.notna(last.get("Z_BROAD_RET_60D")) else None,
        commodity_pressure_score=float(last["COMMODITY_PRESSURE_SCORE"]) if pd.notna(last.get("COMMODITY_PRESSURE_SCORE")) else None,
        energy_shock=bool(last["ENERGY_SHOCK"]) if "ENERGY_SHOCK" in last and pd.notna(last.get("ENERGY_SHOCK")) else None,
        metals_impulse=bool(last["METALS_IMPULSE"]) if "METALS_IMPULSE" in last and pd.notna(last.get("METALS_IMPULSE")) else None,
        components={"window_days": float(252 * 3), "wti_ret_days": 20.0, "copper_ret_days": 60.0, "broad_ret_days": 60.0},
    )

    return CommoditiesState(
        asof=str(pd.to_datetime(last["date"]).date()),
        start_date=start_date,
        inputs=inp,
        notes="Commodities regime MVP: oil+gold (+ optional broad index) returns, z-scored vs recent history; flags energy shocks.",
    )



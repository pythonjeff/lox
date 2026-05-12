"""
Fed balance sheet composition (H.4.1, weekly Wednesday).

Decomposes WALCL into asset-side components so we can see *where* the Fed is
spending while the balance sheet expands — Treasury purchases (bills, notes
and bonds, TIPS), MBS, agency debt, and emergency lending (discount window,
BTFP). Tracks 4w / 13w / YTD deltas per component.

Reading guide:
    Expansion driven by bills + nothing in lending → routine reserve mgmt
        ("not QE") to offset RRP/TGA drain.
    Expansion driven by notes/bonds growth → genuine QE / duration buying;
        risk-on signal.
    Expansion driven by BTFP or discount window spike → bank-stress event,
        not policy choice.
    MBS line shrinks but Treasuries hold → composition shift toward shorter
        duration; passive MBS runoff continuing.

Sources (all FRED, weekly Wed close, $ millions):
    WALCL           — Total assets (anchor)
    WSHOBL          — Treasury bills
    WSHONBNL        — Treasury notes & bonds, nominal
    WSHONBIIL       — Treasury notes & bonds, inflation-indexed (TIPS, face)
    WSHOFADSL       — Federal agency debt
    WSHOMCB         — Mortgage-backed securities
    WORAL           — Asset-side repurchase agreements (standing repo / SRF use)
    WLCFLPCL        — Primary credit (discount window)
    H41RESPPALDKNWW — Bank Term Funding Program (BTFP)

Public API:
    fetch_fed_bs_composition(refresh=False) -> pd.DataFrame
    compute_fed_bs_metrics(refresh=False) -> dict
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from lox.config import load_settings
from lox.data.fred import FredClient


# Component key → (FRED series ID, display label, leg type)
# leg type drives signal coloring: "purchase" lines reflect policy choices,
# "lending" lines reflect bank stress (rising = bad), "total" is neutral.
_COMPONENTS: list[tuple[str, str, str, str]] = [
    ("total_assets",   "WALCL",            "Total (WALCL)",        "total"),
    ("bills",          "WSHOBL",           "Treasury bills",       "purchase"),
    ("notes_bonds",    "WSHONBNL",         "Treasury notes/bonds", "purchase"),
    ("tips",           "WSHONBIIL",        "TIPS (face)",          "purchase"),
    ("agency",         "WSHOFADSL",        "Agency debt",          "purchase"),
    ("mbs",            "WSHOMCB",          "MBS",                  "purchase"),
    ("repo",           "WORAL",            "Repo (asset-side)",    "lending"),
    ("primary_credit", "WLCFLPCL",         "Discount window",      "lending"),
    ("btfp",           "H41RESPPALDKNWW",  "BTFP",                 "lending"),
]


def _component_meta() -> dict[str, dict[str, str]]:
    return {key: {"series": sid, "label": label, "leg": leg} for key, sid, label, leg in _COMPONENTS}


def fetch_fed_bs_composition(
    *,
    refresh: bool = False,
    start_date: str = "2018-01-01",
) -> pd.DataFrame:
    """
    Weekly-aligned DataFrame of H.4.1 balance sheet components in $ millions.

    One row per Wednesday print. Missing components are simply absent columns;
    callers should check `col in df.columns` before reading. Empty DataFrame
    if no FRED key configured or WALCL itself fails to load.
    """
    settings = load_settings()
    if not settings.FRED_API_KEY:
        return pd.DataFrame()

    fred = FredClient(api_key=settings.FRED_API_KEY)
    frames: dict[str, pd.DataFrame] = {}
    for key, sid, _label, _leg in _COMPONENTS:
        try:
            df = fred.fetch_series(series_id=sid, start_date=start_date, refresh=refresh)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df[key] = pd.to_numeric(df["value"], errors="coerce")
        frames[key] = df.dropna(subset=["date", key])[["date", key]]

    if "total_assets" not in frames or frames["total_assets"].empty:
        return pd.DataFrame()

    out = frames["total_assets"]
    for key, f in frames.items():
        if key == "total_assets":
            continue
        out = out.merge(f, on="date", how="left")
    return out.sort_values("date").reset_index(drop=True)


def _delta_n_weeks(df: pd.DataFrame, col: str, n_weeks: int) -> Optional[float]:
    if col not in df.columns or len(df) <= n_weeks:
        return None
    cur = df[col].iloc[-1]
    prior = df[col].iloc[-1 - n_weeks]
    if pd.isna(cur) or pd.isna(prior):
        return None
    return float(cur - prior)


def _delta_ytd(df: pd.DataFrame, col: str, asof: pd.Timestamp) -> Optional[float]:
    if col not in df.columns:
        return None
    year_rows = df[df["date"].dt.year == asof.year]
    if year_rows.empty:
        return None
    first = year_rows.iloc[0][col]
    cur = df.iloc[-1][col]
    if pd.isna(first) or pd.isna(cur):
        return None
    return float(cur - first)


def compute_fed_bs_metrics(*, refresh: bool = False) -> dict:
    """
    Per-component level + 4w / 13w / YTD deltas, all in $ millions.

    Returns dict shape:
        {
            "asof": "YYYY-MM-DD" | None,
            "components": {
                "<key>": {
                    "label": str,
                    "leg": "total" | "purchase" | "lending",
                    "level_m": float,
                    "delta_4w_m": float | None,
                    "delta_13w_m": float | None,
                    "delta_ytd_m": float | None,
                },
                ...
            } | None,
        }

    Deltas are absolute $-millions changes (not %). Convert with /1000 for $B
    or /1_000_000 for $T at the display layer.
    """
    empty = {"asof": None, "components": None}
    try:
        df = fetch_fed_bs_composition(refresh=refresh)
    except Exception:
        return empty
    if df.empty:
        return empty

    last = df.iloc[-1]
    asof_ts = pd.to_datetime(last["date"])
    meta = _component_meta()

    components: dict[str, dict] = {}
    for key, info in meta.items():
        if key not in df.columns:
            continue
        lvl = last[key]
        if pd.isna(lvl):
            continue
        components[key] = {
            "label": info["label"],
            "leg": info["leg"],
            "level_m": float(lvl),
            "delta_4w_m": _delta_n_weeks(df, key, 4),
            "delta_13w_m": _delta_n_weeks(df, key, 13),
            "delta_ytd_m": _delta_ytd(df, key, asof_ts),
        }

    return {
        "asof": str(asof_ts.date()),
        "components": components,
    }

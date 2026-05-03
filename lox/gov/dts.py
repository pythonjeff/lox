"""
Daily Treasury Statement (DTS) — daily TGA flow tracker.

Endpoint: /v1/accounting/dts/operating_cash_balance
Source:   fiscaldata.treasury.gov

The DTS is published every business day around 4 PM ET and reports the
prior business day's Treasury General Account closing balance. This gives
us a daily-resolution view that the weekly FRED WTREGEN series can't.

Public API:
    fetch_tga_daily(refresh=False) -> pd.DataFrame
        columns: date (datetime.date), tga_close_b (float, $B)

    compute_tga_daily_metrics(refresh=False) -> dict
        {
            "asof": "YYYY-MM-DD",
            "level_b": float,            # most recent close
            "delta_1d_b": float | None,  # change vs previous business day
            "series_5d_b": list[float],  # last 5 business days, oldest→newest
            "floor_distance_b": float,   # level minus historical panic floor
            "floor_label": str,          # which floor we're comparing to
        }

Historical "panic floor" reference points:
    Oct 2015 debt ceiling crisis: TGA bottomed near $50B
    Jun 2023 debt ceiling crisis: TGA bottomed near $23B
We use $50B as the conservative comparison floor.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd

from lox.data.fiscaldata import FiscalDataClient, FiscalDataEndpoint


_DTS_OCB_ENDPOINT = FiscalDataEndpoint(
    path="/v1/accounting/dts/operating_cash_balance",
)

# DTS reports TGA in millions ($). Convert to $B for display.
_MILLIONS_PER_BILLION = 1000.0

# Historical debt-ceiling panic floor (conservative — Oct 2015 low).
TGA_PANIC_FLOOR_B: float = 50.0
TGA_PANIC_FLOOR_LABEL: str = "Oct 2015 debt-ceiling low"


def fetch_tga_daily(
    *,
    refresh: bool = False,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """
    Fetch daily TGA closing balances from DTS.

    The DTS operating_cash_balance endpoint returns multiple account_type rows
    per record_date. We filter to TGA closing-balance rows and aggregate any
    duplicates (the schema has changed historically across reformats).
    """
    client = FiscalDataClient()
    today = date.today()
    start = (pd.Timestamp(today) - pd.Timedelta(days=lookback_days * 2)).date().isoformat()

    df = client.fetch(
        endpoint=_DTS_OCB_ENDPOINT,
        params={
            "filter": f"record_date:gte:{start}",
            "sort": "-record_date",
            "fields": "record_date,account_type,close_today_bal,open_today_bal",
        },
        cache_key=f"dts_ocb_{start}",
        refresh=refresh,
    )
    if df.empty:
        return pd.DataFrame(columns=["date", "tga_close_b"])

    df["account_type"] = df["account_type"].astype(str)
    closing_mask = df["account_type"].str.contains("Closing Balance", case=False, na=False)
    df = df[closing_mask].copy()

    # DTS schema quirk: for "Closing Balance" rows the value lives in
    # `open_today_bal` (close_today_bal is published as null). Fall back
    # to close_today_bal in case the schema flips back.
    df["close_today_bal"] = pd.to_numeric(df["close_today_bal"], errors="coerce")
    df["open_today_bal"] = pd.to_numeric(df["open_today_bal"], errors="coerce")
    df["value_m"] = df["open_today_bal"].where(df["open_today_bal"].notna(), df["close_today_bal"])
    df = df.dropna(subset=["value_m"])

    df["date"] = pd.to_datetime(df["record_date"]).dt.date
    daily = (
        df.groupby("date", as_index=False)["value_m"]
        .max()
        .rename(columns={"value_m": "tga_close_m"})
    )
    daily["tga_close_b"] = daily["tga_close_m"] / _MILLIONS_PER_BILLION
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily[["date", "tga_close_b"]]


def compute_tga_daily_metrics(*, refresh: bool = False) -> dict:
    """Daily-resolution TGA metrics for the gov panel."""
    empty = {
        "asof": None,
        "level_b": None,
        "delta_1d_b": None,
        "series_5d_b": [],
        "floor_distance_b": None,
        "floor_label": TGA_PANIC_FLOOR_LABEL,
    }
    try:
        df = fetch_tga_daily(refresh=refresh)
    except Exception:
        return empty

    if df.empty:
        return empty

    last = df.iloc[-1]
    level_b = float(last["tga_close_b"])
    delta_1d: Optional[float] = None
    if len(df) >= 2:
        delta_1d = level_b - float(df.iloc[-2]["tga_close_b"])

    series_5d = [float(v) for v in df["tga_close_b"].tail(5).tolist()]

    return {
        "asof": str(last["date"]),
        "level_b": level_b,
        "delta_1d_b": delta_1d,
        "series_5d_b": series_5d,
        "floor_distance_b": level_b - TGA_PANIC_FLOOR_B,
        "floor_label": TGA_PANIC_FLOOR_LABEL,
    }

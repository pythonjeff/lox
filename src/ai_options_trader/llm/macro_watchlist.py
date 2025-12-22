from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

from ai_options_trader.config import Settings
from ai_options_trader.econ_release import fetch_fmp_economic_calendar, normalize_fmp_economic_calendar, next_release
from ai_options_trader.data.fred import FredClient
import pandas as pd


@dataclass(frozen=True)
class WatchItem:
    name: str
    source_url: str
    schedule_url: str
    default_time_et: str | None = None
    # Optional label to help BEA parsing
    bea_hint: str | None = None
    # Optional FRED series id to show current value + 12y change
    fred_series_id: str | None = None
    fred_transform: str | None = None  # "level" | "yoy_pct"


_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def _extract_dates_from_html(text: str) -> list[datetime]:
    """
    Extract candidate dates from an HTML page.
    Supports patterns like:
    - January 14, 2026
    - Jan. 14, 2026
    - January 28-29, 2026  (takes first day)
    """
    if not text:
        return []
    month_alt = "|".join([m for m in _MONTHS] + [m[:3] + r"\.?" for m in _MONTHS])
    # Month Day(-Day)?, Year
    pat = re.compile(rf"\b({month_alt})\s+(\d{{1,2}})(?:-\d{{1,2}})?\,\s+(\d{{4}})\b")
    out: list[datetime] = []
    for m, d, y in pat.findall(text):
        mm = m.replace(".", "")
        # Normalize short months
        if len(mm) <= 3:
            mm_full = next((x for x in _MONTHS if x.lower().startswith(mm.lower())), None)
            if not mm_full:
                continue
            mm = mm_full
        try:
            dt = datetime.strptime(f"{mm} {d} {y}", "%B %d %Y").replace(tzinfo=timezone.utc)
            out.append(dt)
        except Exception:
            continue
    return out


def _next_date_from_schedule_page(
    *,
    url: str,
    now_utc: datetime,
    hint: str | None = None,
    timeout_s: int = 20,
) -> datetime | None:
    """
    Best-effort: fetch schedule page and return the next date after now_utc.
    If `hint` is provided, we try to look near that phrase (useful for BEA schedule).
    """
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    html = resp.text

    if hint:
        idx = html.lower().find(hint.lower())
        if idx != -1:
            window = html[idx : idx + 4000]
            dates = _extract_dates_from_html(window)
        else:
            dates = _extract_dates_from_html(html)
    else:
        dates = _extract_dates_from_html(html)

    future = sorted({d for d in dates if d.date() >= now_utc.date()})
    for d in future:
        # Treat "today" as future only if still upcoming; we don't know time from page reliably.
        if d.date() > now_utc.date():
            return d
    # If we only saw "today" dates, return today (rare but OK).
    if future:
        return future[0]
    return None


def build_watchlist(now_utc: datetime, settings: Settings | None = None) -> list[str]:
    """
    Generate point (5) bullets with real next scheduled dates by scraping official schedule pages.

    Returns bullets like:
    "- CPI inflation — Source: https://www.bls.gov/schedule/news_release/cpi.htm — Next: 2026-01-14 08:30 ET"
    """
    items = [
        WatchItem(
            name="CPI inflation",
            source_url="https://www.bls.gov/cpi/",
            schedule_url="https://www.bls.gov/schedule/news_release/cpi.htm",
            default_time_et="08:30 ET",
            fred_series_id="CPIAUCSL",
            fred_transform="yoy_pct",
        ),
        WatchItem(
            name="Jobs report (Employment Situation)",
            source_url="https://www.bls.gov/news.release/empsit.htm",
            schedule_url="https://www.bls.gov/schedule/news_release/empsit.htm",
            default_time_et="08:30 ET",
            fred_series_id="UNRATE",
            fred_transform="level",
        ),
        WatchItem(
            name="PPI inflation",
            source_url="https://www.bls.gov/ppi/",
            schedule_url="https://www.bls.gov/schedule/news_release/ppi.htm",
            default_time_et="08:30 ET",
            fred_series_id="PPIACO",
            fred_transform="yoy_pct",
        ),
        WatchItem(
            name="PCE inflation / Personal Income & Outlays",
            source_url="https://www.bea.gov/data/personal-consumption-expenditures-price-index",
            schedule_url="https://www.bea.gov/news/schedule",
            default_time_et="08:30 ET",
            bea_hint="Personal Income and Outlays",
            fred_series_id="PCEPI",
            fred_transform="yoy_pct",
        ),
        WatchItem(
            name="GDP (advance/second/third estimate)",
            source_url="https://www.bea.gov/data/gdp/gross-domestic-product",
            schedule_url="https://www.bea.gov/news/schedule",
            default_time_et="08:30 ET",
            bea_hint="Gross Domestic Product",
            fred_series_id="GDPC1",
            fred_transform="yoy_pct",
        ),
        WatchItem(
            name="FOMC decision / SEP",
            source_url="https://www.federalreserve.gov/monetarypolicy/fomc.htm",
            schedule_url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
            default_time_et="14:00 ET",
            fred_series_id="DFF",
            fred_transform="level",
        ),
    ]

    bullets: list[str] = []
    # Prefer FMP economic calendar when configured; fall back to scraping official schedule pages.
    fmp_events = None
    if settings and settings.fmp_api_key:
        try:
            end = (now_utc + timedelta(days=180)).date().isoformat()
            rows = fetch_fmp_economic_calendar(
                api_key=settings.fmp_api_key,
                from_date=now_utc.date().isoformat(),
                to_date=end,
            )
            fmp_events = normalize_fmp_economic_calendar(rows)
        except Exception:
            fmp_events = None

    fred_client = None
    if settings:
        # FredClient prefers cache when present; api_key can be empty and still work with cache.
        fred_client = FredClient(api_key=settings.fred_api_key or "")

    def _asof_value(df: pd.DataFrame, when: pd.Timestamp) -> float | None:
        if df.empty:
            return None
        d = df.sort_values("date")
        d = d[d["date"] <= when]
        if d.empty:
            return None
        return float(d.iloc[-1]["value"])

    def _fred_current_and_delta12y(series_id: str, transform: str) -> tuple[str | None, str | None]:
        if not fred_client:
            return None, None
        try:
            df = fred_client.fetch_series(series_id=series_id, start_date="2000-01-01", refresh=False)
        except Exception:
            return None, None
        if df.empty:
            return None, None
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        # FRED cache uses tz-naive timestamps; compare using tz-naive times.
        now_ts = pd.Timestamp(now_utc.replace(tzinfo=None))
        t12y = now_ts - pd.DateOffset(years=12)

        if transform == "level":
            cur = _asof_value(df, now_ts)
            past = _asof_value(df, t12y)
            if cur is None or past is None:
                return None, None
            cur_s = f"{cur:.2f}%"
            dpp = cur - past
            delta_s = f"{dpp:+.2f}pp"
            return cur_s, delta_s

        if transform == "yoy_pct":
            # Compute YoY% at a given time: (level / level_12m_ago - 1) * 100
            def yoy_at(ts: pd.Timestamp) -> float | None:
                cur = _asof_value(df, ts)
                prev = _asof_value(df, ts - pd.DateOffset(months=12))
                if cur is None or prev is None or prev == 0:
                    return None
                return (cur / prev - 1.0) * 100.0

            cur_yoy = yoy_at(now_ts)
            past_yoy = yoy_at(t12y)
            if cur_yoy is None or past_yoy is None:
                return None, None
            cur_s = f"{cur_yoy:.2f}%"
            delta_s = f"{(cur_yoy - past_yoy):+.2f}pp"
            return cur_s, delta_s

        return None, None

    for it in items:
        # Stats should be shown even if we can't fetch/parse the next release date.
        cur = None
        d12 = None
        if it.fred_series_id and it.fred_transform:
            cur, d12 = _fred_current_and_delta12y(it.fred_series_id, it.fred_transform)
        stats = f" — Current: {cur} — Δ12y: {d12}" if cur and d12 else ""

        dt = None
        try:
            if fmp_events is not None:
                # Map watch item -> FMP keyword filters (best-effort)
                kw = []
                if it.name.startswith("CPI"):
                    kw = ["cpi"]
                elif it.name.startswith("Jobs report"):
                    kw = ["nonfarm"]
                elif it.name.startswith("PPI"):
                    kw = ["ppi"]
                elif it.name.startswith("PCE"):
                    kw = ["pce"]
                elif it.name.startswith("GDP"):
                    kw = ["gdp"]
                elif it.name.startswith("FOMC"):
                    kw = ["fed"]
                ev = next_release(events=fmp_events, after_utc=now_utc, country="US", event_contains=kw)
                if ev:
                    dt = ev.datetime_utc
            if dt is None:
                dt = _next_date_from_schedule_page(
                    url=it.schedule_url,
                    now_utc=now_utc,
                    hint=it.bea_hint,
                )
        except Exception:
            dt = None

        if dt is None:
            bullets.append(f"- {it.name} — Source: {it.source_url} — Next: see schedule: {it.schedule_url}{stats}")
            continue

        dstr = dt.date().isoformat()
        tstr = f" {it.default_time_et}" if it.default_time_et else ""
        bullets.append(f"- {it.name} — Source: {it.source_url} — Next: {dstr}{tstr}{stats}")

    # Keep it short and stable
    return bullets[:5]



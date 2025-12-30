from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


FMP_BASE_URL = "https://financialmodelingprep.com/api"


@dataclass(frozen=True)
class EconCalendarEvent:
    """
    Normalized event shape across providers.
    """

    provider: str  # "fmp"
    country: str | None
    event: str
    category: str | None
    datetime_utc: datetime
    importance: str | None = None
    actual: str | None = None
    previous: str | None = None
    forecast: str | None = None
    raw: dict[str, Any] | None = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt_any(value: Any) -> datetime | None:
    """
    Parse common datetime formats returned by calendar providers.
    Trading Economics usually returns ISO-ish strings; we accept a few variants.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    s = str(value).strip()
    if not s:
        return None
    # Normalize Z suffix
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Common: "2025-12-22T13:30:00" (no tz) or "2025-12-22T13:30:00+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    # Common: "2025-12-22 13:30:00"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            continue
    return None


def _cache_path(key: str) -> Path:
    root = Path(os.environ.get("AOT_CACHE_DIR", "data/cache"))
    root.mkdir(parents=True, exist_ok=True)
    return root / "econ_calendar" / f"{key}.json"


def _read_cache(path: Path, max_age: timedelta) -> Any | None:
    try:
        if not path.exists():
            return None
        age = _utc_now() - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if age > max_age:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def fetch_fmp_economic_calendar(
    *,
    api_key: str,
    from_date: str,  # YYYY-MM-DD
    to_date: str,  # YYYY-MM-DD
    timeout_s: int = 30,
    cache_max_age: timedelta = timedelta(hours=6),
) -> list[dict[str, Any]]:
    """
    Fetch FMP economic calendar items.

    Endpoint:
    `/api/v3/economic_calendar?from=YYYY-MM-DD&to=YYYY-MM-DD&apikey=...`
    """
    if not api_key:
        raise ValueError("Missing FMP api_key")
    key = f"fmp_economic_calendar_{from_date}_{to_date}"
    cache = _cache_path(key)
    cached = _read_cache(cache, cache_max_age)
    if isinstance(cached, list):
        return cached

    url = f"{FMP_BASE_URL}/v3/economic_calendar"
    resp = requests.get(
        url,
        params={"from": from_date, "to": to_date, "apikey": api_key},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError("Unexpected FMP economic_calendar response shape")
    _write_cache(cache, data)
    return data


def normalize_fmp_economic_calendar(rows: list[dict[str, Any]]) -> list[EconCalendarEvent]:
    out: list[EconCalendarEvent] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        dt = _parse_dt_any(r.get("date") or r.get("Date"))
        if dt is None:
            continue
        event = str(r.get("event") or r.get("Event") or "").strip()
        if not event:
            continue
        out.append(
            EconCalendarEvent(
                provider="fmp",
                country=(str(r.get("country") or r.get("Country") or "").strip() or None),
                event=event,
                category=(str(r.get("currency") or r.get("category") or "").strip() or None),
                datetime_utc=dt,
                importance=(str(r.get("impact") or r.get("importance") or "").strip() or None),
                actual=(str(r.get("actual") or r.get("Actual") or "").strip() or None),
                previous=(str(r.get("previous") or r.get("Previous") or "").strip() or None),
                forecast=(str(r.get("estimate") or r.get("forecast") or "").strip() or None),
                raw=r,
            )
        )
    out.sort(key=lambda x: x.datetime_utc)
    return out


def next_release(
    *,
    events: list[EconCalendarEvent],
    after_utc: datetime,
    country: str | None = None,
    event_contains: list[str] | None = None,
    category_contains: list[str] | None = None,
) -> EconCalendarEvent | None:
    """
    Find the next event after `after_utc` matching optional filters.
    """
    c = country.lower() if country else None
    ev_words = [w.lower() for w in (event_contains or []) if w.strip()]
    cat_words = [w.lower() for w in (category_contains or []) if w.strip()]

    for e in events:
        if e.datetime_utc <= after_utc:
            continue
        if c and (not e.country or e.country.lower() != c):
            continue
        if ev_words and not all(w in e.event.lower() for w in ev_words):
            continue
        if cat_words:
            cat = (e.category or "").lower()
            if not all(w in cat for w in cat_words):
                continue
        return e
    return None



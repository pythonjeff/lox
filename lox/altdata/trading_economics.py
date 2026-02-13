"""Alternative data sources for indicators not on FRED.

Primary: FMP economic calendar (extracts latest actuals from past events).
Fallback: Trading Economics API (when available).

All functions return None gracefully if data is unavailable.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

TE_BASE_URL = "https://api.tradingeconomics.com"


# ── FMP Economic Calendar Lookback ─────────────────────────────────────────

def _fmp_latest_actual(
    event_keywords: list[str],
    country_keywords: list[str] | None = None,
    lookback_days: int = 45,
) -> float | None:
    """Search FMP economic calendar for the most recent actual value matching keywords.

    Looks backward from today to find the latest released actual for an event
    whose name contains ALL of ``event_keywords`` (case-insensitive).
    Optionally filters by country keywords as well.

    Returns the ``actual`` value as a float, or None.
    """
    try:
        from lox.config import load_settings
        settings = load_settings()
        if not settings.fmp_api_key:
            return None

        from lox.data.econ_release import fetch_fmp_economic_calendar

        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        events = fetch_fmp_economic_calendar(
            api_key=settings.fmp_api_key,
            from_date=from_date,
            to_date=to_date,
        )

        kw_lower = [k.lower() for k in event_keywords]
        ctry_lower = [c.lower() for c in (country_keywords or ["us", "united states"])]

        # Walk newest → oldest
        for ev in reversed(events):
            name = (ev.get("event") or "").lower()
            country = (ev.get("country") or "").lower()

            if not all(k in name for k in kw_lower):
                continue
            if ctry_lower and not any(c in country for c in ctry_lower):
                continue

            actual = ev.get("actual")
            if actual is not None and str(actual).strip() not in ("", "None"):
                try:
                    return float(actual)
                except (TypeError, ValueError):
                    continue

        return None
    except Exception as exc:
        logger.debug("FMP calendar lookup failed for %s: %s", event_keywords, exc)
        return None


# ── Trading Economics helpers (kept for future paid tier) ──────────────────

def _get_te_api_key() -> str | None:
    """Get TE API key from environment."""
    from lox.config import load_settings
    try:
        settings = load_settings()
        return settings.trading_economics_api_key
    except Exception:
        return os.getenv("TRADING_ECONOMICS_API_KEY")


def _te_indicator_snapshot(
    indicator: str,
    country: str = "united states",
) -> float | None:
    """Try Trading Economics for a single indicator value.

    Returns float or None.  Fails silently on 403 / free-tier errors.
    """
    api_key = _get_te_api_key()
    if not api_key:
        return None

    import requests

    try:
        url = f"{TE_BASE_URL}/country/{country}/{indicator}"
        resp = requests.get(url, params={"c": api_key}, timeout=10)
        if resp.status_code == 403:
            # Free tier — don't log as warning, just debug
            logger.debug("TE free-tier block for %s (403)", indicator)
            return None
        resp.raise_for_status()
        data = resp.json()
        if data and isinstance(data, list) and len(data) > 0:
            val = data[0].get("LatestValue")
            if val is not None:
                return float(val)
    except Exception as exc:
        logger.debug("TE API error for %s: %s", indicator, exc)

    return None


# ── Public convenience functions ───────────────────────────────────────────

def get_ism_manufacturing() -> float | None:
    """ISM Manufacturing PMI (latest value).

    Primary: FMP economic calendar (lookback for latest actual).
    Fallback: Trading Economics API.
    """
    # FMP first
    val = _fmp_latest_actual(["ism", "manufacturing", "pmi"])
    if val is not None:
        return val

    # TE fallback
    return _te_indicator_snapshot("ISM Manufacturing PMI")


def get_michigan_sentiment() -> float | None:
    """University of Michigan Consumer Sentiment.

    Primary: FMP economic calendar.
    Fallback: Trading Economics.
    """
    val = _fmp_latest_actual(["michigan", "consumer", "sentiment"])
    if val is not None:
        return val
    return _te_indicator_snapshot("consumer confidence")


def get_michigan_expectations() -> float | None:
    """University of Michigan Consumer Expectations sub-index.

    Primary: FMP economic calendar.
    Fallback: Trading Economics.
    """
    val = _fmp_latest_actual(["michigan", "consumer", "expectations"])
    if val is not None:
        return val
    return _te_indicator_snapshot("consumer expectations")


def get_retail_sales_control() -> float | None:
    """Retail Sales Control Group MoM %.

    Primary: FMP economic calendar.
    Fallback: Trading Economics.
    """
    val = _fmp_latest_actual(["retail", "sales"])
    if val is not None:
        return val
    return _te_indicator_snapshot("retail sales ex autos")


def get_personal_spending() -> float | None:
    """Personal Spending MoM %.

    Primary: FMP economic calendar.
    Fallback: Trading Economics.
    """
    val = _fmp_latest_actual(["personal", "spending"])
    if val is not None:
        return val
    return _te_indicator_snapshot("personal spending")


def get_aaii_bullish_sentiment() -> float | None:
    """AAII Bullish Sentiment % (for positioning regime).

    Not available on FRED or FMP.
    Falls back to Trading Economics (paid tier only).
    """
    return _te_indicator_snapshot("AAII Bullish Sentiment")


# ── Legacy public API (kept for backward compat) ──────────────────────────

def get_indicator_snapshot(
    indicator: str,
    country: str = "united states",
) -> dict[str, Any] | None:
    """Get latest value for a Trading Economics indicator.

    Returns dict with keys: value, previous, date, frequency, unit
    or None if unavailable.
    """
    api_key = _get_te_api_key()
    if not api_key:
        return None

    import requests

    try:
        url = f"{TE_BASE_URL}/country/{country}/{indicator}"
        resp = requests.get(url, params={"c": api_key}, timeout=10)
        if resp.status_code == 403:
            logger.debug("TE free-tier block for %s (403)", indicator)
            return None
        resp.raise_for_status()
        data = resp.json()
        if data and isinstance(data, list) and len(data) > 0:
            return {
                "value": data[0].get("LatestValue"),
                "previous": data[0].get("PreviousValue"),
                "date": data[0].get("LatestValueDate"),
                "frequency": data[0].get("Frequency"),
                "unit": data[0].get("Unit"),
            }
    except Exception as exc:
        logger.debug("TE API error for %s: %s", indicator, exc)

    return None


def get_indicator_history(
    indicator: str,
    country: str = "united states",
    start_date: str | None = None,
    periods: int = 24,
) -> list[dict[str, Any]]:
    """Get historical values for a Trading Economics indicator."""
    api_key = _get_te_api_key()
    if not api_key:
        return []

    import requests

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=periods * 30)).strftime("%Y-%m-%d")

    try:
        url = f"{TE_BASE_URL}/historical/country/{country}/indicator/{indicator}"
        resp = requests.get(
            url, params={"c": api_key, "d1": start_date}, timeout=10
        )
        if resp.status_code == 403:
            logger.debug("TE free-tier block for %s history (403)", indicator)
            return []
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("TE history error for %s: %s", indicator, exc)
        return []

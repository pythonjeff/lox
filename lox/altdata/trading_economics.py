"""Trading Economics API client for consumer and macro indicators.

Provides real-time snapshot and historical data for indicators not available via FRED.
Falls back gracefully when API key is not configured.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)

TE_BASE_URL = "https://api.tradingeconomics.com"


def _get_api_key() -> str | None:
    """Get TE API key from environment."""
    from lox.config import load_settings
    try:
        settings = load_settings()
        return settings.trading_economics_api_key
    except Exception:
        return os.getenv("TRADING_ECONOMICS_API_KEY")


def get_indicator_snapshot(
    indicator: str,
    country: str = "united states",
) -> dict[str, Any] | None:
    """Get latest value for a Trading Economics indicator.

    Returns dict with keys: value, previous, date, frequency, unit
    or None if unavailable.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.debug("No TRADING_ECONOMICS_API_KEY configured")
        return None

    import requests

    try:
        url = f"{TE_BASE_URL}/country/{country}/{indicator}"
        resp = requests.get(url, params={"c": api_key}, timeout=10)
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
    except Exception as e:
        logger.warning(f"Trading Economics API error for {indicator}: {e}")

    return None


def get_indicator_history(
    indicator: str,
    country: str = "united states",
    start_date: str | None = None,
    periods: int = 24,
) -> list[dict[str, Any]]:
    """Get historical values for a Trading Economics indicator."""
    api_key = _get_api_key()
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
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Trading Economics history error for {indicator}: {e}")
        return []


# ── Convenience functions for specific indicators ─────────────────────────

def get_ism_manufacturing() -> float | None:
    """ISM Manufacturing PMI (latest value)."""
    snap = get_indicator_snapshot("ISM Manufacturing PMI")
    return float(snap["value"]) if snap and snap.get("value") is not None else None


def get_michigan_sentiment() -> float | None:
    """University of Michigan Consumer Sentiment."""
    snap = get_indicator_snapshot("consumer confidence")
    return float(snap["value"]) if snap and snap.get("value") is not None else None


def get_michigan_expectations() -> float | None:
    """University of Michigan Consumer Expectations sub-index."""
    snap = get_indicator_snapshot("consumer expectations")
    return float(snap["value"]) if snap and snap.get("value") is not None else None


def get_retail_sales_control() -> float | None:
    """Retail Sales Control Group MoM %."""
    snap = get_indicator_snapshot("retail sales ex autos")
    return float(snap["value"]) if snap and snap.get("value") is not None else None


def get_personal_spending() -> float | None:
    """Personal Spending MoM %."""
    snap = get_indicator_snapshot("personal spending")
    return float(snap["value"]) if snap and snap.get("value") is not None else None


def get_aaii_bullish_sentiment() -> float | None:
    """AAII Bullish Sentiment % (for positioning regime)."""
    snap = get_indicator_snapshot("AAII Bullish Sentiment")
    return float(snap["value"]) if snap and snap.get("value") is not None else None

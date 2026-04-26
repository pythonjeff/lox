"""
USDA WASDE / PSD data client.

Fetches commodity supply-demand balances (production, ending stocks,
total use) from the USDA Foreign Agricultural Service PSD Online API.

The PSD API is public but requires a free API key from:
    https://apps.fas.usda.gov/opendata/register

Set USDA_FAS_API_KEY in .env to enable. Without a key the agriculture
dashboard still works — WASDE fields simply show as unavailable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

PSD_API_BASE = "https://apps.fas.usda.gov/opendata/api/psd"

COMMODITY_CODES: dict[str, str] = {
    "corn": "0440000",
    "wheat": "0410000",
    "soybeans": "2222000",
}

US_COUNTRY_CODE = "US"

# WASDE attribute descriptions we parse from PSD records
_ATTR_MAP: dict[str, str] = {
    "ending stocks": "ending_stocks",
    "total use": "total_use",
    "total domestic cons.": "total_use",
    "domestic consumption": "domestic_consumption",
    "production": "production",
    "exports": "exports",
    "beginning stocks": "beginning_stocks",
}


@dataclass(frozen=True)
class WASDEBalance:
    commodity: str
    market_year: int
    ending_stocks: float | None = None
    total_use: float | None = None
    production: float | None = None
    exports: float | None = None
    stocks_to_use_pct: float | None = None
    unit: str = ""


def _current_market_year(commodity: str) -> int:
    """Determine the current USDA marketing year.

    Marketing years start in different months depending on the commodity:
    - Corn & soybeans: September
    - Wheat: June
    """
    now = datetime.now(timezone.utc)
    year = now.year
    month = now.month

    if commodity == "wheat":
        return year if month >= 6 else year - 1
    # Corn and soybeans: Sep start
    return year if month >= 9 else year - 1


def _parse_psd_records(records: list[dict[str, Any]]) -> dict[str, float]:
    """Extract named attributes from PSD commodity records."""
    result: dict[str, float] = {}
    for rec in records:
        desc = str(rec.get("attributeDescription") or rec.get("attribute_description") or "").strip().lower()
        val = rec.get("value")
        mapped = _ATTR_MAP.get(desc)
        if mapped and val is not None:
            try:
                fval = float(val)
                if mapped not in result:
                    result[mapped] = fval
            except (ValueError, TypeError):
                continue
    return result


def fetch_wasde_balance(
    *,
    api_key: str,
    commodity: str,
    market_year: int | None = None,
) -> WASDEBalance | None:
    """Fetch a single commodity's WASDE supply-demand balance from USDA PSD."""
    code = COMMODITY_CODES.get(commodity.lower())
    if not code:
        return None

    if market_year is None:
        market_year = _current_market_year(commodity)

    from lox.altdata.cache import cache_path, read_cache, write_cache
    import requests

    cache_key = f"usda_wasde_{commodity}_{market_year}"
    p = cache_path(cache_key)
    cached = read_cache(p, max_age=timedelta(hours=24))
    if isinstance(cached, dict) and "commodity" in cached:
        return WASDEBalance(**cached)

    headers = {"API_KEY": api_key, "Accept": "application/json"}
    params = {
        "commodityCode": code,
        "countryCode": US_COUNTRY_CODE,
        "marketYear": str(market_year),
    }

    try:
        resp = requests.get(
            f"{PSD_API_BASE}/commodityData",
            headers=headers,
            params=params,
            timeout=30,
        )
        # If the API 500s on the current MY, fall back to prior year
        if resp.status_code >= 500:
            fallback_year = market_year - 1
            logger.info("USDA PSD 500 for %s MY%s, falling back to MY%s", commodity, market_year, fallback_year)
            params["marketYear"] = str(fallback_year)
            market_year = fallback_year
            resp = requests.get(
                f"{PSD_API_BASE}/commodityData",
                headers=headers,
                params=params,
                timeout=30,
            )
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list) or not data:
            return None

        # Use the latest month's estimates
        months = {}
        for rec in data:
            m = rec.get("month") or rec.get("calendarYear")
            if m is not None:
                months.setdefault(int(m), []).append(rec)

        latest_month = max(months.keys()) if months else None
        records = months.get(latest_month, data) if latest_month else data

        attrs = _parse_psd_records(records)
        ending = attrs.get("ending_stocks")
        total_use = attrs.get("total_use")
        if total_use is None:
            dom = attrs.get("domestic_consumption", 0)
            exp = attrs.get("exports", 0)
            if dom or exp:
                total_use = dom + exp

        stu = None
        if ending is not None and total_use and total_use > 0:
            stu = (ending / total_use) * 100.0

        unit = ""
        for rec in records[:1]:
            unit = str(rec.get("unitDescription") or rec.get("unit_description") or "")

        bal = WASDEBalance(
            commodity=commodity,
            market_year=market_year,
            ending_stocks=ending,
            total_use=total_use,
            production=attrs.get("production"),
            exports=attrs.get("exports"),
            stocks_to_use_pct=stu,
            unit=unit,
        )

        write_cache(p, {
            "commodity": bal.commodity,
            "market_year": bal.market_year,
            "ending_stocks": bal.ending_stocks,
            "total_use": bal.total_use,
            "production": bal.production,
            "exports": bal.exports,
            "stocks_to_use_pct": bal.stocks_to_use_pct,
            "unit": bal.unit,
        })
        return bal

    except Exception as e:
        logger.warning("USDA PSD fetch failed for %s MY%s: %s", commodity, market_year, e)
        return None


def fetch_all_wasde(
    api_key: str,
    commodities: list[str] | None = None,
) -> dict[str, WASDEBalance]:
    """Fetch WASDE balances for all tracked agriculture commodities."""
    if not api_key:
        return {}

    if commodities is None:
        commodities = list(COMMODITY_CODES.keys())

    results: dict[str, WASDEBalance] = {}
    for comm in commodities:
        bal = fetch_wasde_balance(api_key=api_key, commodity=comm)
        if bal is not None:
            results[comm] = bal
    return results


# ── Historical S/U averages for context ─────────────────────────────────
# 5-year averages (approx 2019-2023) for stocks-to-use ratio interpretation
HISTORICAL_STU_AVG: dict[str, float] = {
    "corn": 13.8,
    "wheat": 33.5,
    "soybeans": 8.2,
}


def stu_context(commodity: str, stu_pct: float | None) -> str:
    if stu_pct is None:
        return "no data"
    avg = HISTORICAL_STU_AVG.get(commodity.lower(), 15.0)
    ratio = stu_pct / avg if avg > 0 else 1.0
    if ratio < 0.6:
        return f"very tight — well below 5yr avg {avg:.1f}%"
    if ratio < 0.85:
        return f"tight — below 5yr avg {avg:.1f}%"
    if ratio < 1.15:
        return "adequate"
    if ratio < 1.4:
        return f"comfortable — above 5yr avg {avg:.1f}%"
    return f"burdensome — well above 5yr avg {avg:.1f}%"

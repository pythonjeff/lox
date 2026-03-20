"""
Assemble the scan universe: S&P 500 + Dow 30 + macro ETF basket, deduplicated.

Usage:
    from lox.universe.sp500 import build_scan_universe

    tickers = build_scan_universe(settings)  # ~550 unique tickers
"""
from __future__ import annotations

import csv
import logging
from datetime import timedelta
from pathlib import Path

from lox.altdata.cache import cache_path, read_cache, write_cache
from lox.config import Settings

logger = logging.getLogger(__name__)

_CSV_FALLBACK = Path("data/cache/universe/sp500_constituents.csv")


def _parse_symbols_from_fmp(data: list) -> list[str]:
    """Parse FMP constituent JSON into a clean ticker list.

    Handles mixed-case keys, uppercases symbols, skips blanks/non-dicts.
    """
    out: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol") or item.get("Symbol") or "").strip().upper()
        if sym:
            out.append(sym)
    return out


def fetch_sp500_symbols(settings: Settings) -> list[str]:
    """Fetch S&P 500 constituents from FMP, cached 7 days.

    Falls back to local CSV if API unavailable.
    """
    key = "universe_sp500_constituents"
    p = cache_path(key)
    cached = read_cache(p, max_age=timedelta(days=7))
    if isinstance(cached, list) and cached:
        return cached

    if settings.fmp_api_key:
        import requests
        try:
            url = "https://financialmodelingprep.com/api/v3/sp500_constituent"
            resp = requests.get(
                url, params={"apikey": settings.fmp_api_key}, timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                symbols = _parse_symbols_from_fmp(data)
                if symbols:
                    write_cache(p, symbols)
                    return symbols
        except Exception as e:
            logger.warning("FMP S&P 500 constituents fetch failed: %s", e)

    # Fallback: local CSV
    if _CSV_FALLBACK.exists():
        try:
            with open(_CSV_FALLBACK) as f:
                reader = csv.DictReader(f)
                symbols = [
                    row.get("Symbol", "").strip().upper()
                    for row in reader
                    if row.get("Symbol", "").strip()
                ]
                if symbols:
                    return symbols
        except Exception as e:
            logger.warning("CSV fallback read failed: %s", e)

    return []


def fetch_dow30_symbols(settings: Settings) -> list[str]:
    """Fetch Dow Jones 30 constituents from FMP, cached 7 days."""
    key = "universe_dow30_constituents"
    p = cache_path(key)
    cached = read_cache(p, max_age=timedelta(days=7))
    if isinstance(cached, list) and cached:
        return cached

    if not settings.fmp_api_key:
        return []

    import requests
    try:
        url = "https://financialmodelingprep.com/api/v3/dowjones_constituent"
        resp = requests.get(
            url, params={"apikey": settings.fmp_api_key}, timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            symbols = _parse_symbols_from_fmp(data)
            if symbols:
                write_cache(p, symbols)
                return symbols
    except Exception as e:
        logger.warning("FMP Dow 30 constituents fetch failed: %s", e)

    return []


def build_scan_universe(settings: Settings) -> list[str]:
    """Build the full scan universe: macro ETFs + S&P 500 + Dow 30, deduplicated.

    ETFs come first (they have factor mappings), then equities.
    """
    from lox.suggest.cross_asset import CANDIDATE_UNIVERSE

    # Start with macro ETFs (they have rich factor mappings)
    seen: set[str] = set()
    ordered: list[str] = []

    for t in CANDIDATE_UNIVERSE:
        if t not in seen:
            seen.add(t)
            ordered.append(t)

    # Add S&P 500
    for t in fetch_sp500_symbols(settings):
        if t not in seen:
            seen.add(t)
            ordered.append(t)

    # Add Dow 30 (mostly already in S&P, but catches edge cases)
    for t in fetch_dow30_symbols(settings):
        if t not in seen:
            seen.add(t)
            ordered.append(t)

    return ordered

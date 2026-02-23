"""Ticker data fetching — FMP API calls for price, fundamentals, IV, peers."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def fetch_price_data(settings, symbol: str) -> dict:
    """Fetch historical price data."""
    try:
        import requests
        from datetime import datetime, timedelta

        if not settings.fmp_api_key:
            return {}

        # Get quote
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
        resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=15)
        quote = {}
        if resp.ok:
            data = resp.json()
            if data and isinstance(data, list):
                quote = data[0]

        # Get historical
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
        resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=15)
        historical = []
        if resp.ok:
            data = resp.json()
            historical = data.get("historical", [])[:756]  # 3 years

        return {
            "symbol": symbol,
            "quote": quote,
            "historical": historical,
        }
    except Exception:
        logger.debug("Failed to fetch price data for %s", symbol, exc_info=True)
        return {}


def fetch_fundamentals(settings, symbol: str) -> dict:
    """Fetch fundamental data (auto-detects ETFs vs stocks)."""
    try:
        import requests

        if not settings.fmp_api_key:
            return {}

        result = {}

        # Company profile (works for both stocks and ETFs)
        url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
        resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=15)
        if resp.ok:
            data = resp.json()
            if data and isinstance(data, list):
                result["profile"] = data[0]

        # Check if ETF — fetch ETF-specific data instead of stock ratios
        is_etf = result.get("profile", {}).get("isEtf", False)

        if is_etf:
            # ETF info (AUM, expense ratio, holdings count, etc.)
            url = "https://financialmodelingprep.com/api/v4/etf-info"
            resp = requests.get(url, params={"symbol": symbol, "apikey": settings.fmp_api_key}, timeout=15)
            if resp.ok:
                data = resp.json()
                if data and isinstance(data, list):
                    result["etf_info"] = data[0]
        else:
            # Stock: key metrics and ratios
            url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}"
            resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=15)
            if resp.ok:
                data = resp.json()
                if data and isinstance(data, list):
                    result["metrics"] = data[0]

            url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}"
            resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=15)
            if resp.ok:
                data = resp.json()
                if data and isinstance(data, list):
                    result["ratios"] = data[0]

            # Income statement (revenue, gross profit for margin and growth)
            url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}"
            resp = requests.get(url, params={"apikey": settings.fmp_api_key, "limit": 3}, timeout=15)
            if resp.ok:
                data = resp.json()
                if data and isinstance(data, list):
                    result["income_statement"] = data

            # Cash flow (FCF)
            url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}"
            resp = requests.get(url, params={"apikey": settings.fmp_api_key, "limit": 2}, timeout=15)
            if resp.ok:
                data = resp.json()
                if data and isinstance(data, list):
                    result["cash_flow"] = data

            # Balance sheet (cash, debt)
            url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}"
            resp = requests.get(url, params={"apikey": settings.fmp_api_key, "limit": 1}, timeout=15)
            if resp.ok:
                data = resp.json()
                if data and isinstance(data, list):
                    result["balance_sheet"] = data[0]

            # Revenue growth (YoY)
            url = f"https://financialmodelingprep.com/api/v3/income-statement-growth/{symbol}"
            resp = requests.get(url, params={"apikey": settings.fmp_api_key, "limit": 2}, timeout=15)
            if resp.ok:
                data = resp.json()
                if data and isinstance(data, list):
                    result["income_growth"] = data

        return result
    except Exception:
        logger.debug("Failed to fetch fundamentals for %s", symbol, exc_info=True)
        return {}


def fetch_atm_implied_vol(settings, symbol: str, current_price: float | None) -> float | None:
    """Fetch ATM implied vol from options chain (Polygon) if available. Returns annualized IV as decimal or None."""
    if not current_price or current_price <= 0:
        return None
    try:
        from datetime import date, timedelta
        from lox.data.polygon import fetch_options_chain_polygon
        expiry_lte = (date.today() + timedelta(days=60)).isoformat()
        expiry_gte = (date.today() + timedelta(days=20)).isoformat()
        chain = fetch_options_chain_polygon(
            settings, symbol, expiration_date_gte=expiry_gte, expiration_date_lte=expiry_lte, limit=100
        )
        if not chain:
            return None
        # Near ATM: strike within ~5% of spot
        ivs = []
        for c in chain:
            if c.iv is None:
                continue
            strike = c.strike
            if strike and abs(strike - current_price) / current_price <= 0.05:
                ivs.append(c.iv)
        if not ivs:
            return None
        return float(sum(ivs) / len(ivs))
    except Exception:
        logger.debug("Failed to fetch ATM IV for %s", symbol, exc_info=True)
        return None


def fetch_peers(settings, symbol: str) -> list[str]:
    """Fetch peer symbols from FMP (same sector, similar cap). Returns up to 5 symbols."""
    try:
        import requests
        url = "https://financialmodelingprep.com/api/v4/stock_peers"
        resp = requests.get(url, params={"symbol": symbol, "apikey": settings.fmp_api_key}, timeout=10)
        if not resp.ok:
            return []
        data = resp.json()
        if not data or not isinstance(data, list):
            return []
        peers = data[0].get("peersList", []) if isinstance(data[0], dict) else []
        return [p for p in peers if p and p != symbol][:5]
    except Exception:
        logger.debug("Failed to fetch peers for %s", symbol, exc_info=True)
        return []

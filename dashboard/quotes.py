"""
Quote fetching layer for the LOX FUND Dashboard.

FMP commodity prices, Alpaca stock/option quotes (single + batch).
"""

import os
import time

import threading

# Local cache for indicator source values (small, no need for central registry)
INDICATOR_SOURCE_CACHE: dict = {}
INDICATOR_SOURCE_CACHE_LOCK = threading.Lock()
INDICATOR_SOURCE_TTL = 1800  # 30 minutes


def fetch_fmp_commodity(symbol: str) -> float | None:
    """Fetch a commodity quote from FMP. Returns price or None."""
    try:
        fmp_key = os.environ.get("FMP_API_KEY")
        if not fmp_key:
            return None
        import requests as _req
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={fmp_key}"
        r = _req.get(url, timeout=10)
        data = r.json()
        if data and isinstance(data, list) and len(data) > 0:
            return float(data[0].get("price", 0))
    except Exception as e:
        print(f"[Indicators] FMP commodity fetch error ({symbol}): {e}")
    return None


def get_live_source_value(source_key: str) -> str | None:
    """
    Resolve a source key to a live display value, with caching.

    Supported sources:
        silver_spot    — LBMA silver via FMP XAGUSD
        silver_futures — Silver futures via FMP SIUSD
        gold_spot      — Gold spot via FMP XAUUSD
    """
    now = time.time()
    with INDICATOR_SOURCE_CACHE_LOCK:
        cached = INDICATOR_SOURCE_CACHE.get(source_key)
        if cached and (now - cached["ts"]) < INDICATOR_SOURCE_TTL:
            return cached["value"]

    value = None
    if source_key == "silver_spot":
        price = fetch_fmp_commodity("XAGUSD")
        if price:
            value = f"${price:.2f}"
    elif source_key == "silver_futures":
        price = fetch_fmp_commodity("SIUSD")
        if price:
            value = f"${price:.2f}"
    elif source_key == "gold_spot":
        price = fetch_fmp_commodity("XAUUSD")
        if price:
            value = f"${price:.2f}"
    else:
        return None

    if value:
        with INDICATOR_SOURCE_CACHE_LOCK:
            INDICATOR_SOURCE_CACHE[source_key] = {"value": value, "ts": now}
    return value


def fetch_option_quote(data_client, symbol: str) -> dict | None:
    """Fetch bid/ask quote for an option. Returns {'bid': float, 'ask': float} or None."""
    try:
        from alpaca.data.requests import OptionLatestQuoteRequest
        req = OptionLatestQuoteRequest(symbol_or_symbols=[symbol])
        quotes = data_client.get_option_latest_quote(req)
        if quotes and symbol in quotes:
            q = quotes[symbol]
            return {
                "bid": float(getattr(q, 'bid_price', 0) or 0),
                "ask": float(getattr(q, 'ask_price', 0) or 0),
            }
    except Exception as e:
        print(f"[Quote] Option quote fetch error for {symbol}: {e}")
    return None


def fetch_stock_quote(data_client, symbol: str) -> dict | None:
    """Fetch bid/ask quote for a stock/ETF. Returns {'bid': float, 'ask': float} or None."""
    try:
        from alpaca.data.requests import StockLatestQuoteRequest
        req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        quotes = data_client.get_stock_latest_quote(req)
        if quotes and symbol in quotes:
            q = quotes[symbol]
            return {
                "bid": float(getattr(q, 'bid_price', 0) or 0),
                "ask": float(getattr(q, 'ask_price', 0) or 0),
            }
    except Exception as e:
        print(f"[Quote] Stock quote fetch error for {symbol}: {e}")
    return None


def fetch_batch_option_quotes(data_client, symbols: list[str]) -> dict[str, dict]:
    """Batch fetch bid/ask quotes for multiple options in a single API call."""
    if not symbols:
        return {}
    result = {}
    try:
        from alpaca.data.requests import OptionLatestQuoteRequest
        req = OptionLatestQuoteRequest(symbol_or_symbols=symbols)
        quotes = data_client.get_option_latest_quote(req)
        if quotes:
            for symbol, q in quotes.items():
                result[symbol] = {
                    "bid": float(getattr(q, 'bid_price', 0) or 0),
                    "ask": float(getattr(q, 'ask_price', 0) or 0),
                }
    except Exception as e:
        print(f"[Quote] Batch option quote fetch error: {e}")
    return result


def fetch_batch_stock_quotes(data_client, symbols: list[str], settings=None) -> dict[str, dict]:
    """
    Batch fetch bid/ask quotes for multiple stocks/ETFs in a single API call.
    Note: data_client is OptionHistoricalDataClient, so we create a StockHistoricalDataClient here.
    """
    if not symbols:
        return {}
    result = {}
    try:
        from alpaca.data.requests import StockLatestQuoteRequest
        from alpaca.data.historical import StockHistoricalDataClient

        if settings:
            api_key = getattr(settings, 'alpaca_data_key', None) or getattr(settings, 'alpaca_api_key', None)
            secret_key = getattr(settings, 'alpaca_data_secret', None) or getattr(settings, 'alpaca_api_secret', None)
        else:
            api_key = os.environ.get("ALPACA_API_KEY")
            secret_key = os.environ.get("ALPACA_API_SECRET")

        stock_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        req = StockLatestQuoteRequest(symbol_or_symbols=symbols)
        quotes = stock_client.get_stock_latest_quote(req)
        if quotes:
            for symbol, q in quotes.items():
                result[symbol] = {
                    "bid": float(getattr(q, 'bid_price', 0) or 0),
                    "ask": float(getattr(q, 'ask_price', 0) or 0),
                }
    except Exception as e:
        print(f"[Quote] Batch stock quote fetch error: {e}")
    return result

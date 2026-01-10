from __future__ import annotations

import csv
import os
from pathlib import Path

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from ai_options_trader.config import Settings


def fetch_equity_daily_closes_alpaca(
    api_key: str,
    api_secret: str,
    symbols: list[str],
    start: str,
) -> pd.DataFrame:
    """
    Returns a dataframe indexed by date with columns = symbols, values = close.
    """
    client = StockHistoricalDataClient(api_key, api_secret)
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=pd.Timestamp(start, tz="UTC"),
    )
    bars = client.get_stock_bars(req).df  # multiindex: (symbol, timestamp)

    if bars is None or len(bars) == 0:
        raise RuntimeError("No stock bars returned from Alpaca.")

    # bars index is typically MultiIndex: (symbol, timestamp)
    bars = bars.reset_index()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"]).dt.tz_convert(None)
    bars["date"] = bars["timestamp"].dt.date
    pivot = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    pivot.index = pd.to_datetime(pivot.index)
    return pivot


def _fmp_prices_cache_path(*, symbol: str, start: str) -> Path:
    safe = f"{symbol.strip().upper()}_{str(start).strip()}".replace("/", "_").replace(":", "_")
    return Path("data/cache/fmp_prices") / f"{safe}.csv"


def fetch_equity_daily_closes_fmp(
    *,
    settings: Settings,
    symbols: list[str],
    start: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch daily close history from Financial Modeling Prep (FMP), with local CSV caching.

    Endpoint used (per symbol):
      /api/v3/historical-price-full/{SYMBOL}?from=YYYY-MM-DD&apikey=...

    Returns:
      DataFrame indexed by date (datetime64), columns=symbols, values=close.
    """
    if not settings.fmp_api_key:
        raise RuntimeError("Missing FMP_API_KEY (required when AOT_PRICE_SOURCE=fmp).")

    out: dict[str, pd.Series] = {}
    syms = [s.strip().upper() for s in (symbols or []) if s and s.strip()]
    if not syms:
        return pd.DataFrame()

    base_url = "https://financialmodelingprep.com/api/v3/historical-price-full"

    for sym in syms:
        cache_path = _fmp_prices_cache_path(symbol=sym, start=start)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        rows: list[dict] | None = None
        if cache_path.exists() and not refresh:
            try:
                dfc = pd.read_csv(cache_path)
                if "date" in dfc.columns and "close" in dfc.columns:
                    s = pd.to_numeric(dfc["close"], errors="coerce")
                    idx = pd.to_datetime(dfc["date"], errors="coerce")
                    ser = pd.Series(s.values, index=idx).dropna().sort_index()
                    out[sym] = ser
                    continue
            except Exception:
                rows = None

        import requests

        url = f"{base_url}/{sym}"
        resp = requests.get(url, params={"from": str(start), "apikey": settings.fmp_api_key}, timeout=45)
        resp.raise_for_status()
        js = resp.json()
        hist = js.get("historical") if isinstance(js, dict) else None
        if not isinstance(hist, list):
            raise RuntimeError(f"FMP returned no historical data for {sym}.")

        # Normalize to {date, close}
        norm = []
        for r in hist:
            if not isinstance(r, dict):
                continue
            d = r.get("date")
            c = r.get("close")
            if d is None or c is None:
                continue
            norm.append({"date": str(d)[:10], "close": c})

        # Cache (oldest->newest)
        try:
            norm_sorted = sorted(norm, key=lambda x: x.get("date") or "")
            with open(cache_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["date", "close"])
                w.writeheader()
                for r in norm_sorted:
                    w.writerow({"date": r["date"], "close": r["close"]})
        except Exception:
            pass

        df = pd.DataFrame(norm)
        if df.empty:
            raise RuntimeError(f"FMP returned empty historical list for {sym}.")
        idx = pd.to_datetime(df["date"], errors="coerce")
        ser = pd.Series(pd.to_numeric(df["close"], errors="coerce").values, index=idx).dropna().sort_index()
        out[sym] = ser

    px = pd.DataFrame(out).sort_index()
    px.index = pd.to_datetime(px.index)
    return px


def fetch_equity_daily_closes(
    *,
    settings: Settings,
    symbols: list[str],
    start: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Unified historical daily close fetcher.

    - If settings.price_source == "fmp": fetch from FMP (cached)
    - If settings.price_source == "alpaca": fetch from Alpaca market data
    """
    src = (settings.price_source or "fmp").strip().lower()
    if src == "alpaca":
        api_key = settings.alpaca_data_key or settings.alpaca_api_key
        api_secret = settings.alpaca_data_secret or settings.alpaca_api_secret
        return fetch_equity_daily_closes_alpaca(api_key=api_key, api_secret=api_secret, symbols=symbols, start=start)
    return fetch_equity_daily_closes_fmp(settings=settings, symbols=symbols, start=start, refresh=refresh)


def fetch_crypto_daily_closes(
    api_key: str,
    api_secret: str,
    symbols: list[str],
    start: str,
) -> pd.DataFrame:
    """
    Returns a dataframe indexed by date with columns = symbols, values = close.

    Symbols should be Alpaca crypto pair symbols like "BTC/USD", "ETH/USD".
    """
    client = CryptoHistoricalDataClient(api_key, api_secret)
    req = CryptoBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=pd.Timestamp(start, tz="UTC"),
    )
    bars = client.get_crypto_bars(req).df  # multiindex: (symbol, timestamp)

    if bars is None or len(bars) == 0:
        raise RuntimeError("No crypto bars returned from Alpaca.")

    bars = bars.reset_index()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"]).dt.tz_convert(None)
    bars["date"] = bars["timestamp"].dt.date
    pivot = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    pivot.index = pd.to_datetime(pivot.index)
    return pivot



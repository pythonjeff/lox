from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable


@dataclass(frozen=True)
class HistoricalVolumeResult:
    # Aggregated contract volume over the requested window.
    volume_by_symbol: dict[str, int]
    # Useful diagnostics (best-effort).
    chunks: int
    symbols_requested: int
    symbols_returned: int
    start: datetime
    end: datetime


def _chunks(items: list[str], n: int) -> Iterable[list[str]]:
    n = max(1, int(n))
    for i in range(0, len(items), n):
        yield items[i : i + n]


def fetch_option_bar_volumes(
    data_client: Any,
    *,
    option_symbols: list[str],
    start: datetime,
    end: datetime,
    feed: str | None = None,
    chunk_size: int = 200,
) -> HistoricalVolumeResult:
    """
    Fetch option bar data and aggregate volume by option contract symbol.

    This is used to rank "most traded" contracts even when option chain snapshots omit volume/OI.

    Notes:
    - Alpaca historical options data availability depends on your subscription.
    - Request/response schemas in alpaca-py have changed; this function is defensive.
    """
    # Lazy imports so tests and constrained envs can import the module without alpaca/pandas installed.
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame

    try:
        import pandas as pd  # type: ignore
    except Exception:  # pragma: no cover
        pd = None  # type: ignore

    vol: dict[str, int] = {}
    symbols_returned = 0
    chunks = 0

    for batch in _chunks(option_symbols, chunk_size):
        chunks += 1
        req_kwargs = {
            "symbol_or_symbols": batch,
            "timeframe": TimeFrame.Day,
            "start": start,
            "end": end,
        }
        # Feed is optional and may not be supported depending on SDK version.
        if feed:
            req_kwargs["feed"] = feed

        try:
            req = OptionBarsRequest(**req_kwargs)  # type: ignore[arg-type]
        except TypeError:
            # Older SDK that doesn't accept feed
            req_kwargs.pop("feed", None)
            req = OptionBarsRequest(**req_kwargs)  # type: ignore[arg-type]

        bars = data_client.get_option_bars(req)
        df = getattr(bars, "df", None)
        if df is None:
            continue

        # Most common: MultiIndex [symbol, timestamp] with a 'volume' column.
        if pd is not None and isinstance(df, pd.DataFrame) and "volume" in df.columns:
            if isinstance(df.index, pd.MultiIndex) and "symbol" in (df.index.names or []):
                s = df["volume"].groupby(level="symbol").sum()
                for sym, v in s.items():
                    if v is None:
                        continue
                    vol[str(sym)] = int(v)
                symbols_returned += len(s)
                continue

            # Alternative: symbol is a normal column
            if "symbol" in df.columns:
                g = df.groupby("symbol")["volume"].sum()
                for sym, v in g.items():
                    if v is None:
                        continue
                    vol[str(sym)] = int(v)
                symbols_returned += len(g)
                continue

            # Fallback: assume single symbol
            if len(batch) == 1:
                vol[batch[0]] = int(df["volume"].sum())
                symbols_returned += 1

    return HistoricalVolumeResult(
        volume_by_symbol=vol,
        chunks=chunks,
        symbols_requested=len(option_symbols),
        symbols_returned=symbols_returned,
        start=start,
        end=end,
    )



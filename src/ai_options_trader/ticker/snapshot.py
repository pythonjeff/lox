from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ai_options_trader.config import Settings
from ai_options_trader.data.market import fetch_equity_daily_closes
from ai_options_trader.ticker.models import TickerSnapshot


def _pct_return(series: pd.Series, periods: int) -> Optional[float]:
    series = series.dropna()
    if series.empty:
        return None
    if len(series) <= periods:
        return None
    a = float(series.iloc[-1])
    b = float(series.iloc[-1 - periods])
    if b == 0:
        return None
    return (a / b - 1.0) * 100.0


def _ann_vol_pct(rets: pd.Series, window: int) -> Optional[float]:
    r = rets.dropna()
    if len(r) < window:
        return None
    s = float(r.iloc[-window:].std())
    if not math.isfinite(s):
        return None
    return s * math.sqrt(252.0) * 100.0


def _max_drawdown_pct(px: pd.Series, window: int) -> Optional[float]:
    s = px.dropna()
    if len(s) < window:
        return None
    w = s.iloc[-window:]
    running_max = w.cummax()
    dd = (w / running_max - 1.0)
    return float(dd.min() * 100.0)


def build_ticker_snapshot(
    *,
    settings: Settings,
    ticker: str,
    benchmark: str = "SPY",
    start: str = "2011-01-01",
) -> TickerSnapshot:
    """
    Build a simple quantitative snapshot for a single ticker.

    Uses Alpaca daily closes.
    """
    sym = ticker.strip().upper()
    bench = benchmark.strip().upper() if benchmark else "SPY"

    api_key = settings.ALPACA_DATA_KEY or settings.ALPACA_API_KEY
    api_secret = settings.ALPACA_DATA_SECRET or settings.ALPACA_API_SECRET

    px = fetch_equity_daily_closes(api_key=api_key, api_secret=api_secret, symbols=[sym, bench], start=start)
    px = px.sort_index().ffill().dropna(how="all")

    if sym not in px.columns:
        raise RuntimeError(f"No price history returned for {sym}.")
    if bench not in px.columns:
        raise RuntimeError(f"No price history returned for benchmark {bench}.")

    s = px[sym].dropna()
    b = px[bench].dropna()
    asof = str(s.index[-1].date())

    rets = s.pct_change()
    brets = b.pct_change()

    # Trading-day approximations
    d_1m, d_3m, d_6m, d_12m = 21, 63, 126, 252

    ret_1m = _pct_return(s, d_1m)
    ret_3m = _pct_return(s, d_3m)
    ret_6m = _pct_return(s, d_6m)
    ret_12m = _pct_return(s, d_12m)

    b_ret_3m = _pct_return(b, d_3m)
    b_ret_12m = _pct_return(b, d_12m)

    rel_3m = (ret_3m - b_ret_3m) if (ret_3m is not None and b_ret_3m is not None) else None
    rel_12m = (ret_12m - b_ret_12m) if (ret_12m is not None and b_ret_12m is not None) else None

    out = TickerSnapshot(
        ticker=sym,
        asof=asof,
        last_close=float(s.iloc[-1]) if not s.empty else None,
        ret_1m_pct=ret_1m,
        ret_3m_pct=ret_3m,
        ret_6m_pct=ret_6m,
        ret_12m_pct=ret_12m,
        vol_20d_ann_pct=_ann_vol_pct(rets, 20),
        vol_60d_ann_pct=_ann_vol_pct(rets, 60),
        max_drawdown_12m_pct=_max_drawdown_pct(s, d_12m),
        benchmark=bench,
        rel_ret_3m_pct=rel_3m,
        rel_ret_12m_pct=rel_12m,
        components={
            "benchmark_ret_3m_pct": b_ret_3m,
            "benchmark_ret_12m_pct": b_ret_12m,
            "benchmark_vol_60d_ann_pct": _ann_vol_pct(brets, 60),
        },
    )
    return out



"""
Snapshot builders for quantitative analysis of different asset types.

Consolidated from:
- crypto/snapshot.py (crypto pairs)
- ticker/snapshot.py (equity tickers)
"""
from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd

from lox.config import Settings
from lox.data.market import fetch_equity_daily_closes


# ============================================================================
# Shared utilities
# ============================================================================

def _pct_return(series: pd.Series, periods: int) -> Optional[float]:
    """Calculate percentage return over a given number of periods."""
    series = series.dropna()
    if series.empty or len(series) <= periods:
        return None
    a = float(series.iloc[-1])
    b = float(series.iloc[-1 - periods])
    if b == 0:
        return None
    return (a / b - 1.0) * 100.0


def _ann_vol_pct(rets: pd.Series, window: int) -> Optional[float]:
    """Calculate annualized volatility percentage."""
    r = rets.dropna()
    if len(r) < window:
        return None
    s = float(r.iloc[-window:].std())
    if not math.isfinite(s):
        return None
    return s * math.sqrt(252.0) * 100.0


def _max_drawdown_pct(px: pd.Series, window: int) -> Optional[float]:
    """Calculate maximum drawdown percentage over a window."""
    s = px.dropna()
    if len(s) < window:
        return None
    w = s.iloc[-window:]
    running_max = w.cummax()
    dd = (w / running_max - 1.0)
    return float(dd.min() * 100.0)


# ============================================================================
# Crypto snapshot
# ============================================================================

def build_crypto_quant_snapshot(
    *,
    prices: pd.DataFrame,
    symbol: str,
    benchmark: str | None = None,
) -> dict[str, Any]:
    """
    Build a simple quantitative snapshot for a crypto pair from daily closes.

    Expects `prices` indexed by date with columns including `symbol` and optionally `benchmark`.
    Symbols should match Alpaca pair symbols, e.g. "BTC/USD".
    """
    s = symbol.upper()
    b = benchmark.upper() if benchmark else None

    if s not in prices.columns:
        raise ValueError(f"Missing {s} in price dataframe columns: {list(prices.columns)}")
    cols = [s] + ([b] if b else [])

    px = prices[cols].dropna(how="any").sort_index()
    if len(px) < 260:
        raise ValueError("Not enough history for a 12m snapshot (need ~260 trading days).")

    ret = px.pct_change().dropna()

    def trailing_return(col: str, days: int) -> float | None:
        if len(px) <= days:
            return None
        start = float(px[col].iloc[-days - 1])
        end = float(px[col].iloc[-1])
        if start <= 0:
            return None
        return (end / start - 1.0) * 100.0

    # Approx trading days
    r_3m = trailing_return(s, 63)
    r_6m = trailing_return(s, 126)
    r_12m = trailing_return(s, 252)

    br_3m = trailing_return(b, 63) if b else None
    br_6m = trailing_return(b, 126) if b else None
    br_12m = trailing_return(b, 252) if b else None

    # Trend + moving averages
    ma_50 = px[s].rolling(50).mean()
    ma_200 = px[s].rolling(200).mean()
    above_200 = bool(px[s].iloc[-1] > ma_200.iloc[-1]) if pd.notna(ma_200.iloc[-1]) else None
    cross_50_200 = (
        bool(ma_50.iloc[-1] > ma_200.iloc[-1]) if pd.notna(ma_50.iloc[-1]) and pd.notna(ma_200.iloc[-1]) else None
    )

    # Realized vol (annualized)
    vol_20 = ret[s].rolling(20).std(ddof=0).iloc[-1] * (252**0.5)
    vol_60 = ret[s].rolling(60).std(ddof=0).iloc[-1] * (252**0.5)

    # Relative strength vs benchmark (12m excess return)
    rel_12m = None
    if r_12m is not None and br_12m is not None:
        rel_12m = r_12m - br_12m

    # Simple "asset regime" label
    regime = "neutral"
    if above_200 is True and (r_6m is not None and r_6m > 0):
        regime = "bullish"
    elif above_200 is False and (r_6m is not None and r_6m < 0):
        regime = "bearish"

    return {
        "asof": str(px.index[-1].date()),
        "symbol": s,
        "benchmark": b,
        "regime": regime,
        "returns": {
            "asset_3m": r_3m,
            "asset_6m": r_6m,
            "asset_12m": r_12m,
            "benchmark_3m": br_3m,
            "benchmark_6m": br_6m,
            "benchmark_12m": br_12m,
            "excess_12m": rel_12m,
        },
        "trend": {
            "above_200dma": above_200,
            "ma50_gt_ma200": cross_50_200,
        },
        "volatility": {
            "realized_vol_20d_ann": float(vol_20) if pd.notna(vol_20) else None,
            "realized_vol_60d_ann": float(vol_60) if pd.notna(vol_60) else None,
        },
    }


# ============================================================================
# Equity ticker snapshot
# ============================================================================

def build_ticker_snapshot(
    *,
    settings: Settings,
    ticker: str,
    benchmark: str = "SPY",
    start: str = "2011-01-01",
):
    """
    Build a simple quantitative snapshot for a single ticker.

    Uses the configured historical price source (default: FMP) for daily closes.
    
    Returns a TickerSnapshot dataclass.
    """
    from lox.ticker.models import TickerSnapshot
    
    sym = ticker.strip().upper()
    bench = benchmark.strip().upper() if benchmark else "SPY"

    px = fetch_equity_daily_closes(settings=settings, symbols=[sym, bench], start=start)
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

    return TickerSnapshot(
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

"""
Signal Pillar 2: Flow & Accumulation.

Detects whether buying or selling pressure is picking up using:
- Volume surge ratio with directional weighting (up-volume vs down-volume)
- Money flow intensity (price × volume directional bias)
- Short interest % of float (FMP)
- Multi-day volume trend (is volume building over 5 days, not just today?)

Higher score = stronger flow signal.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from lox.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class FlowSignal:
    ticker: str
    volume_surge: float  # today's volume / 20d avg
    volume_trend_5d: float  # avg volume ratio over last 5 days (sustained flow)
    money_flow_score: float  # directional volume intensity (-100 to +100)
    short_interest_pct: float | None  # % of float
    flow_direction: str  # ACCUMULATION, DISTRIBUTION, NEUTRAL
    sub_score: float  # 0-100


def _fetch_si_single(settings: Settings, ticker: str) -> tuple[str, float | None]:
    """Fetch short interest for a single ticker, with per-ticker caching."""
    from lox.altdata.cache import cache_path, read_cache, write_cache

    key = f"scanner_si_{ticker}"
    p = cache_path(key)
    cached = read_cache(p, max_age=timedelta(hours=24))
    if cached is not None:
        return (ticker, float(cached) if cached is not None else None)

    if not settings.fmp_api_key:
        return (ticker, None)

    import requests
    try:
        resp = requests.get(
            f"https://financialmodelingprep.com/api/v4/short-interest",
            params={"symbol": ticker, "apikey": settings.fmp_api_key},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            row = data[0]
            si_pct = None
            for field in ("shortInterestPercentOfFloat", "shortPercentOfFloat",
                          "shortPercentFloat"):
                val = row.get(field)
                if val is not None:
                    try:
                        si_pct = float(val)
                        break
                    except (ValueError, TypeError):
                        continue
            write_cache(p, si_pct)
            return (ticker, si_pct)
    except Exception as e:
        logger.debug("SI fetch failed for %s: %s", ticker, e)

    write_cache(p, None)
    return (ticker, None)


def _fetch_short_interest_batch(
    settings: Settings,
    tickers: list[str],
    max_workers: int = 5,
) -> dict[str, float | None]:
    """Fetch short interest for many tickers in parallel with per-ticker caching."""
    result: dict[str, float | None] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_si_single, settings, t): t
            for t in tickers
        }
        for future in as_completed(futures):
            try:
                ticker, si = future.result()
                result[ticker] = si
            except Exception:
                result[futures[future]] = None
    return result


def _compute_money_flow(price_panel: pd.DataFrame, ticker: str) -> tuple[float, float]:
    """Compute directional money flow intensity and 5-day volume trend.

    Money flow: sum(volume * sign(daily_return)) over last 10 days,
    normalized to [-100, +100]. Positive = buying pressure, negative = selling.

    Volume trend: average of (daily_volume / 20d_avg) over last 5 days.
    Values > 1.2 indicate sustained elevated volume.
    """
    if price_panel is None or price_panel.empty or ticker not in price_panel.columns:
        return 0.0, 1.0

    col = price_panel[ticker].dropna()
    if len(col) < 25:
        return 0.0, 1.0

    closes = col.values.astype(np.float64)
    returns = np.diff(closes) / closes[:-1]

    # Money flow: last 10 days
    recent_rets = returns[-10:]
    # Weight by magnitude of return (bigger moves = more conviction)
    up_flow = sum(abs(r) for r in recent_rets if r > 0)
    down_flow = sum(abs(r) for r in recent_rets if r < 0)
    total = up_flow + down_flow
    if total > 0:
        mf_score = ((up_flow - down_flow) / total) * 100.0
    else:
        mf_score = 0.0

    # 5-day volume trend (can't compute from closes alone, return 1.0)
    vol_trend = 1.0

    return mf_score, vol_trend


def score_flow(
    *,
    settings: Settings,
    tickers: list[str],
    quote_data: dict[str, dict[str, Any]],
    price_panel: pd.DataFrame | None = None,
    fetch_si: bool = True,
) -> dict[str, FlowSignal]:
    """Score tickers on flow/accumulation signals.

    Args:
        quote_data: ticker -> {volume, avgVolume, changesPercentage, ...} from batch quotes.
        price_panel: historical prices for money flow computation.
        fetch_si: whether to fetch short interest (adds API calls).
    """
    # Compute volume surge from quote data
    volume_surges: dict[str, float] = {}
    change_pcts: dict[str, float] = {}
    for ticker in tickers:
        q = quote_data.get(ticker, {})
        vol = q.get("volume") or 0
        avg_vol = q.get("avgVolume") or 0
        change = q.get("changesPercentage") or 0.0
        try:
            volume_surges[ticker] = float(vol) / float(avg_vol) if avg_vol > 0 else 1.0
        except (ValueError, TypeError, ZeroDivisionError):
            volume_surges[ticker] = 1.0
        try:
            change_pcts[ticker] = float(change)
        except (ValueError, TypeError):
            change_pcts[ticker] = 0.0

    # Fetch short interest
    si_data: dict[str, float | None] = {}
    if fetch_si:
        si_data = _fetch_short_interest_batch(settings, tickers, max_workers=5)

    # Compute money flow from price history
    money_flows: dict[str, float] = {}
    vol_trends: dict[str, float] = {}
    for ticker in tickers:
        mf, vt = _compute_money_flow(price_panel, ticker)
        money_flows[ticker] = mf
        vol_trends[ticker] = vt

    out: dict[str, FlowSignal] = {}
    for ticker in tickers:
        surge = volume_surges.get(ticker, 1.0)
        change = change_pcts.get(ticker, 0.0)
        si = si_data.get(ticker)
        mf = money_flows.get(ticker, 0.0)
        vol_trend = vol_trends.get(ticker, 1.0)

        # ── Classify flow direction ──
        # Use both volume surge AND money flow direction for robust classification
        if (surge > 2.0 and change > 1.0) or (surge > 1.5 and mf > 30):
            flow_dir = "ACCUMULATION"
        elif (surge > 2.0 and change < -1.0) or (surge > 1.5 and mf < -30):
            flow_dir = "DISTRIBUTION"
        elif mf > 50:
            flow_dir = "ACCUMULATION"
        elif mf < -50:
            flow_dir = "DISTRIBUTION"
        else:
            flow_dir = "NEUTRAL"

        # ── Volume component ──
        # Require real conviction: 2x = 20pts, 3x = 40pts, 5x+ = 55pts
        # Below 1.8x gets minimal credit (it's just noise)
        if surge >= 5.0:
            vol_component = 55.0
        elif surge >= 3.0:
            vol_component = 30.0 + (surge - 3.0) * 5.0
        elif surge >= 2.0:
            vol_component = 15.0 + (surge - 2.0) * 15.0
        elif surge >= 1.8:
            vol_component = 8.0
        else:
            vol_component = 0.0

        # ── Money flow component ──
        # Strong directional conviction: |mf| > 60 = 20pts, > 40 = 12pts
        mf_component = 0.0
        abs_mf = abs(mf)
        if abs_mf > 60:
            mf_component = 20.0
        elif abs_mf > 40:
            mf_component = 12.0
        elif abs_mf > 25:
            mf_component = 5.0

        # ── Short interest component ──
        si_component = 0.0
        if si is not None:
            if si > 25:
                si_component = 25.0  # extreme squeeze potential
            elif si > 15:
                si_component = 18.0
            elif si > 10:
                si_component = 12.0
            elif si > 5:
                si_component = 5.0

        sub_score = max(0.0, min(100.0, vol_component + mf_component + si_component))

        out[ticker] = FlowSignal(
            ticker=ticker,
            volume_surge=round(surge, 2),
            volume_trend_5d=round(vol_trend, 2),
            money_flow_score=round(mf, 1),
            short_interest_pct=round(si, 1) if si is not None else None,
            flow_direction=flow_dir,
            sub_score=round(sub_score, 1),
        )

    return out

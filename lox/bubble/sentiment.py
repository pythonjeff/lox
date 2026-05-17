"""
Sentiment pillar — AAII bull-bear + VIX-vs-realized vol gap.

Two cheap, complementary reads:

  1. AAII bullish %.  Retail-investor sentiment survey.  Extreme bullishness
     (>50%) is a contrarian late-cycle indicator; extreme bearishness (<25%)
     fades the bubble case.  Sourced via Trading Economics if a key is wired.

  2. VIX vs realized vol.  When implied vol (VIX) is well below realized
     SPY vol the market is *under-pricing* risk — classic complacency.
     The healthier setup is VIX ≥ realized.  We compute the gap as
     VIX − 20d realized vol of SPY.

Both inputs default to None and the regime classifier ignores any pillar
that lacks data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from lox.config import Settings
from lox.data.fred import FredClient
from lox.data.market import fetch_equity_daily_closes


@dataclass
class SentimentSnapshot:
    asof: str
    aaii_bull_pct: float | None
    vix: float | None
    spy_realized_vol_20d: float | None
    vix_minus_realized: float | None       # pp; negative = complacent
    complacency_flag: bool                 # VIX < realized vol


def _realized_vol_20d_pct(s: pd.Series) -> float | None:
    if s is None or len(s) < 25:
        return None
    rets = np.log(s / s.shift(1)).dropna().iloc[-20:]
    if rets.empty:
        return None
    return float(rets.std() * np.sqrt(252) * 100.0)


def fetch_sentiment_snapshot(*, settings: Settings, refresh: bool = False) -> SentimentSnapshot:
    # ── AAII (best-effort) ──────────────────────────────────────────────
    aaii: float | None = None
    try:
        from lox.altdata.trading_economics import get_aaii_bullish_sentiment
        aaii = get_aaii_bullish_sentiment()
    except Exception:
        pass

    # ── VIX from FRED ───────────────────────────────────────────────────
    vix: float | None = None
    try:
        fred = FredClient(api_key=settings.FRED_API_KEY)
        df = fred.fetch_series("VIXCLS", start_date="2018-01-01", refresh=refresh)
        if not df.empty:
            vix = float(df.iloc[-1]["value"])
    except Exception:
        pass

    # ── Realized vol from SPY ───────────────────────────────────────────
    rv: float | None = None
    asof = ""
    try:
        start = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
        px = fetch_equity_daily_closes(settings=settings, symbols=["SPY"],
                                       start=start, refresh=refresh)
        if "SPY" in px.columns:
            spy = px["SPY"].dropna()
            rv = _realized_vol_20d_pct(spy)
            if not spy.empty:
                asof = str(spy.index[-1].date())
    except Exception:
        pass

    gap = (vix - rv) if (vix is not None and rv is not None) else None
    complacent = bool(gap is not None and gap < 0)

    return SentimentSnapshot(
        asof=asof,
        aaii_bull_pct=aaii,
        vix=vix,
        spy_realized_vol_20d=rv,
        vix_minus_realized=gap,
        complacency_flag=complacent,
    )

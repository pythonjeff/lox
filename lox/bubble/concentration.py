"""
Concentration pillar — how narrow is the bull run?

Three reads:

  1. Top-10 share of SPY: the canonical "X% of the index is 10 stocks"
     number.  Pulled directly from FMP's SPY ETF holdings.

  2. Cap-weight vs equal-weight: SPY minus RSP YTD and 1Y.  Confirms
     whether mega-cap dominance is also showing up in returns.

  3. AI-leadership callout: we already have a full AI basket signal; if
     the basket is smoking SPY *and* breadth is narrow, surface
     "AI concentration" as a tag for the regime classifier.

Historical context for the top-10 share: 2014-2019 averaged ~18%, 2020
hit ~26%, and 2023-2024 pushed past 35%.  >30% is elevated, >35% extreme.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from lox.altdata.etf import fetch_etf_holdings
from lox.config import Settings
from lox.data.market import fetch_equity_daily_closes


@dataclass
class ConcentrationSnapshot:
    asof: str
    spy_ytd: float | None
    rsp_ytd: float | None
    spy_minus_rsp_ytd: float | None       # pp; positive = cap-weight crowded
    spy_minus_rsp_1y: float | None        # pp
    spy_minus_rsp_3m: float | None        # pp; short-term crowding momentum
    # Top-10 share of SPY (from FMP holdings)
    top10_share_pct: float | None = None
    top10_names: list[str] = field(default_factory=list)
    # AI leadership tap-in (best-effort, may be None on failure)
    ai_basket_ytd_excess: float | None = None
    ai_breadth_200d: float | None = None  # % of AI basket above 200dma
    ai_leadership_flag: bool = False      # AI basket >+20pp YTD vs SPY


def _ytd_return(s: pd.Series) -> float | None:
    if s is None or s.empty:
        return None
    year_start = pd.Timestamp(year=s.index[-1].year, month=1, day=1)
    earlier = s.loc[s.index >= year_start]
    if earlier.empty:
        return None
    return (s.iloc[-1] / earlier.iloc[0] - 1.0) * 100.0


def _pct_change_days(s: pd.Series, days: int) -> float | None:
    if s is None or s.empty:
        return None
    target = s.index[-1] - pd.Timedelta(days=days)
    earlier = s.loc[s.index <= target]
    if earlier.empty:
        return None
    return (s.iloc[-1] / earlier.iloc[-1] - 1.0) * 100.0


def fetch_concentration_snapshot(*, settings: Settings, refresh: bool = False) -> ConcentrationSnapshot:
    start = (datetime.utcnow() - timedelta(days=520)).strftime("%Y-%m-%d")

    try:
        px = fetch_equity_daily_closes(settings=settings, symbols=["SPY", "RSP"],
                                       start=start, refresh=refresh)
    except Exception:
        px = pd.DataFrame()

    spy = px["SPY"].dropna() if "SPY" in px.columns else pd.Series(dtype=float)
    rsp = px["RSP"].dropna() if "RSP" in px.columns else pd.Series(dtype=float)

    spy_ytd = _ytd_return(spy)
    rsp_ytd = _ytd_return(rsp)
    spy_1y = _pct_change_days(spy, 365)
    rsp_1y = _pct_change_days(rsp, 365)
    spy_3m = _pct_change_days(spy, 90)
    rsp_3m = _pct_change_days(rsp, 90)

    def _spread(a, b):
        return (a - b) if (a is not None and b is not None) else None

    asof = str(spy.index[-1].date()) if not spy.empty else ""

    # ── Top-10 share of SPY via FMP ETF holdings ────────────────────────
    top10_share: float | None = None
    top10_names: list[str] = []
    try:
        is_etf, holdings = fetch_etf_holdings(settings, "SPY", top_n=10)
        if is_etf and holdings:
            weights = [float(h.get("weightPercentage") or 0.0) for h in holdings]
            top10_share = float(sum(weights)) if weights else None
            top10_names = [str(h.get("asset") or h.get("name") or "")
                           for h in holdings if (h.get("asset") or h.get("name"))]
    except Exception:
        pass

    # AI leadership tap-in (don't fail the pillar if AI module errors)
    ai_excess: float | None = None
    ai_breadth: float | None = None
    try:
        from lox.ai.signals import compute_ai_signals
        ai = compute_ai_signals(settings=settings, refresh=refresh)
        ai_excess = ai.basket_ytd_excess
        ai_breadth = ai.pct_above_200dma
    except Exception:
        pass

    ai_flag = bool(ai_excess is not None and ai_excess > 20)

    return ConcentrationSnapshot(
        asof=asof,
        spy_ytd=spy_ytd,
        rsp_ytd=rsp_ytd,
        spy_minus_rsp_ytd=_spread(spy_ytd, rsp_ytd),
        spy_minus_rsp_1y=_spread(spy_1y, rsp_1y),
        spy_minus_rsp_3m=_spread(spy_3m, rsp_3m),
        top10_share_pct=top10_share,
        top10_names=top10_names,
        ai_basket_ytd_excess=ai_excess,
        ai_breadth_200d=ai_breadth,
        ai_leadership_flag=ai_flag,
    )

"""
"Don't overpay for vol" checker for earnings/catalyst setups.

lox has no historical single-name options/IV data anywhere (only live
Alpaca/Polygon chain snapshots — see lox/backtest/vol_factors.py's module
docstring for the same gap). That means we can't backtest whether "IV was
cheap" predicted a profitable option trade historically.

What we CAN do: compare a live option-implied move against the stock's own
historical realized earnings-day moves. If the market is pricing in a move
much bigger than history supports, that's a signal you're overpaying for
vol; if it's pricing in less, the option may be cheap relative to what
usually happens. This sidesteps the missing-historical-IV gap entirely by
using realized price history instead — but it's a live decision aid, not a
backtested edge. Treat the CHEAP/RICH label as a prior to weigh, not proof.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import pandas as pd
from pandas.tseries.offsets import BDay

from lox.config import Settings


@dataclass
class EarningsVolCheck:
    ticker: str
    earnings_date: date
    n_history: int
    median_historical_move_pct: float
    avg_historical_move_pct: float
    raw_straddle_move_pct: float | None    # unadjusted, whatever the nearest expiry actually prices
    implied_move_pct: float | None         # event-only estimate, adjusted for days-past-event
    expiry_used: date | None
    days_past_event: int | None            # expiry minus earnings date — 0 means a same-week expiry existed
    richness_ratio: float | None   # implied / median_historical — >1.2 rich, <0.85 cheap
    label: str                     # CHEAP | FAIR | RICH | NO DATA


def historical_earnings_moves(
    *, settings: Settings, ticker: str, closes: pd.Series, limit: int = 12,
) -> list[float]:
    """Absolute close-to-close % move spanning each past earnings date
    (T-1 business day to T+1 business day), from FMP earnings history.

    Spans a 2-day window instead of picking bmo/amc precisely because the
    "time" field on FMP's earnings-surprises endpoint isn't reliably
    populated — this slightly overstates the pure reaction (includes a
    little normal drift either side) but is robust to that ambiguity.
    """
    from lox.altdata.fmp import fetch_earnings_history

    rows = fetch_earnings_history(settings=settings, ticker=ticker, limit=limit)
    moves: list[float] = []
    for r in rows:
        d = r.get("date")
        if not d:
            continue
        try:
            edate = pd.Timestamp(str(d)[:10])
        except Exception:
            continue
        pre = closes.asof(edate - BDay(1))
        post = closes.asof(edate + BDay(1))
        if pre is None or post is None or pd.isna(pre) or pd.isna(post) or pre <= 0:
            continue
        moves.append(abs(float(post) / float(pre) - 1.0) * 100.0)
    return moves


def implied_move_pct(
    candidates,
    spot: float,
    target_expiry: date,
    *,
    asof: date | None = None,
    daily_vol_annualized: float | None = None,
) -> tuple[float | None, date | None, float | None]:
    """ATM straddle implied move for the expiry closest to (and on/after)
    target_expiry, since an option must expire after the event to price it in.

    Many names don't have weeklies, so the nearest available expiry is often
    days to weeks past the actual earnings date — that straddle prices in
    the WHOLE period's movement, not just the event, which overstates the
    event-specific move if compared directly against a 1-2 day historical
    earnings move. If daily_vol_annualized (trailing realized vol) is given,
    back out the "ordinary" variance for the extra days using it, isolating
    an estimate of the event-specific move. This degrades gracefully to the
    raw straddle move when the expiry is already close to the event.

    Returns (raw_straddle_move_pct, expiry_used, earnings_only_move_pct).
    earnings_only_move_pct falls back to the raw figure if no vol estimate
    or days-to-expiry is given.
    """
    if not candidates or not spot or spot <= 0:
        return None, None, None

    by_expiry: dict[date, list] = defaultdict(list)
    for c in candidates:
        by_expiry[c.expiry].append(c)
    if not by_expiry:
        return None, None, None

    expiries = sorted(by_expiry.keys())
    covering = [e for e in expiries if e >= target_expiry]
    chosen = covering[0] if covering else expiries[-1]

    chain = by_expiry[chosen]
    calls = [c for c in chain if c.opt_type == "call"]
    puts = [c for c in chain if c.opt_type == "put"]
    if not calls or not puts:
        return None, None, None

    atm_call = min(calls, key=lambda c: abs(c.strike - spot))
    atm_put = min(puts, key=lambda c: abs(c.strike - spot))
    call_px = atm_call.mid or atm_call.last
    put_px = atm_put.mid or atm_put.last
    if not call_px or not put_px:
        return None, None, None

    raw_move_pct = (call_px + put_px) / spot * 100.0

    earnings_move_pct = raw_move_pct
    today = asof or date.today()
    days_to_expiry = (chosen - today).days
    if daily_vol_annualized and daily_vol_annualized > 0 and days_to_expiry > 1:
        daily_var = (daily_vol_annualized / math.sqrt(252)) ** 2
        total_var = (raw_move_pct / 100.0) ** 2
        # Assume 1 day is "the event" and the rest is ordinary daily variance;
        # subtract the ordinary portion out to isolate the event's variance.
        # Keep the sign instead of clamping at 0 — a negative result means the
        # straddle doesn't even cover trailing-RV-implied ordinary variance
        # over the gap, i.e. very cheap (or the RV estimate is stale/too high),
        # and flooring it to 0 would erase that distinction.
        non_event_var = daily_var * max(0, days_to_expiry - 1)
        event_var = total_var - non_event_var
        sign = 1.0 if event_var >= 0 else -1.0
        earnings_move_pct = sign * math.sqrt(abs(event_var)) * 100.0

    return raw_move_pct, chosen, earnings_move_pct


def classify_richness(ratio: float | None) -> str:
    if ratio is None:
        return "NO DATA"
    if ratio < 0.85:
        return "CHEAP"
    if ratio > 1.2:
        return "RICH"
    return "FAIR"

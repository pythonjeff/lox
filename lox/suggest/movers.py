"""
Movement-first trade generator — `lox movers`.

`lox suggest` asks "what is interesting today" and prefilters the universe on a
single session's change and volume. That surfaces one-day news pops. This screen
asks the other question: **which names reliably deliver range**, so there is
something to trade in the first place.

Three passes, cheap to expensive:

1. Batch FMP quotes for the whole universe (1-2 requests). Apply the liquidity
   gate, then rank on a quote-derived movement proxy (52w range width, today's
   true range, volume surge) to pick the ~N most kinetic names.
2. Daily close history for those survivors only → the kinetics pillar
   (amplitude, frequency, persistence, vol expansion, trend efficiency).
3. Momentum for directional bias, then map character + direction onto a concrete
   trade structure and a ready-to-run handoff command.

Usage:
    from lox.suggest.movers import run_movers_scan
    result = run_movers_scan(settings=settings, count=20)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from lox.config import Settings
from lox.suggest.signals.kinetics import (
    DEFAULT_MOVE_THRESHOLD,
    DEFAULT_WINDOW,
    KineticsSignal,
    score_kinetics,
)

logger = logging.getLogger(__name__)

# Liquidity gate defaults — below these, options are unusable and fills are bad.
DEFAULT_MIN_PRICE = 5.0
DEFAULT_MIN_DOLLAR_VOLUME = 20_000_000.0

# How many quote-prefilter survivors get full history pulled.
DEFAULT_DEEP_POOL = 120


@dataclass
class MoverCandidate:
    """One tradeable mover with its movement profile and suggested expression."""

    ticker: str
    name: str
    price: float
    change_pct: float
    dollar_volume: float
    sector: str
    is_etf: bool

    kinetics: KineticsSignal

    # Directional read (from the momentum pillar; empty when unavailable)
    direction: str = "NEUTRAL"       # LONG, SHORT, NEUTRAL
    trend_quality: str = ""
    rsi_14: float = 0.0
    zscore_20d: float = 0.0

    # Trade construction
    structure: str = ""              # human-readable expression
    structure_short: str = ""        # compact version for the ranking table
    structure_kind: str = ""         # DIRECTIONAL, LONG_VOL, SHORT_VOL, EVENT, WATCH
    handoff: str = ""                # copy-pasteable next command
    notes: str = ""

    @property
    def score(self) -> float:
        return self.kinetics.sub_score


@dataclass
class MoversResult:
    candidates: list[MoverCandidate] = field(default_factory=list)
    universe_size: int = 0
    liquidity_survivors: int = 0
    deep_pool: int = 0
    scored: int = 0
    window: int = DEFAULT_WINDOW
    move_threshold: float = DEFAULT_MOVE_THRESHOLD
    missing_history: list[str] = field(default_factory=list)
    scan_timestamp: str = ""


# ── Pass 1: quote-level prefilter ────────────────────────────────────────────

def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default  # NaN guard


def quote_movement_proxy(quote: dict[str, Any]) -> float:
    """Cheap stand-in for realized movement, computed from a single FMP quote.

    Blends the 52-week range width (how much ground this name covers in a year),
    today's true range, and the volume surge. Used only to decide which names are
    worth pulling history for — the real measurement happens in pass 2.
    """
    price = _f(quote.get("price"))
    if price <= 0:
        return 0.0

    year_high = _f(quote.get("yearHigh"))
    year_low = _f(quote.get("yearLow"))
    range_width = (year_high - year_low) / price if year_high > year_low > 0 else 0.0

    day_high = _f(quote.get("dayHigh"))
    day_low = _f(quote.get("dayLow"))
    prev_close = _f(quote.get("previousClose"), price)
    day_range = (day_high - day_low) / prev_close if day_high > day_low > 0 and prev_close > 0 else 0.0

    avg_vol = _f(quote.get("avgVolume"))
    volume = _f(quote.get("volume"))
    vol_surge = volume / avg_vol if avg_vol > 0 else 1.0

    abs_change = abs(_f(quote.get("changesPercentage"))) / 100.0

    # Range width dominates (it is the only multi-month term available here).
    return (
        range_width * 1.0
        + day_range * 6.0
        + abs_change * 4.0
        + min(vol_surge, 4.0) * 0.08
    )


def passes_liquidity(
    quote: dict[str, Any],
    *,
    min_price: float,
    min_dollar_volume: float,
) -> bool:
    """Keep only names that can actually be traded in size with usable options."""
    price = _f(quote.get("price"))
    if price < min_price:
        return False
    avg_vol = _f(quote.get("avgVolume"))
    return price * avg_vol >= min_dollar_volume


def _dollar_volume(quote: dict[str, Any]) -> float:
    return _f(quote.get("price")) * _f(quote.get("avgVolume"))


# ── Pass 3: trade construction ───────────────────────────────────────────────

def build_structure(
    kin: KineticsSignal,
    *,
    direction: str,
    days_to_earnings: int | None = None,
) -> tuple[str, str, str]:
    """Map a movement profile onto a trade expression.

    Returns (structure, structure_kind, notes).

    The logic is deliberately blunt: what the underlying does decides the shape
    of the trade, and the direction read decides which side of it you take.
    """
    exp21 = kin.expected_move_21d_pct
    expanding = kin.vol_expansion >= 1.15
    contracting = kin.vol_expansion <= 0.92

    if kin.character == "QUIET":
        return (
            f"No trade — {kin.avg_abs_move_pct:.1f}%/day is not enough to pay for anything",
            "WATCH",
            "Cleared the prefilter on range history, but it is not moving now.",
        )

    if days_to_earnings is not None and 0 <= days_to_earnings <= 7:
        return (
            f"Earnings in {days_to_earnings}d — size for a {exp21:.0f}% move or stand aside",
            "EVENT",
            "IV is bid into the print; long premium needs a bigger move than the straddle implies.",
        )

    if kin.character == "TRENDER" and direction in ("LONG", "SHORT"):
        side = "call" if direction == "LONG" else "put"
        return (
            f"{direction} — {side} debit spread, 30-45 DTE, ~{exp21:.0f}% wide",
            "DIRECTIONAL",
            f"Trend efficiency {kin.trend_efficiency:.2f}: moves stick, so pay for direction.",
        )

    if kin.character == "GAPPER":
        return (
            f"Event-driven — {kin.max_1d_move_pct:.0f}% max 1d vs {kin.avg_abs_move_pct:.1f}% typical",
            "EVENT",
            "Range is concentrated in a few sessions; define risk, do not hold naked premium through them.",
        )

    if kin.character == "CHOPPY" and expanding:
        return (
            f"Long vol — strangle/straddle 21-45 DTE, needs >{exp21:.0f}%",
            "LONG_VOL",
            f"Vol expanding ({kin.vol_expansion:.2f}x) with no clean trend — own the range, not a side.",
        )

    if kin.character == "CHOPPY" and contracting:
        return (
            f"Short vol — iron condor outside ±{exp21:.0f}%, 21-30 DTE",
            "SHORT_VOL",
            f"Big range ({kin.avg_abs_move_pct:.1f}%/day) but vol contracting ({kin.vol_expansion:.2f}x) and no trend.",
        )

    if direction in ("LONG", "SHORT"):
        side = "call" if direction == "LONG" else "put"
        return (
            f"{direction} lean — {side} spread, 30-45 DTE, ~{exp21:.0f}% wide",
            "DIRECTIONAL",
            "Movement is there; directional read is only moderate, so keep it defined-risk.",
        )

    return (
        f"Watchlist — {kin.avg_abs_move_pct:.1f}%/day, no directional edge yet",
        "LONG_VOL",
        "Plenty of range but nothing pointing a direction; wait for a trigger.",
    )


def short_structure(kin: KineticsSignal, *, kind: str, direction: str) -> str:
    """Compact version of the structure, for the ranking table."""
    exp21 = kin.expected_move_21d_pct
    if kind == "DIRECTIONAL":
        side = "call" if direction == "LONG" else "put"
        return f"{direction} {side} spread ~{exp21:.0f}%"
    if kind == "LONG_VOL":
        return f"Long vol, needs >{exp21:.0f}%"
    if kind == "SHORT_VOL":
        return f"Short vol outside ±{exp21:.0f}%"
    if kind == "EVENT":
        return f"Event risk, {exp21:.0f}% move"
    return "Watch — no edge yet"


def _handoff(ticker: str, structure_kind: str, direction: str) -> str:
    """The next command to run for this name."""
    if structure_kind == "DIRECTIONAL":
        want = "call" if direction == "LONG" else "put"
        return f"lox scan -t {ticker} --want {want} --min-days 30 --max-days 60"
    if structure_kind in ("LONG_VOL", "EVENT"):
        return f"lox scan -t {ticker} --want call --min-days 21 --max-days 45"
    if structure_kind == "SHORT_VOL":
        return f"lox scan -t {ticker} --want put --min-days 21 --max-days 30"
    return f"lox research ticker {ticker}"


def _direction_from_momentum(mom: Any) -> str:
    """Collapse a MomentumSignal into LONG / SHORT / NEUTRAL."""
    if mom is None:
        return "NEUTRAL"
    signal = getattr(mom, "signal", "")
    trend = getattr(mom, "trend_quality", "")

    if signal in ("BREAKOUT", "OVERSOLD_BOUNCE", "TRENDING_UP"):
        return "LONG"
    if signal == "TRENDING_DOWN":
        return "SHORT"
    if signal == "EXTENDED_UP":
        # Stretched to the upside in a healthy trend is still a trend, not a short.
        return "LONG" if trend == "STRONG_UP" else "SHORT"
    if signal == "EXTENDED_DOWN":
        return "LONG" if trend in ("PULLBACK_IN_UPTREND", "STRONG_UP") else "SHORT"

    if trend in ("STRONG_UP", "PULLBACK_IN_UPTREND"):
        return "LONG"
    if trend in ("STRONG_DOWN", "BREAKDOWN"):
        return "SHORT"
    return "NEUTRAL"


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_movers_scan(
    *,
    settings: Settings,
    count: int = 20,
    window: int = DEFAULT_WINDOW,
    move_threshold: float = DEFAULT_MOVE_THRESHOLD,
    min_price: float = DEFAULT_MIN_PRICE,
    min_dollar_volume: float = DEFAULT_MIN_DOLLAR_VOLUME,
    deep_pool: int = DEFAULT_DEEP_POOL,
    universe_name: str = "scan",
    character: str = "",
    etf_only: bool = False,
    ticker: str = "",
    refresh: bool = False,
) -> MoversResult:
    """Rank the universe by how much it moves and how often.

    Args:
        count: how many candidates to return.
        window: kinetics lookback in sessions.
        move_threshold: daily move size that counts as "a move" (0.02 = 2%).
        deep_pool: how many quote-prefilter survivors get full price history.
        universe_name: 'scan' (S&P 500 + Dow + macro ETFs), 'etf' (macro basket),
            or 'core' (~30 liquid macro ETFs).
        character: filter to TRENDER / CHOPPY / GAPPER / QUIET.
        ticker: single-ticker mode — skip the universe entirely.
    """
    from lox.altdata.fmp import fetch_batch_quotes_full
    from lox.suggest.cross_asset import CANDIDATE_UNIVERSE, SECTOR_MAP, TICKER_DESC

    now = datetime.now(timezone.utc).isoformat()
    etf_set = set(CANDIDATE_UNIVERSE)

    # ── Universe ──
    if ticker:
        universe = [ticker.strip().upper()]
    elif universe_name == "core":
        from lox.suggest.reversion import CORE_UNIVERSE
        universe = list(CORE_UNIVERSE)
    elif universe_name == "etf":
        universe = list(CANDIDATE_UNIVERSE)
    else:
        from lox.universe.sp500 import build_scan_universe
        universe = build_scan_universe(settings)

    if etf_only:
        universe = [t for t in universe if t in etf_set]

    if not universe:
        return MoversResult(scan_timestamp=now, window=window, move_threshold=move_threshold)

    # ── Pass 1: batch quotes → liquidity gate → movement proxy ──
    quotes = fetch_batch_quotes_full(settings=settings, tickers=universe)
    quote_lookup: dict[str, dict[str, Any]] = {}
    for q in quotes:
        sym = str(q.get("symbol", "")).upper()
        if sym:
            quote_lookup[sym] = q

    if ticker:
        survivors = [ticker.strip().upper()] if ticker.strip().upper() in quote_lookup else []
        liquidity_survivors = len(survivors)
    else:
        liquid = [
            q for q in quotes
            if passes_liquidity(q, min_price=min_price, min_dollar_volume=min_dollar_volume)
        ]
        liquid.sort(key=quote_movement_proxy, reverse=True)
        survivors = [str(q.get("symbol", "")).upper() for q in liquid[: max(1, deep_pool)]]
        liquidity_survivors = len(liquid)

    if not survivors:
        return MoversResult(
            universe_size=len(universe),
            liquidity_survivors=liquidity_survivors,
            scan_timestamp=now,
            window=window,
            move_threshold=move_threshold,
        )

    # ── Pass 2: price history → kinetics ──
    from lox.data.market import fetch_equity_daily_closes_resilient

    # Pad the window so the 20d/60d vol pair and the trend MAs both have room.
    lookback_days = int(max(window, 60) * 1.8) + 260
    start = (pd.Timestamp.now() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    price_panel, missing = fetch_equity_daily_closes_resilient(
        settings=settings, symbols=survivors, start=start, refresh=refresh,
    )

    kinetics = score_kinetics(
        price_panel=price_panel,
        tickers=survivors,
        window=window,
        move_threshold=move_threshold,
    )
    if not kinetics:
        return MoversResult(
            universe_size=len(universe),
            liquidity_survivors=liquidity_survivors,
            deep_pool=len(survivors),
            missing_history=missing,
            scan_timestamp=now,
            window=window,
            move_threshold=move_threshold,
        )

    # ── Pass 3: direction + structure ──
    momentum_signals: dict[str, Any] = {}
    try:
        from lox.suggest.signals.momentum import score_momentum
        momentum_signals, _ = score_momentum(
            settings=settings,
            tickers=list(kinetics.keys()),
            price_panel=price_panel,
        )
    except Exception as e:
        logger.debug("Momentum scoring unavailable for movers scan: %s", e)

    earnings_days: dict[str, int | None] = {}
    try:
        from lox.suggest.signals.catalyst import score_catalyst
        catalysts = score_catalyst(
            settings=settings,
            tickers=list(kinetics.keys()),
            quote_data=quote_lookup,
            fetch_news=False,
        )
        earnings_days = {
            t: getattr(c, "days_to_earnings", None) for t, c in catalysts.items()
        }
    except Exception as e:
        logger.debug("Catalyst lookup unavailable for movers scan: %s", e)

    names: dict[str, str] = {}
    sectors: dict[str, str] = dict(SECTOR_MAP)
    for sym in kinetics:
        q = quote_lookup.get(sym, {})
        names[sym] = TICKER_DESC.get(sym) or str(q.get("name") or sym)

    candidates: list[MoverCandidate] = []
    for sym, kin in kinetics.items():
        q = quote_lookup.get(sym, {})
        mom = momentum_signals.get(sym)
        direction = _direction_from_momentum(mom)
        structure, kind, notes = build_structure(
            kin, direction=direction, days_to_earnings=earnings_days.get(sym),
        )
        candidates.append(MoverCandidate(
            ticker=sym,
            name=names.get(sym, sym),
            price=_f(q.get("price")),
            change_pct=round(_f(q.get("changesPercentage")), 2),
            dollar_volume=_dollar_volume(q),
            sector=sectors.get(sym, ""),
            is_etf=sym in etf_set,
            kinetics=kin,
            direction=direction,
            trend_quality=getattr(mom, "trend_quality", "") if mom else "",
            rsi_14=getattr(mom, "rsi_14", 0.0) if mom else 0.0,
            zscore_20d=getattr(mom, "zscore_20d", 0.0) if mom else 0.0,
            structure=structure,
            structure_short=short_structure(kin, kind=kind, direction=direction),
            structure_kind=kind,
            handoff=_handoff(sym, kind, direction),
            notes=notes,
        ))

    if character:
        want = character.strip().upper()
        candidates = [c for c in candidates if c.kinetics.character == want]

    candidates.sort(key=lambda c: c.kinetics.sub_score, reverse=True)

    return MoversResult(
        candidates=candidates[:count],
        universe_size=len(universe),
        liquidity_survivors=liquidity_survivors,
        deep_pool=len(survivors),
        scored=len(kinetics),
        window=window,
        move_threshold=move_threshold,
        missing_history=missing,
        scan_timestamp=now,
    )

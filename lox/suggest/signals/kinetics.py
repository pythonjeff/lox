"""
Signal Pillar 5: Kinetics — how much a name moves, and how often.

The other pillars answer "is something happening here *today*". This one answers
"is this a name that reliably delivers range", which is the prerequisite for any
trade that needs the underlying to actually go somewhere.

Measured over a daily-close window (default 60 sessions):

- amplitude   — mean |daily return|, the everyday size of a move
- frequency   — share of sessions clearing a move threshold (default 2%)
- persistence — does it clear that bar in *every* sub-window, or was it one gap?
- expansion   — 20d realized vol vs 60d: is the name waking up right now?
- efficiency  — |net move| / sum|moves|: trending mover vs chop

Higher score = moves more, more often, more recently.

Pure functions — no API calls. Feed it the price panel the scanner already has.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

# Default window (sessions) and the move size that counts as "a move".
DEFAULT_WINDOW = 60
DEFAULT_MOVE_THRESHOLD = 0.02

# Character classification cutoffs
_TRENDER_EFFICIENCY = 0.35
_CHOPPY_EFFICIENCY = 0.18
_QUIET_AMPLITUDE = 0.009
# One session worth this share of the window's total travel = event-driven.
_GAP_CONCENTRATION = 0.15

# Piecewise score curves: (metric value, points). Interpolated, clamped at ends.
_AMPLITUDE_CURVE = [(0.004, 0.0), (0.008, 8.0), (0.012, 18.0),
                    (0.018, 28.0), (0.025, 35.0), (0.040, 40.0)]
_FREQUENCY_CURVE = [(0.00, 0.0), (0.05, 6.0), (0.12, 14.0),
                    (0.22, 22.0), (0.35, 27.0), (0.50, 30.0)]
_EXPANSION_CURVE = [(0.70, 0.0), (0.90, 4.0), (1.00, 7.0),
                    (1.20, 11.0), (1.45, 14.0), (1.80, 15.0)]


@dataclass
class KineticsSignal:
    """Movement profile for one ticker."""

    ticker: str

    # Amplitude
    avg_abs_move_pct: float       # mean |daily return| over the window, as %
    max_1d_move_pct: float        # largest single-session absolute move, as %

    # Frequency / persistence
    move_freq_pct: float          # % of sessions with |move| >= threshold
    big_move_count: int           # raw count of those sessions
    persistence: float            # 0-1, share of sub-windows clearing half the bar

    # Volatility state
    rv_20d: float                 # annualized realized vol, 20 sessions
    rv_60d: float                 # annualized realized vol, 60 sessions
    vol_expansion: float          # rv_20d / rv_60d

    # Shape
    trend_efficiency: float       # 0-1, |net move| / sum |moves|
    net_return_pct: float         # net % move across the window

    # Forward-looking sizing helpers
    expected_daily_move_pct: float   # rv_20d / sqrt(252), as %
    expected_move_21d_pct: float     # ~1 month 1-sigma move, as %

    # Classification
    character: str                # TRENDER, GAPPER, CHOPPY, QUIET
    sample_days: int
    sub_score: float              # 0-100


def _interp_curve(value: float, curve: list[tuple[float, float]]) -> float:
    """Piecewise-linear lookup against a (metric, points) curve, clamped at both ends."""
    if not curve:
        return 0.0
    if value <= curve[0][0]:
        return curve[0][1]
    if value >= curve[-1][0]:
        return curve[-1][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if value <= x1:
            span = x1 - x0
            if span <= 0:
                return y1
            return y0 + (y1 - y0) * (value - x0) / span
    return curve[-1][1]


def _annualized_vol(returns: np.ndarray, window: int) -> float:
    """Annualized stdev of the last `window` daily returns."""
    tail = returns[-window:]
    if len(tail) < max(5, window // 4):
        return 0.0
    sd = float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0
    return sd * float(np.sqrt(TRADING_DAYS))


def _persistence(returns: np.ndarray, threshold: float, n_buckets: int = 3) -> float:
    """Share of equal sub-windows that clear half the frequency bar.

    A name that gapped once on earnings clears the 60-day frequency test but
    fails here; a name that moves every week clears both.
    """
    if len(returns) < n_buckets * 5:
        return 0.0
    buckets = np.array_split(returns, n_buckets)
    hits = 0
    for b in buckets:
        if len(b) == 0:
            continue
        freq = float(np.mean(np.abs(b) >= threshold))
        # half the "interesting" frequency bar (~6% of sessions) counts as a hit
        if freq >= 0.06:
            hits += 1
    return hits / n_buckets


def _classify_character(
    *,
    avg_abs_move: float,
    efficiency: float,
    max_move: float,
    total_travel: float,
    move_freq: float,
) -> str:
    """Label the shape of the movement so it maps to a trade structure.

    GAPPER is about *concentration*, not raw size: a genuinely volatile name has
    fat tails, so its biggest session is naturally several times its average.
    What marks an event name is that one session accounts for a large share of
    everything the stock did all quarter.
    """
    if avg_abs_move < _QUIET_AMPLITUDE:
        return "QUIET"

    concentration = max_move / total_travel if total_travel > 0 else 0.0
    if concentration >= _GAP_CONCENTRATION:
        return "GAPPER"
    if avg_abs_move > 0 and max_move >= 5.0 * avg_abs_move and move_freq < 0.15:
        return "GAPPER"

    if efficiency >= _TRENDER_EFFICIENCY:
        return "TRENDER"
    if efficiency <= _CHOPPY_EFFICIENCY:
        return "CHOPPY"
    return "TRENDER" if efficiency >= 0.26 else "CHOPPY"


def compute_kinetics(
    closes: np.ndarray,
    *,
    ticker: str = "",
    window: int = DEFAULT_WINDOW,
    move_threshold: float = DEFAULT_MOVE_THRESHOLD,
) -> KineticsSignal | None:
    """Compute the movement profile for one close series.

    Returns None when there is not enough history to say anything (<25 sessions).
    """
    closes = np.asarray(closes, dtype=np.float64)
    closes = closes[np.isfinite(closes) & (closes > 0)]
    if len(closes) < 26:
        return None

    returns_all = np.diff(closes) / closes[:-1]
    returns = returns_all[-window:]
    if len(returns) < 25:
        return None

    abs_returns = np.abs(returns)
    avg_abs_move = float(np.mean(abs_returns))
    max_move = float(np.max(abs_returns))

    big_mask = abs_returns >= move_threshold
    big_count = int(np.sum(big_mask))
    move_freq = float(np.mean(big_mask))

    rv_20 = _annualized_vol(returns_all, 20)
    rv_60 = _annualized_vol(returns_all, min(window, 60))
    expansion = rv_20 / rv_60 if rv_60 > 0 else 1.0

    total_travel = float(np.sum(abs_returns))
    net_return = float(closes[-1] / closes[-len(returns) - 1] - 1.0)
    efficiency = abs(net_return) / total_travel if total_travel > 0 else 0.0
    efficiency = min(1.0, efficiency)

    persistence = _persistence(returns, move_threshold)

    # ── Score ──
    amplitude_pts = _interp_curve(avg_abs_move, _AMPLITUDE_CURVE)
    frequency_pts = _interp_curve(move_freq, _FREQUENCY_CURVE)
    expansion_pts = _interp_curve(expansion, _EXPANSION_CURVE)
    persistence_pts = 15.0 * persistence

    sub_score = max(0.0, min(100.0,
        amplitude_pts + frequency_pts + expansion_pts + persistence_pts
    ))

    expected_daily = rv_20 / float(np.sqrt(TRADING_DAYS))
    expected_21d = rv_20 * float(np.sqrt(21.0 / TRADING_DAYS))

    character = _classify_character(
        avg_abs_move=avg_abs_move,
        efficiency=efficiency,
        max_move=max_move,
        total_travel=total_travel,
        move_freq=move_freq,
    )

    return KineticsSignal(
        ticker=ticker,
        avg_abs_move_pct=round(avg_abs_move * 100, 2),
        max_1d_move_pct=round(max_move * 100, 2),
        move_freq_pct=round(move_freq * 100, 1),
        big_move_count=big_count,
        persistence=round(persistence, 2),
        rv_20d=round(rv_20, 4),
        rv_60d=round(rv_60, 4),
        vol_expansion=round(expansion, 2),
        trend_efficiency=round(efficiency, 3),
        net_return_pct=round(net_return * 100, 2),
        expected_daily_move_pct=round(expected_daily * 100, 2),
        expected_move_21d_pct=round(expected_21d * 100, 2),
        character=character,
        sample_days=len(returns),
        sub_score=round(sub_score, 1),
    )


def score_kinetics(
    *,
    price_panel: pd.DataFrame,
    tickers: list[str],
    window: int = DEFAULT_WINDOW,
    move_threshold: float = DEFAULT_MOVE_THRESHOLD,
) -> dict[str, KineticsSignal]:
    """Compute kinetics for every ticker present in the panel.

    Tickers missing from the panel or with too little history are skipped —
    the caller decides what to do with the gaps.
    """
    if price_panel is None or price_panel.empty:
        return {}

    out: dict[str, KineticsSignal] = {}
    for ticker in tickers:
        if ticker not in price_panel.columns:
            continue
        col = price_panel[ticker].dropna()
        if col.empty:
            continue
        sig = compute_kinetics(
            col.values,
            ticker=ticker,
            window=window,
            move_threshold=move_threshold,
        )
        if sig is not None:
            out[ticker] = sig
    return out

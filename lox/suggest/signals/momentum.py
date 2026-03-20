"""
Signal Pillar 1: Momentum, Trend Quality & Relative Strength.

Scores tickers on:
- Price extremity (z-score, RSI, MA distance)
- 52-week positioning (near high = breakout, near low = value)
- Relative strength vs SPY (outperformer/underperformer)
- Trend quality (pullback in uptrend vs breakdown)

Higher score = more interesting setup for a directional trade.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lox.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class MomentumSignal:
    ticker: str
    zscore_20d: float
    rsi_14: float
    dist_200d_pct: float
    ret_5d: float
    ret_20d: float
    ret_60d: float
    pct_from_52w_high: float  # negative = below high (e.g. -0.15 = 15% off)
    pct_from_52w_low: float  # positive = above low (e.g. 0.80 = 80% above)
    rel_strength_20d: float  # excess return vs SPY over 20d
    trend_quality: str  # STRONG_UP, PULLBACK_IN_UPTREND, BREAKDOWN, STRONG_DOWN, RANGE_BOUND
    signal: str  # EXTENDED_UP, EXTENDED_DOWN, BREAKOUT, OVERSOLD_BOUNCE, TRENDING_UP, TRENDING_DOWN, NEUTRAL
    sub_score: float  # 0-100


def _compute_52w_position(closes: np.ndarray) -> tuple[float, float]:
    """Compute distance from 52-week (252-day) high and low."""
    lookback = min(252, len(closes))
    if lookback < 20:
        return 0.0, 0.0
    window = closes[-lookback:]
    high = float(np.max(window))
    low = float(np.min(window))
    current = float(closes[-1])
    pct_from_high = (current - high) / high if high > 0 else 0.0
    pct_from_low = (current - low) / low if low > 0 else 0.0
    return pct_from_high, pct_from_low


def _compute_trend_quality(closes: np.ndarray) -> str:
    """Classify trend quality using 50d and 200d MA relationship + recent action."""
    if len(closes) < 200:
        if len(closes) < 50:
            return "RANGE_BOUND"
        ma50 = float(np.mean(closes[-50:]))
        current = float(closes[-1])
        return "STRONG_UP" if current > ma50 else "STRONG_DOWN"

    current = float(closes[-1])
    ma50 = float(np.mean(closes[-50:]))
    ma200 = float(np.mean(closes[-200:]))

    if current > ma50 > ma200:
        return "STRONG_UP"
    elif current < ma50 and ma50 > ma200 and current > ma200:
        return "PULLBACK_IN_UPTREND"
    elif current < ma200 and ma50 > ma200:
        return "BREAKDOWN"
    elif current < ma50 and ma50 < ma200:
        return "STRONG_DOWN"
    else:
        return "RANGE_BOUND"


def score_momentum(
    *,
    settings: Settings,
    tickers: list[str],
    price_panel: pd.DataFrame | None = None,
    refresh: bool = False,
) -> tuple[dict[str, MomentumSignal], pd.DataFrame]:
    """Score tickers on momentum, trend quality, and relative strength.

    Returns (ticker -> MomentumSignal, price_panel for downstream reuse).
    """
    from lox.suggest.reversion import (
        _rsi, _zscore_rolling_return, _pct_return, _dist_from_ma,
    )

    if price_panel is None or price_panel.empty:
        from lox.data.market import fetch_equity_daily_closes
        start = (pd.Timestamp.now() - pd.DateOffset(days=400)).strftime("%Y-%m-%d")
        price_panel = fetch_equity_daily_closes(
            settings=settings, symbols=tickers, start=start, refresh=refresh,
        )

    if price_panel.empty:
        return {}, price_panel

    # Pre-compute SPY returns for relative strength
    spy_ret_20d = 0.0
    if "SPY" in price_panel.columns:
        spy_col = price_panel["SPY"].dropna()
        if len(spy_col) >= 60:
            spy_closes = spy_col.values.astype(np.float64)
            spy_ret_20d = _pct_return(spy_closes, 20)

    out: dict[str, MomentumSignal] = {}
    for ticker in tickers:
        if ticker not in price_panel.columns:
            continue
        col = price_panel[ticker].dropna()
        if len(col) < 60:
            continue
        closes = col.values.astype(np.float64)

        ret_5d = _pct_return(closes, 5)
        ret_20d = _pct_return(closes, 20)
        ret_60d = _pct_return(closes, 60)
        z = _zscore_rolling_return(closes, 20)
        rsi = _rsi(closes)
        dist_200d = _dist_from_ma(closes, 200)

        # 52-week positioning
        pct_from_high, pct_from_low = _compute_52w_position(closes)

        # Relative strength vs SPY
        rel_strength = ret_20d - spy_ret_20d

        # Trend quality
        trend_quality = _compute_trend_quality(closes)

        # ── Signal classification (richer) ──
        if z > 2.5 or ret_20d > 0.20:
            signal = "EXTENDED_UP"
        elif z < -2.5 or ret_20d < -0.20:
            signal = "EXTENDED_DOWN"
        elif pct_from_high > -0.03 and ret_20d > 0.05:
            signal = "BREAKOUT"
        elif pct_from_high < -0.15 and rsi < 35 and trend_quality in ("PULLBACK_IN_UPTREND", "STRONG_UP"):
            signal = "OVERSOLD_BOUNCE"
        elif z > 0.8 and rsi > 55:
            signal = "TRENDING_UP"
        elif z < -0.8 and rsi < 45:
            signal = "TRENDING_DOWN"
        else:
            signal = "NEUTRAL"

        # ── Scoring ──
        # Extremity
        extremity = min(35.0, abs(z) * 12.0)

        # RSI bonus
        rsi_bonus = 0.0
        if rsi > 80 or rsi < 20:
            rsi_bonus = 15.0
        elif rsi > 70 or rsi < 30:
            rsi_bonus = 8.0

        # 52-week positioning bonus
        w52_bonus = 0.0
        if pct_from_high > -0.03:
            w52_bonus = 12.0  # breakout territory
        elif pct_from_high < -0.25:
            w52_bonus = 10.0  # deep value territory
        elif pct_from_high < -0.15:
            w52_bonus = 5.0

        # Relative strength bonus
        rs_bonus = 0.0
        abs_rs = abs(rel_strength)
        if abs_rs > 0.08:
            rs_bonus = 15.0
        elif abs_rs > 0.04:
            rs_bonus = 8.0
        elif abs_rs > 0.02:
            rs_bonus = 3.0

        # Trend quality bonus
        trend_bonus = 0.0
        if trend_quality == "PULLBACK_IN_UPTREND" and z < -0.5:
            trend_bonus = 15.0  # buying a dip in a healthy trend
        elif trend_quality == "BREAKDOWN":
            trend_bonus = 10.0
        elif trend_quality == "STRONG_UP" and signal == "BREAKOUT":
            trend_bonus = 10.0

        sub_score = max(0.0, min(100.0,
            extremity + rsi_bonus + w52_bonus + rs_bonus + trend_bonus
        ))

        out[ticker] = MomentumSignal(
            ticker=ticker,
            zscore_20d=round(z, 2),
            rsi_14=round(rsi, 1),
            dist_200d_pct=round(dist_200d * 100, 1),
            ret_5d=round(ret_5d, 4),
            ret_20d=round(ret_20d, 4),
            ret_60d=round(ret_60d, 4),
            pct_from_52w_high=round(pct_from_high, 3),
            pct_from_52w_low=round(pct_from_low, 3),
            rel_strength_20d=round(rel_strength, 4),
            trend_quality=trend_quality,
            signal=signal,
            sub_score=round(sub_score, 1),
        )

    return out, price_panel

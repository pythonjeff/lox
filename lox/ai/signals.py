"""
AI bubble regime — signal compute layer (v0).

Pulls daily closes for the AI basket + SPY benchmark and reduces them to a
small set of regime inputs.  Price-only by design: v1 will add capex / news /
GPU debt feeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from lox.ai.capex import CapexPanel, fetch_capex_panel
from lox.ai.news_pulse import NewsPulse, fetch_ai_news_pulse
from lox.config import Settings
from lox.data.market import fetch_equity_daily_closes


# Three sub-groups so we can see divergence.  Power-play cracks usually
# show first; chips lead the trade up.
BASKET: dict[str, list[str]] = {
    "chips":       ["NVDA", "AVGO", "AMD", "TSM", "MU", "CBRS"],
    "hyperscale":  ["MSFT", "GOOGL", "META", "AMZN", "ORCL"],
    "power":       ["VST", "CEG", "GEV", "ETN"],
}
# Subset that meaningfully drives AI capex (used by capex panel)
CAPEX_SPENDERS: list[str] = ["MSFT", "GOOGL", "META", "AMZN", "ORCL"]
ALL_TICKERS: list[str] = sorted({t for group in BASKET.values() for t in group})
BENCHMARK = "SPY"


@dataclass
class AISignals:
    asof: str
    basket_ytd_excess: float | None
    basket_3m_excess: float | None
    pct_above_50dma: float | None
    pct_above_200dma: float | None
    avg_drawdown_from_52w: float | None
    chip_vs_power_spread: float | None
    vol_ratio_vs_spy: float | None
    # Per-group return tables for display
    group_returns: dict[str, dict[str, float | None]]   # {group: {"1m":..., "3m":..., "ytd":...}}
    per_name: list[dict]                                 # one row per ticker for display
    # v1 additions
    capex: CapexPanel | None = None
    news: NewsPulse | None = None


def _safe_pct_change(s: pd.Series, lookback_days: int) -> float | None:
    """Return % change over `lookback_days` calendar days, or None."""
    if s is None or s.empty:
        return None
    end_val = s.iloc[-1]
    end_date = s.index[-1]
    target = end_date - pd.Timedelta(days=lookback_days)
    # Pick the closest date <= target
    earlier = s.loc[s.index <= target]
    if earlier.empty:
        return None
    return (end_val / earlier.iloc[-1] - 1.0) * 100.0


def _ytd_return(s: pd.Series) -> float | None:
    if s is None or s.empty:
        return None
    year_start = pd.Timestamp(year=s.index[-1].year, month=1, day=1)
    earlier = s.loc[s.index >= year_start]
    if earlier.empty:
        return None
    return (s.iloc[-1] / earlier.iloc[0] - 1.0) * 100.0


def _drawdown_from_52w(s: pd.Series) -> float | None:
    if s is None or s.empty:
        return None
    cutoff = s.index[-1] - pd.Timedelta(days=365)
    window = s.loc[s.index >= cutoff]
    if window.empty:
        return None
    peak = window.max()
    return (s.iloc[-1] / peak - 1.0) * 100.0


def _above_ma(s: pd.Series, window: int) -> bool | None:
    if s is None or len(s) < window:
        return None
    ma = s.rolling(window).mean().iloc[-1]
    if not np.isfinite(ma):
        return None
    return bool(s.iloc[-1] > ma)


def _realized_vol_20d(s: pd.Series) -> float | None:
    if s is None or len(s) < 25:
        return None
    rets = np.log(s / s.shift(1)).dropna().iloc[-20:]
    if rets.empty:
        return None
    return float(rets.std() * np.sqrt(252) * 100.0)


def _fetch_closes_tolerant(*, settings: Settings, symbols: list[str], start: str, refresh: bool) -> pd.DataFrame:
    """Bulk fetch; on failure (e.g. CBRS not on FMP), drop offenders and retry per-symbol."""
    try:
        return fetch_equity_daily_closes(settings=settings, symbols=symbols, start=start, refresh=refresh)
    except Exception:
        frames: list[pd.DataFrame] = []
        for s in symbols:
            try:
                frames.append(fetch_equity_daily_closes(settings=settings, symbols=[s], start=start, refresh=refresh))
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index()


def compute_ai_signals(*, settings: Settings, refresh: bool = False) -> AISignals:
    start = (datetime.utcnow() - timedelta(days=520)).strftime("%Y-%m-%d")
    px = _fetch_closes_tolerant(
        settings=settings,
        symbols=ALL_TICKERS + [BENCHMARK],
        start=start,
        refresh=refresh,
    )
    px = px.sort_index().ffill()
    asof = str(px.index[-1].date())

    spy = px[BENCHMARK] if BENCHMARK in px.columns else None
    spy_ytd = _ytd_return(spy) if spy is not None else None
    spy_3m = _safe_pct_change(spy, 90) if spy is not None else None
    spy_vol = _realized_vol_20d(spy) if spy is not None else None

    # ── Per-name stats ─────────────────────────────────────────────────
    per_name: list[dict] = []
    group_of: dict[str, str] = {t: g for g, ts in BASKET.items() for t in ts}
    name_stats: dict[str, dict] = {}
    for t in ALL_TICKERS:
        if t not in px.columns:
            per_name.append({"ticker": t, "group": group_of[t], "ytd": None, "3m": None,
                             "dd_52w": None, "above_50": None, "above_200": None})
            continue
        s = px[t].dropna()
        ytd = _ytd_return(s)
        r3m = _safe_pct_change(s, 90)
        dd = _drawdown_from_52w(s)
        a50 = _above_ma(s, 50)
        a200 = _above_ma(s, 200)
        rv = _realized_vol_20d(s)
        name_stats[t] = {"ytd": ytd, "3m": r3m, "dd": dd, "a50": a50, "a200": a200, "rv": rv}
        per_name.append({
            "ticker": t, "group": group_of[t],
            "ytd": ytd, "3m": r3m, "dd_52w": dd,
            "above_50": a50, "above_200": a200,
        })

    # ── Group averages ─────────────────────────────────────────────────
    def _avg(vals: list[float | None]) -> float | None:
        clean = [v for v in vals if v is not None]
        return sum(clean) / len(clean) if clean else None

    group_returns: dict[str, dict[str, float | None]] = {}
    for g, tickers in BASKET.items():
        ytds = [name_stats.get(t, {}).get("ytd") for t in tickers]
        r3ms = [name_stats.get(t, {}).get("3m") for t in tickers]
        r1ms = [_safe_pct_change(px[t].dropna(), 30) if t in px.columns else None for t in tickers]
        group_returns[g] = {"1m": _avg(r1ms), "3m": _avg(r3ms), "ytd": _avg(ytds)}

    # ── Basket-level reductions ────────────────────────────────────────
    all_ytds = [name_stats.get(t, {}).get("ytd") for t in ALL_TICKERS]
    all_3ms = [name_stats.get(t, {}).get("3m") for t in ALL_TICKERS]
    basket_ytd = _avg(all_ytds)
    basket_3m = _avg(all_3ms)
    basket_ytd_excess = (basket_ytd - spy_ytd) if (basket_ytd is not None and spy_ytd is not None) else None
    basket_3m_excess = (basket_3m - spy_3m) if (basket_3m is not None and spy_3m is not None) else None

    above_50_flags = [name_stats.get(t, {}).get("a50") for t in ALL_TICKERS]
    above_200_flags = [name_stats.get(t, {}).get("a200") for t in ALL_TICKERS]
    pct_above_50 = (100.0 * sum(1 for v in above_50_flags if v is True) /
                    max(1, sum(1 for v in above_50_flags if v is not None)))
    pct_above_200 = (100.0 * sum(1 for v in above_200_flags if v is True) /
                     max(1, sum(1 for v in above_200_flags if v is not None)))

    avg_dd = _avg([name_stats.get(t, {}).get("dd") for t in ALL_TICKERS])

    chip_3m = group_returns["chips"]["3m"]
    power_3m = group_returns["power"]["3m"]
    chip_vs_power = (chip_3m - power_3m) if (chip_3m is not None and power_3m is not None) else None

    basket_rvs = [name_stats.get(t, {}).get("rv") for t in ALL_TICKERS]
    basket_rv = _avg(basket_rvs)
    vol_ratio = (basket_rv / spy_vol) if (basket_rv is not None and spy_vol not in (None, 0)) else None

    # ── v1: hyperscaler capex + news pulse (best-effort) ─────────────────
    capex_panel: CapexPanel | None
    try:
        capex_panel = fetch_capex_panel(settings=settings, symbols=CAPEX_SPENDERS, refresh=refresh)
    except Exception:
        capex_panel = None

    news_pulse: NewsPulse | None
    try:
        news_pulse = fetch_ai_news_pulse(settings=settings, symbols=ALL_TICKERS, lookback_days=5)
    except Exception:
        news_pulse = None

    return AISignals(
        asof=asof,
        basket_ytd_excess=basket_ytd_excess,
        basket_3m_excess=basket_3m_excess,
        pct_above_50dma=pct_above_50,
        pct_above_200dma=pct_above_200,
        avg_drawdown_from_52w=avg_dd,
        chip_vs_power_spread=chip_vs_power,
        vol_ratio_vs_spy=vol_ratio,
        group_returns=group_returns,
        per_name=per_name,
        capex=capex_panel,
        news=news_pulse,
    )

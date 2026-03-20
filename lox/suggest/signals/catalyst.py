"""
Signal Pillar 4: Catalyst Freshness.

Detects upcoming earnings events, recent news activity, and
recent large price moves (gap up/down as proxy for news events).

Scoring is continuous, not binary — more catalytic activity = higher score.

Higher score = more catalytic activity around this ticker.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from lox.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class CatalystSignal:
    ticker: str
    days_to_earnings: int | None
    news_count_7d: int
    recent_gap_pct: float  # largest single-day move in last 5 days
    catalyst_type: str  # EARNINGS_IMMINENT, NEWS_DRIVEN, GAP_MOVE, NONE
    sub_score: float  # 0-100


def _fetch_upcoming_earnings_broad(
    settings: Settings,
    tickers_set: set[str],
) -> dict[str, int]:
    """Fetch ALL earnings for next 30 days, then filter to our ticker set.

    This is more reliable than passing 75 tickers to the endpoint,
    which sometimes filters incorrectly.
    """
    if not settings.fmp_api_key:
        return {}

    from lox.altdata.cache import cache_path, read_cache, write_cache

    now = datetime.now(timezone.utc).date()
    to_date = now + timedelta(days=30)

    # Cache the full calendar (not per-ticker)
    key = f"scanner_earnings_calendar_{now.isoformat()}"
    p = cache_path(key)
    cached = read_cache(p, max_age=timedelta(hours=6))
    if isinstance(cached, list):
        rows = cached
    else:
        import requests
        rows = []
        for ep in ("earning_calendar", "earnings_calendar"):
            try:
                resp = requests.get(
                    f"https://financialmodelingprep.com/api/v3/{ep}",
                    params={
                        "from": now.isoformat(),
                        "to": to_date.isoformat(),
                        "apikey": settings.fmp_api_key,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list) and data:
                    rows = [r for r in data if isinstance(r, dict)]
                    write_cache(p, rows)
                    break
            except Exception as e:
                logger.debug("Earnings calendar endpoint %s failed: %s", ep, e)
                continue

    result: dict[str, int] = {}
    for row in rows:
        sym = str(row.get("symbol") or "").strip().upper()
        if sym not in tickers_set:
            continue
        date_str = str(row.get("date") or "")
        if not date_str:
            continue
        try:
            earn_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
            days = (earn_date - now).days
            if days >= 0 and (sym not in result or days < result[sym]):
                result[sym] = days
        except (ValueError, TypeError):
            continue

    return result


def _compute_recent_gaps(
    quote_data: dict[str, dict[str, Any]],
    tickers: list[str],
) -> dict[str, float]:
    """Compute largest recent gap (intraday range as proxy for news-driven moves).

    Uses FMP quote data which includes dayHigh, dayLow, previousClose.
    A large gap between previousClose and today's range suggests a catalyst.
    """
    result: dict[str, float] = {}
    for ticker in tickers:
        q = quote_data.get(ticker, {})
        try:
            prev_close = float(q.get("previousClose", 0) or 0)
            price = float(q.get("price", 0) or 0)
            if prev_close > 0 and price > 0:
                gap_pct = abs(price - prev_close) / prev_close
                result[ticker] = gap_pct
            else:
                result[ticker] = 0.0
        except (ValueError, TypeError):
            result[ticker] = 0.0
    return result


def _count_recent_news(
    settings: Settings,
    tickers: list[str],
    max_tickers: int = 30,
) -> dict[str, int]:
    """Count news articles in the last 7 days for top tickers."""
    from lox.altdata.fmp import fetch_stock_news

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=7)

    result: dict[str, int] = {}
    for ticker in tickers[:max_tickers]:
        try:
            news = fetch_stock_news(
                settings=settings, ticker=ticker, limit=15,
            )
            count = 0
            for article in news:
                pub_date = str(article.get("publishedDate") or "")
                if pub_date:
                    try:
                        pub = datetime.fromisoformat(
                            pub_date.replace("Z", "+00:00")
                        )
                        if pub >= cutoff:
                            count += 1
                    except (ValueError, TypeError):
                        count += 1
            result[ticker] = count
        except Exception:
            result[ticker] = 0

    return result


def score_catalyst(
    *,
    settings: Settings,
    tickers: list[str],
    quote_data: dict[str, dict[str, Any]] | None = None,
    fetch_news: bool = True,
    max_news_tickers: int = 30,
) -> dict[str, CatalystSignal]:
    """Score tickers on catalyst freshness.

    Combines: earnings proximity + news volume + recent gap moves.
    """
    tickers_set = set(tickers)

    # Earnings proximity (1 broad call, then filter)
    earnings_days = _fetch_upcoming_earnings_broad(settings, tickers_set)

    # News counts (per-ticker, limited)
    news_counts: dict[str, int] = {}
    if fetch_news:
        news_counts = _count_recent_news(settings, tickers, max_tickers=max_news_tickers)

    # Recent gap moves (from existing quote data, no API call)
    gaps: dict[str, float] = {}
    if quote_data:
        gaps = _compute_recent_gaps(quote_data, tickers)

    out: dict[str, CatalystSignal] = {}
    for ticker in tickers:
        days_to_earn = earnings_days.get(ticker)
        news_count = news_counts.get(ticker, 0)
        gap_pct = gaps.get(ticker, 0.0)

        # ── Earnings component (continuous, not binary) ──
        earn_component = 0.0
        if days_to_earn is not None:
            if days_to_earn <= 3:
                earn_component = 45.0
            elif days_to_earn <= 7:
                earn_component = 35.0
            elif days_to_earn <= 14:
                earn_component = 20.0
            elif days_to_earn <= 21:
                earn_component = 10.0
            elif days_to_earn <= 30:
                earn_component = 5.0

        # ── News component (continuous) ──
        news_component = 0.0
        if news_count > 10:
            news_component = 25.0
        elif news_count > 7:
            news_component = 18.0
        elif news_count > 5:
            news_component = 14.0
        elif news_count > 3:
            news_component = 8.0
        elif news_count > 1:
            news_component = 4.0
        elif news_count == 1:
            news_component = 2.0

        # ── Gap move component ──
        gap_component = 0.0
        if gap_pct > 0.05:  # 5%+ gap = major catalyst
            gap_component = 20.0
        elif gap_pct > 0.03:  # 3-5% gap
            gap_component = 12.0
        elif gap_pct > 0.02:  # 2-3% gap
            gap_component = 6.0

        sub_score = max(0.0, min(100.0, earn_component + news_component + gap_component))

        # Classify by dominant catalyst
        if days_to_earn is not None and days_to_earn <= 14:
            catalyst_type = "EARNINGS_IMMINENT"
        elif gap_pct > 0.03:
            catalyst_type = "GAP_MOVE"
        elif news_count > 5:
            catalyst_type = "NEWS_DRIVEN"
        else:
            catalyst_type = "NONE"

        out[ticker] = CatalystSignal(
            ticker=ticker,
            days_to_earnings=days_to_earn,
            news_count_7d=news_count,
            recent_gap_pct=round(gap_pct, 4),
            catalyst_type=catalyst_type,
            sub_score=round(sub_score, 1),
        )

    return out

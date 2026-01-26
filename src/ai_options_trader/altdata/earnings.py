"""
Earnings Data Module

Fetches earnings-related data:
- Earnings transcripts (via FMP)
- Earnings surprises/history
- Upcoming earnings calendar
- Key earnings metrics

Author: Lox Capital Research
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from ai_options_trader.altdata.cache import cache_path, read_cache, write_cache
from ai_options_trader.config import Settings


FMP_BASE_URL = "https://financialmodelingprep.com/api"


@dataclass(frozen=True)
class EarningsTranscript:
    """Represents an earnings call transcript."""
    ticker: str
    quarter: int  # 1-4
    year: int
    date: str
    content: str  # Full transcript text
    summary: str = ""  # Optional summary if available


@dataclass
class EarningsSurprise:
    """Historical earnings surprise data."""
    ticker: str
    date: str
    eps_actual: float | None
    eps_estimated: float | None
    eps_surprise: float | None
    eps_surprise_pct: float | None
    revenue_actual: float | None
    revenue_estimated: float | None
    revenue_surprise: float | None
    revenue_surprise_pct: float | None


@dataclass
class EarningsCalendarEvent:
    """Upcoming earnings event."""
    ticker: str
    date: str
    time: str | None  # "bmo" (before market open), "amc" (after market close)
    eps_estimated: float | None
    revenue_estimated: float | None
    fiscal_quarter: str | None  # e.g., "Q1 2024"


def fetch_earnings_transcript(
    *,
    settings: Settings,
    ticker: str,
    quarter: int,
    year: int,
    cache_max_age: timedelta = timedelta(days=30),
) -> EarningsTranscript | None:
    """
    Fetch earnings call transcript for a specific quarter.
    
    Requires FMP API key.
    
    Args:
        settings: Application settings
        ticker: Stock ticker
        quarter: Quarter number (1-4)
        year: Year (e.g., 2024)
        cache_max_age: How long to cache results
    
    Returns:
        EarningsTranscript or None if not available
    """
    if not settings.fmp_api_key:
        return None
    
    t = ticker.strip().upper()
    if not t:
        return None
    
    cache_key = f"fmp_transcript_{t}_Q{quarter}_{year}"
    p = cache_path(cache_key)
    cached = read_cache(p, max_age=cache_max_age)
    
    if isinstance(cached, dict) and cached.get("content"):
        return EarningsTranscript(
            ticker=cached["ticker"],
            quarter=cached["quarter"],
            year=cached["year"],
            date=cached.get("date", ""),
            content=cached["content"],
            summary=cached.get("summary", ""),
        )
    
    # Try to fetch from FMP
    # Endpoint: /api/v3/earning_call_transcript/{symbol}?quarter={q}&year={y}
    try:
        url = f"{FMP_BASE_URL}/v3/earning_call_transcript/{t}"
        resp = requests.get(
            url,
            params={
                "quarter": int(quarter),
                "year": int(year),
                "apikey": settings.fmp_api_key,
            },
            timeout=60,  # Transcripts can be large
        )
        resp.raise_for_status()
        data = resp.json()
        
        if isinstance(data, list) and data:
            item = data[0]
            transcript = EarningsTranscript(
                ticker=t,
                quarter=int(quarter),
                year=int(year),
                date=str(item.get("date", "")),
                content=str(item.get("content", "")),
                summary="",
            )
            
            # Cache
            write_cache(p, {
                "ticker": transcript.ticker,
                "quarter": transcript.quarter,
                "year": transcript.year,
                "date": transcript.date,
                "content": transcript.content,
                "summary": transcript.summary,
            })
            
            return transcript
            
    except Exception:
        pass
    
    return None


def fetch_recent_transcripts(
    *,
    settings: Settings,
    ticker: str,
    num_quarters: int = 4,
    cache_max_age: timedelta = timedelta(days=7),
) -> list[EarningsTranscript]:
    """
    Fetch the most recent N earnings transcripts for a ticker.
    """
    if not settings.fmp_api_key:
        return []
    
    t = ticker.strip().upper()
    if not t:
        return []
    
    cache_key = f"fmp_recent_transcripts_{t}_n{num_quarters}"
    p = cache_path(cache_key)
    cached = read_cache(p, max_age=cache_max_age)
    
    if isinstance(cached, list):
        return [
            EarningsTranscript(
                ticker=c["ticker"],
                quarter=c["quarter"],
                year=c["year"],
                date=c.get("date", ""),
                content=c["content"],
                summary=c.get("summary", ""),
            )
            for c in cached
            if c.get("content")
        ]
    
    # Try the batch endpoint
    # Endpoint: /api/v4/batch_earning_call_transcript/{symbol}
    transcripts: list[EarningsTranscript] = []
    
    try:
        url = f"{FMP_BASE_URL}/v4/batch_earning_call_transcript/{t}"
        resp = requests.get(
            url,
            params={"apikey": settings.fmp_api_key},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        
        if isinstance(data, list):
            for item in data[:num_quarters]:
                if isinstance(item, dict) and item.get("content"):
                    transcripts.append(EarningsTranscript(
                        ticker=t,
                        quarter=int(item.get("quarter", 0)),
                        year=int(item.get("year", 0)),
                        date=str(item.get("date", "")),
                        content=str(item.get("content", "")),
                        summary="",
                    ))
        
        if transcripts:
            write_cache(p, [
                {
                    "ticker": tr.ticker,
                    "quarter": tr.quarter,
                    "year": tr.year,
                    "date": tr.date,
                    "content": tr.content,
                    "summary": tr.summary,
                }
                for tr in transcripts
            ])
            
    except Exception:
        pass
    
    return transcripts


def fetch_earnings_surprises(
    *,
    settings: Settings,
    ticker: str,
    limit: int = 12,
    cache_max_age: timedelta = timedelta(hours=12),
) -> list[EarningsSurprise]:
    """
    Fetch historical earnings surprises for a ticker.
    
    Shows actual vs estimated EPS and revenue.
    """
    if not settings.fmp_api_key:
        return []
    
    t = ticker.strip().upper()
    if not t:
        return []
    
    cache_key = f"fmp_earnings_surprises_{t}_limit{limit}"
    p = cache_path(cache_key)
    cached = read_cache(p, max_age=cache_max_age)
    
    if isinstance(cached, list):
        return [
            EarningsSurprise(
                ticker=c["ticker"],
                date=c["date"],
                eps_actual=c.get("eps_actual"),
                eps_estimated=c.get("eps_estimated"),
                eps_surprise=c.get("eps_surprise"),
                eps_surprise_pct=c.get("eps_surprise_pct"),
                revenue_actual=c.get("revenue_actual"),
                revenue_estimated=c.get("revenue_estimated"),
                revenue_surprise=c.get("revenue_surprise"),
                revenue_surprise_pct=c.get("revenue_surprise_pct"),
            )
            for c in cached
        ]
    
    surprises: list[EarningsSurprise] = []
    
    # Try multiple endpoint variations
    endpoints = [
        f"{FMP_BASE_URL}/v3/earnings-surprises/{t}",
        f"{FMP_BASE_URL}/v3/earnings_surprises/{t}",
    ]
    
    for url in endpoints:
        try:
            resp = requests.get(
                url,
                params={"apikey": settings.fmp_api_key},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            
            if isinstance(data, list) and data:
                for item in data[:limit]:
                    if not isinstance(item, dict):
                        continue
                    
                    eps_actual = item.get("actualEarningResult") or item.get("actualEps")
                    eps_est = item.get("estimatedEarning") or item.get("estimatedEps")
                    
                    eps_surprise = None
                    eps_surprise_pct = None
                    if eps_actual is not None and eps_est is not None:
                        try:
                            eps_surprise = float(eps_actual) - float(eps_est)
                            if float(eps_est) != 0:
                                eps_surprise_pct = (eps_surprise / abs(float(eps_est))) * 100
                        except (ValueError, TypeError):
                            pass
                    
                    surprises.append(EarningsSurprise(
                        ticker=t,
                        date=str(item.get("date", "")),
                        eps_actual=float(eps_actual) if eps_actual is not None else None,
                        eps_estimated=float(eps_est) if eps_est is not None else None,
                        eps_surprise=eps_surprise,
                        eps_surprise_pct=eps_surprise_pct,
                        revenue_actual=None,  # Not always available
                        revenue_estimated=None,
                        revenue_surprise=None,
                        revenue_surprise_pct=None,
                    ))
                
                if surprises:
                    break
                    
        except Exception:
            continue
    
    # Cache results
    if surprises:
        write_cache(p, [
            {
                "ticker": s.ticker,
                "date": s.date,
                "eps_actual": s.eps_actual,
                "eps_estimated": s.eps_estimated,
                "eps_surprise": s.eps_surprise,
                "eps_surprise_pct": s.eps_surprise_pct,
                "revenue_actual": s.revenue_actual,
                "revenue_estimated": s.revenue_estimated,
                "revenue_surprise": s.revenue_surprise,
                "revenue_surprise_pct": s.revenue_surprise_pct,
            }
            for s in surprises
        ])
    
    return surprises


def fetch_upcoming_earnings(
    *,
    settings: Settings,
    tickers: list[str] | None = None,
    days_ahead: int = 30,
    cache_max_age: timedelta = timedelta(hours=6),
) -> list[EarningsCalendarEvent]:
    """
    Fetch upcoming earnings calendar events.
    
    If tickers is None, returns market-wide earnings calendar.
    """
    if not settings.fmp_api_key:
        return []
    
    now = datetime.now(timezone.utc).date()
    end = now + timedelta(days=days_ahead)
    
    ticker_str = ",".join(sorted([t.upper() for t in tickers])) if tickers else "all"
    cache_key = f"fmp_earnings_calendar_{ticker_str}_{now.isoformat()}_{days_ahead}"
    p = cache_path(cache_key)
    cached = read_cache(p, max_age=cache_max_age)
    
    if isinstance(cached, list):
        events = [
            EarningsCalendarEvent(
                ticker=c["ticker"],
                date=c["date"],
                time=c.get("time"),
                eps_estimated=c.get("eps_estimated"),
                revenue_estimated=c.get("revenue_estimated"),
                fiscal_quarter=c.get("fiscal_quarter"),
            )
            for c in cached
        ]
        if tickers:
            tickers_set = {t.upper() for t in tickers}
            events = [e for e in events if e.ticker in tickers_set]
        return events
    
    # Fetch from FMP
    events: list[EarningsCalendarEvent] = []
    
    try:
        url = f"{FMP_BASE_URL}/v3/earning_calendar"
        resp = requests.get(
            url,
            params={
                "from": now.isoformat(),
                "to": end.isoformat(),
                "apikey": settings.fmp_api_key,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        
        if isinstance(data, list):
            ticker_set = {t.upper() for t in tickers} if tickers else None
            
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                sym = str(item.get("symbol", "")).upper()
                if ticker_set and sym not in ticker_set:
                    continue
                
                events.append(EarningsCalendarEvent(
                    ticker=sym,
                    date=str(item.get("date", "")),
                    time=str(item.get("time", "")) or None,
                    eps_estimated=float(item["epsEstimated"]) if item.get("epsEstimated") is not None else None,
                    revenue_estimated=float(item["revenueEstimated"]) if item.get("revenueEstimated") is not None else None,
                    fiscal_quarter=None,  # Not always provided
                ))
        
        # Cache all events (filter on read)
        write_cache(p, [
            {
                "ticker": e.ticker,
                "date": e.date,
                "time": e.time,
                "eps_estimated": e.eps_estimated,
                "revenue_estimated": e.revenue_estimated,
                "fiscal_quarter": e.fiscal_quarter,
            }
            for e in events
        ])
        
    except Exception:
        pass
    
    return events


def analyze_earnings_history(surprises: list[EarningsSurprise]) -> dict[str, Any]:
    """
    Analyze earnings surprise history for patterns.
    """
    if not surprises:
        return {
            "count": 0,
            "beat_rate": None,
            "avg_surprise_pct": None,
            "streak": 0,
            "streak_type": None,
        }
    
    beats = 0
    misses = 0
    total_surprise_pct = 0.0
    valid_surprises = 0
    
    for s in surprises:
        if s.eps_surprise is not None:
            if s.eps_surprise > 0:
                beats += 1
            elif s.eps_surprise < 0:
                misses += 1
        
        if s.eps_surprise_pct is not None:
            total_surprise_pct += s.eps_surprise_pct
            valid_surprises += 1
    
    # Calculate streak (consecutive beats or misses)
    streak = 0
    streak_type = None
    for s in surprises:
        if s.eps_surprise is None:
            continue
        if s.eps_surprise > 0:
            if streak_type == "beat":
                streak += 1
            elif streak_type is None:
                streak_type = "beat"
                streak = 1
            else:
                break
        elif s.eps_surprise < 0:
            if streak_type == "miss":
                streak += 1
            elif streak_type is None:
                streak_type = "miss"
                streak = 1
            else:
                break
        else:
            break
    
    total = beats + misses
    
    return {
        "count": len(surprises),
        "beats": beats,
        "misses": misses,
        "beat_rate": (beats / total * 100) if total > 0 else None,
        "avg_surprise_pct": (total_surprise_pct / valid_surprises) if valid_surprises > 0 else None,
        "streak": streak,
        "streak_type": streak_type,
    }


def extract_transcript_insights(transcript: EarningsTranscript, max_length: int = 5000) -> dict[str, Any]:
    """
    Extract key insights from a transcript (basic version).
    
    For LLM-based analysis, use llm_analyze_transcript().
    """
    content = transcript.content[:max_length] if transcript.content else ""
    
    # Simple keyword analysis
    content_lower = content.lower()
    
    return {
        "ticker": transcript.ticker,
        "quarter": f"Q{transcript.quarter} {transcript.year}",
        "date": transcript.date,
        "length": len(transcript.content),
        "mentions": {
            "guidance": content_lower.count("guidance"),
            "outlook": content_lower.count("outlook"),
            "growth": content_lower.count("growth"),
            "margin": content_lower.count("margin"),
            "headwind": content_lower.count("headwind"),
            "tailwind": content_lower.count("tailwind"),
            "demand": content_lower.count("demand"),
            "inflation": content_lower.count("inflation"),
            "ai": content_lower.count(" ai ") + content_lower.count("artificial intelligence"),
        },
        "preview": content[:500] + "..." if len(content) > 500 else content,
    }

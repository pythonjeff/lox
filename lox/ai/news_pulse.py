"""
AI news pulse (v1, display-only).

Counts recent AI-tagged headlines and tallies bubble-language vs cracks-language
keywords.  Intentionally not fed into the regime score yet — keyword heuristics
need tuning and we want to eyeball results first.
"""
from __future__ import annotations

from dataclasses import dataclass

from lox.altdata.news import fetch_unified_news
from lox.config import Settings


# Tunable — start simple, expand as we see what trips it
BUBBLE_WORDS = [
    "bubble", "blow-off", "blowoff", "frothy", "froth", "overheated",
    "exuberance", "mania", "stretched valuation", "ai trade unwind",
    "ai winter", "circular financing", "vendor financing",
]
CRACKS_WORDS = [
    "capex cut", "guidance cut", "missed estimate", "warning",
    "layoff", "layoffs", "deferred", "delay", "delays", "downgrade",
    "writedown", "write-down", "impairment", "default",
]


@dataclass
class NewsPulse:
    total_articles: int
    bubble_hits: int
    cracks_hits: int
    top_headlines: list[dict]   # [{title, source, published_at, tags: [bubble|cracks]}]


def fetch_ai_news_pulse(
    *,
    settings: Settings,
    symbols: list[str],
    lookback_days: int = 5,
    limit: int = 40,
) -> NewsPulse:
    try:
        articles = fetch_unified_news(
            settings=settings,
            symbols=symbols,
            keywords=["AI", "GPU", "data center", "capex"],
            lookback_days=lookback_days,
            limit=limit,
            include_content=False,
        )
    except Exception:
        articles = []

    bubble_hits = 0
    cracks_hits = 0
    top: list[dict] = []
    for a in articles:
        text = f"{a.title or ''} {a.snippet or ''}".lower()
        tags: list[str] = []
        if any(w in text for w in BUBBLE_WORDS):
            bubble_hits += 1
            tags.append("bubble")
        if any(w in text for w in CRACKS_WORDS):
            cracks_hits += 1
            tags.append("cracks")
        if tags and len(top) < 8:
            top.append({
                "title": a.title or "",
                "source": a.source or "",
                "published_at": a.published_at or "",
                "tags": tags,
            })

    return NewsPulse(
        total_articles=len(articles),
        bubble_hits=bubble_hits,
        cracks_hits=cracks_hits,
        top_headlines=top,
    )

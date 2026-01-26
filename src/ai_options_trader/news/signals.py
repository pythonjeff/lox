"""
News Sentiment Regime - Data Signals

Fetches and aggregates news sentiment data from multiple sources.

Author: Lox Capital Research
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
import statistics

from ai_options_trader.config import Settings
from ai_options_trader.news.models import NewsSentimentInputs, NewsSentimentState


# Sector tickers for sentiment tracking
SECTOR_TICKERS: dict[str, list[str]] = {
    "tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN"],
    "financials": ["JPM", "BAC", "GS", "MS", "WFC", "C"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY"],
    "healthcare": ["UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY"],
    "consumer": ["WMT", "HD", "MCD", "NKE", "SBUX", "TGT"],
    "industrials": ["CAT", "BA", "UNP", "HON", "GE", "DE"],
}

# Theme keywords for detection
THEME_KEYWORDS: dict[str, list[str]] = {
    "earnings": ["earnings", "eps", "revenue", "quarterly", "beat", "miss", "guidance"],
    "fed": ["fed", "fomc", "powell", "rate cut", "rate hike", "hawkish", "dovish", "monetary policy"],
    "tariff": ["tariff", "trade war", "trade deal", "china trade", "import tax", "duties"],
    "recession": ["recession", "slowdown", "contraction", "economic decline", "hard landing"],
    "ai": ["artificial intelligence", " ai ", "chatgpt", "machine learning", "nvidia ai", "openai"],
    "layoffs": ["layoff", "job cut", "workforce reduction", "restructuring", "downsizing"],
}


def _detect_themes(text: str) -> dict[str, int]:
    """Count theme mentions in text."""
    text_lower = text.lower()
    counts: dict[str, int] = {}
    
    for theme, keywords in THEME_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        counts[theme] = count
    
    return counts


def build_news_sentiment_state(
    settings: Settings,
    lookback_days: int = 7,
    refresh: bool = False,
) -> NewsSentimentState:
    """
    Build news sentiment state by analyzing recent market news.
    
    Args:
        settings: Application settings
        lookback_days: Days of news to analyze
        refresh: Force refresh cached data
    
    Returns:
        NewsSentimentState with aggregate sentiment metrics
    """
    from ai_options_trader.llm.outlooks.ticker_news import fetch_fmp_stock_news
    from ai_options_trader.llm.core.sentiment import (
        analyze_article_sentiment,
        aggregate_sentiment,
        ArticleSentiment,
    )
    
    now = datetime.now(timezone.utc).date()
    from_date = (now - timedelta(days=lookback_days)).isoformat()
    to_date = now.isoformat()
    
    # Collect all articles and sentiment
    all_articles: list[ArticleSentiment] = []
    sector_sentiments: dict[str, list[float]] = {s: [] for s in SECTOR_TICKERS}
    theme_counts: dict[str, int] = {t: 0 for t in THEME_KEYWORDS}
    
    # Fetch news for each sector
    for sector, tickers in SECTOR_TICKERS.items():
        try:
            items = fetch_fmp_stock_news(
                settings=settings,
                tickers=tickers,
                from_date=from_date,
                to_date=to_date,
                max_pages=2,
            )
            
            for item in items[:20]:  # Limit per sector
                article = analyze_article_sentiment(
                    headline=item.title or "",
                    content=item.snippet or "",
                    source=item.source or "",
                    url=item.url or "",
                    published_at=str(item.published_at or ""),
                )
                all_articles.append(article)
                
                # Track sector sentiment
                if article.sentiment.label == "positive":
                    sector_sentiments[sector].append(article.sentiment.confidence)
                elif article.sentiment.label == "negative":
                    sector_sentiments[sector].append(-article.sentiment.confidence)
                else:
                    sector_sentiments[sector].append(0.0)
                
                # Track themes
                full_text = (item.title or "") + " " + (item.snippet or "")
                themes = _detect_themes(full_text)
                for theme, count in themes.items():
                    theme_counts[theme] += count
                    
        except Exception:
            continue
    
    # Compute aggregate sentiment
    if not all_articles:
        return NewsSentimentState(
            asof=to_date,
            lookback_days=lookback_days,
            inputs=NewsSentimentInputs(notes="No news data available"),
            notes="No news data available",
        )
    
    agg = aggregate_sentiment(all_articles)
    
    # Compute sector scores
    sector_scores: dict[str, float | None] = {}
    for sector, scores in sector_sentiments.items():
        if scores:
            sector_scores[sector] = statistics.mean(scores)
        else:
            sector_scores[sector] = None
    
    # Find best/worst sectors
    valid_sectors = [(s, sc) for s, sc in sector_scores.items() if sc is not None]
    best_sector = max(valid_sectors, key=lambda x: x[1])[0] if valid_sectors else None
    worst_sector = min(valid_sectors, key=lambda x: x[1])[0] if valid_sectors else None
    
    # Sector dispersion
    valid_scores = [sc for sc in sector_scores.values() if sc is not None]
    sector_dispersion = statistics.stdev(valid_scores) if len(valid_scores) >= 2 else None
    
    # Dominant theme
    dominant_theme = max(theme_counts, key=theme_counts.get) if any(theme_counts.values()) else None
    
    # High conviction counts
    high_pos = sum(1 for a in all_articles if a.sentiment.label == "positive" and a.sentiment.confidence >= 0.7)
    high_neg = sum(1 for a in all_articles if a.sentiment.label == "negative" and a.sentiment.confidence >= 0.7)
    
    # Risk appetite score (derived from sentiment + sector dispersion)
    risk_appetite = agg.score
    if sector_dispersion is not None and sector_dispersion > 0.3:
        risk_appetite *= 0.8  # Reduce conviction when sectors diverge
    
    # Fear/greed score (0-100)
    # Normalize sentiment score from [-1, 1] to [0, 100]
    fear_greed = (agg.score + 1) * 50
    
    inputs = NewsSentimentInputs(
        market_sentiment_score=agg.score,
        market_sentiment_confidence=agg.confidence,
        tech_sentiment_score=sector_scores.get("tech"),
        financials_sentiment_score=sector_scores.get("financials"),
        energy_sentiment_score=sector_scores.get("energy"),
        healthcare_sentiment_score=sector_scores.get("healthcare"),
        consumer_sentiment_score=sector_scores.get("consumer"),
        industrials_sentiment_score=sector_scores.get("industrials"),
        total_articles=agg.total_articles,
        positive_articles=agg.positive_count,
        negative_articles=agg.negative_count,
        neutral_articles=agg.neutral_count,
        high_conviction_positive=high_pos,
        high_conviction_negative=high_neg,
        earnings_mentions=theme_counts.get("earnings", 0),
        fed_mentions=theme_counts.get("fed", 0),
        tariff_mentions=theme_counts.get("tariff", 0),
        recession_mentions=theme_counts.get("recession", 0),
        ai_mentions=theme_counts.get("ai", 0),
        layoffs_mentions=theme_counts.get("layoffs", 0),
        dominant_theme=dominant_theme,
        sector_dispersion=sector_dispersion,
        best_sector=best_sector,
        worst_sector=worst_sector,
        risk_appetite_score=risk_appetite,
        fear_greed_score=fear_greed,
        components={
            "sector_scores": sector_scores,
            "theme_counts": theme_counts,
            "article_count_by_sector": {s: len(scores) for s, scores in sector_sentiments.items()},
        },
    )
    
    return NewsSentimentState(
        asof=to_date,
        lookback_days=lookback_days,
        inputs=inputs,
        notes=f"Analyzed {agg.total_articles} articles across {len(SECTOR_TICKERS)} sectors.",
    )

"""
Unified News Fetcher - Multi-Source

Aggregates financial news from multiple APIs you already pay for:
1. FMP (Financial Modeling Prep) - Stock news + General news
2. Alpaca Markets - News with full content

Zero additional cost since you're already subscribed to both.

Usage:
    from lox.altdata.news import fetch_unified_news, fetch_alpaca_news
    
    news = fetch_unified_news(
        settings=settings,
        symbols=["AAPL", "SPY"],
        lookback_days=5,
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from lox.config import Settings


@dataclass
class NewsArticle:
    """Unified news article from any source."""
    title: str
    source: str
    provider: str  # "FMP" or "Alpaca"
    published_at: str
    url: str | None = None
    snippet: str | None = None
    content: str | None = None  # Full content (Alpaca only)
    symbols: list[str] | None = None
    sentiment: str | None = None  # If available from API
    author: str | None = None


def fetch_alpaca_news(
    *,
    settings: Settings,
    symbols: list[str] | None = None,
    lookback_days: int = 5,
    limit: int = 50,
    include_content: bool = True,
) -> list[NewsArticle]:
    """
    Fetch news from Alpaca Markets News API.
    
    Alpaca provides:
    - News for specific symbols
    - Full article content (with include_content=True)
    - Multiple news sources aggregated
    
    Requires: ALPACA_API_KEY and ALPACA_API_SECRET
    """
    if not settings.alpaca_api_key or not settings.alpaca_api_secret:
        return []
    
    try:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest
        
        client = NewsClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_api_secret,
        )
        
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=lookback_days)
        
        # Build request
        request_params = NewsRequest(
            start=start,
            end=now,
            limit=limit,
            include_content=include_content,
            exclude_contentless=False,
        )
        
        # Add symbols if provided
        if symbols:
            clean_symbols = [s.strip().upper() for s in symbols if s.strip()]
            if clean_symbols:
                request_params.symbols = ",".join(clean_symbols)
        
        response = client.get_news(request_params)
        
        articles = []
        # Alpaca returns NewsSet with data['news'] containing the list
        news_items = []
        if hasattr(response, 'data') and isinstance(response.data, dict):
            news_items = response.data.get('news', []) or []
        
        for item in news_items:
            # Alpaca uses 'headline' not 'title'
            title = getattr(item, "headline", "") or ""
            if not title:
                continue
            
            # Parse published timestamp (Alpaca uses 'created_at')
            created_at = getattr(item, "created_at", None)
            if created_at:
                if isinstance(created_at, datetime):
                    published = created_at.isoformat()
                else:
                    published = str(created_at)
            else:
                published = ""
            
            # Get symbols from article
            article_symbols = getattr(item, "symbols", []) or []
            
            # Get content (can be very long)
            content = None
            if include_content:
                raw_content = getattr(item, "content", None)
                if raw_content:
                    # Strip HTML tags for cleaner text
                    import re
                    content = re.sub(r'<[^>]+>', '', str(raw_content))
                    content = content[:3000]  # Truncate for LLM context
            
            # Alpaca uses 'summary' for snippet
            snippet = getattr(item, "summary", None)
            if snippet:
                snippet = snippet.strip()
            
            articles.append(NewsArticle(
                title=title,
                source=getattr(item, "source", "") or "Alpaca News",
                provider="Alpaca",
                published_at=published,
                url=getattr(item, "url", None),
                snippet=snippet,
                content=content,
                symbols=article_symbols if article_symbols else None,
                author=getattr(item, "author", None),
            ))
        
        return articles
    
    except Exception as e:
        # Log but don't fail - return empty list
        print(f"[dim]Alpaca news fetch failed: {e}[/dim]")
        return []


def fetch_fmp_news(
    *,
    settings: Settings,
    symbols: list[str] | None = None,
    lookback_days: int = 5,
    limit: int = 50,
) -> list[NewsArticle]:
    """
    Fetch news from FMP (Financial Modeling Prep) API.
    
    FMP provides:
    - Stock-specific news
    - General market news
    - Multiple sources
    
    Requires: FMP_API_KEY
    """
    if not settings.fmp_api_key:
        return []
    
    articles = []
    
    try:
        from lox.altdata.fmp import fetch_stock_news
        
        # Fetch ticker-specific news if symbols provided
        if symbols:
            clean_symbols = [s.strip().upper() for s in symbols if s.strip()]
            for sym in clean_symbols[:5]:
                items = fetch_stock_news(settings=settings, ticker=sym, limit=10)
                for item in items[:limit]:
                    articles.append(NewsArticle(
                        title=item.get("title", "") or "",
                        source=item.get("site", "") or "FMP",
                        provider="FMP",
                        published_at=item.get("publishedDate", "") or "",
                        url=item.get("url", None),
                        snippet=(item.get("text", "") or "")[:300] or None,
                        content=None,
                        symbols=[item.get("symbol", "")] if item.get("symbol") else None,
                    ))
    except Exception as e:
        print(f"[dim]FMP ticker news fetch failed: {e}[/dim]")
    
    # Also fetch general macro news via FMP stock news on broad tickers
    try:
        from lox.altdata.fmp import fetch_stock_news
        
        for macro_ticker in ["SPY", "TLT"]:
            macro_items = fetch_stock_news(settings=settings, ticker=macro_ticker, limit=8)
            for item in macro_items[:limit // 2]:
                articles.append(NewsArticle(
                    title=item.get("title", "") or "",
                    source=item.get("site", "") or "FMP",
                    provider="FMP",
                    published_at=item.get("publishedDate", "") or "",
                    url=item.get("url", None),
                    snippet=(item.get("text", "") or "")[:300] or None,
                    content=None,
                    symbols=None,
                ))
    except Exception as e:
        print(f"[dim]FMP macro news fetch failed: {e}[/dim]")
    
    return articles


def fetch_unified_news(
    *,
    settings: Settings,
    symbols: list[str] | None = None,
    keywords: list[str] | None = None,
    lookback_days: int = 5,
    limit: int = 30,
    include_content: bool = True,
) -> list[NewsArticle]:
    """
    Fetch and merge news from ALL available sources (FMP + Alpaca).
    
    Deduplicates by title similarity and ranks by:
    1. Recency
    2. Relevance to keywords
    3. Source quality
    
    Args:
        settings: App settings with API keys
        symbols: Optional list of tickers to filter by
        keywords: Optional keywords to boost relevance
        lookback_days: How far back to look
        limit: Max articles to return
        include_content: Include full content from Alpaca
    
    Returns:
        List of NewsArticle sorted by relevance/recency
    """
    all_articles: list[NewsArticle] = []
    
    # Fetch from Alpaca (has full content)
    alpaca_news = fetch_alpaca_news(
        settings=settings,
        symbols=symbols,
        lookback_days=lookback_days,
        limit=limit,
        include_content=include_content,
    )
    all_articles.extend(alpaca_news)
    
    # Fetch from FMP (broader coverage)
    fmp_news = fetch_fmp_news(
        settings=settings,
        symbols=symbols,
        lookback_days=lookback_days,
        limit=limit,
    )
    all_articles.extend(fmp_news)
    
    # Deduplicate by title similarity
    seen_titles: set[str] = set()
    unique_articles: list[NewsArticle] = []
    
    for article in all_articles:
        # Normalize title for comparison
        normalized = article.title.lower().strip()[:80]
        if normalized and normalized not in seen_titles:
            seen_titles.add(normalized)
            unique_articles.append(article)
    
    # Score and sort
    def score_article(article: NewsArticle) -> float:
        score = 0.0
        
        # Recency boost (higher = more recent)
        try:
            if article.published_at:
                pub_dt = datetime.fromisoformat(article.published_at.replace("Z", "+00:00"))
                age_hours = (datetime.now(timezone.utc) - pub_dt).total_seconds() / 3600
                score += max(0, 100 - age_hours)  # Newer = higher score
        except Exception:
            pass
        
        # Content boost (Alpaca articles with content get priority)
        if article.content:
            score += 50
        
        # Keyword relevance
        if keywords:
            text = f"{article.title} {article.snippet or ''}".lower()
            keyword_hits = sum(1 for kw in keywords if kw.lower() in text)
            score += keyword_hits * 20
        
        # Symbol match boost
        if symbols and article.symbols:
            matching = set(s.upper() for s in symbols) & set(s.upper() for s in article.symbols)
            score += len(matching) * 30
        
        # Source quality boost
        quality_sources = ["reuters", "bloomberg", "wsj", "financial times", "cnbc", "marketwatch"]
        if any(qs in (article.source or "").lower() for qs in quality_sources):
            score += 25
        
        return score
    
    # Sort by score (descending)
    unique_articles.sort(key=score_article, reverse=True)
    
    return unique_articles[:limit]


def format_news_for_llm(
    articles: list[NewsArticle],
    max_articles: int = 15,
    include_snippets: bool = True,
    include_content: bool = False,
) -> list[dict[str, Any]]:
    """
    Format news articles for LLM consumption.
    
    Returns a list of dicts suitable for JSON serialization.
    """
    formatted = []
    
    for i, article in enumerate(articles[:max_articles]):
        item = {
            "index": i + 1,
            "title": article.title,
            "source": article.source,
            "provider": article.provider,
            "published_at": article.published_at,
            "url": article.url or "",
        }
        
        if include_snippets and article.snippet:
            item["snippet"] = article.snippet[:400]
        
        if include_content and article.content:
            # Truncate content for LLM context window
            item["content"] = article.content[:1500]
        
        if article.symbols:
            item["symbols"] = article.symbols
        
        if article.author:
            item["author"] = article.author
        
        formatted.append(item)
    
    return formatted


def get_news_summary_stats(articles: list[NewsArticle]) -> dict[str, Any]:
    """
    Get summary statistics about fetched news.
    """
    providers = {}
    sources = {}
    
    for article in articles:
        providers[article.provider] = providers.get(article.provider, 0) + 1
        sources[article.source] = sources.get(article.source, 0) + 1
    
    # Sort sources by count
    top_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "total_articles": len(articles),
        "by_provider": providers,
        "top_sources": dict(top_sources),
        "with_content": sum(1 for a in articles if a.content),
        "with_snippets": sum(1 for a in articles if a.snippet),
    }

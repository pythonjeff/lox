"""
Sentiment Analysis Module

Provides multiple levels of sentiment analysis:
1. Rule-based (fast, free) - expanded keyword dictionary
2. LLM-based (slower, paid) - nuanced context understanding
3. Aggregate scoring for news collections

Author: Lox Capital Research
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from openai import OpenAI


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class SentimentResult:
    """Single sentiment classification result."""
    label: str  # 'positive' | 'negative' | 'neutral'
    confidence: float  # 0..1
    magnitude: float = 0.5  # 0..1, how strong the sentiment
    reason: str = ""  # Optional explanation


@dataclass
class ArticleSentiment:
    """Sentiment analysis for a single article/headline."""
    headline: str
    sentiment: SentimentResult
    source: str = ""
    url: str = ""
    published_at: str = ""
    keywords_matched: list[str] = field(default_factory=list)


@dataclass
class AggregateSentiment:
    """Aggregate sentiment across multiple articles."""
    label: str  # Dominant sentiment
    confidence: float  # Weighted average
    magnitude: float  # Average magnitude
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    total_articles: int = 0
    score: float = 0.0  # -1 to +1 continuous score
    articles: list[ArticleSentiment] = field(default_factory=list)


# =============================================================================
# Expanded Keyword Dictionaries
# =============================================================================

# Negative keywords with strength weights (0.5 = mild, 0.8 = strong, 1.0 = extreme)
NEGATIVE_KEYWORDS: dict[str, float] = {
    # Earnings/guidance
    "miss": 0.7, "missed": 0.7, "misses": 0.7,
    "disappoints": 0.7, "disappointing": 0.7, "disappointment": 0.7,
    "lowers guidance": 0.8, "cuts guidance": 0.8, "reduces outlook": 0.8,
    "below expectations": 0.7, "below estimates": 0.7, "below consensus": 0.7,
    "warns": 0.7, "warning": 0.7, "cautions": 0.6,
    "weak": 0.6, "weakness": 0.6, "weaker": 0.6,
    "shortfall": 0.7, "fell short": 0.7,
    
    # Legal/regulatory
    "lawsuit": 0.8, "sued": 0.8, "sues": 0.8, "litigation": 0.7,
    "fraud": 0.9, "fraudulent": 0.9, "scandal": 0.8,
    "investigation": 0.7, "investigated": 0.7, "probe": 0.7,
    "indictment": 0.9, "indicted": 0.9, "charges": 0.7,
    "fine": 0.6, "fined": 0.6, "penalty": 0.6, "settlement": 0.5,
    "recall": 0.7, "recalled": 0.7, "recalls": 0.7,
    "violation": 0.7, "violated": 0.7, "breach": 0.7,
    "sec investigation": 0.8, "doj investigation": 0.9,
    
    # Analyst actions
    "downgrade": 0.8, "downgrades": 0.8, "downgraded": 0.8,
    "sell rating": 0.8, "underperform": 0.7, "underweight": 0.6,
    "price target cut": 0.7, "lowers price target": 0.7,
    "reduces estimates": 0.7, "cuts estimates": 0.7,
    
    # Business/operations
    "layoffs": 0.7, "layoff": 0.7, "job cuts": 0.7, "workforce reduction": 0.7,
    "restructuring": 0.5, "cost cutting": 0.5,
    "closing": 0.6, "closes": 0.6, "shutdown": 0.7, "shuttering": 0.7,
    "bankruptcy": 1.0, "chapter 11": 1.0, "insolvent": 1.0, "insolvency": 1.0,
    "defaults": 0.9, "default": 0.9, "defaulted": 0.9,
    "loss": 0.6, "losses": 0.6, "lost": 0.5,
    "decline": 0.6, "declines": 0.6, "declining": 0.6, "declined": 0.6,
    "drops": 0.6, "drop": 0.6, "dropped": 0.6, "plunges": 0.8, "plunge": 0.8,
    "slump": 0.7, "slumps": 0.7, "tumbles": 0.8, "tumble": 0.8,
    "crashes": 0.9, "crash": 0.9, "collapse": 0.9, "collapses": 0.9,
    "selloff": 0.7, "sell-off": 0.7,
    "delays": 0.6, "delayed": 0.6, "postpones": 0.6, "postponed": 0.6,
    "cancels": 0.7, "cancelled": 0.7, "canceled": 0.7,
    
    # Market/macro
    "bear market": 0.7, "bearish": 0.6,
    "recession": 0.7, "recessionary": 0.7,
    "inflation concerns": 0.6, "inflation fears": 0.6,
    "rate hike": 0.5, "hawkish": 0.5,
    "tariff": 0.6, "tariffs": 0.6, "trade war": 0.7,
    "sanctions": 0.7, "sanctioned": 0.7,
    
    # Leadership/governance
    "ceo resigns": 0.7, "cfo resigns": 0.7, "executive departure": 0.6,
    "accounting irregularities": 0.9, "restates earnings": 0.8, "restatement": 0.8,
    "material weakness": 0.8,
    
    # Competition/market share
    "market share loss": 0.7, "loses market share": 0.7,
    "competitive pressure": 0.5, "pricing pressure": 0.5,
}

# Positive keywords with strength weights
POSITIVE_KEYWORDS: dict[str, float] = {
    # Earnings/guidance
    "beat": 0.7, "beats": 0.7, "exceeds": 0.7, "exceeded": 0.7,
    "tops": 0.7, "topped": 0.7, "surpasses": 0.7, "surpassed": 0.7,
    "outperforms": 0.7, "outperformed": 0.7,
    "raises guidance": 0.8, "lifts guidance": 0.8, "increases outlook": 0.8,
    "above expectations": 0.7, "above estimates": 0.7, "above consensus": 0.7,
    "strong": 0.6, "stronger": 0.6, "strength": 0.6,
    "robust": 0.7, "solid": 0.6, "stellar": 0.8,
    "record": 0.7, "record revenue": 0.8, "record earnings": 0.8, "record profit": 0.8,
    "all-time high": 0.8, "new high": 0.7,
    "blowout": 0.8, "blowout quarter": 0.8,
    
    # Analyst actions
    "upgrade": 0.8, "upgrades": 0.8, "upgraded": 0.8,
    "buy rating": 0.7, "outperform rating": 0.7, "overweight": 0.6,
    "price target raise": 0.7, "raises price target": 0.7, "lifts price target": 0.7,
    "raises estimates": 0.7, "increases estimates": 0.7,
    "initiates coverage": 0.5, "starts at buy": 0.6,
    
    # Business/operations
    "expansion": 0.6, "expands": 0.6, "expanding": 0.6,
    "growth": 0.6, "grows": 0.6, "growing": 0.6,
    "launches": 0.6, "launch": 0.6, "launched": 0.6, "unveils": 0.6,
    "partnership": 0.6, "partners": 0.6, "collaboration": 0.6,
    "acquisition": 0.5, "acquires": 0.5, "acquired": 0.5,
    "merger": 0.5, "merges": 0.5,
    "deal": 0.5, "contract": 0.5, "wins contract": 0.7,
    "breakthrough": 0.8, "innovation": 0.6, "patent": 0.5,
    "approval": 0.7, "approved": 0.7, "fda approval": 0.8, "clearance": 0.7,
    "rally": 0.7, "rallies": 0.7, "surge": 0.7, "surges": 0.7, "soars": 0.8,
    "gains": 0.6, "gain": 0.6, "jumps": 0.7, "spikes": 0.7,
    "rebound": 0.6, "rebounds": 0.6, "recovery": 0.6, "recovers": 0.6,
    
    # Dividends/buybacks
    "dividend increase": 0.7, "raises dividend": 0.7, "special dividend": 0.7,
    "buyback": 0.6, "share repurchase": 0.6, "repurchase program": 0.6,
    
    # Market/macro
    "bull market": 0.6, "bullish": 0.6,
    "rate cut": 0.6, "dovish": 0.5,
    "stimulus": 0.6,
    
    # Market share/competition
    "market share gains": 0.7, "gains market share": 0.7,
    "market leader": 0.6, "leading position": 0.6,
}

# Uncertainty/volatility keywords (contribute to lower confidence, not sentiment)
UNCERTAINTY_KEYWORDS: set[str] = {
    "may", "might", "could", "uncertain", "uncertainty",
    "speculation", "speculated", "rumor", "rumors", "rumored",
    "possible", "possibly", "potential", "potentially",
    "expected", "expecting", "anticipates", "anticipated",
    "volatile", "volatility", "wild", "swings",
}


# =============================================================================
# Rule-Based Sentiment (Enhanced)
# =============================================================================

def rule_based_sentiment(text: str) -> SentimentResult:
    """
    Enhanced rule-based sentiment using weighted keyword matching.
    
    Returns:
        SentimentResult with label, confidence, magnitude, and matched keywords.
    """
    t = text.lower()
    
    positive_score = 0.0
    negative_score = 0.0
    positive_matches = []
    negative_matches = []
    
    # Check for negative keywords
    for keyword, weight in NEGATIVE_KEYWORDS.items():
        # Use word boundary matching for multi-word phrases
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, t):
            negative_score += weight
            negative_matches.append(keyword)
    
    # Check for positive keywords
    for keyword, weight in POSITIVE_KEYWORDS.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, t):
            positive_score += weight
            positive_matches.append(keyword)
    
    # Check for uncertainty keywords (reduces confidence)
    uncertainty_count = sum(1 for kw in UNCERTAINTY_KEYWORDS if kw in t)
    confidence_penalty = min(0.3, uncertainty_count * 0.05)
    
    # Determine label and scores
    net_score = positive_score - negative_score
    total_matches = positive_score + negative_score
    
    if total_matches == 0:
        return SentimentResult(
            label="neutral",
            confidence=0.5 - confidence_penalty,
            magnitude=0.0,
            reason="No sentiment keywords detected"
        )
    
    # Calculate magnitude (how strong is the sentiment)
    magnitude = min(1.0, total_matches / 3.0)  # Normalize: 3+ matches = max magnitude
    
    # Calculate confidence based on clarity of signal
    if abs(net_score) < 0.2:
        # Mixed signals
        confidence = 0.4 - confidence_penalty
        label = "neutral"
        reason = f"Mixed signals: +{positive_score:.1f}/-{negative_score:.1f}"
    elif net_score > 0:
        # Positive
        clarity = positive_score / (positive_score + negative_score) if negative_score > 0 else 1.0
        confidence = min(0.95, 0.5 + clarity * 0.4) - confidence_penalty
        label = "positive"
        reason = f"Positive: {', '.join(positive_matches[:3])}"
    else:
        # Negative
        clarity = negative_score / (positive_score + negative_score) if positive_score > 0 else 1.0
        confidence = min(0.95, 0.5 + clarity * 0.4) - confidence_penalty
        label = "negative"
        reason = f"Negative: {', '.join(negative_matches[:3])}"
    
    return SentimentResult(
        label=label,
        confidence=max(0.1, confidence),
        magnitude=magnitude,
        reason=reason
    )


def analyze_article_sentiment(
    headline: str,
    content: str = "",
    source: str = "",
    url: str = "",
    published_at: str = "",
) -> ArticleSentiment:
    """
    Analyze sentiment for a single article.
    
    Combines headline and content analysis with headline weighted higher.
    """
    # Headline sentiment (weighted higher - 70%)
    headline_sent = rule_based_sentiment(headline)
    
    # Content sentiment (30%) - only if provided
    if content and len(content) > 50:
        content_sent = rule_based_sentiment(content[:2000])  # First 2000 chars
        
        # Weighted combination
        combined_positive = 0.0
        combined_negative = 0.0
        
        if headline_sent.label == "positive":
            combined_positive += headline_sent.confidence * 0.7
        elif headline_sent.label == "negative":
            combined_negative += headline_sent.confidence * 0.7
        
        if content_sent.label == "positive":
            combined_positive += content_sent.confidence * 0.3
        elif content_sent.label == "negative":
            combined_negative += content_sent.confidence * 0.3
        
        if combined_positive > combined_negative + 0.1:
            label = "positive"
            confidence = min(0.95, (combined_positive + combined_negative) / 2)
        elif combined_negative > combined_positive + 0.1:
            label = "negative"
            confidence = min(0.95, (combined_positive + combined_negative) / 2)
        else:
            label = "neutral"
            confidence = 0.5
        
        magnitude = (headline_sent.magnitude * 0.7 + content_sent.magnitude * 0.3)
        sentiment = SentimentResult(
            label=label,
            confidence=confidence,
            magnitude=magnitude,
            reason=f"Headline: {headline_sent.reason}; Content: {content_sent.reason}"
        )
    else:
        sentiment = headline_sent
    
    # Extract matched keywords for transparency
    keywords_matched = []
    text = (headline + " " + content).lower()
    for kw in NEGATIVE_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', text):
            keywords_matched.append(f"-{kw}")
    for kw in POSITIVE_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', text):
            keywords_matched.append(f"+{kw}")
    
    return ArticleSentiment(
        headline=headline,
        sentiment=sentiment,
        source=source,
        url=url,
        published_at=published_at,
        keywords_matched=keywords_matched[:10],  # Top 10
    )


def aggregate_sentiment(articles: Sequence[ArticleSentiment]) -> AggregateSentiment:
    """
    Aggregate sentiment across multiple articles with recency and source weighting.
    """
    if not articles:
        return AggregateSentiment(
            label="neutral",
            confidence=0.5,
            magnitude=0.0,
            total_articles=0,
            score=0.0,
        )
    
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    weighted_score = 0.0
    total_weight = 0.0
    magnitude_sum = 0.0
    confidence_sum = 0.0
    
    for i, article in enumerate(articles):
        # Recency weight: more recent = higher weight
        # Assuming articles are ordered newest first
        recency_weight = 1.0 / (1.0 + i * 0.1)
        
        sent = article.sentiment
        
        if sent.label == "positive":
            positive_count += 1
            weighted_score += sent.confidence * sent.magnitude * recency_weight
        elif sent.label == "negative":
            negative_count += 1
            weighted_score -= sent.confidence * sent.magnitude * recency_weight
        else:
            neutral_count += 1
        
        total_weight += recency_weight
        magnitude_sum += sent.magnitude
        confidence_sum += sent.confidence
    
    total = len(articles)
    
    # Normalized score (-1 to +1)
    score = weighted_score / total_weight if total_weight > 0 else 0.0
    score = max(-1.0, min(1.0, score))
    
    # Determine dominant label
    if positive_count > negative_count and positive_count > neutral_count:
        label = "positive"
    elif negative_count > positive_count and negative_count > neutral_count:
        label = "negative"
    else:
        label = "neutral"
    
    # Override if score is clearly directional
    if score >= 0.3:
        label = "positive"
    elif score <= -0.3:
        label = "negative"
    
    return AggregateSentiment(
        label=label,
        confidence=confidence_sum / total if total > 0 else 0.5,
        magnitude=magnitude_sum / total if total > 0 else 0.0,
        positive_count=positive_count,
        negative_count=negative_count,
        neutral_count=neutral_count,
        total_articles=total,
        score=score,
        articles=list(articles),
    )


# =============================================================================
# LLM-Based Sentiment (Enhanced)
# =============================================================================

def llm_sentiment(api_key: str, model: str, headline_blob: str, base_url: str | None = None) -> SentimentResult:
    """
    Use LLM for nuanced sentiment analysis.
    
    Best for:
    - Complex or ambiguous headlines
    - When rule-based confidence is low
    - High-stakes trading decisions
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt = """You are a senior financial analyst at a hedge fund. Analyze the sentiment of the following news for near-term stock price impact.

Consider:
1. Is this materially positive or negative for the stock?
2. How confident are you (0-1)?
3. How significant is this (magnitude 0-1)?

Return ONLY valid JSON:
{"label": "positive|negative|neutral", "confidence": 0.0-1.0, "magnitude": 0.0-1.0, "reason": "brief explanation"}

News:
"""
    
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt + headline_blob}],
        temperature=0.2,
        max_tokens=200,
    )
    content = resp.choices[0].message.content or ""
    
    # Parse JSON response
    try:
        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        obj = json.loads(content.strip())
        return SentimentResult(
            label=obj.get("label", "neutral"),
            confidence=float(obj.get("confidence", 0.5)),
            magnitude=float(obj.get("magnitude", 0.5)),
            reason=obj.get("reason", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback to rule-based if LLM fails
        return rule_based_sentiment(headline_blob)


def llm_sentiment_batch(
    api_key: str,
    model: str,
    articles: Sequence[dict],
    max_articles: int = 10,
    base_url: str | None = None,
) -> AggregateSentiment:
    """
    Batch LLM sentiment analysis for multiple articles.
    
    More efficient than individual calls for aggregate analysis.
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Build context
    articles_text = "\n\n".join([
        f"[{i+1}] {a.get('headline', a.get('title', ''))}"
        for i, a in enumerate(articles[:max_articles])
    ])
    
    prompt = f"""You are a senior financial analyst. Analyze the aggregate sentiment of these news items for near-term stock price impact.

News items:
{articles_text}

Return ONLY valid JSON:
{{
    "aggregate_label": "positive|negative|neutral",
    "aggregate_confidence": 0.0-1.0,
    "aggregate_score": -1.0 to 1.0 (negative = bearish, positive = bullish),
    "positive_count": number,
    "negative_count": number,
    "neutral_count": number,
    "key_themes": ["theme1", "theme2"],
    "reasoning": "brief explanation"
}}
"""
    
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400,
    )
    content = resp.choices[0].message.content or ""
    
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        obj = json.loads(content.strip())
        
        return AggregateSentiment(
            label=obj.get("aggregate_label", "neutral"),
            confidence=float(obj.get("aggregate_confidence", 0.5)),
            magnitude=0.5,  # Not returned by this prompt
            positive_count=int(obj.get("positive_count", 0)),
            negative_count=int(obj.get("negative_count", 0)),
            neutral_count=int(obj.get("neutral_count", 0)),
            total_articles=len(articles[:max_articles]),
            score=float(obj.get("aggregate_score", 0.0)),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback to rule-based aggregate
        article_sentiments = [
            analyze_article_sentiment(
                headline=a.get("headline", a.get("title", "")),
                content=a.get("content", a.get("text", "")),
            )
            for a in articles[:max_articles]
        ]
        return aggregate_sentiment(article_sentiments)

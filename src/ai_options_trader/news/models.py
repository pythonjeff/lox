"""
News Sentiment Regime - Data Models

Tracks aggregate market news sentiment and its regime implications.
News sentiment is a leading indicator of market behavior and risk appetite.

Author: Lox Capital Research
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NewsSentimentInputs:
    """
    Aggregate news sentiment metrics across market sectors.
    
    Metrics are computed from recent news flow (typically 7-14 days).
    """
    # -------------------------------------------------------------------------
    # Aggregate Sentiment Scores (-1 to +1)
    # -------------------------------------------------------------------------
    # Overall market sentiment
    market_sentiment_score: float | None = None
    market_sentiment_confidence: float | None = None
    
    # Sector-level sentiment
    tech_sentiment_score: float | None = None
    financials_sentiment_score: float | None = None
    energy_sentiment_score: float | None = None
    healthcare_sentiment_score: float | None = None
    consumer_sentiment_score: float | None = None
    industrials_sentiment_score: float | None = None
    
    # -------------------------------------------------------------------------
    # Volume and Flow Metrics
    # -------------------------------------------------------------------------
    # Total articles analyzed
    total_articles: int = 0
    
    # Breakdown by sentiment
    positive_articles: int = 0
    negative_articles: int = 0
    neutral_articles: int = 0
    
    # High-conviction articles (confidence > 0.7)
    high_conviction_positive: int = 0
    high_conviction_negative: int = 0
    
    # -------------------------------------------------------------------------
    # Momentum / Change Metrics
    # -------------------------------------------------------------------------
    # Sentiment change vs prior period
    sentiment_momentum_7d: float | None = None  # Current vs 7d ago
    sentiment_momentum_14d: float | None = None  # Current vs 14d ago
    
    # Volatility of sentiment (std of daily scores)
    sentiment_volatility: float | None = None
    
    # -------------------------------------------------------------------------
    # Topic/Theme Metrics
    # -------------------------------------------------------------------------
    # Count of articles mentioning key themes
    earnings_mentions: int = 0
    fed_mentions: int = 0
    tariff_mentions: int = 0
    recession_mentions: int = 0
    ai_mentions: int = 0
    layoffs_mentions: int = 0
    
    # Dominant theme (highest mention count)
    dominant_theme: str | None = None
    
    # -------------------------------------------------------------------------
    # Sector Dispersion
    # -------------------------------------------------------------------------
    # Std dev of sector sentiment scores (high = divergence)
    sector_dispersion: float | None = None
    
    # Best and worst performing sectors by sentiment
    best_sector: str | None = None
    worst_sector: str | None = None
    
    # -------------------------------------------------------------------------
    # Composite Scores
    # -------------------------------------------------------------------------
    # Risk appetite score (positive = risk-on, negative = risk-off)
    risk_appetite_score: float | None = None
    
    # Fear/greed indicator (0-100 scale)
    fear_greed_score: float | None = None
    
    # Debug / transparency
    components: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NewsSentimentState:
    """
    Complete news sentiment regime snapshot.
    """
    asof: str
    lookback_days: int
    inputs: NewsSentimentInputs
    notes: str = ""

"""
News Sentiment Regime Module

Tracks aggregate market news sentiment and classifies it into actionable regimes.
News sentiment is a leading indicator of market behavior and risk appetite.

Regimes:
- NEWS_EUPHORIA: Extreme positive sentiment (contrarian sell)
- NEWS_BULLISH: Strong positive sentiment (risk-on)
- NEWS_NEUTRAL: Mixed or balanced sentiment
- NEWS_CAUTIOUS: Moderately negative sentiment (defensive)
- NEWS_FEARFUL: Extreme negative sentiment (contrarian buy)

Author: Lox Capital Research
"""
from __future__ import annotations

from ai_options_trader.news.models import (
    NewsSentimentInputs,
    NewsSentimentState,
)
from ai_options_trader.news.regime import (
    NewsSentimentRegime,
    classify_news_regime,
    classify_news_regime_from_state,
)
from ai_options_trader.news.signals import (
    build_news_sentiment_state,
    SECTOR_TICKERS,
    THEME_KEYWORDS,
)
from ai_options_trader.news.features import (
    news_feature_vector,
    NEWS_REGIME_NAMES,
)

__all__ = [
    # Models
    "NewsSentimentInputs",
    "NewsSentimentState",
    # Regime
    "NewsSentimentRegime",
    "classify_news_regime",
    "classify_news_regime_from_state",
    # Signals
    "build_news_sentiment_state",
    "SECTOR_TICKERS",
    "THEME_KEYWORDS",
    # Features
    "news_feature_vector",
    "NEWS_REGIME_NAMES",
]

"""
Tests for the news sentiment regime module.
"""
from __future__ import annotations

import pytest

from ai_options_trader.news.models import NewsSentimentInputs, NewsSentimentState
from ai_options_trader.news.regime import NewsSentimentRegime, classify_news_regime
from ai_options_trader.news.features import news_feature_vector, NEWS_REGIME_NAMES


class TestNewsSentimentClassification:
    """Test news sentiment regime classification logic."""
    
    def test_euphoria_regime(self):
        """Extreme positive sentiment should yield euphoria."""
        inputs = NewsSentimentInputs(
            market_sentiment_score=0.6,
            fear_greed_score=80,
            positive_articles=15,
            negative_articles=2,
            neutral_articles=3,
        )
        regime = classify_news_regime(inputs)
        assert regime.name == "news_euphoria"
        assert "extreme" in regime.tags
    
    def test_bullish_regime(self):
        """Strong positive sentiment should yield bullish."""
        inputs = NewsSentimentInputs(
            market_sentiment_score=0.35,
            fear_greed_score=60,
            positive_articles=10,
            negative_articles=3,
            neutral_articles=5,
        )
        regime = classify_news_regime(inputs)
        assert regime.name == "news_bullish"
        assert "risk_on" in regime.tags
    
    def test_fearful_regime(self):
        """Extreme negative sentiment should yield fearful."""
        inputs = NewsSentimentInputs(
            market_sentiment_score=-0.6,
            fear_greed_score=20,
            positive_articles=2,
            negative_articles=15,
            neutral_articles=3,
        )
        regime = classify_news_regime(inputs)
        assert regime.name == "news_fearful"
        assert "panic" in regime.tags
    
    def test_cautious_regime(self):
        """Moderately negative sentiment should yield cautious."""
        inputs = NewsSentimentInputs(
            market_sentiment_score=-0.3,
            fear_greed_score=40,
            positive_articles=4,
            negative_articles=10,
            neutral_articles=6,
        )
        regime = classify_news_regime(inputs)
        assert regime.name == "news_cautious"
        assert "defensive" in regime.tags
    
    def test_neutral_regime(self):
        """Mixed sentiment should yield neutral."""
        inputs = NewsSentimentInputs(
            market_sentiment_score=0.05,
            fear_greed_score=50,
            positive_articles=5,
            negative_articles=5,
            neutral_articles=10,
        )
        regime = classify_news_regime(inputs)
        assert regime.name == "news_neutral"
    
    def test_unknown_regime_no_data(self):
        """Missing data should yield unknown."""
        inputs = NewsSentimentInputs(
            market_sentiment_score=None,
        )
        regime = classify_news_regime(inputs)
        assert regime.name == "unknown"


class TestNewsFeatureVector:
    """Test ML feature extraction."""
    
    def test_feature_vector_structure(self):
        """Feature vector should contain expected feature groups."""
        inputs = NewsSentimentInputs(
            market_sentiment_score=0.3,
            market_sentiment_confidence=0.7,
            fear_greed_score=60,
            tech_sentiment_score=0.4,
            financials_sentiment_score=-0.1,
            total_articles=20,
            positive_articles=10,
            negative_articles=5,
            neutral_articles=5,
            earnings_mentions=8,
            fed_mentions=3,
        )
        state = NewsSentimentState(
            asof="2024-01-15",
            lookback_days=7,
            inputs=inputs,
        )
        regime = NewsSentimentRegime(
            name="news_bullish",
            label="News Bullish",
            description="Test",
            tags=("news", "risk_on"),
        )
        
        vec = news_feature_vector(state, regime)
        
        # Check sentiment features
        assert "news.sentiment.market_score" in vec.features
        assert vec.features["news.sentiment.market_score"] == 0.3
        
        # Check sector features
        assert "news.sector.tech" in vec.features
        assert vec.features["news.sector.tech"] == 0.4
        
        # Check volume features
        assert "news.volume.total" in vec.features
        assert vec.features["news.volume.total"] == 20.0
        
        # Check theme features
        assert "news.theme.earnings" in vec.features
        assert vec.features["news.theme.earnings"] == 8.0
        
        # Check one-hot encoding
        assert "news.regime.news_bullish" in vec.features
        assert vec.features["news.regime.news_bullish"] == 1.0
    
    def test_all_regime_names_in_one_hot(self):
        """All canonical regime names should be available for one-hot."""
        assert "news_euphoria" in NEWS_REGIME_NAMES
        assert "news_bullish" in NEWS_REGIME_NAMES
        assert "news_neutral" in NEWS_REGIME_NAMES
        assert "news_cautious" in NEWS_REGIME_NAMES
        assert "news_fearful" in NEWS_REGIME_NAMES


class TestSentimentEnhanced:
    """Test enhanced sentiment analysis."""
    
    def test_rule_based_negative_keywords(self):
        """Test negative keyword detection."""
        from ai_options_trader.llm.core.sentiment import rule_based_sentiment
        
        result = rule_based_sentiment("Company misses earnings, CEO resigns amid fraud investigation")
        assert result.label == "negative"
        assert result.confidence > 0.6
    
    def test_rule_based_positive_keywords(self):
        """Test positive keyword detection."""
        from ai_options_trader.llm.core.sentiment import rule_based_sentiment
        
        result = rule_based_sentiment("Company beats earnings, raises guidance with record revenue")
        assert result.label == "positive"
        assert result.confidence > 0.6
    
    def test_rule_based_mixed_signals(self):
        """Test mixed sentiment handling."""
        from ai_options_trader.llm.core.sentiment import rule_based_sentiment
        
        result = rule_based_sentiment("Strong growth but misses on margins")
        # Could be positive, negative, or neutral depending on weight
        assert result.confidence >= 0.3
    
    def test_aggregate_sentiment(self):
        """Test aggregate sentiment calculation."""
        from ai_options_trader.llm.core.sentiment import (
            analyze_article_sentiment,
            aggregate_sentiment,
        )
        
        articles = [
            analyze_article_sentiment("Company beats earnings, stock surges"),
            analyze_article_sentiment("Record revenue, raises guidance"),
            analyze_article_sentiment("Misses on margins"),
        ]
        
        agg = aggregate_sentiment(articles)
        assert agg.total_articles == 3
        assert agg.positive_count >= 2
        # Score should lean positive
        assert agg.score >= 0

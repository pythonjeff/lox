"""
News Sentiment Regime - ML Feature Extraction

Converts news sentiment state into a flat feature vector for ML models.

Author: Lox Capital Research
"""
from __future__ import annotations

from ai_options_trader.regimes.schema import RegimeVector, add_feature, add_one_hot
from ai_options_trader.news.models import NewsSentimentState
from ai_options_trader.news.regime import NewsSentimentRegime


# Canonical regime names for one-hot encoding
NEWS_REGIME_NAMES = (
    "news_euphoria",
    "news_bullish",
    "news_neutral",
    "news_cautious",
    "news_fearful",
    "unknown",
)


def news_feature_vector(
    state: NewsSentimentState,
    regime: NewsSentimentRegime,
) -> RegimeVector:
    """
    Convert news sentiment state to ML-friendly feature vector.
    
    Feature groups:
    - news.sentiment.*: Aggregate sentiment metrics
    - news.sector.*: Sector-level sentiment
    - news.volume.*: Article counts
    - news.theme.*: Theme mention counts
    - news.composite.*: Derived scores
    - news.regime.*: One-hot regime encoding
    """
    f: dict[str, float] = {}
    inp = state.inputs
    
    # -------------------------------------------------------------------------
    # Aggregate Sentiment
    # -------------------------------------------------------------------------
    add_feature(f, "news.sentiment.market_score", inp.market_sentiment_score)
    add_feature(f, "news.sentiment.confidence", inp.market_sentiment_confidence)
    add_feature(f, "news.sentiment.momentum_7d", inp.sentiment_momentum_7d)
    add_feature(f, "news.sentiment.momentum_14d", inp.sentiment_momentum_14d)
    add_feature(f, "news.sentiment.volatility", inp.sentiment_volatility)
    
    # -------------------------------------------------------------------------
    # Sector Sentiment
    # -------------------------------------------------------------------------
    add_feature(f, "news.sector.tech", inp.tech_sentiment_score)
    add_feature(f, "news.sector.financials", inp.financials_sentiment_score)
    add_feature(f, "news.sector.energy", inp.energy_sentiment_score)
    add_feature(f, "news.sector.healthcare", inp.healthcare_sentiment_score)
    add_feature(f, "news.sector.consumer", inp.consumer_sentiment_score)
    add_feature(f, "news.sector.industrials", inp.industrials_sentiment_score)
    add_feature(f, "news.sector.dispersion", inp.sector_dispersion)
    
    # -------------------------------------------------------------------------
    # Volume Metrics
    # -------------------------------------------------------------------------
    add_feature(f, "news.volume.total", float(inp.total_articles) if inp.total_articles else None)
    add_feature(f, "news.volume.positive", float(inp.positive_articles) if inp.positive_articles else None)
    add_feature(f, "news.volume.negative", float(inp.negative_articles) if inp.negative_articles else None)
    add_feature(f, "news.volume.neutral", float(inp.neutral_articles) if inp.neutral_articles else None)
    add_feature(f, "news.volume.high_conviction_pos", float(inp.high_conviction_positive) if inp.high_conviction_positive else None)
    add_feature(f, "news.volume.high_conviction_neg", float(inp.high_conviction_negative) if inp.high_conviction_negative else None)
    
    # Ratios (avoid division by zero)
    if inp.total_articles and inp.total_articles > 0:
        add_feature(f, "news.volume.positive_ratio", inp.positive_articles / inp.total_articles)
        add_feature(f, "news.volume.negative_ratio", inp.negative_articles / inp.total_articles)
    
    # -------------------------------------------------------------------------
    # Theme Mentions
    # -------------------------------------------------------------------------
    add_feature(f, "news.theme.earnings", float(inp.earnings_mentions) if inp.earnings_mentions else None)
    add_feature(f, "news.theme.fed", float(inp.fed_mentions) if inp.fed_mentions else None)
    add_feature(f, "news.theme.tariff", float(inp.tariff_mentions) if inp.tariff_mentions else None)
    add_feature(f, "news.theme.recession", float(inp.recession_mentions) if inp.recession_mentions else None)
    add_feature(f, "news.theme.ai", float(inp.ai_mentions) if inp.ai_mentions else None)
    add_feature(f, "news.theme.layoffs", float(inp.layoffs_mentions) if inp.layoffs_mentions else None)
    
    # -------------------------------------------------------------------------
    # Composite Scores
    # -------------------------------------------------------------------------
    add_feature(f, "news.composite.risk_appetite", inp.risk_appetite_score)
    add_feature(f, "news.composite.fear_greed", inp.fear_greed_score)
    
    # Normalized fear/greed (0-1 instead of 0-100)
    if inp.fear_greed_score is not None:
        add_feature(f, "news.composite.fear_greed_normalized", inp.fear_greed_score / 100.0)
    
    # -------------------------------------------------------------------------
    # Regime One-Hot Encoding
    # -------------------------------------------------------------------------
    add_one_hot(f, "news.regime", regime.name, NEWS_REGIME_NAMES)
    
    return RegimeVector(
        asof=state.asof,
        features=f,
        notes=(
            f"News sentiment regime features. Analyzed {inp.total_articles} articles "
            f"over {state.lookback_days} days. Dominant theme: {inp.dominant_theme or 'None'}."
        ),
    )

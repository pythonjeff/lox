"""
News Sentiment Regime - Classification Logic

Classifies market news sentiment into actionable regimes.

Regimes:
- NEWS_EUPHORIA: Extreme positive sentiment, potential complacency
- NEWS_BULLISH: Strong positive sentiment, risk-on
- NEWS_NEUTRAL: Mixed or balanced sentiment
- NEWS_CAUTIOUS: Moderately negative sentiment
- NEWS_FEARFUL: Extreme negative sentiment, potential panic

Author: Lox Capital Research
"""
from __future__ import annotations

from dataclasses import dataclass

from ai_options_trader.news.models import NewsSentimentInputs


@dataclass(frozen=True)
class NewsSentimentRegime:
    """
    News sentiment regime classification result.
    
    Canonical fields for all regime types.
    """
    name: str
    label: str
    description: str
    tags: tuple[str, ...] = ()
    market_implications: str = ""
    contrarian_signal: str = ""  # What contrarians might do


def classify_news_regime(inputs: NewsSentimentInputs) -> NewsSentimentRegime:
    """
    Classify news sentiment regime based on aggregate metrics.
    
    Classification uses:
    - market_sentiment_score: Primary signal (-1 to +1)
    - high_conviction counts: Strength of signal
    - sector_dispersion: Consensus vs divergence
    - fear_greed_score: 0-100 scale
    
    Regime thresholds:
    - Euphoria: score >= 0.5 AND fear_greed >= 75
    - Bullish: score >= 0.2
    - Neutral: -0.2 < score < 0.2
    - Cautious: score <= -0.2
    - Fearful: score <= -0.5 AND fear_greed <= 25
    """
    score = inputs.market_sentiment_score
    fg = inputs.fear_greed_score
    dispersion = inputs.sector_dispersion
    high_pos = inputs.high_conviction_positive
    high_neg = inputs.high_conviction_negative
    
    # Handle missing data
    if score is None:
        return NewsSentimentRegime(
            name="unknown",
            label="Unknown",
            description="Insufficient news data to classify sentiment regime.",
            tags=("news", "unknown"),
        )
    
    # Check for extreme readings
    fg_extreme_high = fg is not None and fg >= 75
    fg_extreme_low = fg is not None and fg <= 25
    
    # High dispersion indicates mixed signals
    high_dispersion = dispersion is not None and dispersion > 0.4
    
    # -------------------------------------------------------------------------
    # EUPHORIA: Extreme bullishness (contrarian sell signal)
    # -------------------------------------------------------------------------
    if score >= 0.5 and fg_extreme_high:
        return NewsSentimentRegime(
            name="news_euphoria",
            label="News Euphoria",
            description=(
                "Extreme positive news sentiment with fear/greed above 75. "
                "Market is pricing in good news; limited upside from incremental positives. "
                "High risk of disappointment on any negative surprise."
            ),
            tags=("news", "risk_on", "extreme", "complacency"),
            market_implications=(
                "Favor: Defensive hedges, put spreads. "
                "Avoid: Chasing momentum, adding to winners. "
                "Risk: Mean reversion, any negative catalyst."
            ),
            contrarian_signal="Consider reducing equity exposure or adding tail hedges.",
        )
    
    # -------------------------------------------------------------------------
    # BULLISH: Strong positive sentiment
    # -------------------------------------------------------------------------
    if score >= 0.2:
        label = "News Bullish (Broad)" if not high_dispersion else "News Bullish (Selective)"
        desc = (
            "Positive news sentiment across markets. "
            "Risk appetite is elevated with positive headline flow. "
        )
        if high_dispersion:
            desc += "However, sector sentiment is diverging - be selective on exposure."
        
        return NewsSentimentRegime(
            name="news_bullish",
            label=label,
            description=desc,
            tags=("news", "risk_on"),
            market_implications=(
                "Favor: Beta, growth stocks, cyclicals. "
                f"Best sector: {inputs.best_sector}. "
                "Avoid: Extreme defensives."
            ),
            contrarian_signal="Watch for sentiment exhaustion near resistance levels.",
        )
    
    # -------------------------------------------------------------------------
    # FEARFUL: Extreme bearishness (contrarian buy signal)
    # -------------------------------------------------------------------------
    if score <= -0.5 and fg_extreme_low:
        return NewsSentimentRegime(
            name="news_fearful",
            label="News Fearful",
            description=(
                "Extreme negative news sentiment with fear/greed below 25. "
                "Market is pricing in worst-case scenarios; headlines are maximally negative. "
                "Historically, extreme fear precedes market bottoms."
            ),
            tags=("news", "risk_off", "extreme", "panic"),
            market_implications=(
                "Favor: Cash, short-term bonds, selective dip buying. "
                "Avoid: Leveraged shorts (squeeze risk). "
                "Watch: VIX term structure for capitulation signals."
            ),
            contrarian_signal="Consider nibbling on quality names with long-term thesis intact.",
        )
    
    # -------------------------------------------------------------------------
    # CAUTIOUS: Moderately negative sentiment
    # -------------------------------------------------------------------------
    if score <= -0.2:
        desc = "Negative news sentiment with cautionary headline flow. "
        if inputs.recession_mentions > 5:
            desc += "Recession concerns are elevated in coverage. "
        if inputs.layoffs_mentions > 3:
            desc += "Layoff announcements are above normal. "
        
        return NewsSentimentRegime(
            name="news_cautious",
            label="News Cautious",
            description=desc,
            tags=("news", "risk_off", "defensive"),
            market_implications=(
                "Favor: Quality, low volatility, dividend aristocrats. "
                f"Worst sector: {inputs.worst_sector}. "
                "Avoid: High beta, unprofitable growth."
            ),
            contrarian_signal="May be setting up for relief rally if bad news is priced in.",
        )
    
    # -------------------------------------------------------------------------
    # NEUTRAL: Mixed or balanced sentiment
    # -------------------------------------------------------------------------
    return NewsSentimentRegime(
        name="news_neutral",
        label="News Neutral",
        description=(
            "Mixed or balanced news sentiment across markets. "
            "No strong directional signal from headline flow. "
            "Focus on fundamentals and technicals over news-driven moves."
        ),
        tags=("news", "neutral"),
        market_implications=(
            "Balanced positioning recommended. "
            "Focus on stock-specific catalysts over macro narratives. "
            f"Dominant theme: {inputs.dominant_theme or 'None'}."
        ),
        contrarian_signal="Wait for clearer sentiment signal before adding directional exposure.",
    )


def classify_news_regime_from_state(
    inputs: NewsSentimentInputs,
) -> NewsSentimentRegime:
    """Alias for classify_news_regime for API consistency."""
    return classify_news_regime(inputs)

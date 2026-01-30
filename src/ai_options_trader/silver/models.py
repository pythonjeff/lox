"""
Silver regime data models.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class SilverInputs(BaseModel):
    """
    Silver market inputs and derived features.
    """

    # Price levels
    slv_price: Optional[float] = Field(None, description="SLV ETF last price")
    slv_ma_50: Optional[float] = Field(None, description="SLV 50-day MA")
    slv_ma_200: Optional[float] = Field(None, description="SLV 200-day MA")

    # Returns
    slv_ret_5d_pct: Optional[float] = Field(None, description="SLV 5-day return %")
    slv_ret_20d_pct: Optional[float] = Field(None, description="SLV 20-day return %")
    slv_ret_60d_pct: Optional[float] = Field(None, description="SLV 60-day return %")
    slv_ret_200d_pct: Optional[float] = Field(None, description="SLV 200-day return %")

    # Z-scores (vs 3-year history)
    slv_zscore_20d: Optional[float] = Field(None, description="SLV 20d return z-score")
    slv_zscore_60d: Optional[float] = Field(None, description="SLV 60d return z-score")

    # Moving average positioning
    slv_above_50ma: Optional[bool] = Field(None, description="SLV above 50-day MA")
    slv_above_200ma: Optional[bool] = Field(None, description="SLV above 200-day MA")
    slv_50ma_above_200ma: Optional[bool] = Field(None, description="50-day MA above 200-day MA (golden cross)")
    slv_pct_from_50ma: Optional[float] = Field(None, description="SLV % distance from 50-day MA")
    slv_pct_from_200ma: Optional[float] = Field(None, description="SLV % distance from 200-day MA")

    # Gold/Silver Ratio (GSR)
    gsr: Optional[float] = Field(None, description="Gold/Silver ratio (GLD price / SLV price adjusted)")
    gsr_20d_avg: Optional[float] = Field(None, description="GSR 20-day average")
    gsr_zscore: Optional[float] = Field(None, description="GSR z-score vs 3-year history")
    gsr_expanding: Optional[bool] = Field(None, description="GSR trending higher (silver weakening vs gold)")

    # Volatility
    slv_vol_20d_ann_pct: Optional[float] = Field(None, description="SLV 20-day annualized volatility %")
    slv_vol_60d_ann_pct: Optional[float] = Field(None, description="SLV 60-day annualized volatility %")
    slv_vol_zscore: Optional[float] = Field(None, description="SLV volatility z-score")

    # Volume
    slv_volume_ratio: Optional[float] = Field(None, description="Current volume vs 20-day avg")
    slv_volume_trending_up: Optional[bool] = Field(None, description="Volume trend higher")

    # Correlation with risk assets
    slv_spy_corr_60d: Optional[float] = Field(None, description="SLV-SPY 60-day correlation")
    slv_vix_corr_60d: Optional[float] = Field(None, description="SLV-VIX 60-day correlation")

    # Composite scores
    trend_score: Optional[float] = Field(
        None,
        description="Trend composite: -100 (strong downtrend) to +100 (strong uptrend)",
    )
    momentum_score: Optional[float] = Field(
        None,
        description="Momentum composite: -100 (bearish) to +100 (bullish)",
    )
    relative_value_score: Optional[float] = Field(
        None,
        description="Relative value vs gold: -100 (silver rich) to +100 (silver cheap)",
    )

    # Bubble / Mean Reversion metrics
    bubble_score: Optional[float] = Field(
        None,
        description="Bubble intensity: 0 (normal) to 100 (extreme bubble)",
    )
    mean_reversion_pressure: Optional[float] = Field(
        None,
        description="Mean reversion pressure: 0 (low) to 100 (high)",
    )
    extension_pct: Optional[float] = Field(
        None,
        description="How extended price is vs fair value estimate (%)",
    )
    days_at_extreme: Optional[int] = Field(
        None,
        description="Consecutive days with bubble score > 50",
    )

    # Momentum divergence (leading indicator)
    rsi_14: Optional[float] = Field(None, description="14-day RSI")
    momentum_divergence: Optional[bool] = Field(
        None,
        description="True if price making highs but momentum weakening",
    )
    volume_exhaustion: Optional[bool] = Field(
        None,
        description="True if volume declining on up days",
    )


class ReversionForecast(BaseModel):
    """
    Mean reversion forecast based on historical patterns and current state.
    """
    
    # Timing estimates
    days_to_reversion_low: Optional[int] = Field(None, description="Optimistic estimate (10th percentile)")
    days_to_reversion_mid: Optional[int] = Field(None, description="Median estimate (50th percentile)")
    days_to_reversion_high: Optional[int] = Field(None, description="Conservative estimate (90th percentile)")
    reversion_probability_30d: Optional[float] = Field(None, description="Probability of reversion within 30 days")
    reversion_probability_60d: Optional[float] = Field(None, description="Probability of reversion within 60 days")
    reversion_probability_90d: Optional[float] = Field(None, description="Probability of reversion within 90 days")
    
    # Severity estimates
    drawdown_10th_pct: Optional[float] = Field(None, description="10th percentile drawdown %")
    drawdown_median: Optional[float] = Field(None, description="Median expected drawdown %")
    drawdown_90th_pct: Optional[float] = Field(None, description="90th percentile drawdown %")
    target_fair_value: Optional[float] = Field(None, description="Fair value target (200MA)")
    target_25pct_premium: Optional[float] = Field(None, description="25% above fair value")
    target_50pct_premium: Optional[float] = Field(None, description="50% above fair value")
    
    # Trigger levels
    trigger_50ma: Optional[float] = Field(None, description="50-day MA break level")
    trigger_gsr_expansion: Optional[float] = Field(None, description="GSR level that signals expansion")
    trigger_volume_capitulation: Optional[bool] = Field(None, description="Volume spike threshold")
    
    # Historical analogs
    analog_dates: list[str] = Field(default_factory=list, description="Dates of similar historical setups")
    analog_outcomes: list[dict] = Field(default_factory=list, description="What happened in each analog")
    
    # Confidence
    forecast_confidence: Optional[float] = Field(None, description="Confidence in forecast 0-100")
    
    class Config:
        frozen = True


class SilverState(BaseModel):
    """
    Complete silver regime state snapshot.
    """

    asof: date = Field(..., description="Snapshot date")
    start_date: date = Field(..., description="Start of data window")
    inputs: SilverInputs = Field(..., description="Silver inputs")
    notes: list[str] = Field(default_factory=list, description="Contextual notes")

    class Config:
        frozen = True

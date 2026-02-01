"""
Momentum Metrics - Technical momentum indicators for research.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class MomentumMetrics:
    """Comprehensive momentum metrics for a ticker."""
    ticker: str
    
    # Price momentum
    return_1d: float
    return_5d: float
    return_1m: float
    return_3m: float
    return_6m: float
    return_ytd: float
    return_1y: float
    
    # Relative strength
    rsi_14: float  # 14-day RSI
    rsi_interpretation: str  # overbought/oversold/neutral
    
    # Moving average signals
    price: float
    sma_20: float
    sma_50: float
    sma_200: float
    above_sma_20: bool
    above_sma_50: bool
    above_sma_200: bool
    
    # Golden/Death cross
    golden_cross: bool  # 50 > 200
    death_cross: bool  # 50 < 200
    
    # Rate of change
    roc_10: float  # 10-day rate of change
    roc_20: float  # 20-day rate of change
    
    # Trend strength
    adx_14: Optional[float]  # Average Directional Index
    trend_strength: str  # strong/moderate/weak/no trend
    trend_direction: str  # bullish/bearish/neutral
    
    # 52-week metrics
    high_52w: float
    low_52w: float
    pct_from_52w_high: float
    pct_from_52w_low: float
    
    # Momentum score (composite)
    momentum_score: int  # -100 to +100
    momentum_label: str  # Strong Bullish, Bullish, Neutral, Bearish, Strong Bearish


def calculate_momentum_metrics(
    prices: pd.Series,
    ticker: str,
) -> MomentumMetrics:
    """
    Calculate comprehensive momentum metrics from a price series.
    
    Args:
        prices: Daily price series (index=dates, values=prices)
        ticker: Ticker symbol
    
    Returns:
        MomentumMetrics dataclass
    """
    if prices.empty or len(prices) < 20:
        # Return empty metrics for insufficient data
        return MomentumMetrics(
            ticker=ticker,
            return_1d=0, return_5d=0, return_1m=0, return_3m=0,
            return_6m=0, return_ytd=0, return_1y=0,
            rsi_14=50, rsi_interpretation="neutral",
            price=prices.iloc[-1] if len(prices) > 0 else 0,
            sma_20=0, sma_50=0, sma_200=0,
            above_sma_20=False, above_sma_50=False, above_sma_200=False,
            golden_cross=False, death_cross=False,
            roc_10=0, roc_20=0,
            adx_14=None, trend_strength="unknown", trend_direction="unknown",
            high_52w=0, low_52w=0, pct_from_52w_high=0, pct_from_52w_low=0,
            momentum_score=0, momentum_label="Unknown",
        )
    
    # Ensure sorted by date
    prices = prices.sort_index()
    current_price = prices.iloc[-1]
    
    # Calculate returns
    def safe_return(n: int) -> float:
        if len(prices) > n:
            return (current_price / prices.iloc[-n-1] - 1) * 100
        return 0
    
    return_1d = safe_return(1)
    return_5d = safe_return(5)
    return_1m = safe_return(21)
    return_3m = safe_return(63)
    return_6m = safe_return(126)
    return_1y = safe_return(252)
    
    # YTD return
    ytd_prices = prices[prices.index >= f"{prices.index[-1].year}-01-01"]
    return_ytd = (current_price / ytd_prices.iloc[0] - 1) * 100 if len(ytd_prices) > 1 else 0
    
    # RSI calculation
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_14 = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
    
    if rsi_14 >= 70:
        rsi_interpretation = "overbought"
    elif rsi_14 <= 30:
        rsi_interpretation = "oversold"
    elif rsi_14 >= 60:
        rsi_interpretation = "bullish"
    elif rsi_14 <= 40:
        rsi_interpretation = "bearish"
    else:
        rsi_interpretation = "neutral"
    
    # Moving averages
    sma_20 = float(prices.rolling(20).mean().iloc[-1]) if len(prices) >= 20 else current_price
    sma_50 = float(prices.rolling(50).mean().iloc[-1]) if len(prices) >= 50 else current_price
    sma_200 = float(prices.rolling(200).mean().iloc[-1]) if len(prices) >= 200 else current_price
    
    above_sma_20 = current_price > sma_20
    above_sma_50 = current_price > sma_50
    above_sma_200 = current_price > sma_200
    
    golden_cross = sma_50 > sma_200
    death_cross = sma_50 < sma_200
    
    # Rate of change
    roc_10 = (current_price / prices.iloc[-11] - 1) * 100 if len(prices) > 11 else 0
    roc_20 = (current_price / prices.iloc[-21] - 1) * 100 if len(prices) > 21 else 0
    
    # ADX calculation (simplified)
    adx_14 = None
    if len(prices) >= 28:
        try:
            high = prices.rolling(2).max()
            low = prices.rolling(2).min()
            
            plus_dm = high.diff().where(high.diff() > -low.diff(), 0).where(high.diff() > 0, 0)
            minus_dm = (-low.diff()).where(-low.diff() > high.diff(), 0).where(-low.diff() > 0, 0)
            
            tr = pd.concat([
                high - low,
                (high - prices.shift()).abs(),
                (low - prices.shift()).abs()
            ], axis=1).max(axis=1)
            
            atr = tr.rolling(14).mean()
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
            
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
            adx_14 = float(dx.rolling(14).mean().iloc[-1])
            if pd.isna(adx_14):
                adx_14 = None
        except Exception:
            adx_14 = None
    
    # Trend strength and direction
    if adx_14 is not None:
        if adx_14 >= 40:
            trend_strength = "strong"
        elif adx_14 >= 25:
            trend_strength = "moderate"
        elif adx_14 >= 15:
            trend_strength = "weak"
        else:
            trend_strength = "no trend"
    else:
        trend_strength = "unknown"
    
    # Trend direction from multiple signals
    bullish_signals = sum([
        return_1m > 0,
        return_3m > 0,
        above_sma_20,
        above_sma_50,
        above_sma_200,
        golden_cross,
        rsi_14 > 50,
    ])
    
    if bullish_signals >= 5:
        trend_direction = "bullish"
    elif bullish_signals <= 2:
        trend_direction = "bearish"
    else:
        trend_direction = "neutral"
    
    # 52-week metrics
    if len(prices) >= 252:
        recent_prices = prices.iloc[-252:]
        high_52w = float(recent_prices.max())
        low_52w = float(recent_prices.min())
    else:
        high_52w = float(prices.max())
        low_52w = float(prices.min())
    
    pct_from_52w_high = (current_price / high_52w - 1) * 100 if high_52w > 0 else 0
    pct_from_52w_low = (current_price / low_52w - 1) * 100 if low_52w > 0 else 0
    
    # Composite momentum score (-100 to +100)
    score_components = []
    
    # Returns component (weighted by recency)
    score_components.append(np.clip(return_1m / 3, -20, 20))  # ±20 points
    score_components.append(np.clip(return_3m / 5, -15, 15))  # ±15 points
    score_components.append(np.clip(return_6m / 8, -10, 10))  # ±10 points
    
    # RSI component
    score_components.append((rsi_14 - 50) * 0.6)  # ±30 points
    
    # Moving average component
    ma_score = 0
    ma_score += 5 if above_sma_20 else -5
    ma_score += 5 if above_sma_50 else -5
    ma_score += 5 if above_sma_200 else -5
    ma_score += 5 if golden_cross else -5
    score_components.append(ma_score)  # ±20 points
    
    # Distance from 52w high/low
    if pct_from_52w_high > -5:
        score_components.append(5)
    elif pct_from_52w_high < -30:
        score_components.append(-5)
    else:
        score_components.append(0)
    
    momentum_score = int(np.clip(sum(score_components), -100, 100))
    
    # Label
    if momentum_score >= 50:
        momentum_label = "Strong Bullish"
    elif momentum_score >= 20:
        momentum_label = "Bullish"
    elif momentum_score >= -20:
        momentum_label = "Neutral"
    elif momentum_score >= -50:
        momentum_label = "Bearish"
    else:
        momentum_label = "Strong Bearish"
    
    return MomentumMetrics(
        ticker=ticker,
        return_1d=round(return_1d, 2),
        return_5d=round(return_5d, 2),
        return_1m=round(return_1m, 2),
        return_3m=round(return_3m, 2),
        return_6m=round(return_6m, 2),
        return_ytd=round(return_ytd, 2),
        return_1y=round(return_1y, 2),
        rsi_14=round(rsi_14, 1),
        rsi_interpretation=rsi_interpretation,
        price=round(current_price, 2),
        sma_20=round(sma_20, 2),
        sma_50=round(sma_50, 2),
        sma_200=round(sma_200, 2),
        above_sma_20=above_sma_20,
        above_sma_50=above_sma_50,
        above_sma_200=above_sma_200,
        golden_cross=golden_cross,
        death_cross=death_cross,
        roc_10=round(roc_10, 2),
        roc_20=round(roc_20, 2),
        adx_14=round(adx_14, 1) if adx_14 is not None else None,
        trend_strength=trend_strength,
        trend_direction=trend_direction,
        high_52w=round(high_52w, 2),
        low_52w=round(low_52w, 2),
        pct_from_52w_high=round(pct_from_52w_high, 2),
        pct_from_52w_low=round(pct_from_52w_low, 2),
        momentum_score=momentum_score,
        momentum_label=momentum_label,
    )


def is_oversold(metrics: MomentumMetrics, threshold: int = -30) -> bool:
    """Check if a ticker is oversold (potential bounce candidate)."""
    return (
        metrics.rsi_14 < 35 or
        metrics.momentum_score <= threshold or
        metrics.pct_from_52w_high < -30
    )


def is_overbought(metrics: MomentumMetrics, threshold: int = 30) -> bool:
    """Check if a ticker is overbought (potential pullback candidate)."""
    return (
        metrics.rsi_14 > 65 or
        metrics.momentum_score >= threshold or
        metrics.pct_from_52w_high > -5
    )

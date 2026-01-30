"""
Silver mean reversion forecasting based on historical patterns.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from ai_options_trader.config import Settings
from ai_options_trader.silver.models import SilverInputs, SilverState, ReversionForecast


def find_historical_analogs(
    df: pd.DataFrame,
    current_bubble_score: float,
    current_extension: float,
    current_gsr_zscore: float,
    lookback_years: int = 10,
    top_n: int = 5,
) -> list[dict]:
    """
    Find historical periods with similar bubble characteristics.
    
    Returns list of analog periods with:
    - date: Peak date
    - bubble_score: Bubble score at peak
    - extension_pct: Extension from 200MA at peak
    - gsr_zscore: GSR z-score at peak
    - days_to_reversion: Days until 20%+ drawdown
    - max_drawdown: Maximum drawdown in following 90 days
    - drawdown_30d: Drawdown after 30 days
    - drawdown_60d: Drawdown after 60 days
    """
    analogs = []
    
    if df.empty or "BUBBLE_SCORE" not in df.columns:
        return analogs
    
    # Find local peaks in bubble score (above 50)
    bubble = df["BUBBLE_SCORE"].fillna(0)
    slv = df["SLV"] if "SLV" in df.columns else None
    
    if slv is None:
        return analogs
    
    # Look for bubble peaks (local maxima above 50)
    window = 20  # 20-day window for local max
    is_peak = (
        (bubble > 50) & 
        (bubble == bubble.rolling(window, center=True).max()) &
        (bubble.shift(1) < bubble) &
        (bubble.shift(-1) < bubble)
    )
    
    peak_dates = df.index[is_peak].tolist()
    
    # For each peak, analyze what happened after
    for peak_date in peak_dates:
        try:
            peak_idx = df.index.get_loc(peak_date)
            
            # Skip if too recent (need 90 days of forward data)
            if peak_idx > len(df) - 90:
                continue
            
            peak_bubble = bubble.iloc[peak_idx]
            peak_extension = df["EXTENSION_PCT"].iloc[peak_idx] if "EXTENSION_PCT" in df.columns else 0
            peak_gsr_z = df["GSR_ZSCORE"].iloc[peak_idx] if "GSR_ZSCORE" in df.columns else 0
            peak_price = slv.iloc[peak_idx]
            
            # Calculate similarity to current state
            bubble_diff = abs(peak_bubble - current_bubble_score) / 100
            extension_diff = abs(peak_extension - current_extension) / 100
            gsr_diff = abs((peak_gsr_z or 0) - (current_gsr_zscore or 0)) / 5
            
            similarity = 1 - (bubble_diff * 0.4 + extension_diff * 0.3 + gsr_diff * 0.3)
            
            # Analyze forward returns
            forward_prices = slv.iloc[peak_idx:peak_idx + 91]
            if len(forward_prices) < 30:
                continue
            
            # Find drawdowns
            cummax = forward_prices.expanding().max()
            drawdowns = (forward_prices / cummax - 1) * 100
            
            max_drawdown = drawdowns.min()
            drawdown_30d = ((forward_prices.iloc[min(30, len(forward_prices)-1)] / peak_price) - 1) * 100
            drawdown_60d = ((forward_prices.iloc[min(60, len(forward_prices)-1)] / peak_price) - 1) * 100 if len(forward_prices) > 60 else None
            
            # Days to significant reversion (20%+ drop from peak)
            days_to_reversion = None
            for i, dd in enumerate(drawdowns):
                if dd <= -20:
                    days_to_reversion = i
                    break
            
            analogs.append({
                "date": peak_date.strftime("%Y-%m-%d") if hasattr(peak_date, "strftime") else str(peak_date),
                "bubble_score": round(peak_bubble, 1),
                "extension_pct": round(peak_extension, 1) if peak_extension else None,
                "gsr_zscore": round(peak_gsr_z, 2) if peak_gsr_z else None,
                "peak_price": round(peak_price, 2),
                "days_to_reversion": days_to_reversion,
                "max_drawdown_90d": round(max_drawdown, 1),
                "drawdown_30d": round(drawdown_30d, 1),
                "drawdown_60d": round(drawdown_60d, 1) if drawdown_60d else None,
                "similarity": round(similarity, 2),
            })
            
        except Exception:
            continue
    
    # Sort by similarity and return top N
    analogs = sorted(analogs, key=lambda x: x["similarity"], reverse=True)[:top_n]
    
    return analogs


def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _detect_momentum_divergence(df: pd.DataFrame, lookback: int = 20) -> bool:
    """
    Detect bearish momentum divergence.
    Price making higher highs but RSI making lower highs.
    """
    if "SLV" not in df.columns or len(df) < lookback + 14:
        return False
    
    slv = df["SLV"].iloc[-lookback:]
    rsi = _compute_rsi(df["SLV"], 14).iloc[-lookback:]
    
    # Check if price made new high in last 5 days
    recent_price = slv.iloc[-5:]
    prior_price = slv.iloc[:-5]
    price_new_high = recent_price.max() >= prior_price.max()
    
    # Check if RSI made lower high
    recent_rsi = rsi.iloc[-5:]
    prior_rsi = rsi.iloc[:-5]
    rsi_lower_high = recent_rsi.max() < prior_rsi.max()
    
    return price_new_high and rsi_lower_high


def _detect_volume_exhaustion(df: pd.DataFrame, lookback: int = 20) -> bool:
    """
    Detect volume exhaustion - declining volume on up days.
    Note: Using volatility as proxy since we don't have volume data.
    """
    if "SLV_VOL_20D_ANN_PCT" not in df.columns or len(df) < lookback:
        return False
    
    # Use declining volatility as proxy for exhaustion
    vol = df["SLV_VOL_20D_ANN_PCT"].iloc[-lookback:]
    vol_trend = vol.iloc[-5:].mean() < vol.iloc[:5].mean()
    
    # Also check if returns are slowing
    if "SLV_RET_5D_PCT" in df.columns and "SLV_RET_20D_PCT" in df.columns:
        ret_5d = df["SLV_RET_5D_PCT"].iloc[-1] or 0
        ret_20d = df["SLV_RET_20D_PCT"].iloc[-1] or 0
        # If 5d return is much less than proportional 20d, momentum slowing
        momentum_slowing = ret_5d < (ret_20d / 4) * 0.8 if ret_20d > 0 else False
        return vol_trend or momentum_slowing
    
    return vol_trend


def build_reversion_forecast(
    settings: Settings,
    state: SilverState,
    df: pd.DataFrame,
) -> ReversionForecast:
    """
    Build reversion forecast based on current state and historical patterns.
    """
    inp = state.inputs
    
    # Find historical analogs
    analogs = find_historical_analogs(
        df=df,
        current_bubble_score=inp.bubble_score or 0,
        current_extension=inp.extension_pct or 0,
        current_gsr_zscore=inp.gsr_zscore or 0,
        lookback_years=10,
        top_n=5,
    )
    
    # Compute timing estimates from analogs
    if analogs:
        reversions = [a["days_to_reversion"] for a in analogs if a["days_to_reversion"] is not None]
        drawdowns_90d = [a["max_drawdown_90d"] for a in analogs]
        drawdowns_30d = [a["drawdown_30d"] for a in analogs]
        drawdowns_60d = [a["drawdown_60d"] for a in analogs if a["drawdown_60d"] is not None]
        
        if reversions:
            days_low = int(np.percentile(reversions, 10))
            days_mid = int(np.percentile(reversions, 50))
            days_high = int(np.percentile(reversions, 90))
            
            # Probability estimates based on analog distribution
            prob_30d = sum(1 for r in reversions if r <= 30) / len(reversions) * 100
            prob_60d = sum(1 for r in reversions if r <= 60) / len(reversions) * 100
            prob_90d = sum(1 for r in reversions if r <= 90) / len(reversions) * 100
        else:
            # Default estimates based on bubble score
            bubble = inp.bubble_score or 50
            days_low = max(5, int(30 - bubble * 0.2))
            days_mid = max(15, int(60 - bubble * 0.3))
            days_high = max(30, int(120 - bubble * 0.5))
            prob_30d = min(80, bubble * 0.8)
            prob_60d = min(90, bubble * 0.9)
            prob_90d = min(95, bubble * 0.95)
        
        # Severity estimates
        if drawdowns_90d:
            dd_10 = np.percentile(drawdowns_90d, 10)
            dd_50 = np.percentile(drawdowns_90d, 50)
            dd_90 = np.percentile(drawdowns_90d, 90)
        else:
            # Default based on extension
            ext = abs(inp.extension_pct or 0)
            dd_10 = -ext * 0.3
            dd_50 = -ext * 0.5
            dd_90 = -ext * 0.8
    else:
        # No analogs - use heuristics based on current state
        bubble = inp.bubble_score or 50
        ext = abs(inp.extension_pct or 0)
        
        days_low = max(5, int(30 - bubble * 0.2))
        days_mid = max(15, int(60 - bubble * 0.3))
        days_high = max(30, int(120 - bubble * 0.5))
        
        prob_30d = min(80, bubble * 0.8)
        prob_60d = min(90, bubble * 0.9)
        prob_90d = min(95, bubble * 0.95)
        
        dd_10 = -ext * 0.3
        dd_50 = -ext * 0.5
        dd_90 = -ext * 0.8
    
    # Compute trigger levels
    trigger_50ma = inp.slv_ma_50
    trigger_gsr = (inp.gsr or 70) + 5  # GSR needs to rise 5+ points to signal gold outperformance
    
    # Detect leading indicators
    momentum_div = _detect_momentum_divergence(df) if not df.empty else False
    volume_exhaust = _detect_volume_exhaustion(df) if not df.empty else False
    
    # Adjust probabilities based on leading indicators
    if momentum_div:
        prob_30d = min(100, prob_30d + 15)
        prob_60d = min(100, prob_60d + 10)
    if volume_exhaust:
        prob_30d = min(100, prob_30d + 10)
        prob_60d = min(100, prob_60d + 5)
    
    # Compute confidence based on analog quality and data availability
    if analogs:
        avg_similarity = np.mean([a["similarity"] for a in analogs])
        confidence = avg_similarity * 100 * 0.7 + 30  # Base 30 + up to 70 from similarity
    else:
        confidence = 40  # Low confidence without analogs
    
    # Adjust confidence based on leading indicators
    if momentum_div:
        confidence = min(100, confidence + 10)
    if volume_exhaust:
        confidence = min(100, confidence + 5)
    
    # Target levels
    fair_value = inp.slv_ma_200
    target_25 = fair_value * 1.25 if fair_value else None
    target_50 = fair_value * 1.5 if fair_value else None
    
    return ReversionForecast(
        days_to_reversion_low=days_low,
        days_to_reversion_mid=days_mid,
        days_to_reversion_high=days_high,
        reversion_probability_30d=round(prob_30d, 1),
        reversion_probability_60d=round(prob_60d, 1),
        reversion_probability_90d=round(prob_90d, 1),
        drawdown_10th_pct=round(dd_10, 1),
        drawdown_median=round(dd_50, 1),
        drawdown_90th_pct=round(dd_90, 1),
        target_fair_value=round(fair_value, 2) if fair_value else None,
        target_25pct_premium=round(target_25, 2) if target_25 else None,
        target_50pct_premium=round(target_50, 2) if target_50 else None,
        trigger_50ma=round(trigger_50ma, 2) if trigger_50ma else None,
        trigger_gsr_expansion=round(trigger_gsr, 1) if trigger_gsr else None,
        trigger_volume_capitulation=volume_exhaust,
        analog_dates=[a["date"] for a in analogs],
        analog_outcomes=analogs,
        forecast_confidence=round(confidence, 1),
    )


def get_forecast_summary(forecast: ReversionForecast, inputs: SilverInputs) -> dict:
    """
    Get human-readable forecast summary.
    """
    summary = {
        "timing": {
            "best_case": f"{forecast.days_to_reversion_low} days",
            "likely_case": f"{forecast.days_to_reversion_mid} days",
            "worst_case": f"{forecast.days_to_reversion_high} days",
        },
        "probability": {
            "30d": f"{forecast.reversion_probability_30d}%",
            "60d": f"{forecast.reversion_probability_60d}%",
            "90d": f"{forecast.reversion_probability_90d}%",
        },
        "severity": {
            "mild": f"{forecast.drawdown_10th_pct}%",
            "expected": f"{forecast.drawdown_median}%",
            "severe": f"{forecast.drawdown_90th_pct}%",
        },
        "confidence": f"{forecast.forecast_confidence}%",
        "analogs_count": len(forecast.analog_outcomes),
    }
    
    # Add put-specific guidance
    current_price = inputs.slv_price or 0
    if current_price and forecast.target_fair_value:
        profit_at_fair = ((current_price - forecast.target_fair_value) / current_price) * 100
        summary["put_profit_potential"] = {
            "to_fair_value": f"{profit_at_fair:.1f}%",
            "to_25pct_premium": f"{((current_price - forecast.target_25pct_premium) / current_price) * 100:.1f}%" if forecast.target_25pct_premium else None,
        }
    
    return summary

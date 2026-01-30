"""
Silver data fetching and state building.

Uses SLV and GLD ETF data from Alpaca (or market data provider).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from ai_options_trader.config import Settings
from ai_options_trader.silver.models import SilverInputs, SilverState


# Price multiplier to convert GLD/SLV ratio to approximate gold/silver ratio
# GLD tracks ~1/10 oz gold, SLV tracks ~1 oz silver
# Actual GSR = (gold_price_per_oz) / (silver_price_per_oz)
# We estimate: GSR ≈ (GLD * 10) / SLV
GSR_ADJUSTMENT_FACTOR = 10.0


def build_silver_dataset(
    settings: Settings,
    start_date: str = "2011-01-01",
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Build silver analysis dataset with SLV, GLD, and derived features.
    
    Returns DataFrame with daily data including:
    - SLV and GLD prices
    - Returns at various windows
    - Moving averages
    - Gold/Silver ratio
    - Z-scores
    - Volatility measures
    """
    from ai_options_trader.data.market import fetch_equity_daily_closes

    # Fetch price data
    tickers = ["SLV", "GLD", "SPY"]  # SPY for correlation
    try:
        px = fetch_equity_daily_closes(
            settings=settings,
            symbols=tickers,
            start=start_date,
            refresh=refresh,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to fetch silver data: {e}")

    if px is None or px.empty:
        raise RuntimeError("No price data returned for silver analysis")

    # Ensure we have SLV
    if "SLV" not in px.columns:
        raise RuntimeError("SLV data not available")

    df = px.copy()
    df = df.sort_index().ffill()

    # SLV price series
    slv = pd.to_numeric(df["SLV"], errors="coerce")
    gld = pd.to_numeric(df["GLD"], errors="coerce") if "GLD" in df.columns else None
    spy = pd.to_numeric(df["SPY"], errors="coerce") if "SPY" in df.columns else None

    # Moving averages
    df["SLV_MA_50"] = slv.rolling(50, min_periods=30).mean()
    df["SLV_MA_200"] = slv.rolling(200, min_periods=100).mean()

    # Returns
    df["SLV_RET_5D_PCT"] = (slv / slv.shift(5) - 1) * 100
    df["SLV_RET_20D_PCT"] = (slv / slv.shift(20) - 1) * 100
    df["SLV_RET_60D_PCT"] = (slv / slv.shift(60) - 1) * 100
    df["SLV_RET_200D_PCT"] = (slv / slv.shift(200) - 1) * 100

    # Z-scores (vs 3-year rolling history)
    win = 252 * 3
    df["SLV_ZSCORE_20D"] = _zscore(df["SLV_RET_20D_PCT"], window=win)
    df["SLV_ZSCORE_60D"] = _zscore(df["SLV_RET_60D_PCT"], window=win)

    # MA positioning
    df["SLV_ABOVE_50MA"] = slv > df["SLV_MA_50"]
    df["SLV_ABOVE_200MA"] = slv > df["SLV_MA_200"]
    df["SLV_50MA_ABOVE_200MA"] = df["SLV_MA_50"] > df["SLV_MA_200"]
    df["SLV_PCT_FROM_50MA"] = (slv / df["SLV_MA_50"] - 1) * 100
    df["SLV_PCT_FROM_200MA"] = (slv / df["SLV_MA_200"] - 1) * 100

    # Gold/Silver Ratio (GSR)
    if gld is not None:
        df["GSR"] = (gld * GSR_ADJUSTMENT_FACTOR) / slv
        df["GSR_20D_AVG"] = df["GSR"].rolling(20, min_periods=10).mean()
        df["GSR_ZSCORE"] = _zscore(df["GSR"], window=win)
        df["GSR_EXPANDING"] = df["GSR"] > df["GSR"].shift(20)

    # Volatility (annualized)
    daily_ret = slv.pct_change()
    df["SLV_VOL_20D_ANN_PCT"] = daily_ret.rolling(20, min_periods=10).std() * np.sqrt(252) * 100
    df["SLV_VOL_60D_ANN_PCT"] = daily_ret.rolling(60, min_periods=30).std() * np.sqrt(252) * 100
    df["SLV_VOL_ZSCORE"] = _zscore(df["SLV_VOL_20D_ANN_PCT"], window=win)

    # Volume (if available - using price as proxy for now)
    # Note: Volume would need separate API call; using volatility ratio as proxy
    df["SLV_VOLUME_RATIO"] = df["SLV_VOL_20D_ANN_PCT"] / df["SLV_VOL_60D_ANN_PCT"]

    # Correlations
    if spy is not None:
        spy_ret = spy.pct_change()
        df["SLV_SPY_CORR_60D"] = daily_ret.rolling(60, min_periods=30).corr(spy_ret)

    # Composite scores
    df["TREND_SCORE"] = _compute_trend_score(df, slv)
    df["MOMENTUM_SCORE"] = _compute_momentum_score(df)
    if gld is not None:
        df["RELATIVE_VALUE_SCORE"] = _compute_relative_value_score(df)

    # Bubble / Mean Reversion metrics
    df["BUBBLE_SCORE"] = _compute_bubble_score(df, slv)
    df["MEAN_REVERSION_PRESSURE"] = _compute_mean_reversion_pressure(df, slv)
    df["EXTENSION_PCT"] = _compute_extension_pct(df, slv)
    df["DAYS_AT_EXTREME"] = _compute_days_at_extreme(df)

    # RSI and momentum divergence
    df["RSI_14"] = _compute_rsi(slv, 14)
    df["MOMENTUM_DIVERGENCE"] = _detect_momentum_divergence_series(df, slv)
    df["VOLUME_EXHAUSTION"] = _detect_volume_exhaustion_series(df)

    # Technical breakdown levels
    breakdown_levels = _compute_breakdown_levels(df, slv)
    for key, val in breakdown_levels.items():
        df[key] = val

    return df


def _zscore(series: pd.Series, window: int = 756) -> pd.Series:
    """Compute rolling z-score."""
    mean = series.rolling(window, min_periods=60).mean()
    std = series.rolling(window, min_periods=60).std()
    return (series - mean) / std.replace(0, np.nan)


def _compute_trend_score(df: pd.DataFrame, slv: pd.Series) -> pd.Series:
    """
    Compute trend score from -100 to +100.
    
    Components:
    - MA positioning (above/below 50/200)
    - Golden/death cross
    - Distance from MAs
    - Return momentum
    """
    score = pd.Series(0.0, index=df.index)

    # Above/below 50-day MA (+/-20)
    score = score.where(~df["SLV_ABOVE_50MA"], score + 20)
    score = score.where(df["SLV_ABOVE_50MA"], score - 20)

    # Above/below 200-day MA (+/-25)
    score = score.where(~df["SLV_ABOVE_200MA"], score + 25)
    score = score.where(df["SLV_ABOVE_200MA"], score - 25)

    # Golden/death cross (+/-15)
    score = score.where(~df["SLV_50MA_ABOVE_200MA"], score + 15)
    score = score.where(df["SLV_50MA_ABOVE_200MA"], score - 15)

    # 20-day return z-score contribution (+/-20)
    z20 = df["SLV_ZSCORE_20D"].clip(-2, 2) * 10
    score = score + z20.fillna(0)

    # 60-day return z-score contribution (+/-20)
    z60 = df["SLV_ZSCORE_60D"].clip(-2, 2) * 10
    score = score + z60.fillna(0)

    return score.clip(-100, 100)


def _compute_momentum_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute momentum score from -100 to +100.
    
    Components:
    - Short-term return (5d, 20d)
    - Acceleration (20d vs 60d)
    - Volatility regime
    """
    score = pd.Series(0.0, index=df.index)

    # 5-day return contribution (+/-30)
    ret_5d = df["SLV_RET_5D_PCT"].clip(-15, 15) * 2
    score = score + ret_5d.fillna(0)

    # 20-day return contribution (+/-30)
    ret_20d = df["SLV_RET_20D_PCT"].clip(-30, 30)
    score = score + ret_20d.fillna(0)

    # Acceleration: 20d > 60d/3 means accelerating (+/-20)
    accel = (df["SLV_RET_20D_PCT"] - df["SLV_RET_60D_PCT"] / 3).clip(-20, 20)
    score = score + accel.fillna(0)

    # High volatility dampens confidence (+/-20)
    vol_penalty = (df["SLV_VOL_ZSCORE"].clip(-2, 2) * -10).fillna(0)
    score = score + vol_penalty

    return score.clip(-100, 100)


def _compute_relative_value_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute relative value score from -100 (silver rich) to +100 (silver cheap).
    
    Based on Gold/Silver Ratio:
    - GSR > 80: Silver historically cheap (+50 to +100)
    - GSR 70-80: Silver moderately cheap (+25 to +50)
    - GSR 60-70: Fair value (-25 to +25)
    - GSR 50-60: Silver moderately rich (-50 to -25)
    - GSR < 50: Silver historically rich (-100 to -50)
    """
    gsr = df["GSR"]
    score = pd.Series(0.0, index=df.index)

    # Base score from absolute GSR level
    score = (gsr - 70) * 2.5  # 70 = neutral, +/-25 per 10 points

    # Z-score adjustment (how extreme vs history)
    gsr_z = df["GSR_ZSCORE"].clip(-2, 2) * 15
    score = score + gsr_z.fillna(0)

    return score.clip(-100, 100)


def _compute_bubble_score(df: pd.DataFrame, slv: pd.Series) -> pd.Series:
    """
    Compute bubble intensity score from 0 (normal) to 100 (extreme bubble).
    
    Components (each contributes 0-20 points):
    1. Return z-score extremity (20d)
    2. Volatility z-score extremity
    3. Distance from 200-day MA
    4. GSR extremity (if available)
    5. Acceleration (rate of change of returns)
    """
    score = pd.Series(0.0, index=df.index)
    
    # 1. Return z-score extremity (0-20 points)
    # Bubble = extreme positive returns
    z20 = df["SLV_ZSCORE_20D"].fillna(0)
    ret_component = (z20.clip(0, 4) / 4) * 20  # 0 at z=0, 20 at z>=4
    score = score + ret_component
    
    # 2. Volatility z-score (0-20 points)
    # High vol = unstable, bubble-like
    vol_z = df["SLV_VOL_ZSCORE"].fillna(0)
    vol_component = (vol_z.clip(0, 4) / 4) * 20
    score = score + vol_component
    
    # 3. Distance from 200-day MA (0-20 points)
    # >50% above 200MA = extreme
    pct_from_200 = df["SLV_PCT_FROM_200MA"].fillna(0)
    ma_component = (pct_from_200.clip(0, 100) / 100) * 20
    score = score + ma_component
    
    # 4. GSR extremity (0-20 points)
    # Low GSR = silver rich = bubble territory
    if "GSR_ZSCORE" in df.columns:
        gsr_z = df["GSR_ZSCORE"].fillna(0)
        # Negative GSR z-score = silver outperforming = bubble
        gsr_component = ((-gsr_z).clip(0, 4) / 4) * 20
        score = score + gsr_component
    
    # 5. Acceleration - 5d return vs 20d return rate (0-20 points)
    ret_5d = df["SLV_RET_5D_PCT"].fillna(0)
    ret_20d = df["SLV_RET_20D_PCT"].fillna(0)
    # If 5d return is >25% of 20d return, accelerating
    accel = np.where(ret_20d > 0, ret_5d / (ret_20d / 4 + 0.01), 0)
    accel_series = pd.Series(accel, index=df.index)
    accel_component = (accel_series.clip(0, 2) / 2) * 20
    score = score + accel_component
    
    return score.clip(0, 100)


def _compute_mean_reversion_pressure(df: pd.DataFrame, slv: pd.Series) -> pd.Series:
    """
    Compute mean reversion pressure from 0 (low) to 100 (high).
    
    Higher = more likely to revert. Based on:
    1. How extended the bubble score is
    2. Duration at extreme levels
    3. Momentum deceleration signals
    4. Historical percentile of current price vs range
    """
    score = pd.Series(0.0, index=df.index)
    
    # 1. Bubble score contribution (0-30)
    bubble = df["BUBBLE_SCORE"] if "BUBBLE_SCORE" in df.columns else _compute_bubble_score(df, slv)
    score = score + (bubble / 100) * 30
    
    # 2. Percentile vs 1-year range (0-25)
    rolling_min = slv.rolling(252, min_periods=60).min()
    rolling_max = slv.rolling(252, min_periods=60).max()
    pct_rank = (slv - rolling_min) / (rolling_max - rolling_min + 0.01)
    # If at top of range, high reversion pressure
    pct_component = pct_rank.fillna(0.5) * 25
    score = score + pct_component
    
    # 3. RSI-like overbought (0-25)
    # Simplified: use 20d return z-score as proxy
    z20 = df["SLV_ZSCORE_20D"].fillna(0)
    rsi_proxy = (z20.clip(0, 3) / 3) * 25
    score = score + rsi_proxy
    
    # 4. GSR mean reversion (0-20)
    # When GSR is extremely low, pressure to normalize
    if "GSR_ZSCORE" in df.columns:
        gsr_z = df["GSR_ZSCORE"].fillna(0)
        gsr_reversion = ((-gsr_z).clip(0, 4) / 4) * 20
        score = score + gsr_reversion
    
    return score.clip(0, 100)


def _compute_extension_pct(df: pd.DataFrame, slv: pd.Series) -> pd.Series:
    """
    Compute how extended price is vs estimated fair value (%).
    
    Fair value estimate based on:
    - 200-day MA as baseline
    - Adjusted for typical premium/discount range
    """
    ma_200 = df["SLV_MA_200"]
    
    # Fair value = 200MA (could add GSR adjustment later)
    fair_value = ma_200
    
    extension = ((slv / fair_value) - 1) * 100
    
    return extension.fillna(0)


def _compute_days_at_extreme(df: pd.DataFrame) -> pd.Series:
    """
    Count consecutive days with bubble score > 50.
    """
    bubble = df["BUBBLE_SCORE"] if "BUBBLE_SCORE" in df.columns else pd.Series(0, index=df.index)
    is_extreme = bubble > 50
    
    # Count consecutive True values
    groups = (~is_extreme).cumsum()
    days = is_extreme.groupby(groups).cumsum()
    
    return days.astype(int)


def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _detect_momentum_divergence_series(df: pd.DataFrame, slv: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Detect bearish momentum divergence at each point in time.
    Price making higher highs but RSI making lower highs.
    """
    result = pd.Series(False, index=df.index)
    
    if len(df) < lookback + 14:
        return result
    
    rsi = _compute_rsi(slv, 14)
    
    for i in range(lookback + 14, len(df)):
        window_price = slv.iloc[i-lookback:i+1]
        window_rsi = rsi.iloc[i-lookback:i+1]
        
        if window_rsi.isna().all():
            continue
        
        # Check if price made new high in last 5 days
        recent_price = window_price.iloc[-5:]
        prior_price = window_price.iloc[:-5]
        price_new_high = recent_price.max() >= prior_price.max() * 0.99  # Within 1%
        
        # Check if RSI made lower high
        recent_rsi = window_rsi.iloc[-5:]
        prior_rsi = window_rsi.iloc[:-5]
        
        if prior_rsi.isna().all() or recent_rsi.isna().all():
            continue
            
        rsi_lower_high = recent_rsi.max() < prior_rsi.max() - 2  # At least 2 points lower
        
        result.iloc[i] = price_new_high and rsi_lower_high
    
    return result


def _detect_volume_exhaustion_series(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Detect volume exhaustion at each point.
    Using volatility and momentum as proxies.
    """
    result = pd.Series(False, index=df.index)
    
    if "SLV_VOL_20D_ANN_PCT" not in df.columns:
        return result
    
    vol = df["SLV_VOL_20D_ANN_PCT"]
    ret_5d = df.get("SLV_RET_5D_PCT", pd.Series(0, index=df.index))
    ret_20d = df.get("SLV_RET_20D_PCT", pd.Series(0, index=df.index))
    
    for i in range(lookback, len(df)):
        window_vol = vol.iloc[i-lookback:i+1]
        
        if window_vol.isna().all():
            continue
        
        # Declining volatility
        vol_trend = window_vol.iloc[-5:].mean() < window_vol.iloc[:5].mean()
        
        # Momentum slowing
        r5 = ret_5d.iloc[i] if not pd.isna(ret_5d.iloc[i]) else 0
        r20 = ret_20d.iloc[i] if not pd.isna(ret_20d.iloc[i]) else 0
        momentum_slowing = r5 < (r20 / 4) * 0.8 if r20 > 0 else False
        
        result.iloc[i] = vol_trend or momentum_slowing
    
    return result


def _compute_breakdown_levels(df: pd.DataFrame, slv: pd.Series) -> dict:
    """
    Compute key technical breakdown levels that would trigger selling pressure.
    
    Returns dict of Series for each level type.
    """
    levels = {}
    
    # Moving averages (already computed, but include for completeness)
    levels["BREAKDOWN_10MA"] = slv.rolling(10, min_periods=5).mean()
    levels["BREAKDOWN_20MA"] = slv.rolling(20, min_periods=10).mean()
    
    # Recent swing low (lowest low in last 10 days)
    levels["BREAKDOWN_SWING_LOW_10D"] = slv.rolling(10, min_periods=5).min()
    
    # Recent swing low (lowest low in last 20 days)
    levels["BREAKDOWN_SWING_LOW_20D"] = slv.rolling(20, min_periods=10).min()
    
    # Fibonacci retracements from 60-day rally
    # Find the low 60 days ago and compute fibs from there
    def compute_fibs(row_idx):
        if row_idx < 60:
            return {}
        
        window = slv.iloc[max(0, row_idx-60):row_idx+1]
        low = window.min()
        high = window.max()
        rally = high - low
        
        if rally <= 0:
            return {}
        
        return {
            "fib_236": high - (rally * 0.236),
            "fib_382": high - (rally * 0.382),
            "fib_500": high - (rally * 0.500),
            "fib_618": high - (rally * 0.618),
            "fib_786": high - (rally * 0.786),
        }
    
    # Compute fibs for each row (use last row for current levels)
    fib_236 = []
    fib_382 = []
    fib_500 = []
    fib_618 = []
    fib_786 = []
    
    for i in range(len(slv)):
        fibs = compute_fibs(i)
        fib_236.append(fibs.get("fib_236"))
        fib_382.append(fibs.get("fib_382"))
        fib_500.append(fibs.get("fib_500"))
        fib_618.append(fibs.get("fib_618"))
        fib_786.append(fibs.get("fib_786"))
    
    levels["BREAKDOWN_FIB_236"] = pd.Series(fib_236, index=df.index)
    levels["BREAKDOWN_FIB_382"] = pd.Series(fib_382, index=df.index)
    levels["BREAKDOWN_FIB_500"] = pd.Series(fib_500, index=df.index)
    levels["BREAKDOWN_FIB_618"] = pd.Series(fib_618, index=df.index)
    levels["BREAKDOWN_FIB_786"] = pd.Series(fib_786, index=df.index)
    
    # Round number support (nearest $5 and $10 below current price)
    def round_down_5(x):
        return np.floor(x / 5) * 5 if pd.notna(x) else np.nan
    
    def round_down_10(x):
        return np.floor(x / 10) * 10 if pd.notna(x) else np.nan
    
    levels["BREAKDOWN_ROUND_5"] = slv.apply(round_down_5)
    levels["BREAKDOWN_ROUND_10"] = slv.apply(round_down_10)
    
    # Previous week's low
    levels["BREAKDOWN_PREV_WEEK_LOW"] = slv.rolling(5, min_periods=3).min().shift(1)
    
    # Keltner channel lower band (20-day MA - 2 * ATR)
    # Using high-low range as ATR proxy
    daily_range = slv.diff().abs()
    atr_20 = daily_range.rolling(20, min_periods=10).mean()
    ma_20 = slv.rolling(20, min_periods=10).mean()
    levels["BREAKDOWN_KELTNER_LOWER"] = ma_20 - (2 * atr_20 * np.sqrt(252) / 100)
    
    return levels


def get_breakdown_levels_summary(df: pd.DataFrame, current_price: float) -> list[dict]:
    """
    Get a sorted list of breakdown levels with their significance.
    
    Returns list of dicts with:
    - level: price level
    - name: description
    - pct_away: percentage below current price
    - significance: 1-5 (5 = most significant)
    """
    if df.empty:
        return []
    
    latest = df.iloc[-1]
    levels = []
    
    # Helper to add level if valid
    def add_level(name: str, col: str, significance: int, description: str):
        val = latest.get(col)
        if val is not None and pd.notna(val) and val < current_price:
            pct = ((val - current_price) / current_price) * 100
            levels.append({
                "level": round(val, 2),
                "name": name,
                "pct_away": round(pct, 1),
                "significance": significance,
                "description": description,
            })
    
    # Moving averages
    add_level("10-day MA", "BREAKDOWN_10MA", 3, "Short-term trend break")
    add_level("20-day MA", "BREAKDOWN_20MA", 4, "Momentum traders exit")
    add_level("50-day MA", "SLV_MA_50", 5, "MAJOR: Trend followers sell")
    add_level("200-day MA", "SLV_MA_200", 5, "MAJOR: Long-term trend break")
    
    # Swing lows
    add_level("10-day Swing Low", "BREAKDOWN_SWING_LOW_10D", 4, "Stop-loss cluster trigger")
    add_level("20-day Swing Low", "BREAKDOWN_SWING_LOW_20D", 5, "MAJOR: Significant support break")
    
    # Fibonacci levels
    add_level("Fib 23.6%", "BREAKDOWN_FIB_236", 2, "Shallow retracement")
    add_level("Fib 38.2%", "BREAKDOWN_FIB_382", 3, "Normal retracement")
    add_level("Fib 50.0%", "BREAKDOWN_FIB_500", 4, "Key retracement level")
    add_level("Fib 61.8%", "BREAKDOWN_FIB_618", 4, "Golden ratio - major support")
    add_level("Fib 78.6%", "BREAKDOWN_FIB_786", 3, "Deep retracement")
    
    # Round numbers (compute dynamically)
    round_5 = np.floor(current_price / 5) * 5
    round_10 = np.floor(current_price / 10) * 10
    
    if round_5 < current_price:
        pct = ((round_5 - current_price) / current_price) * 100
        levels.append({
            "level": round_5,
            "name": f"${round_5:.0f} Round",
            "pct_away": round(pct, 1),
            "significance": 3,
            "description": "Psychological support",
        })
    
    if round_10 < current_price and round_10 != round_5:
        pct = ((round_10 - current_price) / current_price) * 100
        levels.append({
            "level": round_10,
            "name": f"${round_10:.0f} Round",
            "pct_away": round(pct, 1),
            "significance": 4,
            "description": "Major psychological level",
        })
    
    # Previous week low
    add_level("Previous Week Low", "BREAKDOWN_PREV_WEEK_LOW", 3, "Short-term support")
    
    # Sort by distance from current price (closest first)
    levels = sorted(levels, key=lambda x: x["pct_away"], reverse=True)
    
    return levels


def build_silver_state(
    settings: Settings,
    start_date: str = "2011-01-01",
    refresh: bool = False,
) -> SilverState:
    """
    Build current silver state snapshot.
    """
    df = build_silver_dataset(settings=settings, start_date=start_date, refresh=refresh)

    if df.empty:
        raise RuntimeError("No silver data available")

    # Get latest row
    latest = df.iloc[-1]
    asof = latest.name.date() if hasattr(latest.name, "date") else date.today()

    # Build inputs
    inputs = SilverInputs(
        slv_price=_safe_float(latest.get("SLV")),
        slv_ma_50=_safe_float(latest.get("SLV_MA_50")),
        slv_ma_200=_safe_float(latest.get("SLV_MA_200")),
        slv_ret_5d_pct=_safe_float(latest.get("SLV_RET_5D_PCT")),
        slv_ret_20d_pct=_safe_float(latest.get("SLV_RET_20D_PCT")),
        slv_ret_60d_pct=_safe_float(latest.get("SLV_RET_60D_PCT")),
        slv_ret_200d_pct=_safe_float(latest.get("SLV_RET_200D_PCT")),
        slv_zscore_20d=_safe_float(latest.get("SLV_ZSCORE_20D")),
        slv_zscore_60d=_safe_float(latest.get("SLV_ZSCORE_60D")),
        slv_above_50ma=_safe_bool(latest.get("SLV_ABOVE_50MA")),
        slv_above_200ma=_safe_bool(latest.get("SLV_ABOVE_200MA")),
        slv_50ma_above_200ma=_safe_bool(latest.get("SLV_50MA_ABOVE_200MA")),
        slv_pct_from_50ma=_safe_float(latest.get("SLV_PCT_FROM_50MA")),
        slv_pct_from_200ma=_safe_float(latest.get("SLV_PCT_FROM_200MA")),
        gsr=_safe_float(latest.get("GSR")),
        gsr_20d_avg=_safe_float(latest.get("GSR_20D_AVG")),
        gsr_zscore=_safe_float(latest.get("GSR_ZSCORE")),
        gsr_expanding=_safe_bool(latest.get("GSR_EXPANDING")),
        slv_vol_20d_ann_pct=_safe_float(latest.get("SLV_VOL_20D_ANN_PCT")),
        slv_vol_60d_ann_pct=_safe_float(latest.get("SLV_VOL_60D_ANN_PCT")),
        slv_vol_zscore=_safe_float(latest.get("SLV_VOL_ZSCORE")),
        slv_volume_ratio=_safe_float(latest.get("SLV_VOLUME_RATIO")),
        slv_spy_corr_60d=_safe_float(latest.get("SLV_SPY_CORR_60D")),
        trend_score=_safe_float(latest.get("TREND_SCORE")),
        momentum_score=_safe_float(latest.get("MOMENTUM_SCORE")),
        relative_value_score=_safe_float(latest.get("RELATIVE_VALUE_SCORE")),
        bubble_score=_safe_float(latest.get("BUBBLE_SCORE")),
        mean_reversion_pressure=_safe_float(latest.get("MEAN_REVERSION_PRESSURE")),
        extension_pct=_safe_float(latest.get("EXTENSION_PCT")),
        days_at_extreme=int(latest.get("DAYS_AT_EXTREME", 0)) if latest.get("DAYS_AT_EXTREME") is not None else None,
        rsi_14=_safe_float(latest.get("RSI_14")),
        momentum_divergence=_safe_bool(latest.get("MOMENTUM_DIVERGENCE")),
        volume_exhaustion=_safe_bool(latest.get("VOLUME_EXHAUSTION")),
    )

    # Build notes
    notes = _build_notes(inputs)

    return SilverState(
        asof=asof,
        start_date=date.fromisoformat(start_date),
        inputs=inputs,
        notes=notes,
    )


def _safe_float(val) -> Optional[float]:
    """Safely convert to float, returning None for NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else round(f, 4)
    except (ValueError, TypeError):
        return None


def _safe_bool(val) -> Optional[bool]:
    """Safely convert to bool."""
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    return None


def _build_notes(inputs: SilverInputs) -> list[str]:
    """Build contextual notes based on inputs."""
    notes = []

    # Price context
    if inputs.slv_price:
        notes.append(f"SLV at ${inputs.slv_price:.2f}")

    # MA positioning
    if inputs.slv_above_200ma is True:
        notes.append("Above 200-day MA (bullish)")
    elif inputs.slv_above_200ma is False:
        notes.append("Below 200-day MA (bearish)")

    if inputs.slv_50ma_above_200ma is True:
        notes.append("Golden cross active")
    elif inputs.slv_50ma_above_200ma is False:
        notes.append("Death cross active")

    # GSR context
    if inputs.gsr is not None:
        if inputs.gsr > 85:
            notes.append(f"GSR at {inputs.gsr:.1f} - silver historically cheap vs gold")
        elif inputs.gsr > 75:
            notes.append(f"GSR at {inputs.gsr:.1f} - silver moderately cheap vs gold")
        elif inputs.gsr < 55:
            notes.append(f"GSR at {inputs.gsr:.1f} - silver historically rich vs gold")
        else:
            notes.append(f"GSR at {inputs.gsr:.1f} - fair value range")

    # Trend score
    if inputs.trend_score is not None:
        if inputs.trend_score > 50:
            notes.append(f"Strong uptrend (score: {inputs.trend_score:.0f})")
        elif inputs.trend_score < -50:
            notes.append(f"Strong downtrend (score: {inputs.trend_score:.0f})")

    # Momentum
    if inputs.momentum_score is not None:
        if inputs.momentum_score > 50:
            notes.append(f"Bullish momentum (score: {inputs.momentum_score:.0f})")
        elif inputs.momentum_score < -50:
            notes.append(f"Bearish momentum (score: {inputs.momentum_score:.0f})")

    return notes

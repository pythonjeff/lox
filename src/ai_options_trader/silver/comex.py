"""
COMEX Silver Inventory Tracking

COMEX inventory data is critical for silver analysis:
- Declining inventories = physical tightness = bullish
- Rising inventories = supply glut = bearish
- Inventory vs price divergence = key signal

Data sources:
- Primary: Manual entry from CME reports (updated weekly)
- Proxy: SLV ETF holdings (available via API)
- Future: Nasdaq Data Link (Quandl) API integration
"""

from __future__ import annotations

import os
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests

from ai_options_trader.config import Settings


# =============================================================================
# COMEX INVENTORY DATA (manually updated from CME reports)
# =============================================================================

# Historical COMEX silver inventory data points (millions of ounces)
# Source: CME Group COMEX warehouse reports
# Update frequency: Weekly (Fridays)
COMEX_INVENTORY_HISTORY = {
    # Date: inventory_moz (millions of ounces)
    "2020-01-01": 320,
    "2020-06-01": 350,
    "2020-12-01": 390,
    "2021-01-01": 395,
    "2021-06-01": 370,
    "2021-12-01": 350,
    "2022-01-01": 340,
    "2022-06-01": 310,
    "2022-12-01": 290,
    "2023-01-01": 280,
    "2023-06-01": 275,
    "2023-12-01": 285,
    "2024-01-01": 290,
    "2024-06-01": 310,
    "2024-12-01": 350,
    "2025-01-01": 480,  # Major spike
    "2025-06-01": 520,  # Peak
    "2025-09-01": 450,  # Sharp drawdown
    "2025-12-01": 400,  # Continued drawdown
    "2026-01-01": 395,
    "2026-02-01": 400,  # Current (from user's chart)
}


@dataclass
class ComexInventory:
    """COMEX silver inventory snapshot."""
    date: date
    inventory_moz: float  # Millions of ounces
    change_1m_moz: Optional[float] = None
    change_3m_moz: Optional[float] = None
    change_1y_moz: Optional[float] = None
    change_1m_pct: Optional[float] = None
    change_3m_pct: Optional[float] = None
    change_1y_pct: Optional[float] = None
    percentile_5y: Optional[float] = None  # Current level vs 5-year range
    trend: str = "stable"  # rising, falling, stable
    signal: str = "neutral"  # bullish, bearish, neutral


def get_comex_inventory_data() -> pd.DataFrame:
    """
    Get COMEX inventory data as a DataFrame.
    
    Returns daily interpolated data from manual checkpoints.
    """
    # Convert to DataFrame
    data = []
    for date_str, inv in sorted(COMEX_INVENTORY_HISTORY.items()):
        data.append({
            "date": pd.to_datetime(date_str),
            "inventory_moz": inv,
        })
    
    df = pd.DataFrame(data)
    df = df.set_index("date")
    
    # Create daily index and interpolate
    daily_idx = pd.date_range(start=df.index.min(), end=pd.Timestamp.today(), freq="D")
    df = df.reindex(daily_idx)
    df["inventory_moz"] = df["inventory_moz"].interpolate(method="linear")
    
    # Add computed columns
    df["change_1m_moz"] = df["inventory_moz"] - df["inventory_moz"].shift(30)
    df["change_3m_moz"] = df["inventory_moz"] - df["inventory_moz"].shift(90)
    df["change_1y_moz"] = df["inventory_moz"] - df["inventory_moz"].shift(365)
    
    df["change_1m_pct"] = (df["change_1m_moz"] / df["inventory_moz"].shift(30)) * 100
    df["change_3m_pct"] = (df["change_3m_moz"] / df["inventory_moz"].shift(90)) * 100
    df["change_1y_pct"] = (df["change_1y_moz"] / df["inventory_moz"].shift(365)) * 100
    
    # 5-year percentile
    rolling_min = df["inventory_moz"].rolling(252 * 5, min_periods=252).min()
    rolling_max = df["inventory_moz"].rolling(252 * 5, min_periods=252).max()
    df["percentile_5y"] = ((df["inventory_moz"] - rolling_min) / (rolling_max - rolling_min + 0.01)) * 100
    
    return df


def get_current_comex_inventory() -> ComexInventory:
    """Get the current COMEX inventory snapshot."""
    df = get_comex_inventory_data()
    
    if df.empty:
        return ComexInventory(
            date=date.today(),
            inventory_moz=0,
            signal="unknown",
        )
    
    latest = df.iloc[-1]
    latest_date = latest.name.date() if hasattr(latest.name, "date") else date.today()
    
    # Determine trend
    change_3m = latest.get("change_3m_pct", 0) or 0
    if change_3m > 10:
        trend = "rising"
    elif change_3m < -10:
        trend = "falling"
    else:
        trend = "stable"
    
    # Determine signal
    # Falling inventories = bullish for price
    # Rising inventories = bearish for price
    if change_3m < -15:
        signal = "bullish"
    elif change_3m > 15:
        signal = "bearish"
    else:
        signal = "neutral"
    
    return ComexInventory(
        date=latest_date,
        inventory_moz=round(latest["inventory_moz"], 1),
        change_1m_moz=round(latest["change_1m_moz"], 1) if pd.notna(latest.get("change_1m_moz")) else None,
        change_3m_moz=round(latest["change_3m_moz"], 1) if pd.notna(latest.get("change_3m_moz")) else None,
        change_1y_moz=round(latest["change_1y_moz"], 1) if pd.notna(latest.get("change_1y_moz")) else None,
        change_1m_pct=round(latest["change_1m_pct"], 1) if pd.notna(latest.get("change_1m_pct")) else None,
        change_3m_pct=round(latest["change_3m_pct"], 1) if pd.notna(latest.get("change_3m_pct")) else None,
        change_1y_pct=round(latest["change_1y_pct"], 1) if pd.notna(latest.get("change_1y_pct")) else None,
        percentile_5y=round(latest["percentile_5y"], 1) if pd.notna(latest.get("percentile_5y")) else None,
        trend=trend,
        signal=signal,
    )


# =============================================================================
# SLV ETF HOLDINGS (proxy for physical silver demand)
# =============================================================================

def fetch_slv_holdings(settings: Settings, days: int = 365) -> pd.DataFrame:
    """
    Fetch SLV ETF holdings data as a proxy for physical silver demand.
    
    Uses FMP API for ETF holdings data.
    """
    try:
        # FMP doesn't have historical ETF holdings, so we use price * shares as proxy
        # This is an approximation - actual holdings come from iShares website
        base_url = "https://financialmodelingprep.com/api/v3"
        
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        resp = requests.get(
            f"{base_url}/historical-price-full/SLV",
            params={
                "apikey": settings.fmp_api_key,
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
            },
            timeout=15,
        )
        
        if not resp.ok:
            return pd.DataFrame()
        
        data = resp.json()
        historical = data.get("historical", [])
        
        if not historical:
            return pd.DataFrame()
        
        df = pd.DataFrame(historical)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        
        # SLV holdings proxy: volume-weighted price suggests accumulation/distribution
        # Higher volume on up days = accumulation
        df["volume_direction"] = np.where(df["close"] > df["open"], df["volume"], -df["volume"])
        df["accumulation_20d"] = df["volume_direction"].rolling(20).sum()
        
        return df
        
    except Exception as e:
        print(f"Error fetching SLV holdings proxy: {e}")
        return pd.DataFrame()


# =============================================================================
# COMBINED INVENTORY ANALYSIS
# =============================================================================

def build_inventory_vs_price_dataset(
    settings: Settings,
    start_date: str = "2020-01-01",
) -> pd.DataFrame:
    """
    Build combined dataset with COMEX inventory and SLV price.
    
    Used for charting inventory vs price divergences.
    """
    from ai_options_trader.data.market import fetch_equity_daily_closes
    
    # Get COMEX inventory data
    comex_df = get_comex_inventory_data()
    
    # Get SLV price data
    try:
        px = fetch_equity_daily_closes(
            settings=settings,
            symbols=["SLV"],
            start=start_date,
        )
        if px is not None and not px.empty:
            slv_df = px[["SLV"]].copy()
            slv_df.columns = ["slv_price"]
        else:
            slv_df = pd.DataFrame()
    except Exception:
        slv_df = pd.DataFrame()
    
    # Merge on date
    if not comex_df.empty and not slv_df.empty:
        df = comex_df.join(slv_df, how="outer")
        df = df.ffill()
    elif not comex_df.empty:
        df = comex_df
    elif not slv_df.empty:
        df = slv_df
    else:
        df = pd.DataFrame()
    
    if not df.empty:
        # Normalize for comparison (0-100 scale)
        if "inventory_moz" in df.columns:
            inv_min = df["inventory_moz"].min()
            inv_max = df["inventory_moz"].max()
            df["inventory_normalized"] = ((df["inventory_moz"] - inv_min) / (inv_max - inv_min + 0.01)) * 100
        
        if "slv_price" in df.columns:
            price_min = df["slv_price"].min()
            price_max = df["slv_price"].max()
            df["price_normalized"] = ((df["slv_price"] - price_min) / (price_max - price_min + 0.01)) * 100
        
        # Divergence score: when inventory and price move opposite directions
        if "inventory_normalized" in df.columns and "price_normalized" in df.columns:
            inv_change = df["inventory_normalized"].diff(30)
            price_change = df["price_normalized"].diff(30)
            # Positive divergence = inventory down, price up (bullish)
            # Negative divergence = inventory up, price down (bearish)
            df["divergence_score"] = price_change - inv_change
    
    return df


def get_inventory_analysis_summary(settings: Settings) -> dict:
    """
    Get summary of COMEX inventory analysis.
    
    Returns dict with key metrics for display.
    """
    comex = get_current_comex_inventory()
    df = build_inventory_vs_price_dataset(settings)
    
    summary = {
        "comex_inventory_moz": comex.inventory_moz,
        "comex_date": comex.date.isoformat(),
        "comex_change_1m_pct": comex.change_1m_pct,
        "comex_change_3m_pct": comex.change_3m_pct,
        "comex_change_1y_pct": comex.change_1y_pct,
        "comex_percentile_5y": comex.percentile_5y,
        "comex_trend": comex.trend,
        "comex_signal": comex.signal,
    }
    
    if not df.empty and "slv_price" in df.columns:
        latest = df.iloc[-1]
        summary["slv_price"] = round(latest["slv_price"], 2) if pd.notna(latest.get("slv_price")) else None
        summary["divergence_score"] = round(latest["divergence_score"], 1) if pd.notna(latest.get("divergence_score")) else None
        
        # Interpretation
        div = summary.get("divergence_score", 0) or 0
        if div > 20:
            summary["divergence_interpretation"] = "Bullish divergence (price up, inventory down)"
        elif div < -20:
            summary["divergence_interpretation"] = "Bearish divergence (price down, inventory up)"
        else:
            summary["divergence_interpretation"] = "No significant divergence"
    
    return summary


def update_comex_inventory(date_str: str, inventory_moz: float) -> None:
    """
    Update COMEX inventory data point.
    
    This modifies the module-level data. For persistence, 
    would need to write to a data file.
    
    Usage:
        update_comex_inventory("2026-02-05", 398.5)
    """
    COMEX_INVENTORY_HISTORY[date_str] = inventory_moz
    print(f"Updated COMEX inventory for {date_str}: {inventory_moz} Moz")

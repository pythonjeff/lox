"""
Silver / SLV Regime Tracker

Tracks silver market conditions using SLV ETF and related metrics.

Key Inputs:
- SLV ETF price and returns
- Gold/Silver ratio (GSR) via GLD/SLV
- Moving average positioning
- Volume trends
- Correlation with VIX/risk assets

Regime States:
- silver_rally: Strong uptrend, above key MAs
- silver_breakdown: Breaking support, trending lower
- silver_squeeze: Rapid short covering rally
- silver_consolidation: Range-bound, low volatility
- silver_capitulation: High volume selloff

GSR (Gold/Silver Ratio) Context:
- GSR > 80: Silver historically cheap vs gold
- GSR 70-80: Silver moderately cheap
- GSR 60-70: Fair value range
- GSR < 60: Silver rich vs gold
"""

from .models import SilverInputs, SilverState, ReversionForecast
from .regime import SilverRegime, classify_silver_regime, SILVER_REGIME_CHOICES
from .signals import build_silver_state, build_silver_dataset
from .forecast import build_reversion_forecast, find_historical_analogs

__all__ = [
    "SilverInputs",
    "SilverState",
    "ReversionForecast",
    "SilverRegime",
    "classify_silver_regime",
    "SILVER_REGIME_CHOICES",
    "build_silver_state",
    "build_silver_dataset",
    "build_reversion_forecast",
    "find_historical_analogs",
]

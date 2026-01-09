"""
DEPRECATED: use `ai_options_trader.funding.signals`.

This module is retained for backwards compatibility.
"""

from ai_options_trader.funding.signals import FUNDING_FRED_SERIES as LIQUIDITY_FRED_SERIES
from ai_options_trader.funding.signals import build_funding_dataset as build_liquidity_dataset
from ai_options_trader.funding.signals import build_funding_state as build_liquidity_state

__all__ = ["LIQUIDITY_FRED_SERIES", "build_liquidity_dataset", "build_liquidity_state"]



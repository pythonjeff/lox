"""
DEPRECATED: `ai_options_trader.liquidity` is now `ai_options_trader.funding`.

This compatibility package exists to reduce churn while callers migrate.
"""

from ai_options_trader.funding.features import funding_feature_vector as liquidity_feature_vector
from ai_options_trader.funding.models import FundingInputs as LiquidityInputs
from ai_options_trader.funding.models import FundingState as LiquidityState
from ai_options_trader.funding.signals import build_funding_dataset as build_liquidity_dataset
from ai_options_trader.funding.signals import build_funding_state as build_liquidity_state

__all__ = [
    "LiquidityInputs",
    "LiquidityState",
    "build_liquidity_dataset",
    "build_liquidity_state",
    "liquidity_feature_vector",
]



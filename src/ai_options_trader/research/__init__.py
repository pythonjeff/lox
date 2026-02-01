"""
Research Module - Comprehensive ticker research with HF-grade metrics.
"""

from ai_options_trader.research.ticker_report import (
    TickerResearchReport,
    build_ticker_research_report,
)
from ai_options_trader.research.momentum import (
    MomentumMetrics,
    calculate_momentum_metrics,
    is_oversold,
    is_overbought,
)
from ai_options_trader.research.hf_metrics import (
    HedgeFundMetrics,
    calculate_hf_metrics,
)
from ai_options_trader.research.ideas import (
    TradeIdea,
    IdeasReport,
    generate_trade_ideas,
    DEFAULT_SCAN_UNIVERSE,
)

__all__ = [
    "TickerResearchReport",
    "build_ticker_research_report",
    "MomentumMetrics",
    "calculate_momentum_metrics",
    "is_oversold",
    "is_overbought",
    "HedgeFundMetrics",
    "calculate_hf_metrics",
    "TradeIdea",
    "IdeasReport",
    "generate_trade_ideas",
    "DEFAULT_SCAN_UNIVERSE",
]

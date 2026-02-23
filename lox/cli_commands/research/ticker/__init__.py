"""Ticker analysis subpackage — data, compute, display, chart, LLM modules."""
from __future__ import annotations

from lox.cli_commands.research.ticker.data import (
    fetch_price_data,
    fetch_fundamentals,
    fetch_atm_implied_vol,
    fetch_peers,
)
from lox.cli_commands.research.ticker.compute import (
    compute_technicals,
    compute_refinancing_wall,
    compute_flow_context,
)
from lox.cli_commands.research.ticker.display import (
    show_price_panel,
    show_fundamentals,
    show_key_risks_summary,
    show_peer_comparison,
    show_etf_flows,
    show_refinancing_wall,
    show_technicals,
)
from lox.cli_commands.research.ticker.chart import (
    generate_chart,
    open_chart,
)
from lox.cli_commands.research.ticker.llm import (
    show_llm_analysis,
)

__all__ = [
    "fetch_price_data",
    "fetch_fundamentals",
    "fetch_atm_implied_vol",
    "fetch_peers",
    "compute_technicals",
    "compute_refinancing_wall",
    "compute_flow_context",
    "show_price_panel",
    "show_fundamentals",
    "show_key_risks_summary",
    "show_peer_comparison",
    "show_etf_flows",
    "show_refinancing_wall",
    "show_technicals",
    "generate_chart",
    "open_chart",
    "show_llm_analysis",
]

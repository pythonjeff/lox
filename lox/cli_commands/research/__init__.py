"""
LOX Research Module

Clean, focused research commands for market analysis.

Commands:
    lox research regimes   - Unified regime view with LLM commentary
    lox research ticker    - Hedge fund level ticker analysis  
    lox research portfolio - LLM outlook on open positions
"""
from __future__ import annotations

import typer

research_app = typer.Typer(
    name="research",
    help="LOX Research - Market analysis and research tools",
    add_completion=False,
)


def register_research_commands(app: typer.Typer) -> None:
    """Register all research commands."""
    from lox.cli_commands.research.regimes_cmd import register as register_regimes
    from lox.cli_commands.research.ticker_cmd import register as register_ticker
    from lox.cli_commands.research.portfolio_cmd import register as register_portfolio
    from lox.cli_commands.research.chat_cmd import register as register_chat
    from lox.cli_commands.research.oi_scan_cmd import register as register_oi_scan
    from lox.cli_commands.research.cvna_cmd import register as register_cvna
    
    register_regimes(research_app)
    register_ticker(research_app)
    register_portfolio(research_app)
    register_chat(research_app)
    register_oi_scan(research_app)
    register_cvna(research_app)
    
    app.add_typer(research_app, name="research")

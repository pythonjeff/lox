"""
LOX Research: Ticker Command

Hedge fund level single-stock analysis with visualizations.

Usage:
    lox research ticker AAPL          # Full analysis
    lox research ticker NVDA --chart  # Generate price chart
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from lox.config import load_settings
from lox.cli_commands.research.ticker.data import (
    fetch_price_data,
    fetch_fundamentals,
    fetch_atm_implied_vol,
)
from lox.cli_commands.research.ticker.compute import compute_technicals
from lox.cli_commands.research.ticker.display import (
    show_price_panel,
    show_fundamentals,
    show_key_risks_summary,
    show_peer_comparison,
    show_etf_flows,
    show_refinancing_wall,
    show_technicals,
    show_hy_default_context,
    _HY_ETF_TICKERS,
)
from lox.cli_commands.research.ticker.chart import generate_chart, open_chart
from lox.cli_commands.research.ticker.llm import show_llm_analysis


def register(app: typer.Typer) -> None:
    """Register the ticker command."""

    @app.command("ticker")
    def ticker_cmd(
        symbol: str = typer.Argument(..., help="Ticker symbol (e.g., AAPL, NVDA, SPY)"),
        chart: bool = typer.Option(True, "--chart/--no-chart", help="Generate price chart"),
        llm: bool = typer.Option(False, "--llm/--no-llm", help="Include LLM analysis (off by default)"),
    ):
        """
        Hedge fund level ticker analysis.

        Includes:
        - Price chart with key levels (matplotlib)
        - Fundamentals snapshot (P/E, revenue, margins)
        - Technical levels (support/resistance, RSI, MAs)
        - LLM synthesis with bull/bear case

        Examples:
            lox research ticker AAPL
            lox research ticker NVDA --no-chart
            lox research ticker SPY --no-llm
        """
        console = Console()
        settings = load_settings()

        symbol = symbol.upper()
        console.print()
        console.print(f"[bold cyan]LOX RESEARCH[/bold cyan]  [bold]{symbol}[/bold]")
        console.print()

        # Fetch all data
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Fetching data...[/bold cyan]"),
            transient=True,
        ) as progress:
            progress.add_task("fetch", total=None)

            price_data = fetch_price_data(settings, symbol)
            fundamentals = fetch_fundamentals(settings, symbol)
            technicals = compute_technicals(price_data)

        # 1. Price box
        if price_data:
            iv = fetch_atm_implied_vol(settings, symbol, technicals.get("current") if technicals else None)
            show_price_panel(console, symbol, price_data, technicals or {}, implied_vol=iv)

        # 2. Key Risks Summary (auto-generated, before fundamentals)
        if fundamentals or technicals:
            show_key_risks_summary(console, symbol, fundamentals or {}, technicals or {})

        # 3. Fundamentals
        if fundamentals:
            show_fundamentals(console, fundamentals, technicals or {}, price_data or {})

        # 4. Peer Comparison (stocks only, data-driven)
        is_etf = fundamentals.get("profile", {}).get("isEtf", False) or fundamentals.get("etf_info")
        if not is_etf and fundamentals:
            show_peer_comparison(console, settings, symbol, fundamentals)

        # 5. Technical levels
        if technicals:
            show_technicals(console, technicals)

        # ETF Flow analysis (only for ETFs)
        if is_etf and price_data.get("historical"):
            show_etf_flows(console, price_data, fundamentals)

        # Bond ETF: Refinancing wall + HY credit stress
        if is_etf:
            asset_class = (fundamentals.get("etf_info", {}).get("assetClass") or "").lower()
            description = (fundamentals.get("profile", {}).get("description") or "").lower()
            is_bond_etf = "fixed income" in asset_class or "bond" in description or "credit" in description
            if is_bond_etf:
                show_refinancing_wall(console, settings, symbol)
            # HY credit stress panel — for HY ETFs or any bond ETF with HY in name/desc
            is_hy = symbol in _HY_ETF_TICKERS or "high yield" in description or "high-yield" in description
            if is_hy and settings.FRED_API_KEY:
                show_hy_default_context(console, settings, symbol)

        # Generate chart
        if chart and price_data:
            chart_path = generate_chart(symbol, price_data, technicals)
            if chart_path:
                console.print(f"\n[green]Chart saved:[/green] {chart_path}")
                open_chart(chart_path)

        # LLM Analysis
        if llm:
            show_llm_analysis(console, settings, symbol, price_data, fundamentals, technicals)

        console.print()

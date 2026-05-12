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
    fetch_earnings_data,
    fetch_futures_depth,
    FUTURES_ETF_MAP,
)
from lox.cli_commands.research.ticker.compute import (
    compute_technicals,
    compute_earnings_outlook,
    compute_flow_context,
    detect_stacked_signals,
)
from lox.cli_commands.research.ticker.display import (
    show_price_panel,
    show_fundamentals,
    show_key_risks_summary,
    show_peer_comparison,
    show_earnings_outlook,
    show_etf_flows,
    show_futures_depth,
    show_refinancing_wall,
    show_stacked_signals,
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
        chart: bool = typer.Option(False, "--chart/--no-chart", help="Generate price chart"),
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
        iv = None
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
        flow_context = None
        if is_etf and price_data.get("historical"):
            show_etf_flows(console, price_data, fundamentals)
            flow_context = compute_flow_context(price_data)

        # E-mini Futures Depth (index ETFs with corresponding CME futures)
        futures_data = None
        if symbol in FUTURES_ETF_MAP:
            profile = fundamentals.get("profile", {})
            etf_price = (price_data.get("quote", {}).get("price")
                         or profile.get("price"))
            last_div = profile.get("lastDiv")
            div_yield_pct = None
            if last_div and etf_price:
                try:
                    div_yield_pct = float(last_div) / float(etf_price) * 100
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
            futures_data = fetch_futures_depth(
                settings, symbol, etf_price, div_yield_pct,
                technicals=technicals,
            )
            if futures_data:
                show_futures_depth(console, futures_data)

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

        # Earnings outlook (stocks only — ETFs don't report earnings)
        earnings_data = None
        outlook = None
        if not is_etf and price_data:
            earnings_data = fetch_earnings_data(settings, symbol)
            if earnings_data:
                spot = (price_data.get("quote", {}).get("price")
                        or (technicals or {}).get("current"))
                # HV decimalized for the fallback path (technicals stores it as percent)
                hv_pct = (technicals or {}).get("volatility_30d") or (technicals or {}).get("volatility")
                hv_decimal = hv_pct / 100.0 if isinstance(hv_pct, (int, float)) else None
                outlook = compute_earnings_outlook(
                    earnings_data=earnings_data,
                    price_data=price_data,
                    spot=spot,
                    implied_vol=iv,
                    realized_vol_fallback=hv_decimal,
                )
                show_earnings_outlook(
                    console,
                    symbol,
                    outlook,
                    ratings_consensus=earnings_data.get("ratings_consensus"),
                    price_target=earnings_data.get("price_target"),
                    current_price=spot,
                )

        # ── Stacked Signals — synthesis across all blocks ────────────────
        current_price = (price_data.get("quote", {}).get("price") if price_data else None) \
            or (technicals or {}).get("current")
        stacks = detect_stacked_signals(
            symbol=symbol,
            is_etf=bool(is_etf),
            technicals=technicals,
            fundamentals=fundamentals,
            earnings_outlook=outlook,
            ratings_consensus=(earnings_data or {}).get("ratings_consensus"),
            price_target=(earnings_data or {}).get("price_target"),
            futures_data=futures_data,
            flow_context=flow_context,
            iv=iv,
            current_price=current_price,
        )
        show_stacked_signals(console, stacks)

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

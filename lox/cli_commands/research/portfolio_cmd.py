"""
LOX Research: Portfolio Command

LLM-powered outlook on each open position.

Usage:
    lox research portfolio           # Review all positions
    lox research portfolio --brief   # Summary only
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from lox.config import load_settings


def register(app: typer.Typer) -> None:
    """Register the portfolio command."""
    
    @app.command("portfolio")
    def portfolio_cmd(
        brief: bool = typer.Option(False, "--brief", "-b", help="Summary table only, no detailed analysis"),
    ):
        """
        LLM outlook on each open position.
        
        For each position, the AI analyst provides:
        - Current thesis assessment
        - Bull/bear/neutral outlook
        - Key risks
        - Suggested action (hold/trim/add)
        
        Examples:
            lox research portfolio         # Full analysis
            lox research portfolio --brief # Quick summary
        """
        console = Console()
        settings = load_settings()
        
        console.print()
        console.print("[bold cyan]LOX RESEARCH[/bold cyan]  [bold]Portfolio Outlook[/bold]")
        console.print()
        
        # Fetch positions
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Fetching positions...[/bold cyan]"),
            transient=True,
        ) as progress:
            progress.add_task("fetch", total=None)
            positions = _fetch_positions(settings)
        
        if not positions:
            console.print("[yellow]No open positions found.[/yellow]")
            return
        
        # Summary table
        _show_positions_table(console, positions)
        
        if brief:
            console.print("\n[dim]Run without --brief for detailed AI analysis of each position.[/dim]")
            return
        
        # LLM analysis for each position
        console.print()
        _analyze_positions(console, settings, positions)


def _fetch_positions(settings) -> list[dict]:
    """Fetch current positions from Alpaca."""
    try:
        from lox.data.alpaca import make_clients
        
        trading, _data = make_clients(settings)
        raw_positions = trading.get_all_positions()
        
        positions = []
        for pos in raw_positions:
            # Handle both dict and object forms
            if isinstance(pos, dict):
                symbol = pos.get("symbol", "")
                qty = float(pos.get("qty", 0))
                current_price = float(pos.get("current_price", 0))
                avg_entry_price = float(pos.get("avg_entry_price", 0))
                market_value = float(pos.get("market_value", 0))
                unrealized_pl = float(pos.get("unrealized_pl", 0))
                unrealized_plpc = float(pos.get("unrealized_plpc", 0)) * 100
                asset_class = pos.get("asset_class", "us_equity")
            else:
                symbol = getattr(pos, "symbol", "")
                qty = float(getattr(pos, "qty", 0) or 0)
                current_price = float(getattr(pos, "current_price", 0) or 0)
                avg_entry_price = float(getattr(pos, "avg_entry_price", 0) or 0)
                market_value = float(getattr(pos, "market_value", 0) or 0)
                unrealized_pl = float(getattr(pos, "unrealized_pl", 0) or 0)
                unrealized_plpc = float(getattr(pos, "unrealized_plpc", 0) or 0) * 100
                asset_class = str(getattr(pos, "asset_class", "us_equity") or "us_equity")
            
            positions.append({
                "symbol": symbol,
                "qty": qty,
                "current_price": current_price,
                "avg_entry_price": avg_entry_price,
                "market_value": market_value,
                "unrealized_pl": unrealized_pl,
                "unrealized_plpc": unrealized_plpc,
                "asset_class": asset_class,
            })
        
        # Sort by market value
        positions.sort(key=lambda x: abs(x["market_value"]), reverse=True)
        return positions
    
    except Exception as e:
        # Print error for debugging
        import sys
        print(f"[red]Error fetching positions: {e}[/red]", file=sys.stderr)
        return []


def _show_positions_table(console: Console, positions: list[dict]):
    """Show summary table of positions."""
    table = Table(
        title="[bold]Open Positions[/bold]",
        box=None,
        padding=(0, 2),
        show_header=True,
        header_style="bold",
    )
    table.add_column("Symbol", style="bold")
    table.add_column("Qty", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("P/L", justify="right")
    table.add_column("P/L %", justify="right")
    table.add_column("Value", justify="right")
    
    total_pl = 0
    total_value = 0
    
    for pos in positions:
        pl_color = "green" if pos["unrealized_pl"] >= 0 else "red"
        
        # Format symbol (trim options symbols)
        symbol = pos["symbol"]
        if len(symbol) > 12:
            # Options symbol - show abbreviated
            display_symbol = symbol[:10] + ".."
        else:
            display_symbol = symbol
        
        table.add_row(
            display_symbol,
            f"{pos['qty']:.0f}" if pos['qty'] == int(pos['qty']) else f"{pos['qty']:.2f}",
            f"${pos['avg_entry_price']:,.2f}",
            f"${pos['current_price']:,.2f}",
            f"[{pl_color}]${pos['unrealized_pl']:+,.2f}[/{pl_color}]",
            f"[{pl_color}]{pos['unrealized_plpc']:+.1f}%[/{pl_color}]",
            f"${pos['market_value']:,.2f}",
        )
        
        total_pl += pos["unrealized_pl"]
        total_value += pos["market_value"]
    
    console.print(table)
    
    # Summary
    pl_color = "green" if total_pl >= 0 else "red"
    console.print()
    console.print(f"[bold]Total Value:[/bold] ${total_value:,.2f}  |  [bold]Unrealized P/L:[/bold] [{pl_color}]${total_pl:+,.2f}[/{pl_color}]")


def _analyze_positions(console: Console, settings, positions: list[dict]):
    """Run LLM analysis on each position."""
    from rich.markdown import Markdown
    
    for i, pos in enumerate(positions):
        symbol = pos["symbol"]
        
        # Progress indicator
        console.print(f"\n[bold cyan]Analyzing {symbol}[/bold cyan] ({i+1}/{len(positions)})")
        
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[dim]Researching {symbol}...[/dim]"),
            transient=True,
        ) as progress:
            progress.add_task("analyze", total=None)
            analysis = _analyze_single_position(settings, pos)
        
        if analysis:
            # Parse outlook from analysis
            outlook, action = _extract_outlook_action(analysis)
            
            # Color coding
            outlook_colors = {
                "bullish": "green",
                "bearish": "red",
                "neutral": "yellow",
            }
            action_colors = {
                "hold": "blue",
                "add": "green",
                "trim": "yellow",
                "close": "red",
            }
            
            outlook_color = outlook_colors.get(outlook.lower(), "white")
            action_color = action_colors.get(action.lower(), "white")
            
            # Header with outlook/action
            pl_color = "green" if pos["unrealized_pl"] >= 0 else "red"
            header = f"""[bold]{symbol}[/bold]  |  P/L: [{pl_color}]${pos['unrealized_pl']:+,.2f} ({pos['unrealized_plpc']:+.1f}%)[/{pl_color}]
Outlook: [{outlook_color}]{outlook.upper()}[/{outlook_color}]  |  Action: [{action_color}]{action.upper()}[/{action_color}]"""
            
            console.print(Panel(
                Markdown(analysis),
                title=header,
                border_style=outlook_color,
            ))
        else:
            console.print(f"[dim]Analysis unavailable for {symbol}[/dim]")


def _analyze_single_position(settings, position: dict) -> str | None:
    """Get LLM analysis for a single position."""
    try:
        if not settings.openai_api_key:
            return None
        
        from openai import OpenAI
        import json
        
        client = OpenAI(api_key=settings.openai_api_key)
        
        symbol = position["symbol"]
        is_option = len(symbol) > 10  # Options have long symbols
        
        # Fetch additional context
        context = _get_position_context(settings, symbol, is_option)
        
        payload = {
            "position": position,
            "is_option": is_option,
            "context": context,
        }
        
        prompt = f"""You are a senior portfolio analyst reviewing a trading position.

POSITION DATA:
{json.dumps(payload, indent=2, default=str)}

Provide a CONCISE analysis (150-200 words max):

1. **THESIS**: What's the likely thesis for this position? (1-2 sentences)

2. **OUTLOOK**: BULLISH / BEARISH / NEUTRAL with 1-sentence rationale

3. **KEY RISK**: The single biggest risk to this position

4. **ACTION**: HOLD / ADD / TRIM / CLOSE with brief rationale

Be direct and actionable. Reference specific data points."""

        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        
        return (resp.choices[0].message.content or "").strip()
    
    except Exception as e:
        return None


def _get_position_context(settings, symbol: str, is_option: bool) -> dict:
    """Get additional context for position analysis."""
    context = {}
    
    try:
        import requests
        
        if is_option:
            # Extract underlying from options symbol
            # Options format: AAPL250117C00200000 (SYMBOL + YYMMDD + C/P + STRIKE)
            underlying = ""
            for i, c in enumerate(symbol):
                if c.isdigit():
                    underlying = symbol[:i]
                    break
            if underlying:
                symbol = underlying
        
        if not settings.fmp_api_key:
            return context
        
        # Quick quote
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
        resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=10)
        if resp.ok:
            data = resp.json()
            if data and isinstance(data, list):
                quote = data[0]
                context["quote"] = {
                    "price": quote.get("price"),
                    "change_pct": quote.get("changesPercentage"),
                    "52w_high": quote.get("yearHigh"),
                    "52w_low": quote.get("yearLow"),
                }
        
        # Recent news count
        try:
            url = f"https://financialmodelingprep.com/api/v3/stock_news"
            resp = requests.get(url, params={"tickers": symbol, "limit": 5, "apikey": settings.fmp_api_key}, timeout=10)
            if resp.ok:
                news = resp.json()
                if news:
                    context["recent_news"] = [
                        {"title": n.get("title", "")[:80], "date": n.get("publishedDate", "")[:10]}
                        for n in news[:3]
                    ]
        except Exception:
            pass
    
    except Exception:
        pass
    
    return context


def _extract_outlook_action(analysis: str) -> tuple[str, str]:
    """Extract outlook and action from analysis text."""
    outlook = "neutral"
    action = "hold"
    
    analysis_upper = analysis.upper()
    
    # Extract outlook
    if "BULLISH" in analysis_upper:
        outlook = "bullish"
    elif "BEARISH" in analysis_upper:
        outlook = "bearish"
    
    # Extract action
    if "CLOSE" in analysis_upper and "ACTION" in analysis_upper:
        action = "close"
    elif "TRIM" in analysis_upper:
        action = "trim"
    elif "ADD" in analysis_upper and "ACTION" in analysis_upper:
        action = "add"
    
    return outlook, action

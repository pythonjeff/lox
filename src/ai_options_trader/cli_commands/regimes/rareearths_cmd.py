"""
Rare Earths & Critical Minerals CLI Command

Provides analysis of rare earth miners, processors, and the critical minerals supply chain.
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def register(app: typer.Typer) -> None:
    """Register rare earth commands."""
    
    @app.command("rareearths")
    def rareearths_cmd(
        basket: str = typer.Option("all", "--basket", "-b", 
            help="all|pure_play|us_focused|lithium|etfs|defense_chain|ev_chain"),
        thesis: bool = typer.Option(False, "--thesis", "-t", help="Show investment thesis summary"),
        llm: bool = typer.Option(False, "--llm", help="Get LLM analysis of the sector"),
    ):
        """
        Rare Earths & Critical Minerals Tracker.
        
        Tracks the rare earth supply chain - miners, processors, and end users.
        Key themes: China dominance, EV/defense demand, supply chain de-risking.
        
        Baskets:
            all          - Complete universe
            pure_play    - Pure rare earth miners (MP, Lynas, REMX)
            us_focused   - US supply chain (MP, UUUU, LAC)
            lithium      - Lithium producers (ALB, SQM, LAC, LIT)
            etfs         - Diversified ETFs (REMX, LIT)
            defense_chain - Defense supply chain (MP, Lynas, LMT, RTX)
            ev_chain     - EV materials supply chain
        
        Examples:
            lox labs rareearths                  # Full overview
            lox labs rareearths -b pure_play     # Pure RE miners only
            lox labs rareearths -b us_focused    # US supply chain
            lox labs rareearths --thesis         # Investment thesis
            lox labs rareearths --llm            # AI analysis
        """
        from ai_options_trader.config import load_settings
        from ai_options_trader.rareearths.tracker import (
            build_rareearth_report,
            RE_BASKETS,
            RE_MARKET_CONTEXT,
            get_thesis_summary,
        )
        
        console = Console()
        settings = load_settings()
        
        # Show thesis summary if requested
        if thesis:
            _show_thesis(console)
            return
        
        basket_info = RE_BASKETS.get(basket, RE_BASKETS["all"])
        
        console.print(f"\n[bold cyan]RARE EARTHS & CRITICAL MINERALS TRACKER[/bold cyan]")
        console.print(f"[dim]{basket_info['name']}: {basket_info['description']}[/dim]\n")
        
        console.print("[dim]Fetching market data...[/dim]")
        report = build_rareearth_report(settings, basket)
        
        # Market context panel
        ctx = RE_MARKET_CONTEXT
        context_lines = [
            f"[bold]China Dominance:[/bold]",
            f"  Mining: {ctx['china_dominance']['mining_pct']}% | Processing: {ctx['china_dominance']['processing_pct']}% | Magnets: {ctx['china_dominance']['magnets_pct']}%",
            f"  Trend: {ctx['china_dominance']['trend']}",
            "",
            f"[bold]Key Demand Drivers:[/bold]",
        ]
        for driver in ctx["demand_drivers"][:3]:
            context_lines.append(f"  • {driver}")
        
        console.print(Panel("\n".join(context_lines), title="Market Context", border_style="cyan", expand=False))
        
        # Main securities table
        table = Table(title=f"Rare Earth Securities ({len(report.securities)} tickers)", expand=False)
        table.add_column("Ticker", style="bold")
        table.add_column("Name", width=20)
        table.add_column("Category", style="cyan")
        table.add_column("RE%", justify="right")
        table.add_column("China", justify="center")
        table.add_column("EV", justify="center")
        table.add_column("Defense", justify="center")
        table.add_column("Price", justify="right")
        table.add_column("Today", justify="right")
        table.add_column("Mkt Cap", justify="right")
        
        exposure_colors = {
            "extreme": "red",
            "high": "orange1",
            "medium": "yellow",
            "low": "green",
            "none": "dim",
            "concern": "red",
            "unknown": "dim",
        }
        
        for sec in report.securities:
            china_color = exposure_colors.get(sec.china_exposure, "white")
            ev_color = exposure_colors.get(sec.ev_exposure, "white")
            def_color = exposure_colors.get(sec.defense_exposure, "white")
            chg_color = "green" if sec.change_1d > 0 else "red" if sec.change_1d < 0 else "white"
            
            table.add_row(
                sec.ticker,
                sec.name[:20],
                sec.category,
                f"{sec.re_revenue_pct}%" if sec.re_revenue_pct > 0 else "—",
                f"[{china_color}]{sec.china_exposure.upper()[:4]}[/{china_color}]",
                f"[{ev_color}]{sec.ev_exposure.upper()[:4]}[/{ev_color}]",
                f"[{def_color}]{sec.defense_exposure.upper()[:4]}[/{def_color}]",
                f"${sec.price:.2f}" if sec.price else "—",
                f"[{chg_color}]{sec.change_1d:+.2f}%[/{chg_color}]" if sec.price else "—",
                f"${sec.market_cap_b:.1f}B" if sec.market_cap_b else "—",
            )
        
        console.print(table)
        
        # Signals panel
        if report.bull_signals or report.bear_signals:
            signal_lines = []
            if report.bull_signals:
                signal_lines.append("[bold green]BULL SIGNALS[/bold green]")
                for s in report.bull_signals:
                    signal_lines.append(f"  [green]▲[/green] {s}")
            if report.bear_signals:
                if signal_lines:
                    signal_lines.append("")
                signal_lines.append("[bold red]BEAR SIGNALS[/bold red]")
                for s in report.bear_signals:
                    signal_lines.append(f"  [red]▼[/red] {s}")
            
            console.print()
            console.print(Panel("\n".join(signal_lines), title="Signals", expand=False))
        
        # Summary panel
        basket_color = "green" if report.basket_change_1d > 0 else "red"
        summary_lines = [
            f"[bold]Basket Today:[/bold] [{basket_color}]{report.basket_change_1d:+.2f}%[/{basket_color}]",
            f"[bold]Total Market Cap:[/bold] ${report.total_market_cap_b:,.1f}B",
        ]
        console.print()
        console.print(Panel("\n".join(summary_lines), title="Summary", expand=False))
        
        # Available baskets
        basket_lines = ["[bold]Available Baskets[/bold] (--basket <name>)", ""]
        for bname, binfo in RE_BASKETS.items():
            marker = " ← current" if bname == basket else ""
            basket_lines.append(f"  [cyan]{bname}[/cyan]: {binfo['description'][:45]}{marker}")
        
        console.print()
        console.print(Panel("\n".join(basket_lines), title="Baskets", expand=False))
        
        # LLM analysis if requested
        if llm:
            _show_llm_analysis(console, settings, report, basket_info)


def _show_thesis(console: Console) -> None:
    """Display the rare earths investment thesis."""
    from ai_options_trader.rareearths.tracker import get_thesis_summary
    
    thesis = get_thesis_summary()
    
    console.print("\n[bold cyan]RARE EARTHS INVESTMENT THESIS[/bold cyan]\n")
    
    # Bull case
    bull_lines = ["[bold green]BULL CASE[/bold green]", ""]
    for point in thesis["bull_case"]:
        bull_lines.append(f"  [green]▲[/green] {point}")
    console.print(Panel("\n".join(bull_lines), expand=False))
    
    # Bear case
    bear_lines = ["[bold red]BEAR CASE[/bold red]", ""]
    for point in thesis["bear_case"]:
        bear_lines.append(f"  [red]▼[/red] {point}")
    console.print(Panel("\n".join(bear_lines), expand=False))
    
    # Key catalysts
    cat_lines = ["[bold yellow]KEY CATALYSTS TO WATCH[/bold yellow]", ""]
    for cat in thesis["key_catalysts"]:
        cat_lines.append(f"  [yellow]⚡[/yellow] {cat}")
    console.print(Panel("\n".join(cat_lines), expand=False))
    
    # Positioning
    pos = thesis["positioning"]
    pos_lines = [
        "[bold cyan]POSITIONING IDEAS[/bold cyan]",
        "",
        f"[bold]Long:[/bold] {pos['long']}",
        f"[bold]Avoid:[/bold] {pos['avoid']}",
        f"[bold]Speculative:[/bold] {pos['speculative']}",
    ]
    console.print(Panel("\n".join(pos_lines), expand=False))


def _show_llm_analysis(console: Console, settings, report, basket_info) -> None:
    """Generate LLM analysis of the rare earths sector."""
    from rich.markdown import Markdown
    
    console.print("\n[bold cyan]Generating AI analysis...[/bold cyan]\n")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        
        # Build context
        securities_summary = []
        for sec in report.securities[:10]:
            securities_summary.append(
                f"- {sec.ticker} ({sec.name}): ${sec.price:.2f}, {sec.change_1d:+.1f}% today, "
                f"RE rev {sec.re_revenue_pct}%, China exp: {sec.china_exposure}"
            )
        
        prompt = f"""You are a senior equity analyst covering rare earths and critical minerals.
        
Analyze the current state of the rare earth sector based on this data:

BASKET: {basket_info['name']} - {basket_info['description']}

SECURITIES:
{chr(10).join(securities_summary)}

BASKET PERFORMANCE: {report.basket_change_1d:+.2f}% today
TOTAL MARKET CAP: ${report.total_market_cap_b:.1f}B

BULL SIGNALS: {', '.join(report.bull_signals)}
BEAR SIGNALS: {', '.join(report.bear_signals)}

MARKET CONTEXT:
- China controls 60% of mining, 90% of processing, 92% of magnets
- Key demand: EVs (30%), Wind (25%), Electronics (20%), Defense (10%)
- US/EU pushing for supply chain independence

Provide a brief (3-4 paragraph) analysis covering:
1. Current sector dynamics and what today's price action signals
2. Key risks and opportunities in the near term
3. Which names look most interesting and why

Be specific about tickers and use quantitative references where possible."""

        response = client.chat.completions.create(
            model=settings.openai_model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
        )
        
        analysis = response.choices[0].message.content.strip()
        console.print(Panel(Markdown(analysis), title="AI Analysis", expand=False))
        
    except Exception as e:
        console.print(f"[red]Error generating analysis: {e}[/red]")

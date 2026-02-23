"""
LOX Research: Regimes Command

Unified regime view with LLM commentary and drill-down capability.

Usage:
    lox research regimes              # Overview of all regimes
    lox research regimes --llm        # Add LLM commentary
    lox research regimes --detail vol # Drill into volatility regime
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from lox.config import load_settings

# ── Score → color thresholds ─────────────────────────────────────────────
SCORE_LOW_THRESHOLD = 35   # below = green (benign)
SCORE_HIGH_THRESHOLD = 65  # above = red (stressed)
DESC_TRUNCATE_LEN = 34     # max chars before truncation in tables


def _get_regime_color(score: float) -> str:
    """Get color based on regime score."""
    if score < SCORE_LOW_THRESHOLD:
        return "green"
    elif score < SCORE_HIGH_THRESHOLD:
        return "yellow"
    return "red"


def _regime_bar(score: float, width: int = 10) -> str:
    """Create a visual progress bar for regime score."""
    filled = int((score / 100) * width)
    empty = width - filled
    color = _get_regime_color(score)
    return f"[{color}]{'█' * filled}{'░' * empty}[/{color}]"


def _format_metrics(metrics: dict, max_items: int = 4) -> str:
    """Format a metrics dict into a compact display string."""
    if not metrics:
        return "[dim]—[/dim]"
    parts = []
    for k, v in list(metrics.items())[:max_items]:
        if v is not None:
            parts.append(f"[bold]{k}[/bold] {v}")
    return "  ".join(parts) if parts else "[dim]—[/dim]"


def register(app: typer.Typer) -> None:
    """Register the regimes command."""
    
    @app.command("regimes")
    def regimes_cmd(
        llm: bool = typer.Option(False, "--llm", "-l", help="Add LLM commentary"),
        detail: str = typer.Option(None, "--detail", "-d", help="Drill into specific regime (macro, vol, rates, funding, etc.)"),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh data"),
        book: bool = typer.Option(False, "--book", "-b", help="Show impact on open positions"),
    ):
        """
        Unified regime dashboard with LLM commentary.

        Shows current state of all 12 regime pillars:
        - Core: Growth, Inflation, Volatility, Credit, Rates, Funding
        - Extended: Consumer, Fiscal, Positioning, Monetary, USD, Commodities

        Examples:
            lox research regimes              # Quick overview
            lox research regimes --llm        # With AI commentary
            lox research regimes -d vol       # Drill into volatility
            lox research regimes --book       # Show position exposure
        """
        console = Console()
        settings = load_settings()
        
        console.print()
        
        # Build unified regime state
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Building regime state...[/bold cyan]"),
            transient=True,
        ) as progress:
            progress.add_task("loading", total=None)
            
            from lox.regimes import build_unified_regime_state
            state = build_unified_regime_state(
                settings=settings,
                start_date="2020-01-01",
                refresh=refresh,
            )
        
        # Book impact — fetch positions and correlate with regimes
        book_impacts = None
        if book:
            from lox.cli_commands.shared.book_impact import analyze_book_impact, render_book_impact
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]Analyzing position exposure...[/bold cyan]"),
                transient=True,
            ) as progress:
                progress.add_task("book", total=None)
                book_impacts = analyze_book_impact(settings, state)

        # If drilling into specific regime, show detailed view
        if detail:
            _show_regime_detail(console, settings, state, detail.lower(), llm, book_impacts)
            return

        # Show overview
        _show_regime_overview(console, settings, state, llm, book_impacts)
    
    @app.callback(invoke_without_command=True)
    def default(ctx: typer.Context):
        """Default: show regime overview."""
        if ctx.invoked_subcommand is None:
            # Call regimes command with defaults
            ctx.invoke(regimes_cmd)


def _show_regime_overview(console: Console, settings, state, include_llm: bool, book_impacts=None):
    """Show unified regime overview."""
    from datetime import datetime
    
    # Header
    overall_color = _get_regime_color(state.overall_risk_score)
    
    header_content = f"""[bold]REGIME STATE[/bold]  {state.overall_category.upper()}
[dim]Risk Score: [{overall_color}]{state.overall_risk_score:.0f}/100[/{overall_color}][/dim]"""
    
    console.print(Panel(
        header_content,
        title=f"[bold cyan]LOX RESEARCH[/bold cyan]  [dim]{state.asof}[/dim]",
        border_style="cyan",
        padding=(0, 2),
    ))
    
    console.print()
    
    # Core Pillars
    core_table = Table(
        title="[bold]Core Pillars[/bold]",
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 2),
    )
    core_table.add_column("Pillar", style="bold", min_width=10, no_wrap=True)
    core_table.add_column("Score", justify="center", min_width=5, no_wrap=True)
    core_table.add_column("Regime", min_width=14)
    core_table.add_column("Key Metrics", ratio=2)
    core_table.add_column("Signal", ratio=1)
    
    core_pillars = [
        ("Growth", state.growth),
        ("Inflation", state.inflation),
        ("Volatility", state.volatility),
        ("Credit", state.credit),
        ("Rates", state.rates),
        ("Liquidity", state.liquidity),
    ]
    
    for name, regime in core_pillars:
        if regime:
            color = _get_regime_color(regime.score)
            desc = regime.description[:DESC_TRUNCATE_LEN] + "..." if len(regime.description) > DESC_TRUNCATE_LEN else regime.description
            metrics_str = _format_metrics(regime.metrics) if regime.metrics else "[dim]—[/dim]"
            core_table.add_row(
                name,
                f"[{color}]{regime.score:.0f}[/{color}]",
                f"[{color}]{regime.label}[/{color}]",
                metrics_str,
                f"[dim]{desc}[/dim]",
            )
        else:
            core_table.add_row(name, "[dim]—[/dim]", "[dim]No data[/dim]", "", "")
    
    console.print(core_table)
    console.print()
    
    # Extended Pillars
    ext_table = Table(
        title="[bold]Extended Pillars[/bold]",
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 2),
    )
    ext_table.add_column("Pillar", style="bold", min_width=10, no_wrap=True)
    ext_table.add_column("Score", justify="center", min_width=5, no_wrap=True)
    ext_table.add_column("Regime", min_width=14)
    ext_table.add_column("Key Metrics", ratio=2)
    ext_table.add_column("Signal", ratio=1)
    
    extended = [
        ("Consumer", state.consumer),
        ("Fiscal", state.fiscal),
        ("USD", state.usd),
        ("Commodities", state.commodities),
    ]
    
    for name, regime in extended:
        if regime:
            color = _get_regime_color(regime.score)
            desc = regime.description[:DESC_TRUNCATE_LEN] + "..." if len(regime.description) > DESC_TRUNCATE_LEN else regime.description
            metrics_str = _format_metrics(regime.metrics) if regime.metrics else "[dim]—[/dim]"
            ext_table.add_row(
                name,
                f"[{color}]{regime.score:.0f}[/{color}]",
                f"[{color}]{regime.label}[/{color}]",
                metrics_str,
                f"[dim]{desc}[/dim]",
            )
        else:
            ext_table.add_row(name, "[dim]—[/dim]", "[dim]No data[/dim]", "", "")
    
    console.print(ext_table)
    console.print()

    # Book Impact
    if book_impacts is not None:
        from lox.cli_commands.shared.book_impact import render_book_impact
        render_book_impact(console, book_impacts)
        console.print()

    # LLM Commentary
    if include_llm:
        _show_llm_commentary(console, settings, state, book_impacts)
    else:
        console.print("[dim]Add --llm for AI commentary  |  --detail <pillar> to drill down  |  --book for position exposure[/dim]")
    
    console.print()


def _show_regime_detail(console: Console, settings, state, pillar: str, include_llm: bool = True, book_impacts=None):
    """Show detailed view of a specific regime."""
    # Map pillar names
    pillar_map = {
        "growth": ("growth", state.growth),
        "inflation": ("inflation", state.inflation),
        "macro": ("growth", state.growth),  # alias for backwards compat
        "vol": ("volatility", state.volatility),
        "volatility": ("volatility", state.volatility),
        "credit": ("credit", state.credit),
        "rates": ("rates", state.rates),
        "liquidity": ("liquidity", state.liquidity),
        "consumer": ("consumer", state.consumer),
        "fiscal": ("fiscal", state.fiscal),
        "usd": ("usd", state.usd),
        "commodities": ("commodities", state.commodities),
    }
    
    if pillar not in pillar_map:
        console.print(f"[red]Unknown pillar: {pillar}[/red]")
        console.print(f"[dim]Available: {', '.join(pillar_map.keys())}[/dim]")
        return
    
    domain, regime = pillar_map[pillar]
    
    if not regime:
        console.print(f"[yellow]No data available for {domain}[/yellow]")
        return
    
    # Header with key metrics
    color = _get_regime_color(regime.score)
    header_parts = [
        f"[bold]{domain.upper()} REGIME[/bold]\n",
        f"Status: [{color}]{regime.label}[/{color}]",
        f"Score: [{color}]{regime.score:.0f}/100[/{color}]\n",
        f"{regime.description}",
    ]
    
    # Show metrics in the header panel
    if regime.metrics:
        header_parts.append("")
        metrics_line = "  ".join(
            f"[bold]{k}[/bold]: {v}" for k, v in regime.metrics.items() if v is not None
        )
        header_parts.append(metrics_line)
    
    console.print(Panel(
        "\n".join(header_parts),
        title=f"[bold cyan]LOX RESEARCH[/bold cyan]  [dim]{state.asof}[/dim]",
        border_style=color,
    ))
    console.print()
    
    # Build snapshot from regime metrics (real data for LLM)
    snapshot = {
        "regime": regime.label,
        "score": regime.score,
        "description": regime.description,
    }
    if regime.metrics:
        snapshot["current_metrics"] = {k: v for k, v in regime.metrics.items() if v is not None}
    
    # Book Impact for this domain
    if book_impacts is not None:
        from lox.cli_commands.shared.book_impact import render_book_impact
        # Filter to impacts relevant to this domain
        domain_filtered = []
        for pi in book_impacts:
            filtered_impacts = [di for di in pi.impacts if di.domain == domain]
            if filtered_impacts:
                from copy import copy
                pi_copy = copy(pi)
                pi_copy.impacts = filtered_impacts
                domain_filtered.append(pi_copy)
        if domain_filtered:
            render_book_impact(console, domain_filtered)
            console.print()

    # LLM Analysis — detail view always includes it (streaming, no spinner)
    try:
        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis
        print_llm_regime_analysis(
            settings=settings,
            domain=domain,
            snapshot=snapshot,
            regime_label=regime.label,
            regime_description=regime.description,
            console=console,
            book_impacts=book_impacts,
        )
    except Exception as e:
        console.print(f"[red]LLM analysis failed: {e}[/red]")
    
    console.print()


def _show_llm_commentary(console: Console, settings, state, book_impacts=None):
    """Show brief LLM commentary on overall regime state."""
    from rich.progress import Progress, SpinnerColumn, TextColumn
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Generating commentary...[/bold cyan]"),
        transient=True,
    ) as progress:
        progress.add_task("llm", total=None)
        
        try:
            from lox.llm.core.analyst import quick_llm_summary
            
            # Build summary data
            summary_data = {
                "overall": {
                    "category": state.overall_category,
                    "risk_score": state.overall_risk_score,
                },
                "pillars": {},
            }
            
            for name, regime in [
                ("growth", state.growth),
                ("inflation", state.inflation),
                ("volatility", state.volatility),
                ("credit", state.credit),
                ("rates", state.rates),
                ("liquidity", state.liquidity),
            ]:
                if regime:
                    summary_data["pillars"][name] = {
                        "label": regime.label,
                        "score": regime.score,
                        "description": regime.description,
                    }
            
            commentary = quick_llm_summary(
                settings=settings,
                title="Regime Overview",
                data=summary_data,
                question="Provide a 3-4 sentence summary of the current macro environment and key trading implications. Be direct and actionable.",
            )
            
            console.print(Panel(
                commentary,
                title="[bold]AI Commentary[/bold]",
                border_style="blue",
                padding=(1, 2),
            ))
            
        except Exception as e:
            console.print(f"[dim]Commentary unavailable: {e}[/dim]")

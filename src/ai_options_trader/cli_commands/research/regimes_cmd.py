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

from ai_options_trader.config import load_settings


def _regime_bar(score: float, width: int = 10) -> str:
    """Create a visual progress bar for regime score."""
    filled = int((score / 100) * width)
    empty = width - filled
    
    if score < 35:
        color = "green"
    elif score < 65:
        color = "yellow"
    else:
        color = "red"
    
    return f"[{color}]{'█' * filled}{'░' * empty}[/{color}]"


def _get_regime_color(score: float) -> str:
    """Get color based on regime score."""
    if score < 35:
        return "green"
    elif score < 65:
        return "yellow"
    return "red"


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
    ):
        """
        Unified regime dashboard with LLM commentary.
        
        Shows current state of all regime pillars:
        - Core: Macro, Volatility, Rates, Funding
        - Extended: Fiscal, Commodities, Housing, Monetary, USD, Crypto
        
        Examples:
            lox research regimes              # Quick overview
            lox research regimes --llm        # With AI commentary
            lox research regimes -d vol       # Drill into volatility
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
            
            from ai_options_trader.regimes import build_unified_regime_state
            state = build_unified_regime_state(
                settings=settings,
                start_date="2020-01-01",
                refresh=refresh,
            )
        
        # If drilling into specific regime, show detailed view
        if detail:
            _show_regime_detail(console, settings, state, detail.lower(), llm)
            return
        
        # Show overview
        _show_regime_overview(console, settings, state, llm)
    
    @app.callback(invoke_without_command=True)
    def default(ctx: typer.Context):
        """Default: show regime overview."""
        if ctx.invoked_subcommand is None:
            # Call regimes command with defaults
            ctx.invoke(regimes_cmd)


def _show_regime_overview(console: Console, settings, state, include_llm: bool):
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
        ("Macro", state.macro),
        ("Volatility", state.volatility),
        ("Rates", state.rates),
        ("Funding", state.funding),
    ]
    
    for name, regime in core_pillars:
        if regime:
            color = _get_regime_color(regime.score)
            desc = regime.description[:34] + "..." if len(regime.description) > 34 else regime.description
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
        ("Fiscal", state.fiscal),
        ("Commodities", state.commodities),
        ("Housing", state.housing),
        ("Monetary", state.monetary),
        ("USD", state.usd),
        ("Crypto", state.crypto),
    ]
    
    for name, regime in extended:
        if regime:
            color = _get_regime_color(regime.score)
            desc = regime.description[:34] + "..." if len(regime.description) > 34 else regime.description
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
    
    # LLM Commentary
    if include_llm:
        _show_llm_commentary(console, settings, state)
    else:
        console.print("[dim]Add --llm for AI commentary  |  --detail <pillar> to drill down[/dim]")
    
    console.print()


def _show_regime_detail(console: Console, settings, state, pillar: str, include_llm: bool):
    """Show detailed view of a specific regime."""
    from ai_options_trader.llm.core.analyst import llm_analyze_regime
    
    # Map pillar names
    pillar_map = {
        "macro": ("macro", state.macro),
        "vol": ("volatility", state.volatility),
        "volatility": ("volatility", state.volatility),
        "rates": ("rates", state.rates),
        "funding": ("funding", state.funding),
        "fiscal": ("fiscal", state.fiscal),
        "commodities": ("commodities", state.commodities),
        "housing": ("housing", state.housing),
        "monetary": ("monetary", state.monetary),
        "usd": ("usd", state.usd),
        "crypto": ("crypto", state.crypto),
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
    
    # LLM Analysis
    if include_llm or True:  # Always show LLM for detail view
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]Analyzing {domain}...[/bold cyan]"),
            transient=True,
        ) as progress:
            progress.add_task("analyzing", total=None)
            
            try:
                analysis = llm_analyze_regime(
                    settings=settings,
                    domain=domain,
                    snapshot=snapshot,
                    regime_label=regime.label,
                    regime_description=regime.description,
                    include_news=True,
                    include_prices=True,
                    include_calendar=True,
                )
                
                from rich.markdown import Markdown
                console.print(Markdown(analysis))
                
            except Exception as e:
                console.print(f"[red]LLM analysis failed: {e}[/red]")
    
    console.print()


def _show_llm_commentary(console: Console, settings, state):
    """Show brief LLM commentary on overall regime state."""
    from rich.progress import Progress, SpinnerColumn, TextColumn
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Generating commentary...[/bold cyan]"),
        transient=True,
    ) as progress:
        progress.add_task("llm", total=None)
        
        try:
            from ai_options_trader.llm.core.analyst import quick_llm_summary
            
            # Build summary data
            summary_data = {
                "overall": {
                    "category": state.overall_category,
                    "risk_score": state.overall_risk_score,
                },
                "pillars": {},
            }
            
            for name, regime in [
                ("macro", state.macro),
                ("volatility", state.volatility),
                ("rates", state.rates),
                ("funding", state.funding),
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

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


def _format_metrics(metrics: dict, max_items: int = 3) -> str:
    """Format a metrics dict into a compact display string.

    Only shows the first `max_items` non-None values.
    Keys should already be human-readable (e.g. 'HY OAS', 'CPI YoY').
    Values should already be formatted strings (e.g. '288bp', '2.8%').
    """
    if not metrics:
        return "[dim]—[/dim]"
    parts = []
    for k, v in metrics.items():
        if v is not None:
            parts.append(f"[bold]{k}[/bold] {v}")
            if len(parts) >= max_items:
                break
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
    
    # ── Pillar table builder (shared for Core + Extended) ──────────────
    def _build_pillar_table(title: str, pillars: list) -> Table:
        tbl = Table(
            title=f"[bold]{title}[/bold]",
            show_header=True,
            header_style="bold",
            box=None,
            padding=(0, 2),
        )
        tbl.add_column("Pillar", style="bold", min_width=10, no_wrap=True)
        tbl.add_column("", min_width=12, no_wrap=True)       # score bar
        tbl.add_column("Score", justify="center", min_width=5, no_wrap=True)
        tbl.add_column("Regime", min_width=16, no_wrap=True)
        tbl.add_column("Key Metrics", ratio=1)

        for name, regime in pillars:
            if regime:
                color = _get_regime_color(regime.score)
                bar = _regime_bar(regime.score)
                metrics_str = _format_metrics(regime.metrics) if regime.metrics else "[dim]—[/dim]"
                tbl.add_row(
                    name,
                    bar,
                    f"[{color}]{regime.score:.0f}[/{color}]",
                    f"[{color}]{regime.label}[/{color}]",
                    metrics_str,
                )
            else:
                tbl.add_row(name, _regime_bar(0), "[dim]—[/dim]", "[dim]No data[/dim]", "")
        return tbl

    core_pillars = [
        ("Growth", state.growth),
        ("Inflation", state.inflation),
        ("Volatility", state.volatility),
        ("Credit", state.credit),
        ("Rates", state.rates),
        ("Liquidity", state.liquidity),
    ]
    console.print(_build_pillar_table("Core Pillars", core_pillars))
    console.print()

    extended = [
        ("Consumer", state.consumer),
        ("Fiscal", state.fiscal),
        ("USD", state.usd),
        ("Commodities", state.commodities),
    ]
    console.print(_build_pillar_table("Extended Pillars", extended))
    console.print()

    # Active Macro Scenarios
    if state.active_scenarios:
        from lox.cli_commands.shared.scenario_display import render_scenarios_summary
        render_scenarios_summary(state.active_scenarios, console)
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

    # Scenarios involving this domain
    if state.active_scenarios:
        domain_scenarios = [
            s for s in state.active_scenarios
            if domain in s.domains_involved
        ]
        if domain_scenarios:
            from lox.cli_commands.shared.scenario_display import render_scenario_panel
            for s in domain_scenarios:
                render_scenario_panel(s, console)
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
            active_scenarios=state.active_scenarios if state.active_scenarios else None,
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
            
            # Inject active scenarios into LLM context
            if state.active_scenarios:
                from lox.regimes.scenarios import format_scenarios_for_llm
                summary_data["active_scenarios"] = format_scenarios_for_llm(state.active_scenarios)

            scenario_hint = ""
            if state.active_scenarios:
                names = [s.name for s in state.active_scenarios]
                scenario_hint = f" Active macro scenarios: {', '.join(names)}. Reference these in your analysis."

            commentary = quick_llm_summary(
                settings=settings,
                title="Regime Overview",
                data=summary_data,
                question=f"Provide a 3-4 sentence summary of the current macro environment and key trading implications. Be direct and actionable.{scenario_hint}",
            )
            
            console.print(Panel(
                commentary,
                title="[bold]AI Commentary[/bold]",
                border_style="blue",
                padding=(1, 2),
            ))
            
        except Exception as e:
            console.print(f"[dim]Commentary unavailable: {e}[/dim]")

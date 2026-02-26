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
        trend: bool = typer.Option(False, "--trend", "-t", help="Show trend & momentum dashboard"),
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
            lox research regimes --trend      # Trend & momentum dashboard
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

        # If trend dashboard requested, show it
        if trend:
            _show_trend_dashboard(console, state)
            return

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


def _show_trend_dashboard(console: Console, state) -> None:
    """Show full trend & momentum dashboard."""
    from lox.data.regime_history import get_all_score_series
    from lox.cli_commands.shared.trend_display import render_trend_dashboard

    all_series = get_all_score_series()

    # Header
    overall_color = _get_regime_color(state.overall_risk_score)
    header_content = f"""[bold]REGIME TREND & MOMENTUM[/bold]  {state.overall_category.upper()}
[dim]Risk Score: [{overall_color}]{state.overall_risk_score:.0f}/100[/{overall_color}][/dim]"""

    console.print(Panel(
        header_content,
        title=f"[bold cyan]LOX RESEARCH[/bold cyan]  [dim]{state.asof}[/dim]",
        border_style="cyan",
        padding=(0, 2),
    ))
    console.print()

    render_trend_dashboard(state.trends, all_series, console)
    console.print()
    console.print("[dim]--detail <pillar> for single-pillar trend detail  |  --llm for AI commentary[/dim]")
    console.print()


def _fmt_trend_arrow(trend) -> str:
    """Format a compact trend arrow for the overview table."""
    if trend is None:
        return "[dim]—[/dim]"
    return f"[{trend.trend_color}]{trend.trend_arrow}[/{trend.trend_color}]"


def _fmt_delta_compact(val) -> str:
    """Format a 7d delta for the overview table."""
    if val is None:
        return "[dim]—[/dim]"
    sign = "+" if val > 0 else ""
    if abs(val) < 0.5:
        return f"[dim]{sign}{val:.0f}[/dim]"
    c = "red" if val > 0 else "green"
    return f"[{c}]{sign}{val:.0f}[/{c}]"


def _show_regime_overview(console: Console, settings, state, include_llm: bool, book_impacts=None):
    """Show unified regime overview."""
    from datetime import datetime

    # Header
    overall_color = _get_regime_color(state.overall_risk_score)

    header_content = f"""[bold]REGIME STATE[/bold]  {state.overall_category.upper()}  {state.macro_quadrant}
[dim]Risk Score: [{overall_color}]{state.overall_risk_score:.0f}/100[/{overall_color}][/dim]"""

    console.print(Panel(
        header_content,
        title=f"[bold cyan]LOX RESEARCH[/bold cyan]  [dim]{state.asof}[/dim]",
        border_style="cyan",
        padding=(0, 2),
    ))

    console.print()

    trends = state.trends or {}

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
        tbl.add_column("", justify="center", min_width=3, no_wrap=True)  # trend arrow
        tbl.add_column("Δ7d", justify="right", min_width=4, no_wrap=True)
        tbl.add_column("Regime", min_width=16, no_wrap=True)
        tbl.add_column("Key Metrics", ratio=1)

        for name, regime in pillars:
            if regime:
                color = _get_regime_color(regime.score)
                bar = _regime_bar(regime.score)
                metrics_str = _format_metrics(regime.metrics) if regime.metrics else "[dim]—[/dim]"
                domain_key = name.lower()
                t = trends.get(domain_key)
                arrow = _fmt_trend_arrow(t)
                delta7 = _fmt_delta_compact(t.score_chg_7d if t else None)
                tbl.add_row(
                    name,
                    bar,
                    f"[{color}]{regime.score:.0f}[/{color}]",
                    arrow,
                    delta7,
                    f"[{color}]{regime.label}[/{color}]",
                    metrics_str,
                )
            else:
                tbl.add_row(name, _regime_bar(0), "[dim]—[/dim]", "[dim]—[/dim]", "[dim]—[/dim]", "[dim]No data[/dim]", "")
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

    # Market Dislocations
    dislocations = getattr(state, "active_dislocations", [])
    if dislocations:
        from lox.cli_commands.shared.inconsistency_display import render_dislocations_panel
        render_dislocations_panel(dislocations, console)
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
        console.print("[dim]--trend for momentum dashboard  |  --llm for AI commentary  |  --detail <pillar> to drill down  |  --book for position exposure[/dim]")
    
    console.print()


def _show_regime_detail(console: Console, settings, state, pillar: str, include_llm: bool = True, book_impacts=None):
    """Dispatch to the canonical `lox regime <domain>` snapshot function."""
    # Map aliases → canonical domain name
    pillar_aliases = {
        "growth": "growth",
        "inflation": "inflation",
        "macro": "growth",
        "vol": "volatility",
        "volatility": "volatility",
        "credit": "credit",
        "rates": "rates",
        "liquidity": "liquidity",
        "funding": "liquidity",
        "consumer": "consumer",
        "fiscal": "fiscal",
        "usd": "usd",
        "commodities": "commodities",
    }

    if pillar not in pillar_aliases:
        console.print(f"[red]Unknown pillar: {pillar}[/red]")
        console.print(f"[dim]Available: {', '.join(sorted(set(pillar_aliases.values())))}[/dim]")
        return

    domain = pillar_aliases[pillar]

    console.print(f"[dim]→ lox regime {domain}[/dim]\n")

    # Dispatch to the canonical snapshot function for each domain.
    # Each already renders the full panel with trend data + LLM chat.
    if domain == "growth":
        from lox.cli_commands.regimes.growth_cmd import growth_snapshot
        growth_snapshot(llm=include_llm, refresh=False)
    elif domain == "inflation":
        from lox.cli_commands.regimes.inflation_cmd import inflation_snapshot
        inflation_snapshot(llm=include_llm, refresh=False)
    elif domain == "volatility":
        from lox.cli_commands.regimes.volatility_cmd import volatility_snapshot
        volatility_snapshot(llm=include_llm, refresh=False)
    elif domain == "credit":
        from lox.cli_commands.regimes.credit_cmd import credit_snapshot
        credit_snapshot(llm=include_llm, refresh=False)
    elif domain == "rates":
        from lox.cli_commands.regimes.rates_cmd import rates_snapshot
        rates_snapshot(llm=include_llm, refresh=False)
    elif domain == "liquidity":
        from lox.cli_commands.regimes.funding_cmd import funding_snapshot
        funding_snapshot(llm=include_llm, refresh=False)
    elif domain == "consumer":
        from lox.cli_commands.regimes.consumer_cmd import consumer_snapshot
        consumer_snapshot(llm=include_llm, refresh=False)
    elif domain == "fiscal":
        from lox.cli_commands.regimes.fiscal_cmd import fiscal_snapshot
        fiscal_snapshot(llm=include_llm, refresh=False)
    elif domain == "usd":
        from lox.cli_commands.regimes.usd_cmd import run_usd_snapshot
        run_usd_snapshot(llm=include_llm, refresh=False)
    elif domain == "commodities":
        from lox.cli_commands.regimes.commodities_cmd import _run_commodities_snapshot
        _run_commodities_snapshot(llm=include_llm, refresh=False)


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
            
            trends = state.trends or {}
            for name, regime in [
                ("growth", state.growth),
                ("inflation", state.inflation),
                ("volatility", state.volatility),
                ("credit", state.credit),
                ("rates", state.rates),
                ("liquidity", state.liquidity),
            ]:
                if regime:
                    pillar_data = {
                        "label": regime.label,
                        "score": regime.score,
                        "description": regime.description,
                    }
                    t = trends.get(name)
                    if t:
                        pillar_data["trend"] = t.to_dict()
                    summary_data["pillars"][name] = pillar_data
            
            # Inject active scenarios into LLM context
            if state.active_scenarios:
                from lox.regimes.scenarios import format_scenarios_for_llm
                summary_data["active_scenarios"] = format_scenarios_for_llm(state.active_scenarios)

            # Inject dislocations into LLM context
            dislocations = getattr(state, "active_dislocations", [])
            if dislocations:
                from lox.regimes.inconsistencies import format_dislocations_for_llm
                summary_data["dislocations"] = format_dislocations_for_llm(dislocations)

            scenario_hint = ""
            if state.active_scenarios:
                names = [s.name for s in state.active_scenarios]
                scenario_hint = f" Active macro scenarios: {', '.join(names)}. Reference these in your analysis."

            dislocation_hint = ""
            if dislocations:
                dnames = [d.name for d in dislocations]
                dislocation_hint = f" Market dislocations detected: {', '.join(dnames)}. Explain which are most actionable and how to capitalize."

            commentary = quick_llm_summary(
                settings=settings,
                title="Regime Overview",
                data=summary_data,
                question=f"Provide a 3-4 sentence summary of the current macro environment and key trading implications. Be direct and actionable.{scenario_hint}{dislocation_hint}",
            )
            
            console.print(Panel(
                commentary,
                title="[bold]AI Commentary[/bold]",
                border_style="blue",
                padding=(1, 2),
            ))
            
        except Exception as e:
            console.print(f"[dim]Commentary unavailable: {e}[/dim]")

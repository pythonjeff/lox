"""
Scenario analysis CLI commands for stress testing portfolio.

Includes:
- Basic predefined scenarios
- ML-enhanced forward-looking scenarios
- Historical event replay
- Factor-based scenarios
- Monte Carlo simulation
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ai_options_trader.config import Settings
from ai_options_trader.utils.settings import safe_load_settings
from ai_options_trader.utils.regimes import get_current_macro_regime
from ai_options_trader.llm.scenarios.scenarios import list_scenarios, apply_scenario, SCENARIOS
from ai_options_trader.llm.scenarios.scenario_impact import estimate_portfolio_impact

# Import ML scenario dependencies
from ai_options_trader.llm.scenarios.scenario_ml import (
    HISTORICAL_EVENTS,
    FACTOR_SCENARIOS,
    ScenarioFactors,
    apply_historical_event,
    apply_factor_scenario,
)
from ai_options_trader.llm.scenarios.scenario_forward import (
    generate_plausible_scenarios,
    ForwardScenario,
)
from ai_options_trader.llm.scenarios.scenario_custom import build_custom_scenario
from ai_options_trader.llm.scenarios.scenario_diagnostics import diagnose_scenario_estimate

# Import Monte Carlo v3 registration
from ai_options_trader.cli_commands.analysis.monte_carlo_cmd import register_v3


def register(labs_app: typer.Typer) -> None:
    """Register all scenario commands including ML-enhanced ones."""
    # Register Monte Carlo v3
    register_v3(labs_app)
    
    # Register basic scenarios command
    _register_basic_scenarios(labs_app)
    
    # Register ML-enhanced scenario commands
    _register_ml_scenarios(labs_app)


def _register_basic_scenarios(labs_app: typer.Typer) -> None:
    """Register the basic predefined scenarios command."""
    
    @labs_app.command("scenarios")
    def scenario_analysis(
        list_only: bool = typer.Option(False, "--list", help="List all available scenarios"),
        scenario_ids: str = typer.Option(
            "rates_rise_moderate,ten_year_collapse,vix_spike,liquidity_drain",
            "--scenarios",
            help="Comma-separated list of scenario IDs to run",
        ),
        net_delta: float = typer.Option(
            -0.2,
            "--net-delta",
            help="Portfolio net delta as % of NAV (e.g., -0.2 for 20% net short)",
        ),
        vega: float = typer.Option(
            0.1,
            "--vega",
            help="Portfolio vega (rough estimate, normalized to NAV)",
        ),
        theta: float = typer.Option(
            -0.0005,
            "--theta",
            help="Portfolio theta per day (as % of NAV)",
        ),
        has_tail_hedges: bool = typer.Option(
            True,
            "--tail-hedges/--no-tail-hedges",
            help="Whether portfolio has convex tail hedges",
        ),
        start: str = typer.Option("2011-01-01", "--start", help="Start date for historical data"),
        refresh: bool = typer.Option(False, "--refresh", help="Refresh all data (ignore cache)"),
    ):
        """
        Run scenario analysis to stress test portfolio under different market conditions.
        
        This tool:
        1. Fetches current market state (regimes + data)
        2. Applies user-selected scenarios (e.g., "rates rise", "VIX spike")
        3. Estimates portfolio P&L under each scenario
        4. Displays comparison table
        
        Examples:
            # List all available scenarios
            lox labs scenarios --list
            
            # Run default scenarios
            lox labs scenarios
            
            # Run custom scenarios with custom portfolio params
            lox labs scenarios --scenarios "vix_spike,credit_stress,stagflation" --net-delta -0.3 --vega 0.15
            
            # Run with no tail hedges
            lox labs scenarios --no-tail-hedges
        """
        settings = safe_load_settings()
        c = Console()
        
        # --- List scenarios ---
        if list_only:
            c.print(Panel("[bold cyan]Available Scenarios[/bold cyan]", expand=False))
            
            categories = ["rates", "inflation", "growth", "volatility", "liquidity", "credit"]
            for category in categories:
                scenarios = list_scenarios(category=category)
                if not scenarios:
                    continue
                
                c.print(f"\n[bold yellow]{category.upper()}[/bold yellow]")
                for s in scenarios:
                    severity_color = {"mild": "green", "moderate": "yellow", "severe": "red"}.get(s.severity, "white")
                    c.print(f"  [{severity_color}]{s.id:25s}[/{severity_color}] {s.name:30s} - {s.description}")
            
            return
        
        # --- Get current market state ---
        c.print("[dim]Loading...[/dim]")
        regimes = get_current_macro_regime(settings, start=start, refresh=refresh)
        baseline_macro = regimes["macro_state"]
        baseline_funding = regimes["liquidity_state"]
        macro_regime = regimes["macro_regime"]
        
        c.print(f"[green]✓[/green] Current regime: [bold]{macro_regime}[/bold]")
        
        # --- Run scenarios ---
        scenario_ids_list = [s.strip() for s in scenario_ids.split(",")]
        
        c.print(f"\n[cyan]Running {len(scenario_ids_list)} scenario(s)...[/cyan]")
        
        results = []
        for sid in scenario_ids_list:
            if sid not in SCENARIOS:
                c.print(f"[yellow]Warning: Unknown scenario '{sid}', skipping[/yellow]")
                continue
            
            scenario = SCENARIOS[sid]
            
            # Apply scenario
            scenario_macro, scenario_funding = apply_scenario(sid, baseline_macro, baseline_funding)
            
            # Estimate impact
            impact = estimate_portfolio_impact(
                baseline_macro=baseline_macro,
                baseline_funding=baseline_funding,
                scenario_macro=scenario_macro,
                scenario_funding=scenario_funding,
                portfolio_net_delta=net_delta,
                portfolio_vega=vega,
                portfolio_theta=theta,
                has_tail_hedges=has_tail_hedges,
            )
            impact.scenario_id = sid
            impact.scenario_name = scenario.name
            
            results.append((scenario, impact))
        
        # --- Display results ---
        c.print("\n")
        c.print(Panel(
            f"[bold]Baseline Regime:[/bold] {macro_regime}\n"
            f"[bold]Portfolio:[/bold] Net Delta {net_delta*100:+.0f}%, Vega {vega:.2f}, Tail Hedges: {'Yes' if has_tail_hedges else 'No'}",
            title="Scenario Analysis",
            border_style="cyan",
        ))
        
        # Build comparison table
        table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
        table.add_column("Scenario", style="bold", width=30)
        table.add_column("Severity", width=10)
        table.add_column("Portfolio P&L", justify="right", width=15)
        table.add_column("Confidence", width=10)
        table.add_column("Summary", width=50)
        
        for scenario, impact in results:
            # Color code P&L
            pnl_pct = impact.expected_pnl_pct * 100
            if pnl_pct > 5:
                pnl_color = "bold green"
            elif pnl_pct > 1:
                pnl_color = "green"
            elif pnl_pct < -5:
                pnl_color = "bold red"
            elif pnl_pct < -1:
                pnl_color = "red"
            else:
                pnl_color = "white"
            
            # Severity color
            severity_color = {"mild": "green", "moderate": "yellow", "severe": "red"}.get(scenario.severity, "white")
            
            # Confidence color
            conf_color = {"high": "green", "medium": "yellow", "low": "red"}.get(impact.confidence, "white")
            
            table.add_row(
                scenario.name,
                f"[{severity_color}]{scenario.severity.upper()}[/{severity_color}]",
                f"[{pnl_color}]{pnl_pct:+.1f}%[/{pnl_color}]",
                f"[{conf_color}]{impact.confidence}[/{conf_color}]",
                impact.summary,
            )
        
        c.print(table)
        
        # --- Detailed breakdown ---
        c.print("\n[bold cyan]Detailed Breakdown[/bold cyan]\n")
        for scenario, impact in results:
            c.print(f"[bold yellow]{scenario.name}[/bold yellow] ({scenario.description})")
            
            if impact.key_drivers:
                c.print("  [cyan]Key Drivers:[/cyan]")
                for driver in impact.key_drivers:
                    c.print(f"    • {driver}")
            
            if impact.risks:
                c.print("  [red]Risks:[/red]")
                for risk in impact.risks:
                    c.print(f"    • {risk}")
            
            c.print()
        
        # --- Best/worst scenarios ---
        sorted_results = sorted(results, key=lambda x: x[1].expected_pnl_pct, reverse=True)
        best = sorted_results[0]
        worst = sorted_results[-1]
        
        c.print(Panel(
            f"[bold green]Best scenario:[/bold green] {best[0].name} ({best[1].expected_pnl_pct*100:+.1f}%)\n"
            f"[bold red]Worst scenario:[/bold red] {worst[0].name} ({worst[1].expected_pnl_pct*100:+.1f}%)\n"
            f"[bold]Range:[/bold] {(best[1].expected_pnl_pct - worst[1].expected_pnl_pct)*100:.1f}% spread",
            title="Risk Summary",
            border_style="yellow",
        ))


def _register_ml_scenarios(labs_app: typer.Typer) -> None:
    """Register ML-enhanced scenario commands."""
    
    @labs_app.command("scenarios-custom")
    def scenarios_custom(
        name: str = typer.Option("My Custom Scenario", "--name", help="Scenario name"),
        ust_10y: float = typer.Option(None, "--10y", help="Target 10Y yield"),
        ust_2y: float = typer.Option(None, "--2y", help="Target 2Y yield"),
        cpi: float = typer.Option(None, "--cpi", help="Target CPI YoY"),
        unemployment: float = typer.Option(None, "--unemployment", help="Target unemployment"),
        vix: float = typer.Option(None, "--vix", help="Target VIX level"),
        hy_oas: float = typer.Option(None, "--hy-oas", help="Target HY OAS (bps)"),
        net_delta: float = typer.Option(-0.2, "--net-delta"),
        vega: float = typer.Option(0.1, "--vega"),
        theta: float = typer.Option(-0.0005, "--theta"),
        has_tail_hedges: bool = typer.Option(True, "--tail-hedges/--no-tail-hedges"),
        horizon_months: int = typer.Option(3, "--horizon"),
        diagnose: bool = typer.Option(False, "--diagnose"),
        start: str = typer.Option("2011-01-01", "--start"),
        refresh: bool = typer.Option(False, "--refresh"),
    ):
        """Test portfolio under YOUR specific macro view."""
        settings = safe_load_settings()
        c = Console()
        
        if all(v is None for v in [ust_10y, ust_2y, cpi, unemployment, vix, hy_oas]):
            c.print("[red]Error: Must specify at least one target value[/red]")
            return
        
        import warnings
        warnings.filterwarnings("ignore", message=".*Optional series.*unavailable.*")
        
        regimes = get_current_macro_regime(settings, start=start, refresh=refresh)
        baseline_macro = regimes["macro_state"]
        baseline_funding = regimes["liquidity_state"]
        
        changes = []
        if ust_10y is not None: changes.append(f"10Y → {ust_10y:.2f}%")
        if ust_2y is not None: changes.append(f"2Y → {ust_2y:.2f}%")
        if cpi is not None: changes.append(f"CPI → {cpi:.2f}%")
        if unemployment is not None: changes.append(f"Unemployment → {unemployment:.1f}%")
        if vix is not None: changes.append(f"VIX → {vix:.1f}")
        if hy_oas is not None: changes.append(f"HY OAS → {hy_oas:.0f} bps")
        
        scenario_macro, scenario_funding = build_custom_scenario(
            name=name, description=", ".join(changes),
            baseline_macro=baseline_macro, baseline_funding=baseline_funding,
            ust_10y_target=ust_10y, ust_2y_target=ust_2y, cpi_yoy_target=cpi,
            unemployment_target=unemployment, vix_target=vix, hy_oas_target=hy_oas,
        )
        
        impact = estimate_portfolio_impact(
            baseline_macro=baseline_macro, baseline_funding=baseline_funding,
            scenario_macro=scenario_macro, scenario_funding=scenario_funding,
            portfolio_net_delta=net_delta, portfolio_vega=vega,
            portfolio_theta=theta, has_tail_hedges=has_tail_hedges,
            horizon_months=horizon_months,
        )
        
        pnl_pct = impact.expected_pnl_pct * 100
        pnl_color = "green" if pnl_pct > 0 else "red"
        
        c.print(Panel(
            f"[bold]Scenario:[/bold] {name}\n[bold]Changes:[/bold] {', '.join(changes)}",
            title="Custom Scenario", border_style="cyan",
        ))
        c.print(Panel(
            f"[bold {pnl_color}]Portfolio P&L: {pnl_pct:+.1f}%[/bold {pnl_color}]\n{impact.summary}",
            title="Impact", border_style="yellow",
        ))
    
    @labs_app.command("scenarios-historical")
    def scenarios_historical(
        event_ids: str = typer.Option("gfc_2008,covid_crash_2020,volmageddon_2018", "--events"),
        severity: float = typer.Option(1.0, "--severity"),
        net_delta: float = typer.Option(-0.2, "--net-delta"),
        vega: float = typer.Option(0.1, "--vega"),
        theta: float = typer.Option(-0.0005, "--theta"),
        has_tail_hedges: bool = typer.Option(True, "--tail-hedges/--no-tail-hedges"),
        list_events: bool = typer.Option(False, "--list"),
        start: str = typer.Option("2011-01-01", "--start"),
        refresh: bool = typer.Option(False, "--refresh"),
    ):
        """Stress test using REAL historical events (2008 GFC, 2020 COVID, etc.)."""
        settings = safe_load_settings()
        c = Console()
        
        if list_events:
            c.print("[bold cyan]Historical Events:[/bold cyan]")
            for eid, event in HISTORICAL_EVENTS.items():
                c.print(f"  {eid}: {event.name} ({event.date_range})")
            return
        
        regimes = get_current_macro_regime(settings, start=start, refresh=refresh)
        baseline_macro = regimes["macro_state"]
        baseline_funding = regimes["liquidity_state"]
        
        results = []
        for eid in event_ids.split(","):
            eid = eid.strip()
            if eid not in HISTORICAL_EVENTS: continue
            event = HISTORICAL_EVENTS[eid]
            scenario_macro, scenario_funding = apply_historical_event(eid, baseline_macro, baseline_funding, severity)
            impact = estimate_portfolio_impact(
                baseline_macro=baseline_macro, baseline_funding=baseline_funding,
                scenario_macro=scenario_macro, scenario_funding=scenario_funding,
                portfolio_net_delta=net_delta, portfolio_vega=vega,
                portfolio_theta=theta, has_tail_hedges=has_tail_hedges,
            )
            results.append((event, impact))
        
        table = Table(title="Historical Scenarios", show_header=True)
        table.add_column("Event"); table.add_column("P&L", justify="right")
        for event, impact in results:
            pnl = impact.expected_pnl_pct * 100
            color = "green" if pnl > 0 else "red"
            table.add_row(event.name, f"[{color}]{pnl:+.1f}%[/{color}]")
        c.print(table)
    
    @labs_app.command("scenarios-factors")
    def scenarios_factors(
        risk_appetite: float = typer.Option(-0.8, "--risk-appetite"),
        growth_shock: float = typer.Option(-0.3, "--growth"),
        inflation_shock: float = typer.Option(0.0, "--inflation"),
        liquidity_stress: float = typer.Option(0.5, "--liquidity"),
        policy_shock: float = typer.Option(0.0, "--policy"),
        net_delta: float = typer.Option(-0.2, "--net-delta"),
        vega: float = typer.Option(0.1, "--vega"),
        has_tail_hedges: bool = typer.Option(True, "--tail-hedges/--no-tail-hedges"),
        list_presets: bool = typer.Option(False, "--list"),
        preset: str = typer.Option(None, "--preset"),
        start: str = typer.Option("2011-01-01", "--start"),
        refresh: bool = typer.Option(False, "--refresh"),
    ):
        """Stress test using high-level factors (risk appetite, growth, inflation, etc.)."""
        settings = safe_load_settings()
        c = Console()
        
        if list_presets:
            c.print("[bold cyan]Preset Factor Scenarios:[/bold cyan]")
            for pid, factors in FACTOR_SCENARIOS.items():
                c.print(f"  {pid}: {factors.name}")
            return
        
        regimes = get_current_macro_regime(settings, start=start, refresh=refresh)
        baseline_macro = regimes["macro_state"]
        baseline_funding = regimes["liquidity_state"]
        
        if preset and preset in FACTOR_SCENARIOS:
            factors = FACTOR_SCENARIOS[preset]
        else:
            factors = ScenarioFactors(
                risk_appetite=risk_appetite, growth_shock=growth_shock,
                inflation_shock=inflation_shock, liquidity_stress=liquidity_stress,
                policy_shock=policy_shock,
            )
        
        scenario_macro, scenario_funding = apply_factor_scenario(factors, baseline_macro, baseline_funding)
        impact = estimate_portfolio_impact(
            baseline_macro=baseline_macro, baseline_funding=baseline_funding,
            scenario_macro=scenario_macro, scenario_funding=scenario_funding,
            portfolio_net_delta=net_delta, portfolio_vega=vega,
            portfolio_theta=-0.0005, has_tail_hedges=has_tail_hedges,
        )
        
        pnl = impact.expected_pnl_pct * 100
        color = "green" if pnl > 0 else "red"
        c.print(Panel(
            f"[bold]Factors:[/bold] {factors.name}\n[bold {color}]P&L: {pnl:+.1f}%[/bold {color}]",
            title="Factor Scenario", border_style="cyan",
        ))

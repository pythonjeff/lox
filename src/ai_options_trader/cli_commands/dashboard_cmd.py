"""
Dashboard CLI Command - Unified regime dashboard.

Usage:
    lox dashboard                    # Full dashboard
    lox dashboard --focus inflation  # Deep-dive into inflation
    lox dashboard --features         # Export ML features
    lox dashboard --json             # JSON output

Author: Lox Capital Research
"""
from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console

from ai_options_trader.config import load_settings


def register(app: typer.Typer) -> None:
    """Register dashboard commands."""
    
    @app.command("dashboard")
    def dashboard(
        focus: Optional[str] = typer.Option(
            None, "--focus", "-f",
            help="Deep-dive into specific pillar: inflation|growth|liquidity|volatility"
        ),
        features: bool = typer.Option(
            False, "--features",
            help="Output ML features as JSON"
        ),
        output_json: bool = typer.Option(
            False, "--json",
            help="Output full dashboard as JSON"
        ),
    ):
        """
        Unified regime dashboard - all economic factors at a glance.
        
        Shows:
        - Inflation (CPI, breakevens, stickiness)
        - Growth (payrolls, unemployment, claims)
        - Liquidity (Fed balance sheet, funding rates)
        - Volatility (VIX, term structure)
        
        Examples:
            lox dashboard                     # Full summary
            lox dashboard --focus inflation   # Inflation deep-dive
            lox dashboard --focus liquidity   # Liquidity deep-dive
            lox dashboard --features          # ML features JSON
        """
        console = Console()
        settings = load_settings()
        
        # Create and compute dashboard
        from ai_options_trader.regimes.dashboard import create_default_dashboard
        
        console.print("[dim]Loading regime data...[/dim]\n")
        
        dash = create_default_dashboard()
        dash.compute_all(settings)
        
        # Output modes
        if features:
            feature_dict = dash.to_features()
            console.print(json.dumps(feature_dict, indent=2, default=str))
            return
        
        if output_json:
            output = {
                "asof": dash.asof.isoformat() if dash.asof else None,
                "overall_regime": dash.overall_regime(),
                "pillars": {},
            }
            for name, pillar in dash.pillars.items():
                output["pillars"][name] = {
                    "regime": pillar.regime,
                    "score": pillar.composite_score,
                    "metrics": {m.name: m.value for m in pillar.metrics},
                }
            console.print(json.dumps(output, indent=2, default=str))
            return
        
        # Visual dashboard
        dash.render(console, focus=focus)
        
        # Tips
        console.print("\n[dim]Deep-dive: lox dashboard --focus <pillar>[/dim]")
        console.print("[dim]ML features: lox dashboard --features[/dim]")


def register_pillar_commands(app: typer.Typer) -> None:
    """Register individual pillar commands for quick access."""
    
    @app.command("inflation")
    def inflation_cmd():
        """Quick inflation regime view."""
        console = Console()
        settings = load_settings()
        
        from ai_options_trader.regimes.pillars import InflationPillar
        
        pillar = InflationPillar()
        pillar.compute(settings)
        
        console.print(f"\n[bold cyan]Inflation Regime: {pillar.regime}[/bold cyan]")
        console.print(f"Score: {pillar.composite_score:.0f}/100\n" if pillar.composite_score else "")
        
        for m in pillar.metrics:
            delta = f" ({m.format_delta('3m')} 3m)" if m.delta_3m else ""
            console.print(f"  {m.name}: {m.format_value()}{delta}")
        
        console.print(f"\n[dim]{pillar._interpret()}[/dim]")
    
    @app.command("growth")
    def growth_cmd():
        """Quick growth regime view."""
        console = Console()
        settings = load_settings()
        
        from ai_options_trader.regimes.pillars import GrowthPillar
        
        pillar = GrowthPillar()
        pillar.compute(settings)
        
        console.print(f"\n[bold cyan]Growth Regime: {pillar.regime}[/bold cyan]")
        console.print(f"Score: {pillar.composite_score:.0f}/100\n" if pillar.composite_score else "")
        
        for m in pillar.metrics:
            console.print(f"  {m.name}: {m.format_value()}")
        
        console.print(f"\n[dim]{pillar._interpret()}[/dim]")
    
    @app.command("liquidity")
    def liquidity_cmd():
        """Quick liquidity regime view."""
        console = Console()
        settings = load_settings()
        
        from ai_options_trader.regimes.pillars import LiquidityPillar
        
        pillar = LiquidityPillar()
        pillar.compute(settings)
        
        console.print(f"\n[bold cyan]Liquidity Regime: {pillar.regime}[/bold cyan]")
        console.print(f"Score: {pillar.composite_score:.0f}/100\n" if pillar.composite_score else "")
        console.print(f"Buffer: {pillar.buffer_status()}\n")
        
        for m in pillar.metrics:
            delta = f" ({m.format_delta('1m')} 1m)" if m.delta_1m else ""
            console.print(f"  {m.name}: {m.format_value()}{delta}")
        
        triggers = pillar._triggers() if hasattr(pillar, "_triggers") else []
        if triggers:
            console.print(f"\n[yellow]Triggers: {', '.join(triggers)}[/yellow]")
        
        console.print(f"\n[dim]{pillar._interpret()}[/dim]")
    
    @app.command("vol")
    def vol_cmd():
        """Quick volatility regime view."""
        console = Console()
        settings = load_settings()
        
        from ai_options_trader.regimes.pillars import VolatilityPillar
        
        pillar = VolatilityPillar()
        pillar.compute(settings)
        
        console.print(f"\n[bold cyan]Volatility Regime: {pillar.regime}[/bold cyan]")
        console.print(f"Score: {pillar.composite_score:.0f}/100\n" if pillar.composite_score else "")
        console.print(f"Term Structure: {pillar.term_structure_status()}\n")
        
        for m in pillar.metrics:
            pct = f" ({m.percentile:.0f}%ile)" if m.percentile else ""
            console.print(f"  {m.name}: {m.format_value()}{pct}")
        
        implications = pillar._trade_implications() if hasattr(pillar, "_trade_implications") else []
        if implications:
            console.print(f"\n[dim]Trade implications:[/dim]")
            for impl in implications:
                console.print(f"  • {impl}")
        
        console.print(f"\n[dim]{pillar._interpret()}[/dim]")

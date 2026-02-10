"""
Unified Regime Dashboard - Aggregate all pillars into one view.

Provides:
1. Quick summary across all pillars
2. Deep-dive into individual pillars
3. ML feature extraction
4. Portfolio implications

Author: Lox Capital Research
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lox.regimes.core import RegimePillar, Metric


@dataclass
class RegimeDashboard:
    """
    Unified dashboard aggregating all regime pillars.
    """
    pillars: Dict[str, RegimePillar] = field(default_factory=dict)
    computed: bool = False
    asof: Optional[datetime] = None
    
    def add_pillar(self, pillar: RegimePillar) -> None:
        """Add a pillar to the dashboard."""
        self.pillars[pillar.name.lower()] = pillar
    
    def compute_all(self, settings: Any) -> None:
        """Compute all pillars."""
        for pillar in self.pillars.values():
            try:
                pillar.compute(settings)
            except Exception as e:
                print(f"Error computing {pillar.name}: {e}")
        
        self.computed = True
        self.asof = datetime.now()
    
    def get_pillar(self, name: str) -> Optional[RegimePillar]:
        """Get a pillar by name."""
        return self.pillars.get(name.lower())
    
    def to_features(self) -> Dict[str, float]:
        """Extract all ML features from all pillars."""
        features = {}
        for pillar in self.pillars.values():
            features.update(pillar.to_features())
        return features
    
    def overall_regime(self) -> str:
        """
        Compute overall market regime from pillar combination.
        
        Returns one of: RISK_ON, NEUTRAL, RISK_OFF, CRISIS
        """
        scores = []
        
        # Weight pillars by importance for risk assessment
        weights = {
            "liquidity": 0.25,
            "volatility": 0.25,
            "growth": 0.20,
            "inflation": 0.15,
            "credit": 0.15,
        }
        
        for name, pillar in self.pillars.items():
            if pillar.composite_score is not None:
                weight = weights.get(name, 0.10)
                # Invert scores where high = risky (vol, inflation)
                if name in ["volatility", "inflation"]:
                    risk_score = pillar.composite_score
                else:
                    risk_score = 100 - pillar.composite_score
                scores.append((risk_score, weight))
        
        if not scores:
            return "UNKNOWN"
        
        weighted_risk = sum(s * w for s, w in scores) / sum(w for _, w in scores)
        
        if weighted_risk >= 70:
            return "CRISIS"
        elif weighted_risk >= 55:
            return "RISK_OFF"
        elif weighted_risk >= 40:
            return "NEUTRAL"
        else:
            return "RISK_ON"
    
    def portfolio_implications(self) -> Dict[str, str]:
        """
        Generate portfolio implications for each pillar.
        
        Specific to a tail-risk hedging fund.
        """
        implications = {}
        
        for name, pillar in self.pillars.items():
            regime = pillar.regime or "UNKNOWN"
            
            if name == "inflation":
                if "HIGH" in regime or "ELEVATED" in regime:
                    implications[name] = "Supports real asset hedges; bonds vulnerable"
                elif "LOW" in regime:
                    implications[name] = "Duration-friendly; nominal bonds attractive"
                else:
                    implications[name] = "Neutral inflation impact"
            
            elif name == "growth":
                if regime in ["RECESSION_RISK", "CONTRACTION"]:
                    implications[name] = "Supports defensive positioning; equity puts attractive"
                elif regime == "EXPANSION":
                    implications[name] = "Risk-on bias; hedges as insurance"
                else:
                    implications[name] = "Monitor for inflection"
            
            elif name == "liquidity":
                if regime in ["SCARCE", "STRESSED"]:
                    implications[name] = "Elevated volatility risk; convex hedges attractive"
                elif regime == "FRAGILE":
                    implications[name] = "Buffer depleted; watch for funding stress"
                else:
                    implications[name] = "Funding orderly; standard hedge sizing"
            
            elif name == "volatility":
                if regime in ["CRISIS", "STRESSED"]:
                    implications[name] = "Vol expensive; consider vol-selling or patience"
                elif regime in ["COMPLACENT", "LOW"]:
                    implications[name] = "Vol cheap; attractive entry for protection"
                else:
                    implications[name] = "Vol fair; standard sizing"
            
            else:
                implications[name] = f"Regime: {regime}"
        
        return implications
    
    def render(self, console: Console, focus: Optional[str] = None) -> None:
        """
        Render dashboard to console.
        
        Args:
            console: Rich console
            focus: Optional pillar name for deep-dive
        """
        if not self.computed:
            console.print("[yellow]Dashboard not computed. Call compute_all() first.[/yellow]")
            return
        
        # Header
        overall = self.overall_regime()
        header_style = {
            "RISK_ON": "green",
            "NEUTRAL": "yellow",
            "RISK_OFF": "red",
            "CRISIS": "bold red",
        }.get(overall, "white")
        
        console.print(Panel(
            f"[{header_style}]Overall: {overall}[/{header_style}]\n"
            f"As of: {self.asof.strftime('%Y-%m-%d %H:%M') if self.asof else 'N/A'}",
            title="[bold]Lox Regime Dashboard[/bold]",
            expand=False,
        ))
        
        # If focus, show deep-dive
        if focus:
            self._render_deep_dive(console, focus)
            return
        
        # Summary table
        self._render_summary(console)
        
        # Portfolio implications
        self._render_implications(console)
    
    def _render_summary(self, console: Console) -> None:
        """Render summary table of all pillars."""
        table = Table(title="Regime Summary", show_header=True)
        table.add_column("Pillar", style="cyan")
        table.add_column("Regime", style="bold")
        table.add_column("Score", justify="right")
        table.add_column("Key Metric")
        table.add_column("Trend")
        
        for name, pillar in self.pillars.items():
            regime = pillar.regime or "N/A"
            score = f"{pillar.composite_score:.0f}" if pillar.composite_score else "N/A"
            
            # Pick key metric
            key_metric = ""
            trend = ""
            if pillar.metrics:
                m = pillar.metrics[0]
                key_metric = f"{m.name}: {m.format_value()}"
                trend = m.trend
            
            # Color regime
            regime_style = "white"
            if "HIGH" in regime or "CRISIS" in regime or "STRESSED" in regime:
                regime_style = "red"
            elif "LOW" in regime or "AMPLE" in regime or "EXPANSION" in regime:
                regime_style = "green"
            
            table.add_row(
                name.title(),
                f"[{regime_style}]{regime}[/{regime_style}]",
                score,
                key_metric,
                trend,
            )
        
        console.print(table)
    
    def _render_implications(self, console: Console) -> None:
        """Render portfolio implications."""
        implications = self.portfolio_implications()
        
        table = Table(title="Portfolio Relevance (Tail-Risk Fund)", show_header=False)
        table.add_column("Pillar", style="bold yellow", width=12)
        table.add_column("Implication")
        
        for name, impl in implications.items():
            table.add_row(name.title(), impl)
        
        console.print(table)
    
    def _render_deep_dive(self, console: Console, pillar_name: str) -> None:
        """Render deep-dive for a specific pillar."""
        pillar = self.get_pillar(pillar_name)
        if not pillar:
            console.print(f"[red]Pillar '{pillar_name}' not found.[/red]")
            console.print(f"Available: {', '.join(self.pillars.keys())}")
            return
        
        # Header
        console.print(Panel(
            f"[bold]{pillar.name}[/bold]: {pillar.description}\n"
            f"Regime: [bold]{pillar.regime}[/bold] | Score: {pillar.composite_score:.0f}" if pillar.composite_score else "",
            title=f"Deep Dive: {pillar.name}",
            expand=False,
        ))
        
        # Metrics table
        table = Table(title="Metrics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Z-Score", justify="right")
        table.add_column("1M Change", justify="right")
        table.add_column("Source")
        
        for m in pillar.metrics:
            z = f"{m.z_score:.2f}" if m.z_score else "—"
            delta = m.format_delta("1m") if m.delta_1m else "—"
            
            table.add_row(
                m.name,
                m.format_value(),
                z,
                delta,
                m.source,
            )
        
        console.print(table)
        
        # Deep dive analysis if available
        if hasattr(pillar, "deep_dive"):
            analysis = pillar.deep_dive()
            
            if "interpretation" in analysis:
                console.print(Panel(
                    analysis["interpretation"],
                    title="Interpretation",
                    expand=False,
                ))
            
            if "triggers" in analysis and analysis["triggers"]:
                console.print(Panel(
                    "\n".join(f"• {t}" for t in analysis["triggers"]),
                    title="[yellow]Active Triggers[/yellow]",
                    expand=False,
                ))


def create_default_dashboard() -> RegimeDashboard:
    """Create dashboard with all standard pillars."""
    from lox.regimes.pillars import (
        InflationPillar,
        GrowthPillar,
        LiquidityPillar,
        VolatilityPillar,
    )
    
    dashboard = RegimeDashboard()
    dashboard.add_pillar(InflationPillar())
    dashboard.add_pillar(GrowthPillar())
    dashboard.add_pillar(LiquidityPillar())
    dashboard.add_pillar(VolatilityPillar())
    
    return dashboard

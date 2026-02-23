"""
Rich rendering for cross-regime scenario triggers.

Follows display conventions from regime_display.py and book_impact.py:
- Rich Tables with box=None, padding=(0, 2)
- Direction color coding: Long=green, Short=red, Reduce=yellow, Hedge=blue
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from lox.regimes.scenarios import ScenarioResult


_CONVICTION_STYLES: dict[str, tuple[str, str]] = {
    "HIGH": ("bold red", "red"),
    "MEDIUM": ("bold yellow", "yellow"),
    "LOW": ("bold dim", "dim"),
}


def _direction_color(direction: str) -> str:
    d = direction.upper()
    if d == "LONG":
        return "green"
    if d == "SHORT":
        return "red"
    if d == "REDUCE":
        return "yellow"
    if d == "HEDGE":
        return "blue"
    return "dim"


def render_scenario_panel(scenario: "ScenarioResult", console: Console) -> None:
    """Render a single scenario as a detailed Rich Panel with trade table."""
    parts: list = []

    style, border = _CONVICTION_STYLES.get(scenario.conviction, ("dim", "dim"))

    parts.append(Text.from_markup(
        f"[{style}]{scenario.conviction} CONVICTION[/{style}]  "
        f"({scenario.conditions_met}/{scenario.conditions_total} conditions met)\n"
    ))
    parts.append(Text.from_markup(f"{scenario.thesis}\n"))

    if scenario.trigger_metrics:
        triggers = "  ".join(
            f"[bold]{d}[/bold] {m}" for d, m in scenario.trigger_metrics.items()
        )
        parts.append(Text.from_markup(f"\n[cyan]Triggers:[/cyan]  {triggers}\n"))

    trade_table = Table(
        show_header=True, header_style="bold",
        box=None, padding=(0, 2),
    )
    trade_table.add_column("Direction", min_width=7)
    trade_table.add_column("Ticker", style="cyan", min_width=6)
    trade_table.add_column("Via", min_width=8, style="dim")
    trade_table.add_column("Rationale", ratio=2)
    trade_table.add_column("Size", min_width=9, style="dim")

    for t in scenario.trades:
        c = _direction_color(t.direction)
        trade_table.add_row(
            f"[{c}]{t.direction}[/{c}]",
            t.ticker,
            t.instrument,
            t.rationale,
            t.sizing_hint.upper(),
        )

    parts.append(Text.from_markup("\n"))
    parts.append(trade_table)
    parts.append(Text.from_markup(f"\n[dim]Risk: {scenario.primary_risk}[/dim]"))

    console.print(Panel.fit(
        Group(*parts),
        title=f"[bold]{scenario.name}[/bold]",
        border_style=border,
        padding=(0, 2),
    ))


def render_scenarios_summary(
    scenarios: list["ScenarioResult"],
    console: Console,
) -> None:
    """Compact summary table of all active scenarios for the unified dashboard."""
    if not scenarios:
        return

    table = Table(
        title="[bold]Active Macro Scenarios[/bold]",
        show_header=True, header_style="bold",
        box=None, padding=(0, 2),
    )
    table.add_column("Scenario", style="bold", min_width=22)
    table.add_column("Conviction", min_width=10)
    table.add_column("Hit", justify="center", min_width=5)
    table.add_column("Pillars", ratio=1)
    table.add_column("Top Trade", ratio=1)

    for s in scenarios:
        style = _CONVICTION_STYLES.get(s.conviction, ("dim", "dim"))[0]

        top = s.trades[0] if s.trades else None
        top_str = ""
        if top:
            c = _direction_color(top.direction)
            top_str = f"[{c}]{top.direction}[/{c}] {top.ticker}"

        table.add_row(
            s.name,
            f"[{style}]{s.conviction}[/{style}]",
            f"{s.conditions_met}/{s.conditions_total}",
            ", ".join(s.domains_involved),
            top_str,
        )

    console.print(table)

    # Drill-down hint with scenario details
    console.print(
        f"\n[dim]{len(scenarios)} scenario(s) active  |  "
        f"Use --detail <pillar> for scenario trades  |  "
        f"--llm for AI scenario analysis[/dim]"
    )

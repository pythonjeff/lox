"""
Rich rendering for cross-pillar market dislocations.

Follows display conventions from scenario_display.py:
- Rich Tables with box=None, padding=(0, 2)
- Severity color coding: HIGH=red, MEDIUM=yellow
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from lox.regimes.inconsistencies import MarketDislocation


_SEVERITY_STYLES: dict[str, tuple[str, str]] = {
    "HIGH": ("bold red", "red"),
    "MEDIUM": ("bold yellow", "yellow"),
}


def render_dislocations_panel(
    dislocations: list["MarketDislocation"],
    console: Console,
) -> None:
    """Render market dislocations as a summary table + detail panels for HIGH."""
    if not dislocations:
        return

    # ── Summary table ────────────────────────────────────────────────────
    table = Table(
        title="[bold]Market Dislocations[/bold]",
        show_header=True, header_style="bold",
        box=None, padding=(0, 2),
    )
    table.add_column("Dislocation", style="bold", no_wrap=True)
    table.add_column("", min_width=6)
    table.add_column("Pillars")
    table.add_column("Signal", ratio=1)

    for d in dislocations:
        style = _SEVERITY_STYLES.get(d.severity, ("dim", "dim"))[0]

        # Build a short signal: just the key metrics
        metrics_parts = [f"{k}: {v}" for k, v in d.metrics_snapshot.items()]
        signal = "  ".join(metrics_parts) if metrics_parts else ""

        table.add_row(
            d.name,
            f"[{style}]{d.severity}[/{style}]",
            ", ".join(d.domains_involved),
            signal,
        )

    console.print(table)

    # ── Detail panels for all dislocations ───────────────────────────────
    console.print()
    for d in dislocations:
        style, border = _SEVERITY_STYLES.get(d.severity, ("dim", "dim"))
        parts: list = []
        parts.append(Text.from_markup(f"[{style}]{d.severity}[/{style}]  {d.thesis}\n"))
        parts.append(Text.from_markup(f"\n[cyan]Trade:[/cyan] {d.trade_implication}"))

        console.print(Panel.fit(
            Group(*parts),
            title=f"[bold]{d.name}[/bold]",
            border_style=border,
            padding=(0, 2),
        ))

    high_count = sum(1 for d in dislocations if d.severity == "HIGH")
    med_count = len(dislocations) - high_count

    severity_parts = []
    if high_count:
        severity_parts.append(f"[bold red]{high_count} HIGH[/bold red]")
    if med_count:
        severity_parts.append(f"[yellow]{med_count} MEDIUM[/yellow]")

    console.print(
        f"\n[dim]{len(dislocations)} dislocation(s): {', '.join(severity_parts)}  |  "
        f"--llm for AI analysis of trade opportunities[/dim]"
    )

"""
Uniform regime output for all `lox regime <domain>` commands.

CFA/quant-style: consistent Panel layout, score bar, metrics table, and description.
"""
from __future__ import annotations

from typing import Any

from rich.panel import Panel
from rich.table import Table


def score_color(score: float) -> str:
    """Return Rich color tag for score value. Higher = more stress (red)."""
    if score < 30:
        return "green"
    if score < 60:
        return "yellow"
    return "red"


def render_score_bar(score: float, width: int = 50) -> str:
    """Render a colored score bar. score 0-100."""
    filled = int(round((score / 100.0) * width))
    filled = max(0, min(width, filled))
    empty = width - filled
    color = score_color(score)
    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"


def render_regime_panel(
    *,
    domain: str,
    asof: str,
    regime_label: str,
    score: float | None = None,
    percentile: float | None = None,
    description: str = "",
    metrics: list[dict[str, Any]],
    sub_scores: list[dict[str, Any]] | None = None,
) -> Panel:
    """
    Render a standardized regime panel for CLI output.

    Args:
        domain: Display name, e.g. "Fiscal", "Volatility".
        asof: As-of date string (e.g. ISO date).
        regime_label: Regime name, e.g. "STRESS BUILDING".
        score: 0-100 composite score (None = omit score bar).
        percentile: 0-100 historical percentile (None = omit from bar line).
        description: 1-2 sentence regime explanation.
        metrics: List of {"name": str, "value": str, "context": str} for main table.
        sub_scores: Optional list of {"name": str, "score": float, "percentile": float | None} for breakdown table.
    """
    from rich.console import Group
    from rich.text import Text

    parts: list[Any] = []
    parts.append(Text.from_markup(f"As of: {asof}\n"))
    if score is not None:
        color = score_color(score)
        pct_suffix = f"  {percentile:.0f}th %ile" if percentile is not None else ""
        parts.append(Text.from_markup(
            f"[bold]{regime_label}[/bold]   "
            f"Score: [{color}]{score:.0f}/100[/{color}]   "
            f"{render_score_bar(score)}{pct_suffix}\n\n"
        ))
    else:
        parts.append(Text.from_markup(f"[bold]{regime_label}[/bold]\n\n"))
    if description:
        parts.append(Text.from_markup(f"{description}\n\n"))
    if metrics:
        mt = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
        mt.add_column("Metric", style="cyan")
        mt.add_column("Value", justify="right")
        mt.add_column("Context", style="dim")
        for m in metrics:
            mt.add_row(
                str(m.get("name", "—")),
                str(m.get("value", "—")),
                str(m.get("context", "")),
            )
        parts.append(mt)
    if sub_scores:
        parts.append(Text.from_markup("\n"))
        st = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
        st.add_column("Pillar", style="magenta")
        st.add_column("Score", justify="right")
        # Show weight if any sub_score has it; otherwise show percentile
        has_weight = any(s.get("weight") is not None for s in sub_scores)
        if has_weight:
            st.add_column("Weight", justify="right", style="dim")
            st.add_column("Bar", min_width=20)
        else:
            st.add_column("%ile", justify="right", style="dim")
        for s in sub_scores:
            sc = s.get("score")
            sc_str = f"{sc:.0f}" if sc is not None else "—"
            if has_weight:
                wt = s.get("weight", "—")
                bar = render_score_bar(float(sc), width=20) if sc is not None else ""
                st.add_row(str(s.get("name", "—")), sc_str, str(wt), bar)
            else:
                pct = s.get("percentile")
                st.add_row(
                    str(s.get("name", "—")),
                    sc_str,
                    f"{pct:.0f}%" if pct is not None else "—",
                )
        parts.append(st)

    return Panel.fit(
        Group(*parts),
        title=f"{domain} Regime",
        border_style="cyan",
    )


def print_llm_regime_analysis(
    *,
    settings: Any,
    domain: str,
    snapshot: dict[str, Any] | Any,
    regime_label: str | None = None,
    regime_description: str | None = None,
    panel_title: str = "Analysis",
    console: Any = None,
    **llm_kwargs: Any,
) -> None:
    """
    Uniform LLM analysis block for all regime commands.
    Uses the compact, metrics-focused prompt (300–400 words, tables, minimal prose).
    """
    from rich import print as rprint
    from rich.markdown import Markdown

    from lox.llm.core.analyst import llm_analyze_regime

    _print = console.print if console is not None else rprint
    _print("\n[bold cyan]Generating LLM analysis...[/bold cyan]\n")

    analysis = llm_analyze_regime(
        settings=settings,
        domain=domain,
        snapshot=snapshot,
        regime_label=regime_label,
        regime_description=regime_description,
        **llm_kwargs,
    )

    _print(Panel(Markdown(analysis), title=panel_title, expand=False))

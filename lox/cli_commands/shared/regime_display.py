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
    if score < 35:
        return "green"
    if score < 65:
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
    trend: Any | None = None,
) -> Any:
    """
    Render a standardized regime display for CLI output.

    Returns a Group containing a compact header Panel (regime label, score,
    trend) followed by borderless detail tables (description, metrics,
    sub-scores).  All regime commands use ``from rich import print`` so the
    Group renders correctly with no caller changes.
    """
    from rich.console import Group
    from rich.text import Text

    # ── Compact header panel (verdict) ───────────────────────────────────
    header_parts: list[Any] = []
    if score is not None:
        color = score_color(score)
        pct_suffix = f"  {percentile:.0f}th %ile" if percentile is not None else ""
        header_parts.append(Text.from_markup(
            f"[bold]{regime_label}[/bold]   "
            f"Score: [{color}]{score:.0f}/100[/{color}]   "
            f"{render_score_bar(score)}{pct_suffix}"
        ))
    else:
        header_parts.append(Text.from_markup(f"[bold]{regime_label}[/bold]"))

    if trend is not None:
        markup = _trend_markup(trend)
        header_parts.append(Text.from_markup(f"\n{markup}"))

    panel = Panel.fit(
        Group(*header_parts),
        title=f"{domain} Regime",
        subtitle=f"[dim]As of: {asof}[/dim]",
        border_style="cyan",
    )

    # ── Detail sections (outside the panel border) ───────────────────────
    all_parts: list[Any] = [panel]

    if description:
        all_parts.append(Text.from_markup(f"\n{description}"))

    if metrics:
        has_change = any(m.get("change") for m in metrics)
        mt = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2))
        mt.add_column("Metric", style="bold")
        mt.add_column("Value", justify="right")
        if has_change:
            mt.add_column("Chg", justify="right", min_width=10)
        mt.add_column("Context", style="dim")
        for m in metrics:
            row = [str(m.get("name", "—")), str(m.get("value", "—"))]
            if has_change:
                row.append(str(m.get("change", "")))
            row.append(str(m.get("context", "")))
            mt.add_row(*row)
        all_parts.append(Text(""))
        all_parts.append(mt)

    if sub_scores:
        st = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2))
        st.add_column("Pillar", style="bold")
        st.add_column("Score", justify="right")
        has_weight = any(s.get("weight") is not None for s in sub_scores)
        has_contrib = any(s.get("contrib") is not None for s in sub_scores)
        if has_weight:
            st.add_column("Weight", justify="right", style="dim")
            if has_contrib:
                st.add_column("Contrib", justify="right", style="dim")
            st.add_column("Bar", min_width=20)
        else:
            st.add_column("%ile", justify="right", style="dim")

        prev_section: str | None = None
        for s in sub_scores:
            # Optional section divider (e.g. weighted pillars vs diagnostic sub-scores).
            # When `section` is provided and changes between rows, draw a labelled
            # separator row so the reader can see the boundary at a glance.
            section = s.get("section")
            if section is not None and section != prev_section and prev_section is not None:
                label = s.get("section_label", section)
                # Render the divider as an empty row (visual whitespace) followed
                # by a labelled row that sits in the Pillar column. Avoids the
                # awkward column-width tug-of-war that wrapped long labels.
                pad_cols = 4 if (has_weight and has_contrib) else (3 if has_weight else 2)
                st.add_row(*(["" for _ in range(pad_cols + 1)]))
                st.add_row(f"[dim italic]{label}[/dim italic]", *(["" for _ in range(pad_cols)]))
            prev_section = section if section is not None else prev_section

            sc = s.get("score")
            sc_str = f"{sc:.0f}" if sc is not None else "—"
            name = str(s.get("name", "—"))
            if has_weight:
                wt = s.get("weight", "—")
                bar = render_score_bar(float(sc), width=20) if sc is not None else ""
                if has_contrib:
                    cb = s.get("contrib")
                    cb_str = f"{cb:.1f}" if isinstance(cb, (int, float)) else "—"
                    st.add_row(name, sc_str, str(wt), cb_str, bar)
                else:
                    st.add_row(name, sc_str, str(wt), bar)
            else:
                pct = s.get("percentile")
                st.add_row(
                    name,
                    sc_str,
                    f"{pct:.0f}%" if pct is not None else "—",
                )
        all_parts.append(Text(""))
        all_parts.append(st)

    return Group(*all_parts)


def _trend_markup(trend: Any) -> str:
    """Return Rich markup string for a trend/momentum summary line."""
    tc = trend.trend_color
    arrow = trend.trend_arrow

    prev_str = ""
    if trend.prev_label and trend.prev_label != trend.current_label:
        prev_c = score_color(trend.prev_score) if trend.prev_score is not None else "dim"
        prev_str = f"  Prev: [{prev_c}]{trend.prev_label}[/{prev_c}]"

    deltas = []
    for label, val in [("1d", trend.score_chg_1d), ("7d", trend.score_chg_7d), ("30d", trend.score_chg_30d)]:
        if val is not None:
            sign = "+" if val > 0 else ""
            if abs(val) < 0.5:
                deltas.append(f"[dim]{label} {sign}{val:.1f}[/dim]")
            else:
                c = "red" if val > 0 else "green"
                deltas.append(f"[{c}]{label} {sign}{val:.1f}[/{c}]")
    delta_str = "  ".join(deltas)

    momo_str = ""
    if trend.momentum_z is not None:
        sign = "+" if trend.momentum_z > 0 else ""
        if abs(trend.momentum_z) >= 1.5:
            mc = "bold red" if trend.momentum_z > 0 else "bold green"
        elif abs(trend.momentum_z) >= 1.0:
            mc = "red" if trend.momentum_z > 0 else "green"
        else:
            mc = "dim"
        momo_str = f"  Momo: [{mc}]{sign}{trend.momentum_z:.1f}σ[/{mc}]"

    days_str = f"  [{tc}]{trend.days_in_regime}d in regime[/{tc}]"

    return f"[{tc}]{arrow} {trend.trend_label}[/{tc}]{prev_str}  {delta_str}{momo_str}{days_str}"


def _render_trend_line(parts: list, trend: Any) -> None:
    """Inject a compact trend summary line into a parts list (legacy helper)."""
    from rich.text import Text
    parts.append(Text.from_markup(_trend_markup(trend) + "\n\n"))


def print_llm_regime_analysis(
    *,
    settings: Any,
    domain: str,
    snapshot: dict[str, Any] | Any,
    regime_label: str | None = None,
    regime_description: str | None = None,
    panel_title: str = "Analysis",
    console: Any = None,
    ticker: str = "",
    book_impacts: list | None = None,
    active_scenarios: list | None = None,
    **llm_kwargs: Any,
) -> None:
    """
    Interactive LLM chat for regime commands.

    Launches a conversation pre-loaded with the regime snapshot (and optional
    ticker context) so the user can ask follow-up questions.
    """
    from rich.console import Console as _Console

    from lox.cli_commands.shared.regime_chat import start_regime_chat

    start_regime_chat(
        settings=settings,
        domain=domain,
        snapshot=snapshot,
        regime_label=regime_label,
        regime_description=regime_description,
        ticker=ticker,
        console=console if isinstance(console, _Console) else None,
        book_impacts=book_impacts,
        active_scenarios=active_scenarios,
    )

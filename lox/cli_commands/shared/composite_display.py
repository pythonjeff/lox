"""
Rich rendering for composite regime classification.

Follows display conventions from regime_display.py and scenario_display.py:
- Rich Tables with box=None, padding=(0, 2)
- Panel.fit() with border_style="cyan"
- Score colors: green <35, yellow 35-64, red >=65
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from lox.regimes.composite import CompositeRegimeResult, SwingFactor


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _score_color(score: float) -> str:
    if score >= 65:
        return "red"
    if score >= 35:
        return "yellow"
    return "green"


def _confidence_color(conf: float) -> str:
    if conf >= 0.5:
        return "green"
    if conf >= 0.25:
        return "yellow"
    return "red"


def _stance_color(stance: str) -> str:
    s = stance.upper()
    if s in ("OVERWEIGHT", "LONG", "SELL", "FULL"):
        return "green"
    if s in ("UNDERWEIGHT", "SHORT", "BUY", "MINIMAL"):
        return "red"
    return "yellow"


def _direction_color(direction: str) -> str:
    d = direction.upper()
    if d == "LONG":
        return "green"
    if d in ("SHORT", "REDUCE"):
        return "red"
    return "yellow"


def _prob_bar(prob: float, width: int = 20) -> str:
    filled = int(prob * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def _regime_style(regime: str) -> str:
    styles = {
        "RISK_ON": "bold green",
        "REFLATION": "bold yellow",
        "STAGFLATION": "bold red",
        "RISK_OFF": "bold magenta",
        "TRANSITION": "bold cyan",
    }
    return styles.get(regime, "bold")


# ═════════════════════════════════════════════════════════════════════════════
# Dashboard (full output for `lox regime composite`)
# ═════════════════════════════════════════════════════════════════════════════

def render_composite_dashboard(
    result: CompositeRegimeResult,
    console: Console,
) -> None:
    """Full composite regime dashboard — PM morning briefing format."""

    style = _regime_style(result.regime)
    conf_color = _confidence_color(result.confidence)

    # ── Header Panel ──────────────────────────────────────────────────────
    header_parts = []
    header_parts.append(Text.from_markup(
        f"[{style}]REGIME: {result.label}[/{style}]  "
        f"[{conf_color}]({result.confidence:.0%} confidence)[/{conf_color}]\n"
    ))
    header_parts.append(Text.from_markup(f"{result.description}\n"))
    header_parts.append(Text.from_markup(
        f"\n[dim]Pillar Dispersion: {result.pillar_dispersion:.1f}"
        f"  ({'high — mixed signals' if result.pillar_dispersion > 18 else 'moderate' if result.pillar_dispersion > 12 else 'low — clear regime'})[/dim]"
    ))

    console.print()
    console.print(Panel.fit(
        Group(*header_parts),
        title="[bold]COMPOSITE REGIME[/bold]",
        border_style="cyan",
        padding=(0, 2),
    ))

    # ── Pillar Score Vector ───────────────────────────────────────────────
    console.print()
    pillar_table = Table(
        show_header=True, header_style="bold",
        box=None, padding=(0, 2),
        title="[bold]Pillar Scores[/bold]",
    )
    pillar_table.add_column("Pillar", style="bold", min_width=12)
    pillar_table.add_column("Score", justify="right", min_width=6)
    pillar_table.add_column("Bar", min_width=22)

    from lox.regimes.composite import CLASSIFICATION_PILLARS, REGIME_PROTOTYPES
    proto = REGIME_PROTOTYPES.get(result.regime, {})

    for pillar in CLASSIFICATION_PILLARS:
        score = result.score_vector.get(pillar, 50.0)
        ideal = proto.get(pillar, 50.0)
        color = _score_color(score)
        bar_len = int(score / 100 * 20)
        bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)

        delta = score - ideal
        if abs(delta) < 5:
            alignment = "[green]aligned[/green]"
        elif abs(delta) < 15:
            alignment = f"[yellow]{delta:+.0f} from ideal[/yellow]"
        else:
            alignment = f"[red]{delta:+.0f} from ideal[/red]"

        pillar_table.add_row(
            pillar.upper(),
            f"[{color}]{score:.0f}[/{color}]",
            f"[{color}]{bar}[/{color}]  {alignment}",
        )

    console.print(pillar_table)

    # ── Regime Probabilities + Transition Outlook ─────────────────────────
    console.print()
    prob_table = Table(
        show_header=True, header_style="bold",
        box=None, padding=(0, 2),
        title="[bold]Regime Probabilities[/bold]",
    )
    prob_table.add_column("Regime", style="bold", min_width=24)
    prob_table.add_column("Now", justify="right", min_width=6)
    prob_table.add_column("", min_width=22)
    prob_table.add_column("Next Mo", justify="right", min_width=8)
    prob_table.add_column("Shift", justify="right", min_width=8)

    from lox.regimes.composite import COMPOSITE_REGIMES, COMPOSITE_LABELS

    # Sort by current probability descending
    sorted_regimes = sorted(
        COMPOSITE_REGIMES,
        key=lambda r: result.regime_probabilities.get(r, 0),
        reverse=True,
    )

    for r in sorted_regimes:
        now_prob = result.regime_probabilities.get(r, 0)
        next_prob = result.transition_outlook.get(r, 0)
        shift = next_prob - now_prob

        bar = _prob_bar(now_prob)
        r_style = _regime_style(r)
        label = COMPOSITE_LABELS[r]

        marker = " *" if r == result.regime else ""
        shift_color = "green" if shift < -0.02 else "red" if shift > 0.02 else "dim"
        shift_arrow = "\u2191" if shift > 0.02 else "\u2193" if shift < -0.02 else "\u2014"

        prob_table.add_row(
            f"[{r_style}]{label}{marker}[/{r_style}]",
            f"{now_prob:.0%}",
            f"[dim]{bar}[/dim]",
            f"{next_prob:.0%}",
            f"[{shift_color}]{shift:+.0%} {shift_arrow}[/{shift_color}]",
        )

    console.print(prob_table)

    # ── Swing Factors ─────────────────────────────────────────────────────
    if result.swing_factors:
        console.print()
        swing_table = Table(
            show_header=True, header_style="bold",
            box=None, padding=(0, 2),
            title="[bold]Swing Factors[/bold] — What Flips the Regime",
        )
        swing_table.add_column("Pillar", style="bold", min_width=12)
        swing_table.add_column("Score", justify="right", min_width=6)
        swing_table.add_column("Target", justify="right", min_width=7)
        swing_table.add_column("Regime if Flip", min_width=22)
        swing_table.add_column("Distance", justify="right", min_width=9)
        swing_table.add_column("Velocity", justify="right", min_width=9)
        swing_table.add_column("ETA", justify="right", min_width=8)

        for sf in result.swing_factors[:6]:
            dir_arrow = "\u2191" if sf.direction == "UP" else "\u2193"
            vel_str = f"{sf.velocity_7d:+.1f}/d" if sf.velocity_7d is not None else "[dim]n/a[/dim]"

            if sf.days_to_flip is not None:
                if sf.days_to_flip < 14:
                    eta_str = f"[red]~{sf.days_to_flip:.0f}d[/red]"
                elif sf.days_to_flip < 30:
                    eta_str = f"[yellow]~{sf.days_to_flip:.0f}d[/yellow]"
                else:
                    eta_str = f"[dim]~{sf.days_to_flip:.0f}d[/dim]"
            else:
                eta_str = "[dim]n/a[/dim]"

            target_style = _regime_style(sf.target_regime)
            swing_table.add_row(
                sf.pillar.upper(),
                f"[{_score_color(sf.current_score)}]{sf.current_score:.0f}[/{_score_color(sf.current_score)}]",
                f"{sf.target_score:.0f} {dir_arrow}",
                f"[{target_style}]{COMPOSITE_LABELS[sf.target_regime]}[/{target_style}]",
                f"{sf.distance_to_flip:.1f}",
                vel_str,
                eta_str,
            )

        console.print(swing_table)

    # ── Playbook Panel ────────────────────────────────────────────────────
    console.print()
    pb = result.playbook
    playbook_parts = []

    # Stances row
    stances = [
        ("Equity", pb.equity_stance),
        ("Duration", pb.duration_stance),
        ("Credit", pb.credit_stance),
        ("Commodities", pb.commodity_stance),
        ("Vol", pb.vol_stance),
        ("Cash", f"{pb.cash_target_pct:.0f}%"),
        ("Gross", pb.gross_exposure),
    ]
    stance_str = "  ".join(
        f"[bold]{name}:[/bold] [{_stance_color(val)}]{val}[/{_stance_color(val)}]"
        for name, val in stances
    )
    playbook_parts.append(Text.from_markup(stance_str + "\n"))

    # Key expressions table
    expr_table = Table(
        show_header=True, header_style="bold",
        box=None, padding=(0, 2),
    )
    expr_table.add_column("Direction", min_width=7)
    expr_table.add_column("Ticker", style="cyan", min_width=6)
    expr_table.add_column("Rationale", ratio=2)

    for direction, ticker, rationale in pb.key_expressions:
        c = _direction_color(direction)
        expr_table.add_row(
            f"[{c}]{direction}[/{c}]",
            ticker,
            rationale,
        )

    playbook_parts.append(Text.from_markup("\n"))
    playbook_parts.append(expr_table)

    console.print(Panel.fit(
        Group(*playbook_parts),
        title=f"[bold]CANONICAL PLAYBOOK: {result.label}[/bold]",
        border_style="cyan",
        padding=(0, 2),
    ))
    console.print()


# ═════════════════════════════════════════════════════════════════════════════
# Headline (compact one-liner for PM report / overview integration)
# ═════════════════════════════════════════════════════════════════════════════

def render_composite_headline(
    result: CompositeRegimeResult,
    console: Console,
) -> None:
    """
    One-line headline for embedding in PM report or overview headers.

    Format: REGIME: STAGFLATION (67%) -> RISK-OFF rising (22%)
    """
    from lox.regimes.composite import COMPOSITE_LABELS

    style = _regime_style(result.regime)
    conf_color = _confidence_color(result.confidence)

    # Find the regime with the biggest positive shift in transition outlook
    biggest_shift_regime = None
    biggest_shift = 0.0
    for r in result.transition_outlook:
        if r == result.regime:
            continue
        shift = result.transition_outlook[r] - result.regime_probabilities.get(r, 0)
        if shift > biggest_shift:
            biggest_shift = shift
            biggest_shift_regime = r

    line = (
        f"[{style}]{result.label}[/{style}]  "
        f"[{conf_color}]({result.confidence:.0%} confidence)[/{conf_color}]"
    )
    if biggest_shift_regime and biggest_shift > 0.02:
        t_style = _regime_style(biggest_shift_regime)
        t_prob = result.transition_outlook[biggest_shift_regime]
        line += f"  [dim]\u2192[/dim]  [{t_style}]{COMPOSITE_LABELS[biggest_shift_regime]}[/{t_style}] [dim]rising ({t_prob:.0%})[/dim]"

    console.print(f"[bold]Regime:[/bold] {line}")

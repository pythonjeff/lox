"""
Quant-style regime trend & momentum display.

Bloomberg/Citadel-style dashboard showing:
- Current state vs previous state for every pillar
- Score deltas at 1d/7d/30d horizons
- Trend arrows, momentum z-scores, velocity
- Regime persistence and transition frequency
- 30-day score sparkline
"""
from __future__ import annotations

from typing import Any, Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lox.regimes.trend import (
    RegimeTrend,
    TREND_DETERIORATING,
    TREND_WEAKENING,
    TREND_STABLE,
    TREND_IMPROVING,
    TREND_STRENGTHENING,
)

# ── Score coloring (consistent with regime_display.py) ────────────────────
def _score_color(score: float) -> str:
    if score < 35:
        return "green"
    if score < 65:
        return "yellow"
    return "red"


def _fmt_delta(val: Optional[float], invert: bool = False) -> str:
    """Format a score delta with color. Positive = more stress = red by default.
    Set invert=True if positive is good (e.g., for improving pillars).
    """
    if val is None:
        return "[dim]—[/dim]"
    sign = "+" if val > 0 else ""
    if abs(val) < 0.5:
        return f"[dim]{sign}{val:.1f}[/dim]"
    # Positive delta = score UP = more stress (bad) → red
    # Unless inverted
    if (val > 0 and not invert) or (val < 0 and invert):
        color = "red"
    elif (val < 0 and not invert) or (val > 0 and invert):
        color = "green"
    else:
        color = "dim"
    return f"[{color}]{sign}{val:.1f}[/{color}]"


def _fmt_momentum_z(z: Optional[float]) -> str:
    """Format momentum z-score."""
    if z is None:
        return "[dim]—[/dim]"
    sign = "+" if z > 0 else ""
    if abs(z) >= 2.0:
        color = "bold red" if z > 0 else "bold green"
    elif abs(z) >= 1.0:
        color = "red" if z > 0 else "green"
    else:
        color = "dim"
    return f"[{color}]{sign}{z:.1f}σ[/{color}]"


def _fmt_velocity(v: Optional[float]) -> str:
    """Format velocity (daily slope)."""
    if v is None:
        return "[dim]—[/dim]"
    sign = "+" if v > 0 else ""
    if abs(v) >= 2.0:
        color = "bold red" if v > 0 else "bold green"
    elif abs(v) >= 0.5:
        color = "red" if v > 0 else "green"
    else:
        color = "dim"
    return f"[{color}]{sign}{v:.1f}/d[/{color}]"


def _sparkline(series_scores: list[float], width: int = 20) -> str:
    """Generate a unicode sparkline from score history."""
    if not series_scores or len(series_scores) < 2:
        return "[dim]—[/dim]"
    # Resample to `width` points
    n = len(series_scores)
    if n > width:
        step = n / width
        sampled = [series_scores[int(i * step)] for i in range(width)]
    else:
        sampled = series_scores

    # Map scores (0-100) to sparkline characters
    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx = 0, 100  # Fixed scale for consistency across pillars
    out = []
    for v in sampled:
        idx = int((v - mn) / (mx - mn) * (len(blocks) - 1))
        idx = max(0, min(len(blocks) - 1, idx))
        out.append(blocks[idx])

    # Color the whole sparkline based on latest value
    color = _score_color(sampled[-1])
    return f"[{color}]{''.join(out)}[/{color}]"


def render_trend_dashboard(
    trends: dict[str, RegimeTrend],
    all_series: dict[str, list[dict]],
    console: Console,
    title: str = "REGIME TREND & MOMENTUM",
) -> None:
    """
    Render the full trend dashboard — all pillars with deltas, momentum, sparklines.

    This is the main entry point called from the regime command.
    """
    if not trends:
        console.print("[dim]No trend data available yet. Run regimes again to start building history.[/dim]")
        return

    # ═══════════════════════════════════════════════════════════════════════
    # Main Trend Table
    # ═══════════════════════════════════════════════════════════════════════
    tbl = Table(
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 1),
        title=f"[bold]{title}[/bold]",
        title_style="bold cyan",
    )
    tbl.add_column("Pillar", style="bold", min_width=10, no_wrap=True)
    tbl.add_column("Score", justify="center", min_width=5, no_wrap=True)
    tbl.add_column("Trend", justify="center", min_width=5, no_wrap=True)
    tbl.add_column("Regime", min_width=14, no_wrap=True)
    tbl.add_column("Prev", min_width=14, no_wrap=True)
    tbl.add_column("Δ1d", justify="right", min_width=5, no_wrap=True)
    tbl.add_column("Δ7d", justify="right", min_width=5, no_wrap=True)
    tbl.add_column("Δ30d", justify="right", min_width=6, no_wrap=True)
    tbl.add_column("Momo", justify="right", min_width=6, no_wrap=True)
    tbl.add_column("Vel", justify="right", min_width=6, no_wrap=True)
    tbl.add_column("Days", justify="right", min_width=4, no_wrap=True)
    tbl.add_column("30d Range", justify="center", min_width=10, no_wrap=True)
    tbl.add_column("Spark", min_width=20, no_wrap=True)

    # Sort: core first, then extended (match ALL_DOMAINS order)
    from lox.regimes.features import ALL_DOMAINS
    domain_order = {d: i for i, d in enumerate(ALL_DOMAINS)}
    sorted_domains = sorted(trends.keys(), key=lambda d: domain_order.get(d, 99))

    for domain in sorted_domains:
        t = trends[domain]
        color = _score_color(t.current_score)

        # Trend arrow with color
        arrow = f"[{t.trend_color}]{t.trend_arrow}[/{t.trend_color}]"

        # Current regime
        regime_str = f"[{color}]{t.current_label}[/{color}]"

        # Previous regime (highlight if changed)
        if t.prev_label and t.prev_label != t.current_label:
            prev_color = _score_color(t.prev_score) if t.prev_score is not None else "dim"
            prev_str = f"[{prev_color}]{t.prev_label}[/{prev_color}]"
        elif t.prev_label:
            prev_str = f"[dim]{t.prev_label}[/dim]"
        else:
            prev_str = "[dim]—[/dim]"

        # Days in regime
        days_str = f"{t.days_in_regime}d" if t.days_in_regime > 0 else "[dim]new[/dim]"

        # 30d range
        if t.score_30d_low is not None and t.score_30d_high is not None:
            range_str = f"{t.score_30d_low:.0f}-{t.score_30d_high:.0f}"
        else:
            range_str = "[dim]—[/dim]"

        # Sparkline from score_series
        series = all_series.get(domain, [])
        spark_scores = [float(e["score"]) for e in series]
        spark_str = _sparkline(spark_scores)

        tbl.add_row(
            domain.upper(),
            f"[{color}]{t.current_score:.0f}[/{color}]",
            arrow,
            regime_str,
            prev_str,
            _fmt_delta(t.score_chg_1d),
            _fmt_delta(t.score_chg_7d),
            _fmt_delta(t.score_chg_30d),
            _fmt_momentum_z(t.momentum_z),
            _fmt_velocity(t.velocity_7d),
            days_str,
            range_str,
            spark_str,
        )

    console.print(tbl)

    # ═══════════════════════════════════════════════════════════════════════
    # Alerts: regime transitions and extreme momentum
    # ═══════════════════════════════════════════════════════════════════════
    alerts = []
    for domain in sorted_domains:
        t = trends[domain]
        # Regime changed from previous snapshot
        if t.prev_label and t.prev_label != t.current_label:
            prev_c = _score_color(t.prev_score) if t.prev_score is not None else "dim"
            curr_c = _score_color(t.current_score)
            alerts.append(
                f"  [{prev_c}]{t.prev_label}[/{prev_c}] → [{curr_c}]{t.current_label}[/{curr_c}]  "
                f"({domain.upper()} score {t.prev_score:.0f} → {t.current_score:.0f})"
            )
        # Extreme momentum (|z| >= 2)
        if t.momentum_z is not None and abs(t.momentum_z) >= 2.0:
            direction = "accelerating stress" if t.momentum_z > 0 else "rapid improvement"
            alerts.append(
                f"  [bold]{domain.upper()}[/bold] momentum {t.momentum_z:+.1f}σ — {direction}"
            )
        # High transition count (unstable regime)
        if t.transitions_30d >= 3:
            alerts.append(
                f"  [bold]{domain.upper()}[/bold] unstable: {t.transitions_30d} transitions in 30d"
            )

    if alerts:
        console.print()
        console.print(Panel(
            "\n".join(alerts),
            title="[bold yellow]Regime Alerts[/bold yellow]",
            border_style="yellow",
            padding=(0, 2),
        ))


def render_trend_detail(
    trend: RegimeTrend,
    series: list[dict],
    console: Console,
) -> None:
    """
    Render detailed trend view for a single pillar.

    Shows expanded metrics, score history, and transition log.
    """
    color = _score_color(trend.current_score)

    parts: list[Any] = []

    # ── Header line ───────────────────────────────────────────────────────
    header = Text.from_markup(
        f"[bold]{trend.domain.upper()} TREND & MOMENTUM[/bold]\n"
    )
    parts.append(header)

    # ── Current vs Previous ───────────────────────────────────────────────
    state_tbl = Table(box=None, padding=(0, 2), show_header=False)
    state_tbl.add_column("", style="dim", min_width=16)
    state_tbl.add_column("", min_width=30)

    state_tbl.add_row("Current", f"[{color}]{trend.current_label}[/{color}]  [{color}]{trend.current_score:.0f}/100[/{color}]")
    if trend.prev_label:
        prev_c = _score_color(trend.prev_score) if trend.prev_score is not None else "dim"
        prev_score_str = f"  [{prev_c}]{trend.prev_score:.0f}/100[/{prev_c}]" if trend.prev_score is not None else ""
        state_tbl.add_row("Previous", f"[{prev_c}]{trend.prev_label}[/{prev_c}]{prev_score_str}  [dim]({trend.prev_date})[/dim]")
    state_tbl.add_row("Trend", f"[{trend.trend_color}]{trend.trend_arrow} {trend.trend_direction}[/{trend.trend_color}]")
    state_tbl.add_row("Days in Regime", f"{trend.days_in_regime}d")
    parts.append(state_tbl)
    parts.append(Text(""))

    # ── Score deltas table ────────────────────────────────────────────────
    delta_tbl = Table(
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 2),
        title="[bold]Score Dynamics[/bold]",
    )
    delta_tbl.add_column("Window", style="cyan", min_width=10)
    delta_tbl.add_column("Δ Score", justify="right", min_width=8)
    delta_tbl.add_column("Signal", min_width=20)

    for label, val in [("1-Day", trend.score_chg_1d), ("7-Day", trend.score_chg_7d), ("30-Day", trend.score_chg_30d)]:
        signal = ""
        if val is not None:
            if abs(val) > 10:
                signal = "[bold red]LARGE MOVE[/bold red]" if val > 0 else "[bold green]LARGE IMPROVEMENT[/bold green]"
            elif abs(val) > 5:
                signal = "[red]Meaningful shift[/red]" if val > 0 else "[green]Meaningful shift[/green]"
            else:
                signal = "[dim]Within normal range[/dim]"
        delta_tbl.add_row(label, _fmt_delta(val), signal)
    parts.append(delta_tbl)
    parts.append(Text(""))

    # ── Momentum & Velocity ───────────────────────────────────────────────
    momo_tbl = Table(
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 2),
        title="[bold]Momentum & Velocity[/bold]",
    )
    momo_tbl.add_column("Metric", style="cyan", min_width=16)
    momo_tbl.add_column("Value", justify="right", min_width=10)
    momo_tbl.add_column("Interpretation", min_width=24)

    # Momentum z
    momo_interp = "[dim]Neutral[/dim]"
    if trend.momentum_z is not None:
        if abs(trend.momentum_z) >= 2.0:
            momo_interp = "[bold red]Extreme — regime change risk[/bold red]" if trend.momentum_z > 0 else "[bold green]Extreme — rapid normalization[/bold green]"
        elif abs(trend.momentum_z) >= 1.0:
            momo_interp = "[red]Above average stress acceleration[/red]" if trend.momentum_z > 0 else "[green]Above average improvement[/green]"
    momo_tbl.add_row("Momentum (z)", _fmt_momentum_z(trend.momentum_z), momo_interp)

    # Velocity
    vel_interp = "[dim]Flat[/dim]"
    if trend.velocity_7d is not None:
        if abs(trend.velocity_7d) >= 2.0:
            vel_interp = "[bold]High velocity — watch for breakout[/bold]"
        elif abs(trend.velocity_7d) >= 0.5:
            vel_interp = "Moderate directional drift"
    momo_tbl.add_row("Velocity (7d)", _fmt_velocity(trend.velocity_7d), vel_interp)

    # Score volatility
    if trend.score_volatility is not None:
        vol_interp = "[dim]Stable readings[/dim]"
        if trend.score_volatility > 10:
            vol_interp = "[yellow]High score dispersion — noisy signal[/yellow]"
        elif trend.score_volatility > 5:
            vol_interp = "Moderate variation"
        momo_tbl.add_row("Score Vol (30d)", f"{trend.score_volatility:.1f}", vol_interp)

    # Transitions
    if trend.transitions_30d > 0:
        trans_interp = "[yellow]Unstable — low conviction[/yellow]" if trend.transitions_30d >= 3 else "[dim]Some regime churn[/dim]"
        momo_tbl.add_row("Transitions (30d)", str(trend.transitions_30d), trans_interp)

    # Range
    if trend.score_30d_low is not None and trend.score_30d_high is not None:
        rng = trend.score_30d_high - trend.score_30d_low
        range_interp = "[yellow]Wide range — high uncertainty[/yellow]" if rng > 25 else "[dim]Contained[/dim]"
        momo_tbl.add_row("30d Range", f"{trend.score_30d_low:.0f} – {trend.score_30d_high:.0f}", range_interp)

    parts.append(momo_tbl)
    parts.append(Text(""))

    # ── Sparkline history ─────────────────────────────────────────────────
    if series and len(series) >= 2:
        spark_scores = [float(e["score"]) for e in series]
        spark = _sparkline(spark_scores, width=40)
        parts.append(Text.from_markup(f"[bold]Score History:[/bold]  {spark}"))
        # Show date range
        first_date = series[0].get("date", "?")
        last_date = series[-1].get("date", "?")
        parts.append(Text.from_markup(f"[dim]{first_date} → {last_date}  ({len(series)} observations)[/dim]"))

    console.print(Panel.fit(
        Group(*parts),
        title=f"[bold cyan]{trend.domain.upper()} Trend[/bold cyan]",
        border_style="cyan",
    ))

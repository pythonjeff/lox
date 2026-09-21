"""
Rich display for the movement scanner (`lox movers`).

Follows existing conventions: Table(box=None, padding=(0,2)),
Panel.fit(border_style="cyan"), green/yellow/dim score bands.
"""
from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lox.suggest.movers import MoverCandidate, MoversResult

_CHARACTER_STYLE = {
    "TRENDER": ("TREND", "green"),
    "CHOPPY": ("CHOP", "yellow"),
    "GAPPER": ("GAP", "magenta"),
    "QUIET": ("QUIET", "dim"),
}

_KIND_ORDER = ["DIRECTIONAL", "LONG_VOL", "SHORT_VOL", "EVENT", "WATCH"]
_KIND_LABEL = {
    "DIRECTIONAL": ("Directional — trade the side", "green"),
    "LONG_VOL": ("Long vol — trade the range", "cyan"),
    "SHORT_VOL": ("Short vol — sell the range", "yellow"),
    "EVENT": ("Event-driven — dated risk", "magenta"),
    "WATCH": ("Watchlist — nothing to pay for yet", "dim"),
}


def _score_cell(score: float) -> str:
    if score >= 65:
        return f"[bold green]{score:.0f}[/bold green]"
    if score >= 45:
        return f"[yellow]{score:.0f}[/yellow]"
    return f"[dim]{score:.0f}[/dim]"


def _dir_cell(direction: str) -> str:
    if direction == "LONG":
        return "[bold green]LONG[/bold green]"
    if direction == "SHORT":
        return "[bold red]SHORT[/bold red]"
    return "[dim]—[/dim]"


def _char_cell(character: str) -> str:
    label, style = _CHARACTER_STYLE.get(character, (character, "dim"))
    return f"[{style}]{label}[/{style}]"


def _expansion_cell(ratio: float) -> str:
    if ratio >= 1.15:
        return f"[green]{ratio:.2f}x[/green]"
    if ratio <= 0.90:
        return f"[dim]{ratio:.2f}x[/dim]"
    return f"{ratio:.2f}x"


def render_movers_header(console: Console, result: MoversResult) -> None:
    thr = f"{result.move_threshold:.0%}"
    lines = [
        f"[bold]Movement screen[/bold]  ·  {result.window}d window  ·  "
        f"move bar {thr}",
        f"[dim]{result.universe_size} universe → {result.liquidity_survivors} liquid → "
        f"{result.deep_pool} deep → {result.scored} scored[/dim]",
    ]
    if result.missing_history:
        n = len(result.missing_history)
        sample = ", ".join(result.missing_history[:5])
        lines.append(f"[dim]no history: {n} ({sample}{'…' if n > 5 else ''})[/dim]")
    console.print(Panel.fit("\n".join(lines), border_style="cyan"))
    console.print()


def render_movers_table(console: Console, candidates: list[MoverCandidate]) -> None:
    """The ranking table — one row per name, movement stats front and centre."""
    t = Table(box=None, padding=(0, 1))
    t.add_column("Ticker", style="bold", min_width=5, no_wrap=True)
    t.add_column("Name", style="dim", min_width=10, max_width=14, no_wrap=True, overflow="ellipsis")
    t.add_column("Scr", justify="right", min_width=3)
    t.add_column("Move", justify="right", min_width=10, no_wrap=True)
    t.add_column("E[1m]", justify="right", min_width=5)
    t.add_column("Vol", justify="right", min_width=5)
    t.add_column("Type", min_width=5, no_wrap=True)
    t.add_column("Dir", min_width=5, no_wrap=True)
    t.add_column("Trade", min_width=22, no_wrap=True, overflow="ellipsis")

    for c in candidates:
        k = c.kinetics
        t.add_row(
            c.ticker,
            c.name,
            _score_cell(k.sub_score),
            f"{k.avg_abs_move_pct:.1f}%[dim]/{k.move_freq_pct:.0f}%[/dim]",
            f"{k.expected_move_21d_pct:.0f}%",
            _expansion_cell(k.vol_expansion),
            _char_cell(k.character),
            _dir_cell(c.direction),
            f"[dim]{c.structure_short or c.structure}[/dim]",
        )

    console.print(t)
    console.print()
    console.print(
        "[dim]Move = mean daily move / share of sessions over the move bar · "
        "E[1m] = 1-sigma monthly move · Vol = 20d vs 60d realized vol[/dim]"
    )
    console.print()


def render_playbook(console: Console, candidates: list[MoverCandidate], limit: int = 5) -> None:
    """Top names expanded into an actionable block with the next command."""
    if not candidates:
        return

    console.print("[bold cyan]How to trade them[/bold cyan]\n")

    by_kind: dict[str, list[MoverCandidate]] = {}
    for c in candidates[:limit]:
        by_kind.setdefault(c.structure_kind, []).append(c)

    for kind in _KIND_ORDER:
        group = by_kind.get(kind, [])
        if not group:
            continue
        label, style = _KIND_LABEL.get(kind, (kind, "cyan"))
        console.print(f"[bold {style}]{label}[/bold {style}]")
        for c in group:
            k = c.kinetics
            console.print(
                f"  [bold]{c.ticker}[/bold] [dim]${c.price:,.2f}[/dim]  {c.structure}"
            )
            console.print(
                f"    [dim]{k.avg_abs_move_pct:.1f}%/day · {k.big_move_count} big moves "
                f"in {k.sample_days} sessions · {c.notes}[/dim]"
            )
            console.print(f"    [cyan]{c.handoff}[/cyan]")
        console.print()


def render_movers_dashboard(console: Console, result: MoversResult) -> None:
    console.print()
    render_movers_header(console, result)

    if not result.candidates:
        console.print("[dim]No names cleared the liquidity and movement filters.[/dim]")
        console.print("[dim]Try: lox movers --min-dollar-volume 5000000 --move 0.015[/dim]")
        return

    render_movers_table(console, result.candidates)
    render_playbook(console, result.candidates)


def format_movers_json(result: MoversResult) -> dict[str, Any]:
    return {
        "scan_timestamp": result.scan_timestamp,
        "window": result.window,
        "move_threshold": result.move_threshold,
        "universe_size": result.universe_size,
        "liquidity_survivors": result.liquidity_survivors,
        "deep_pool": result.deep_pool,
        "scored": result.scored,
        "missing_history": result.missing_history,
        "candidates": [
            {
                "ticker": c.ticker,
                "name": c.name,
                "price": c.price,
                "change_pct": c.change_pct,
                "dollar_volume": c.dollar_volume,
                "sector": c.sector,
                "is_etf": c.is_etf,
                "direction": c.direction,
                "trend_quality": c.trend_quality,
                "rsi_14": c.rsi_14,
                "zscore_20d": c.zscore_20d,
                "structure": c.structure,
                "structure_short": c.structure_short,
                "structure_kind": c.structure_kind,
                "handoff": c.handoff,
                "notes": c.notes,
                "kinetics": {
                    "score": c.kinetics.sub_score,
                    "avg_abs_move_pct": c.kinetics.avg_abs_move_pct,
                    "max_1d_move_pct": c.kinetics.max_1d_move_pct,
                    "move_freq_pct": c.kinetics.move_freq_pct,
                    "big_move_count": c.kinetics.big_move_count,
                    "persistence": c.kinetics.persistence,
                    "rv_20d": c.kinetics.rv_20d,
                    "rv_60d": c.kinetics.rv_60d,
                    "vol_expansion": c.kinetics.vol_expansion,
                    "trend_efficiency": c.kinetics.trend_efficiency,
                    "net_return_pct": c.kinetics.net_return_pct,
                    "expected_daily_move_pct": c.kinetics.expected_daily_move_pct,
                    "expected_move_21d_pct": c.kinetics.expected_move_21d_pct,
                    "character": c.kinetics.character,
                    "sample_days": c.kinetics.sample_days,
                },
            }
            for c in result.candidates
        ],
    }


def format_movers_for_llm(result: MoversResult) -> str:
    """Compact text block for the regime chat context."""
    lines = [
        f"MOVEMENT SCREEN ({result.window}d window, "
        f"{result.move_threshold:.0%} move bar, {result.scored} names scored)",
        "",
    ]
    for c in result.candidates:
        k = c.kinetics
        lines.append(
            f"{c.ticker} ({c.name}) score {k.sub_score:.0f} | "
            f"{k.avg_abs_move_pct:.1f}%/day, {k.move_freq_pct:.0f}% of sessions over "
            f"{result.move_threshold:.0%} | 1m expected move {k.expected_move_21d_pct:.0f}% | "
            f"vol {k.vol_expansion:.2f}x | {k.character} | {c.direction} | {c.structure}"
        )
    return "\n".join(lines)

"""
Rich display panels for the opportunity scanner (lox suggest --universe scan).

Follows existing conventions: Table(box=None, padding=(0,2)),
Panel.fit(border_style="cyan"), score colors green<35 yellow<65 red>=65.
"""
from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lox.suggest.opportunity import ScoredOpportunity
from lox.suggest.scanner import ScannerResult


def _score_dot(score: float) -> str:
    """Colored score with dot prefix."""
    if score >= 65:
        return f"[bold green]{score:.0f}[/bold green]"
    if score >= 45:
        return f"[yellow]{score:.0f}[/yellow]"
    return f"[dim]{score:.0f}[/dim]"


def _dir_style(direction: str) -> str:
    return "bold green" if direction == "LONG" else "bold red"


def _conv_style(conviction: str) -> str:
    if conviction == "HIGH":
        return "bold green"
    if conviction == "MEDIUM":
        return "yellow"
    return "dim"


def _short_name(name: str, sector: str, is_etf: bool) -> str:
    """Build a compact display name: company short name or ETF description + sector."""
    if not name or name == "":
        return sector or ""

    # For ETFs, the TICKER_DESC is already short (e.g. "crude oil", "gold miners")
    if is_etf:
        return name

    # For stocks, strip common suffixes and truncate
    short = name
    for suffix in (
        " Inc.", ", Inc.", " Inc", " Corporation", " Corp.", " Corp",
        " Company", " Co.", " Holdings", " Holding", " Ltd.", " Ltd",
        " PLC", " plc", " N.V.", " SE", " S.A.", " Group",
        " International", " Intl", " Worldwide",
    ):
        if short.endswith(suffix):
            short = short[: -len(suffix)]
            break

    # Strip trailing commas and "The " prefix
    short = short.rstrip(",").strip()
    if short.startswith("The "):
        short = short[4:]

    # If still long, shorten
    if len(short) > 16 and sector:
        first_word = short.split()[0] if short.split() else short
        sector_short = sector.split()[0].title() if sector else ""
        return f"{first_word} ({sector_short})"

    return short


def _signal_type_label(signal_type: str) -> tuple[str, str]:
    """Return (display_label, border_style) for a signal type."""
    labels = {
        "REGIME_TAILWIND": ("Regime Tailwinds", "cyan"),
        "FLOW_ACCELERATION": ("Flow Acceleration", "magenta"),
        "REVERSION_SETUP": ("Reversion Setups", "yellow"),
        "CATALYST_DRIVEN": ("Catalyst Driven", "green"),
    }
    return labels.get(signal_type, (signal_type, "white"))


def render_scanner_header(console: Console, result: ScannerResult) -> None:
    """One-line scan summary header."""
    regime = result.regime_headline or "UNKNOWN"
    regime_upper = regime.upper()
    if "RISK_ON" in regime_upper or "GOLDILOCKS" in regime_upper:
        style = "green"
    elif "RISK_OFF" in regime_upper or "STAG" in regime_upper:
        style = "red"
    else:
        style = "yellow"

    rotation_flag = " | rotation applied" if result.rotation_applied else ""

    console.print(
        f"  [{style}]{regime}[/{style}]"
        f"  |  {result.universe_size} scanned"
        f"  |  {result.pass1_survivors} filtered"
        f"  |  {len(result.candidates)} opportunities"
        f"{rotation_flag}",
        style="dim",
    )
    console.print()


def render_signal_group(
    console: Console,
    group_name: str,
    border_style: str,
    candidates: list[ScoredOpportunity],
) -> None:
    """Render one signal group as a Rich table."""
    if not candidates:
        return

    t = Table(
        box=None,
        padding=(0, 2),
        title=f"[bold {border_style}]{group_name}[/bold {border_style}]",
        title_style=f"bold {border_style}",
    )
    t.add_column("Ticker", style="bold", min_width=5, no_wrap=True)
    t.add_column("Name", style="dim", min_width=14, max_width=16, no_wrap=True, overflow="ellipsis")
    t.add_column("Dir", min_width=5, no_wrap=True)
    t.add_column("Scr", justify="right", min_width=3)
    t.add_column("Thesis", min_width=30, no_wrap=True, overflow="ellipsis")

    for c in candidates:
        dir_str = f"[{_dir_style(c.direction)}]{c.direction}[/{_dir_style(c.direction)}]"
        short_name = _short_name(c.name, c.sector, c.is_etf)

        t.add_row(
            c.ticker,
            short_name,
            dir_str,
            _score_dot(c.composite_score),
            f"[dim]{c.thesis}[/dim]",
        )

    console.print(t)
    console.print()


def render_scanner_dashboard(console: Console, result: ScannerResult) -> None:
    """Main display entry point. Groups candidates by signal type."""
    console.print()
    render_scanner_header(console, result)

    if not result.candidates:
        console.print("[dim]No opportunities found in this scan.[/dim]")
        return

    # Group by signal type
    groups: dict[str, list[ScoredOpportunity]] = {}
    for c in result.candidates:
        groups.setdefault(c.signal_type, []).append(c)

    # Render in priority order
    for signal_type in [
        "REGIME_TAILWIND", "FLOW_ACCELERATION",
        "REVERSION_SETUP", "CATALYST_DRIVEN",
    ]:
        candidates = groups.get(signal_type, [])
        if candidates:
            label, style = _signal_type_label(signal_type)
            render_signal_group(console, label, style, candidates)


def render_track_record(console: Console) -> None:
    """Render the track record performance dashboard."""
    from lox.suggest.track_record import compute_track_record

    tr = compute_track_record(lookback_days=60)
    if tr.total_recs == 0:
        console.print("[dim]No recommendation history yet. Run `lox suggest` to start building a track record.[/dim]")
        return

    console.print()

    # Signal type performance table
    t = Table(
        box=None,
        padding=(0, 2),
        title="[bold cyan]Suggestion Track Record (60 days)[/bold cyan]",
        title_style="bold cyan",
    )
    t.add_column("Signal Type", style="bold", min_width=20)
    t.add_column("Recs", justify="right")
    t.add_column("Hit 5d", justify="right")
    t.add_column("Hit 20d", justify="right")
    t.add_column("Avg Ret", justify="right")
    t.add_column("Adj", justify="right")

    from lox.suggest.track_record import get_weight_adjustments
    adjustments = get_weight_adjustments()

    # Map signal types to display
    signal_order = [
        "REGIME_TAILWIND", "FLOW_ACCELERATION",
        "REVERSION_SETUP", "CATALYST_DRIVEN",
    ]
    signal_to_pillar = {
        "REGIME_TAILWIND": "regime",
        "FLOW_ACCELERATION": "flow",
        "REVERSION_SETUP": "momentum",
        "CATALYST_DRIVEN": "catalyst",
    }

    for st in signal_order:
        stats = tr.by_signal_type.get(st, {})
        count = stats.get("count", 0)
        hit5 = stats.get("hit_rate_5d")
        hit20 = stats.get("hit_rate_20d")
        avg_ret = stats.get("avg_return")
        pillar = signal_to_pillar.get(st, "")
        adj = adjustments.get(pillar, 1.0)

        hit5_str = f"[{'green' if hit5 and hit5 > 0.5 else 'red'}]{hit5:.0%}[/]" if hit5 is not None else "[dim]—[/dim]"
        hit20_str = f"[{'green' if hit20 and hit20 > 0.5 else 'red'}]{hit20:.0%}[/]" if hit20 is not None else "[dim]—[/dim]"
        ret_str = f"[{'green' if avg_ret and avg_ret > 0 else 'red'}]{avg_ret:+.1%}[/]" if avg_ret is not None else "[dim]—[/dim]"
        adj_str = f"[yellow]{adj:.1f}x[/yellow]" if adj != 1.0 else "[dim]1.0x[/dim]"

        label, _ = _signal_type_label(st)
        t.add_row(label, str(count), hit5_str, hit20_str, ret_str, adj_str)

    console.print(t)

    # Overall summary
    parts = [f"[bold]{tr.total_recs}[/bold] total recs"]
    if tr.hit_rate_5d is not None:
        color = "green" if tr.hit_rate_5d > 0.5 else "red"
        parts.append(f"[{color}]{tr.hit_rate_5d:.0%} hit (5d)[/{color}]")
    if tr.hit_rate_20d is not None:
        color = "green" if tr.hit_rate_20d > 0.5 else "red"
        parts.append(f"[{color}]{tr.hit_rate_20d:.0%} hit (20d)[/{color}]")
    if tr.avg_return_20d is not None:
        color = "green" if tr.avg_return_20d > 0 else "red"
        parts.append(f"[{color}]{tr.avg_return_20d:+.1%} avg[/{color}]")

    console.print()
    console.print("  " + "  |  ".join(parts))
    console.print()


def format_scanner_json(result: ScannerResult) -> dict[str, Any]:
    """JSON-serializable output."""
    return {
        "regime_headline": result.regime_headline,
        "composite_regime": result.composite_regime,
        "regime_confidence": result.regime_confidence,
        "universe_size": result.universe_size,
        "pass1_survivors": result.pass1_survivors,
        "scan_timestamp": result.scan_timestamp,
        "rotation_applied": result.rotation_applied,
        "candidates": [
            {
                "ticker": c.ticker,
                "name": c.name,
                "direction": c.direction,
                "composite_score": c.composite_score,
                "conviction": c.conviction,
                "signal_type": c.signal_type,
                "momentum_score": c.momentum_score,
                "flow_score": c.flow_score,
                "regime_score": c.regime_score,
                "catalyst_score": c.catalyst_score,
                "price": c.price,
                "change_pct": c.change_pct,
                "volume_surge": c.volume_surge,
                "zscore_20d": c.zscore_20d,
                "sector": c.sector,
                "is_etf": c.is_etf,
                "thesis": c.thesis,
            }
            for c in result.candidates
        ],
    }


def format_scanner_for_llm(result: ScannerResult) -> str:
    """Format scanner results as text for LLM context injection."""
    lines = [
        f"## Opportunity Scanner Results",
        f"Regime: {result.regime_headline}",
        f"Universe: {result.universe_size} tickers scanned, {result.pass1_survivors} filtered, {len(result.candidates)} opportunities",
        "",
    ]

    for c in result.candidates:
        lines.append(
            f"- **{c.ticker}** ({c.direction}, {c.conviction}): "
            f"score {c.composite_score:.0f} "
            f"[regime {c.regime_score:.0f}, flow {c.flow_score:.0f}, "
            f"mom {c.momentum_score:.0f}, cat {c.catalyst_score:.0f}] "
            f"— {c.thesis}"
        )

    return "\n".join(lines)

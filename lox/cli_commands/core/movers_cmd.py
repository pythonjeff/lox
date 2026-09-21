"""
`lox movers` — rank the universe by how much it moves and how often.

Where `lox suggest` looks for what is happening today, this looks for names that
reliably deliver range, then maps each one onto a trade structure.

Usage:
    lox movers                         # top 20 movers across S&P 500 + Dow + ETFs
    lox movers -n 30                   # deeper list
    lox movers --move 0.03             # only count 3%+ days as "a move"
    lox movers --window 120            # measure over ~6 months instead of ~3
    lox movers --character TRENDER     # only names whose moves stick
    lox movers --character CHOPPY      # only range traders (long/short vol)
    lox movers --etf-only              # macro ETFs, no single names
    lox movers --universe core         # ~30 liquid macro ETFs (fast)
    lox movers -t NVDA                 # single-name movement profile
    lox movers --llm                   # hand the screen to the analyst chat
    lox movers --json                  # machine-readable
"""
from __future__ import annotations

import json
import logging

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from lox.config import load_settings

logger = logging.getLogger(__name__)

_VALID_CHARACTERS = {"TRENDER", "CHOPPY", "GAPPER", "QUIET"}


def register(app: typer.Typer) -> None:
    """Register `lox movers` on the main app."""

    @app.command("movers")
    def movers_cmd(
        count: int = typer.Option(20, "--count", "-n", help="Number of names to return"),
        window: int = typer.Option(60, "--window", "-w", help="Lookback in trading sessions"),
        move: float = typer.Option(
            0.02, "--move", "-m",
            help="Daily move that counts as 'a move' (0.02 = 2%)",
        ),
        universe: str = typer.Option(
            "scan", "--universe", "-u",
            help="'scan' (S&P 500 + Dow + ETFs), 'etf' (macro basket), 'core' (~30 ETFs)",
        ),
        character: str = typer.Option(
            "", "--character", "-c",
            help="Filter: TRENDER, CHOPPY, GAPPER, QUIET",
        ),
        ticker: str = typer.Option("", "--ticker", "-t", help="Single-name movement profile"),
        etf_only: bool = typer.Option(False, "--etf-only", help="Exclude individual stocks"),
        min_price: float = typer.Option(5.0, "--min-price", help="Minimum share price"),
        min_dollar_volume: float = typer.Option(
            20_000_000.0, "--min-dollar-volume",
            help="Minimum average daily dollar volume",
        ),
        deep_pool: int = typer.Option(
            120, "--pool",
            help="How many prefilter survivors get full price history",
        ),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh cached prices"),
        llm: bool = typer.Option(False, "--llm", help="Hand the screen to the analyst chat"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
    ):
        """Find names that move a lot, often — and how to trade them."""
        console = Console()
        settings = load_settings()

        char = character.strip().upper()
        if char and char not in _VALID_CHARACTERS:
            console.print(
                f"[yellow]Unknown character '{character}'. "
                f"Valid: {', '.join(sorted(_VALID_CHARACTERS))}[/yellow]"
            )
            raise typer.Exit(code=1)

        if window < 25:
            console.print("[yellow]--window must be at least 25 sessions.[/yellow]")
            raise typer.Exit(code=1)

        from lox.suggest.movers import run_movers_scan
        from lox.cli_commands.shared.movers_display import (
            format_movers_for_llm,
            format_movers_json,
            render_movers_dashboard,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console,
        ) as prog:
            prog.add_task("Measuring how much each name moves...", total=None)
            result = run_movers_scan(
                settings=settings,
                count=count,
                window=window,
                move_threshold=move,
                min_price=min_price,
                min_dollar_volume=min_dollar_volume,
                deep_pool=deep_pool,
                universe_name=universe.strip().lower(),
                character=char,
                etf_only=etf_only,
                ticker=ticker.strip(),
                refresh=refresh,
            )

        if json_out:
            console.print_json(json.dumps(format_movers_json(result)))
            return

        render_movers_dashboard(console, result)

        if llm and result.candidates:
            from lox.cli_commands.shared.regime_chat import start_regime_chat
            start_regime_chat(
                domain="movers",
                snapshot={"movement_screen": format_movers_for_llm(result)},
                state=None,
                settings=settings,
            )

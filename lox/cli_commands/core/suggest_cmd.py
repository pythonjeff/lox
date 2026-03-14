"""
Mean-reversion screener command — find overextended ETFs with macro attribution.

Usage:
    lox suggest                      # Core universe, top 5 candidates
    lox suggest --universe full      # Expanded 100+ ticker scan
    lox suggest -t USO               # Single-ticker deep dive
    lox suggest --threshold 2.0      # Stricter z-score cutoff
    lox suggest --llm                # LLM analysis of results
"""
from __future__ import annotations

import json
import logging

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from lox.config import load_settings

logger = logging.getLogger(__name__)


def register(app: typer.Typer) -> None:
    """Register `lox suggest` on the main app."""

    @app.command("suggest")
    def suggest_cmd(
        universe: str = typer.Option(
            "core", "--universe", "-u",
            help="Universe: 'core' (~30 macro ETFs) or 'full' (~100+ ETFs)",
        ),
        llm: bool = typer.Option(False, "--llm", help="LLM analysis of top candidates"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
        ticker: str = typer.Option("", "--ticker", "-t", help="Single-ticker reversion analysis"),
        threshold: float = typer.Option(1.5, "--threshold", help="Z-score threshold for extended (default 1.5)"),
        lookback: int = typer.Option(20, "--lookback", help="Return lookback in trading days (default 20)"),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh cached price data"),
        count: int = typer.Option(5, "--count", "-n", help="Number of top candidates to show (default 5)"),
    ):
        """Mean-reversion screener — find overextended ETFs with macro attribution."""
        from lox.suggest.reversion import run_reversion_screen
        from lox.cli_commands.shared.reversion_display import (
            render_reversion_dashboard,
            format_reversion_json,
            format_reversion_for_llm,
        )

        console = Console()
        settings = load_settings()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console,
        ) as prog:
            prog.add_task("Scanning for reversion candidates...", total=None)
            result = run_reversion_screen(
                settings=settings,
                universe=universe,
                threshold=threshold,
                lookback=lookback,
                count=count,
                ticker=ticker.strip(),
                refresh=refresh,
            )

        if json_out:
            console.print_json(json.dumps(format_reversion_json(result)))
            return

        render_reversion_dashboard(console, result)

        if llm and result.candidates:
            from lox.cli_commands.shared.regime_chat import start_regime_chat
            llm_context = format_reversion_for_llm(result)
            start_regime_chat(
                domain="suggest",
                snapshot={"reversion_screen": llm_context},
                state=None,
                settings=settings,
            )

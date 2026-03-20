"""
Opportunity scanner — quant-level idea generation across S&P 500 + Dow + macro ETFs.

Usage:
    lox suggest                       # Full scanner (S&P 500 + Dow + ETFs)
    lox suggest --universe core       # Legacy: 30 macro ETFs reversion screen
    lox suggest --universe full       # Legacy: 100+ ETF reversion screen
    lox suggest --signal flow         # Filter to flow acceleration candidates
    lox suggest --signal tailwind     # Filter to regime tailwind candidates
    lox suggest --etf-only            # Exclude individual stocks
    lox suggest --deep                # MC + extended flow analysis
    lox suggest --track-record        # Show suggestion performance dashboard
    lox suggest -n 15                 # Top 15 instead of default 10
    lox suggest -t AAPL               # Single-ticker deep dive
    lox suggest --llm                 # LLM analysis of results
    lox suggest --json                # Machine-readable output
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
            "scan", "--universe", "-u",
            help="'scan' (S&P 500+ETFs, default), 'core' (~30 ETFs), 'full' (~100 ETFs)",
        ),
        llm: bool = typer.Option(False, "--llm", help="LLM analysis of top candidates"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
        ticker: str = typer.Option("", "--ticker", "-t", help="Single-ticker analysis"),
        threshold: float = typer.Option(1.5, "--threshold", help="Z-score threshold (reversion mode only)"),
        lookback: int = typer.Option(20, "--lookback", help="Return lookback days (reversion mode only)"),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh cached data"),
        count: int = typer.Option(10, "--count", "-n", help="Number of top candidates (default 10)"),
        deep: bool = typer.Option(False, "--deep", help="Run Monte Carlo + extended flow analysis"),
        signal: str = typer.Option("", "--signal", "-s", help="Filter by signal: tailwind, flow, reversion, catalyst"),
        etf_only: bool = typer.Option(False, "--etf-only", help="Exclude individual stocks"),
        track_record: bool = typer.Option(False, "--track-record", help="Show suggestion performance dashboard"),
    ):
        """Opportunity scanner — find actionable ideas across equities and macro ETFs."""
        console = Console()
        settings = load_settings()

        # ── Track record mode ──
        if track_record:
            from lox.cli_commands.shared.scanner_display import render_track_record
            render_track_record(console)
            return

        # ── Legacy reversion mode (backward compatible) ──
        if universe in ("core", "full"):
            from lox.suggest.reversion import run_reversion_screen
            from lox.cli_commands.shared.reversion_display import (
                render_reversion_dashboard,
                format_reversion_json,
                format_reversion_for_llm,
            )

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
            return

        # ── Full opportunity scanner (default) ──
        from lox.suggest.scanner import run_opportunity_scan
        from lox.cli_commands.shared.scanner_display import (
            render_scanner_dashboard,
            format_scanner_json,
            format_scanner_for_llm,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console,
        ) as prog:
            prog.add_task("Scanning S&P 500 + Dow + ETFs for opportunities...", total=None)
            result = run_opportunity_scan(
                settings=settings,
                count=count,
                refresh=refresh,
                deep=deep,
                ticker=ticker.strip(),
                signal_filter=signal.strip(),
                etf_only=etf_only,
            )

        if json_out:
            console.print_json(json.dumps(format_scanner_json(result)))
            return

        render_scanner_dashboard(console, result)

        if llm and result.candidates:
            from lox.cli_commands.shared.regime_chat import start_regime_chat
            llm_context = format_scanner_for_llm(result)
            start_regime_chat(
                domain="suggest",
                snapshot={"opportunity_scan": llm_context},
                state=None,
                settings=settings,
            )

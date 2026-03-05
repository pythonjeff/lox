"""
LOX Research: Scenario Prediction Tracker

Review past scenario predictions against actual outcomes.

Usage:
    lox research scenario-track              # Full scorecard
    lox research scenario-track --active     # Pending predictions only
    lox research scenario-track --symbol SPY # Filter to one ticker
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lox.config import load_settings


def register(app: typer.Typer) -> None:
    """Register the scenario-track command."""

    @app.command("scenario-track")
    def scenario_track_cmd(
        symbol: str | None = typer.Option(None, "--symbol", "-s", help="Filter to a specific ticker"),
        active: bool = typer.Option(False, "--active", help="Show only pending (unresolved) predictions"),
    ):
        """
        Review scenario predictions and track accuracy.

        Shows all past scenario runs with predicted vs actual outcomes.
        Predictions are auto-scored once their horizon has elapsed.

        Examples:
            lox research scenario-track
            lox research scenario-track --symbol SPY
            lox research scenario-track --active
        """
        console = Console()
        settings = load_settings()

        from lox.scenarios.tracking import load_predictions, score_predictions, compute_scorecard
        from rich.progress import Progress, SpinnerColumn, TextColumn

        # Score any completed predictions
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Scoring predictions...[/bold cyan]"),
            transient=True,
        ) as progress:
            progress.add_task("score", total=None)
            predictions = score_predictions(settings)

        if symbol:
            predictions = [p for p in predictions if p.get("symbol") == symbol.upper()]

        if not predictions:
            console.print("[dim]No scenario predictions logged yet.[/dim]")
            console.print("[dim]Run `lox research scenario SPY --oil 90` to create one.[/dim]")
            return

        if active:
            predictions = [p for p in predictions if p.get("actual_price") is None]
            if not predictions:
                console.print("[dim]No pending predictions.[/dim]")
                return

        console.print()
        console.print("[bold cyan]LOX SCENARIO TRACKER[/bold cyan]")
        console.print()

        _show_predictions_table(console, predictions)

        # Scorecard (only if we have scored predictions)
        scored = [p for p in predictions if p.get("actual_price") is not None]
        if scored and not active:
            scorecard = compute_scorecard(predictions)
            _show_scorecard(console, scorecard)

        console.print()

    return


def _show_predictions_table(console: Console, predictions: list[dict]) -> None:
    """Render the predictions table."""
    from datetime import date

    table = Table(
        box=None,
        padding=(0, 2),
        show_header=True,
        header_style="bold",
    )
    table.add_column("Date", min_width=10)
    table.add_column("Ticker", style="bold", min_width=6)
    table.add_column("Entry $", justify="right", min_width=8)
    table.add_column("p50 Target", justify="right", min_width=10)
    table.add_column("Horizon", justify="right", min_width=8)
    table.add_column("Actual $", justify="right", min_width=9)
    table.add_column("Percentile", justify="right", min_width=10)
    table.add_column("Scenario Hit", min_width=12)
    table.add_column("Status", min_width=10)

    today = date.today()

    for p in predictions:
        ts = p["timestamp"][:10]
        sym = p["symbol"]
        entry = p["current_price"]
        p50 = p["full_distribution"]["p50"]
        days = p["horizon_days"]

        end_date = date.fromisoformat(p["horizon_end_date"])
        days_left = (end_date - today).days

        if p.get("actual_price") is not None:
            actual = p["actual_price"]
            pct = p.get("actual_percentile", 0)
            scenario = p.get("actual_scenario", "?")

            # Color the percentile
            if 25 <= pct <= 75:
                pct_color = "green"
            elif 10 <= pct <= 90:
                pct_color = "yellow"
            else:
                pct_color = "red"

            scenario_label = scenario.replace("_", " ") if scenario else "?"

            actual_ret = (actual / entry - 1) * 100
            actual_color = "green" if actual_ret > 0 else "red"

            table.add_row(
                ts,
                sym,
                f"${entry:,.0f}",
                f"${p50:,.0f}",
                f"{days}d",
                f"[{actual_color}]${actual:,.0f}[/{actual_color}]",
                f"[{pct_color}]p{pct:.0f}[/{pct_color}]",
                scenario_label,
                "[green]scored[/green]",
            )
        elif days_left > 0:
            table.add_row(
                ts,
                sym,
                f"${entry:,.0f}",
                f"${p50:,.0f}",
                f"{days}d",
                "[dim]—[/dim]",
                "[dim]—[/dim]",
                "[dim]—[/dim]",
                f"[yellow]{days_left}d left[/yellow]",
            )
        else:
            table.add_row(
                ts,
                sym,
                f"${entry:,.0f}",
                f"${p50:,.0f}",
                f"{days}d",
                "[dim]pending[/dim]",
                "[dim]—[/dim]",
                "[dim]—[/dim]",
                "[cyan]awaiting[/cyan]",
            )

    console.print(table)
    console.print()


def _show_scorecard(console: Console, scorecard: dict) -> None:
    """Render the aggregate accuracy scorecard."""
    n = scorecard.get("n_scored", 0)
    if n == 0:
        return

    pending = scorecard.get("n_pending", 0)

    lines = [
        f"  [bold]Predictions scored:[/bold]  {n}    [dim]({pending} pending)[/dim]",
        "",
        f"  [bold]Actual in BASE range:[/bold]     {scorecard.get('pct_in_base', 0):.0f}%  [dim](ideal ~45%)[/dim]",
        f"  [bold]Actual in p25–p75:[/bold]        {scorecard.get('pct_in_iqr', 0):.0f}%  [dim](ideal ~50%)[/dim]",
        f"  [bold]Actual in p10–p90:[/bold]        {scorecard.get('pct_in_p10_p90', 0):.0f}%  [dim](ideal ~80%)[/dim]",
        f"  [bold]Mean actual percentile:[/bold]   {scorecard.get('mean_percentile', 50):.0f}  [dim](ideal 50)[/dim]",
        "",
        f"  [bold]Assessment:[/bold]  {scorecard.get('calibration_note', '')}",
    ]

    console.print(Panel(
        "\n".join(lines),
        title="[bold]Model Calibration Scorecard[/bold]",
        border_style="green",
        padding=(0, 1),
    ))

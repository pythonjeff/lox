from __future__ import annotations

import typer
from rich import print

from ai_options_trader.config import load_settings


def register(liquidity_app: typer.Typer) -> None:
    @liquidity_app.command("snapshot")
    def liquidity_snapshot(
        start: str = typer.Option("2011-01-01", "--start", help="Start date YYYY-MM-DD"),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh FRED downloads"),
        features: bool = typer.Option(False, "--features", help="Print ML-friendly scalar feature vector too"),
    ):
        """
        DEPRECATED: Use `lox funding snapshot`.

        This command remains as a backwards-compatible alias.
        """
        from ai_options_trader.cli_commands.funding_cmd import register as register_funding

        # Mount funding snapshot under the legacy `liquidity` command group.
        tmp = typer.Typer(add_completion=False)
        register_funding(tmp)
        # Call the funding snapshot command directly.
        tmp.commands["snapshot"](start=start, refresh=refresh, features=features)  # type: ignore[misc]



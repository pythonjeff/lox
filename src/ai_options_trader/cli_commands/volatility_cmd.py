from __future__ import annotations

import typer
from rich import print
from rich.panel import Panel


def register(vol_app: typer.Typer) -> None:
    @vol_app.command("snapshot")
    def snapshot(
        start: str = typer.Option("2011-01-01", "--start"),
        refresh: bool = typer.Option(False, "--refresh"),
    ):
        """
        Volatility snapshot (VIX-based): level, momentum, term structure, and spike/persistence indicators.
        """
        from ai_options_trader.config import load_settings
        from ai_options_trader.volatility.signals import build_volatility_state
        from ai_options_trader.volatility.regime import classify_volatility_regime

        settings = load_settings()
        state = build_volatility_state(settings=settings, start_date=start, refresh=refresh)
        regime = classify_volatility_regime(state.inputs)

        print(
            Panel(
                f"[b]Regime:[/b] {regime.label}\n"
                f"[b]VIX:[/b] {state.inputs.vix}\n"
                f"[b]5d chg%:[/b] {state.inputs.vix_chg_5d_pct}\n"
                f"[b]Term (VIX-3m):[/b] {state.inputs.vix_term_spread}\n"
                f"[b]Z VIX:[/b] {state.inputs.z_vix}\n"
                f"[b]Z 5d chg:[/b] {state.inputs.z_vix_chg_5d}\n"
                f"[b]Persist 20d:[/b] {state.inputs.persist_20d}\n"
                f"[b]Pressure score:[/b] {state.inputs.vol_pressure_score}\n\n"
                f"[dim]{regime.description}[/dim]",
                title="Volatility snapshot",
                expand=False,
            )
        )



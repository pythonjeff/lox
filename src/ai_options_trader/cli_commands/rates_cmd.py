from __future__ import annotations

import typer
from rich import print
from rich.panel import Panel

from ai_options_trader.config import load_settings
from ai_options_trader.rates.regime import classify_rates_regime
from ai_options_trader.rates.signals import RATES_FRED_SERIES, build_rates_state


def register(rates_app: typer.Typer) -> None:
    def _fmt_pct(x: object) -> str:
        return f"{float(x):.2f}%" if isinstance(x, (int, float)) else "n/a"

    def _fmt_bps(x: object) -> str:
        return f"{float(x) * 100.0:+.0f}bp" if isinstance(x, (int, float)) else "n/a"

    def _fmt_z(x: object) -> str:
        return f"{float(x):+.2f}" if isinstance(x, (int, float)) else "n/a"

    @rates_app.command("snapshot")
    def rates_snapshot(
        start: str = typer.Option("2011-01-01", "--start", help="Start date YYYY-MM-DD"),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh FRED downloads"),
    ):
        """Rates / yield curve regime snapshot."""
        settings = load_settings()
        state = build_rates_state(settings=settings, start_date=start, refresh=refresh)
        ri = state.inputs
        regime = classify_rates_regime(ri)

        body = "\n".join(
            [
                f"As of: [bold]{state.asof}[/bold]",
                f"2y: [bold]{_fmt_pct(ri.ust_2y)}[/bold] | 10y: [bold]{_fmt_pct(ri.ust_10y)}[/bold] | 3m: [bold]{_fmt_pct(ri.ust_3m)}[/bold]",
                f"Curve 2s10s: [bold]{_fmt_bps(ri.curve_2s10s)}[/bold] (z={_fmt_z(ri.z_curve_2s10s)})",
                f"10y Δ20d: [bold]{_fmt_bps(ri.ust_10y_chg_20d)}[/bold] (z={_fmt_z(ri.z_ust_10y_chg_20d)})",
                f"Regime: [bold]{regime.label or regime.name}[/bold]",
                f"Answer: {regime.description}",
                f"Series (FRED): [dim]{', '.join(sorted(set(RATES_FRED_SERIES.values())))}[/dim]",
            ]
        )
        print(Panel.fit(body, title="US Rates / Curve (MVP)", border_style="magenta"))



from __future__ import annotations

import typer
from rich import print
from rich.panel import Panel

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings
from lox.rates.regime import classify_rates_regime
from lox.rates.signals import RATES_FRED_SERIES, build_rates_state


def _fmt_pct(x: object) -> str:
    return f"{float(x):.2f}%" if isinstance(x, (int, float)) else "n/a"

def _fmt_bps(x: object) -> str:
    return f"{float(x) * 100.0:+.0f}bp" if isinstance(x, (int, float)) else "n/a"

def _fmt_z(x: object) -> str:
    return f"{float(x):+.2f}" if isinstance(x, (int, float)) else "n/a"


def _run_rates_snapshot(
    start: str = "2011-01-01",
    refresh: bool = False,
    llm: bool = False,
    ticker: str = "",
    features: bool = False,
    json_out: bool = False,
    delta: str = "",
    alert: bool = False,
    calendar: bool = False,
    trades: bool = False,
):
    """Shared implementation for rates snapshot."""
    from lox.cli_commands.shared.labs_utils import (
        handle_output_flags, parse_delta_period, show_delta_summary,
        show_alert_output, show_calendar_output, show_trades_output,
    )
    from rich.console import Console
    
    console = Console()
    settings = load_settings()
    state = build_rates_state(settings=settings, start_date=start, refresh=refresh)
    ri = state.inputs
    regime = classify_rates_regime(ri)

    # Build snapshot and features
    snapshot_data = {
        "ust_2y": ri.ust_2y,
        "ust_10y": ri.ust_10y,
        "ust_3m": ri.ust_3m,
        "curve_2s10s": ri.curve_2s10s,
        "z_curve_2s10s": ri.z_curve_2s10s,
        "ust_10y_chg_20d": ri.ust_10y_chg_20d,
        "z_ust_10y_chg_20d": ri.z_ust_10y_chg_20d,
        "regime": regime.label or regime.name,
    }
    
    feature_dict = {
        "ust_2y": ri.ust_2y,
        "ust_10y": ri.ust_10y,
        "ust_3m": ri.ust_3m,
        "curve_2s10s": ri.curve_2s10s,
        "z_curve_2s10s": ri.z_curve_2s10s,
        "ust_10y_chg_20d": ri.ust_10y_chg_20d,
        "z_ust_10y_chg_20d": ri.z_ust_10y_chg_20d,
    }

    # Handle --features and --json flags
    if handle_output_flags(
        domain="rates",
        snapshot=snapshot_data,
        features=feature_dict,
        regime=regime.label or regime.name,
        regime_description=regime.description,
        asof=state.asof,
        output_json=json_out,
        output_features=features,
    ):
        return

    # Handle --alert flag (silent unless extreme)
    if alert:
        show_alert_output("rates", regime.label or regime.name, snapshot_data, regime.description)
        return

    # Handle --calendar flag
    if calendar:
        print(Panel.fit(f"[b]Regime:[/b] {regime.label or regime.name}", title="US Rates", border_style="magenta"))
        show_calendar_output("rates")
        return

    # Handle --trades flag
    if trades:
        print(Panel.fit(f"[b]Regime:[/b] {regime.label or regime.name}", title="US Rates", border_style="magenta"))
        show_trades_output("rates", regime.label or regime.name)
        return

    # Handle --delta flag
    if delta:
        from lox.cli_commands.shared.labs_utils import get_delta_metrics
        
        delta_days = parse_delta_period(delta)
        metric_keys = [
            "UST 2Y:ust_2y:%",
            "UST 10Y:ust_10y:%",
            "UST 3M:ust_3m:%",
            "Curve 2s10s:curve_2s10s:",
            "Curve z-score:z_curve_2s10s:",
            "10Y Δ20d z:z_ust_10y_chg_20d:",
        ]
        metrics_for_delta, prev_regime = get_delta_metrics("rates", snapshot_data, metric_keys, delta_days)
        show_delta_summary("rates", regime.label or regime.name, prev_regime, metrics_for_delta, delta_days)
        
        if prev_regime is None:
            console.print(f"\n[dim]No cached data from {delta_days}d ago. Run `lox labs rates` daily to build history.[/dim]")
        return

    # Uniform regime panel
    score = 80 if "shock" in regime.name else (60 if "inverted" in regime.name else 40)
    metrics = [
        {"name": "2Y", "value": _fmt_pct(ri.ust_2y), "context": "UST"},
        {"name": "10Y", "value": _fmt_pct(ri.ust_10y), "context": "UST"},
        {"name": "3M", "value": _fmt_pct(ri.ust_3m), "context": "UST"},
        {"name": "Curve 2s10s", "value": _fmt_bps(ri.curve_2s10s), "context": f"z={_fmt_z(ri.z_curve_2s10s)}"},
        {"name": "10Y Δ20d", "value": _fmt_bps(ri.ust_10y_chg_20d), "context": f"z={_fmt_z(ri.z_ust_10y_chg_20d)}"},
    ]
    print(render_regime_panel(
        domain="Rates",
        asof=state.asof,
        regime_label=regime.label or regime.name,
        score=score,
        percentile=None,
        description=regime.description,
        metrics=metrics,
    ))

    if llm:
        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis
        print_llm_regime_analysis(
            settings=settings,
            domain="rates",
            snapshot=snapshot_data,
            regime_label=regime.label or regime.name,
            regime_description=regime.description,
            ticker=ticker,
        )


def rates_snapshot(**kwargs) -> None:
    """Entry point for `lox regime rates` (no subcommand)."""
    _run_rates_snapshot(**kwargs)


def register(rates_app: typer.Typer) -> None:
    # Default callback so `lox labs rates --llm` works without `snapshot`
    @rates_app.callback(invoke_without_command=True)
    def rates_default(
        ctx: typer.Context,
        llm: bool = typer.Option(False, "--llm", help="Chat with LLM analyst"),
        ticker: str = typer.Option("", "--ticker", "-t", help="Ticker for focused chat (used with --llm)"),
        features: bool = typer.Option(False, "--features", help="Export ML-ready feature vector (JSON)"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
        delta: str = typer.Option("", "--delta", help="Show changes vs N days ago (e.g., 7d, 1w, 1m)"),
        alert: bool = typer.Option(False, "--alert", help="Only output if regime is extreme (for cron/monitoring)"),
        calendar: bool = typer.Option(False, "--calendar", help="Show upcoming events that could shift this regime"),
        trades: bool = typer.Option(False, "--trades", help="Show quick trade expressions for current regime"),
    ):
        """Rates / yield curve regime (UST level/slope/momentum)"""
        if ctx.invoked_subcommand is None:
            _run_rates_snapshot(llm=llm, ticker=ticker, features=features, json_out=json_out, delta=delta, alert=alert, calendar=calendar, trades=trades)

    @rates_app.command("snapshot")
    def rates_snapshot(
        start: str = typer.Option("2011-01-01", "--start", help="Start date YYYY-MM-DD"),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh FRED downloads"),
        llm: bool = typer.Option(False, "--llm", help="Chat with LLM analyst"),
        ticker: str = typer.Option("", "--ticker", "-t", help="Ticker for focused chat (used with --llm)"),
        features: bool = typer.Option(False, "--features", help="Export ML-ready feature vector (JSON)"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
        delta: str = typer.Option("", "--delta", help="Show changes vs N days ago (e.g., 7d, 1w, 1m)"),
        alert: bool = typer.Option(False, "--alert", help="Only output if regime is extreme (for cron/monitoring)"),
        calendar: bool = typer.Option(False, "--calendar", help="Show upcoming events that could shift this regime"),
        trades: bool = typer.Option(False, "--trades", help="Show quick trade expressions for current regime"),
    ):
        """Rates / yield curve regime snapshot."""
        _run_rates_snapshot(start=start, refresh=refresh, llm=llm, ticker=ticker, features=features, json_out=json_out, delta=delta, alert=alert, calendar=calendar, trades=trades)

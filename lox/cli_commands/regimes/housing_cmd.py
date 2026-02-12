from __future__ import annotations

import typer
from rich import print
from rich.panel import Panel

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings


def run_housing_snapshot(
    *,
    start: str = "2011-01-01",
    refresh: bool = False,
    features: bool = False,
    json_out: bool = False,
    delta: str = "",
    llm: bool = False,
    alert: bool = False,
    calendar: bool = False,
    trades: bool = False,
) -> None:
    """Shared implementation for housing snapshot."""
    from lox.cli_commands.shared.labs_utils import (
        handle_output_flags, parse_delta_period, show_delta_summary,
        show_alert_output, show_calendar_output, show_trades_output,
        get_delta_metrics,
    )
    from lox.housing.signals import build_housing_state
    from lox.housing.regime import classify_housing_regime

    settings = load_settings()
    state = build_housing_state(settings=settings, start_date=start, refresh=refresh)
    regime = classify_housing_regime(state.inputs)

    # Build snapshot data
    snapshot_data = {
        "mortgage_30y": state.inputs.mortgage_30y,
        "ust_10y": state.inputs.ust_10y,
        "mortgage_spread": state.inputs.mortgage_spread,
        "z_mortgage_spread": state.inputs.z_mortgage_spread,
        "z_mbs_rel_ret_60d": state.inputs.z_mbs_rel_ret_60d,
        "z_homebuilder_rel_ret_60d": state.inputs.z_homebuilder_rel_ret_60d,
        "z_reit_rel_ret_60d": state.inputs.z_reit_rel_ret_60d,
        "housing_pressure_score": state.inputs.housing_pressure_score,
        "regime": regime.label,
    }

    feature_dict = {
        "mortgage_30y": state.inputs.mortgage_30y,
        "ust_10y": state.inputs.ust_10y,
        "mortgage_spread": state.inputs.mortgage_spread,
        "z_mortgage_spread": state.inputs.z_mortgage_spread,
        "z_mbs_rel_ret_60d": state.inputs.z_mbs_rel_ret_60d,
        "z_homebuilder_rel_ret_60d": state.inputs.z_homebuilder_rel_ret_60d,
        "z_reit_rel_ret_60d": state.inputs.z_reit_rel_ret_60d,
        "housing_pressure_score": state.inputs.housing_pressure_score,
    }

    # Handle --features and --json flags
    if handle_output_flags(
        domain="housing",
        snapshot=snapshot_data,
        features=feature_dict,
        regime=regime.label,
        regime_description=regime.description,
        asof=state.asof,
        output_json=json_out,
        output_features=features,
    ):
        return

    # Handle --alert flag
    if alert:
        show_alert_output("housing", regime.label, snapshot_data, regime.description)
        return

    # Handle --calendar flag
    if calendar:
        print(Panel.fit(f"[b]Regime:[/b] {regime.label}", title="Housing / MBS", border_style="cyan"))
        show_calendar_output("housing")
        return

    # Handle --trades flag
    if trades:
        print(Panel.fit(f"[b]Regime:[/b] {regime.label}", title="Housing / MBS", border_style="cyan"))
        show_trades_output("housing", regime.label)
        return

    # Handle --delta flag
    if delta:
        delta_days = parse_delta_period(delta)
        metric_keys = [
            "Mortgage 30Y:mortgage_30y:%",
            "UST 10Y:ust_10y:%",
            "Mortgage Spread:mortgage_spread:bp",
            "Z Mortgage Spread:z_mortgage_spread:",
            "Housing Pressure:housing_pressure_score:",
        ]
        metrics_for_delta, prev_regime = get_delta_metrics("housing", snapshot_data, metric_keys, delta_days)
        show_delta_summary("housing", regime.label, prev_regime, metrics_for_delta, delta_days)
        return

    # Uniform regime panel
    inp = state.inputs
    score = 70 if "stress" in regime.name else (30 if "easing" in regime.name else 50)
    def _v(x):
        return f"{x:.2f}" if x is not None and isinstance(x, (int, float)) else "n/a"
    metrics = [
        {"name": "Mortgage 30y", "value": _v(inp.mortgage_30y), "context": "%" if inp.mortgage_30y is not None else "—"},
        {"name": "UST 10y", "value": _v(inp.ust_10y), "context": "%" if inp.ust_10y is not None else "—"},
        {"name": "Mortgage spread", "value": _v(inp.mortgage_spread), "context": "bp" if inp.mortgage_spread is not None else "—"},
        {"name": "Z mortgage spread", "value": _v(inp.z_mortgage_spread), "context": "vs history"},
        {"name": "Z MBS rel 60d", "value": _v(inp.z_mbs_rel_ret_60d), "context": "MBB-IEF"},
        {"name": "Z Homebuilders 60d", "value": _v(inp.z_homebuilder_rel_ret_60d), "context": "ITB-SPY"},
        {"name": "Pressure score", "value": _v(inp.housing_pressure_score), "context": "composite"},
    ]
    print(render_regime_panel(
        domain="Housing",
        asof=state.asof,
        regime_label=regime.label,
        score=score,
        percentile=None,
        description=regime.description + (" " + (state.notes or "")) if state.notes else regime.description,
        metrics=metrics,
    ))

    if llm:
        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis
        print_llm_regime_analysis(
            settings=settings,
            domain="housing",
            snapshot=snapshot_data,
            regime_label=regime.label,
            regime_description=regime.description,
        )


def register(housing_app: typer.Typer) -> None:
    # Default callback so `lox labs housing --llm` works without `snapshot`
    @housing_app.callback(invoke_without_command=True)
    def housing_default(
        ctx: typer.Context,
        llm: bool = typer.Option(False, "--llm", help="Get LLM analysis with real-time data"),
        features: bool = typer.Option(False, "--features", help="Export ML-ready feature vector (JSON)"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
        delta: str = typer.Option("", "--delta", help="Show changes vs N days ago (e.g., 7d, 1w, 1m)"),
        alert: bool = typer.Option(False, "--alert", help="Only output if regime is extreme (for cron/monitoring)"),
        calendar: bool = typer.Option(False, "--calendar", help="Show upcoming events that could shift this regime"),
        trades: bool = typer.Option(False, "--trades", help="Show quick trade expressions for current regime"),
    ):
        """Housing / MBS regime (mortgage spreads + housing proxies)"""
        if ctx.invoked_subcommand is None:
            run_housing_snapshot(llm=llm, features=features, json_out=json_out, delta=delta, alert=alert, calendar=calendar, trades=trades)

    @housing_app.command("snapshot")
    def snapshot(
        start: str = typer.Option("2011-01-01", "--start"),
        refresh: bool = typer.Option(False, "--refresh"),
        features: bool = typer.Option(False, "--features", help="Export ML-ready feature vector (JSON)"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
        delta: str = typer.Option("", "--delta", help="Show changes vs N days ago (e.g., 7d, 1w, 1m)"),
        llm: bool = typer.Option(False, "--llm", help="Get LLM analysis with real-time data"),
        alert: bool = typer.Option(False, "--alert", help="Only output if regime is extreme (for cron/monitoring)"),
        calendar: bool = typer.Option(False, "--calendar", help="Show upcoming events that could shift this regime"),
        trades: bool = typer.Option(False, "--trades", help="Show quick trade expressions for current regime"),
    ):
        """Housing / MBS snapshot: mortgage spread stress + market proxies."""
        run_housing_snapshot(
            start=start, refresh=refresh, features=features, json_out=json_out,
            delta=delta, llm=llm, alert=alert, calendar=calendar, trades=trades,
        )


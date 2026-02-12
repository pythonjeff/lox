from __future__ import annotations

import typer
from rich import print
from rich.panel import Panel


def _run_commodities_snapshot(
    start: str = "2011-01-01",
    refresh: bool = False,
    llm: bool = False,
    features: bool = False,
    json_out: bool = False,
    delta: str = "",
    alert: bool = False,
    calendar: bool = False,
    trades: bool = False,
):
    """Shared implementation for commodities snapshot."""
    import numpy as np
    import pandas as pd
    from rich.console import Console

from lox.config import load_settings
from lox.data.market import fetch_equity_daily_closes
    from lox.cli_commands.shared.regime_display import render_regime_panel
    from lox.commodities.signals import build_commodities_state
    from lox.commodities.regime import classify_commodities_regime
    from lox.cli_commands.shared.labs_utils import (
        handle_output_flags, parse_delta_period, show_delta_summary,
        show_alert_output, show_calendar_output, show_trades_output,
    )

    console = Console()
    settings = load_settings()
    state = build_commodities_state(settings=settings, start_date=start, refresh=refresh)
    regime = classify_commodities_regime(state.inputs)

    # Build snapshot and features
    snapshot_data = {
        "wti": state.inputs.wti,
        "wti_ret_20d_pct": state.inputs.wti_ret_20d_pct,
        "z_wti_ret_20d": state.inputs.z_wti_ret_20d,
        "gold": state.inputs.gold,
        "gold_ret_20d_pct": state.inputs.gold_ret_20d_pct,
        "z_gold_ret_20d": state.inputs.z_gold_ret_20d,
        "copper": state.inputs.copper,
        "copper_ret_60d_pct": state.inputs.copper_ret_60d_pct,
        "z_copper_ret_60d": state.inputs.z_copper_ret_60d,
        "broad_index": state.inputs.broad_index,
        "broad_ret_60d_pct": state.inputs.broad_ret_60d_pct,
        "z_broad_ret_60d": state.inputs.z_broad_ret_60d,
        "commodity_pressure_score": state.inputs.commodity_pressure_score,
        "energy_shock": state.inputs.energy_shock,
        "metals_impulse": state.inputs.metals_impulse,
        "regime": regime.label,
    }

    feature_dict = {
        "wti": state.inputs.wti,
        "wti_ret_20d_pct": state.inputs.wti_ret_20d_pct,
        "z_wti_ret_20d": state.inputs.z_wti_ret_20d,
        "gold": state.inputs.gold,
        "gold_ret_20d_pct": state.inputs.gold_ret_20d_pct,
        "z_gold_ret_20d": state.inputs.z_gold_ret_20d,
        "copper": state.inputs.copper,
        "copper_ret_60d_pct": state.inputs.copper_ret_60d_pct,
        "z_copper_ret_60d": state.inputs.z_copper_ret_60d,
        "broad_index": state.inputs.broad_index,
        "broad_ret_60d_pct": state.inputs.broad_ret_60d_pct,
        "z_broad_ret_60d": state.inputs.z_broad_ret_60d,
        "commodity_pressure_score": state.inputs.commodity_pressure_score,
        "energy_shock": state.inputs.energy_shock,
        "metals_impulse": state.inputs.metals_impulse,
    }

    # Handle --features and --json flags
    if handle_output_flags(
        domain="commodities",
        snapshot=snapshot_data,
        features=feature_dict,
        regime=regime.label,
        regime_description=regime.description,
        asof=state.asof if hasattr(state, 'asof') else None,
        output_json=json_out,
        output_features=features,
    ):
        return

    # Handle --alert flag (silent unless extreme)
    if alert:
        show_alert_output("commodities", regime.label, snapshot_data, regime.description)
        return

    # Handle --calendar flag
    if calendar:
        print(Panel(f"[b]Regime:[/b] {regime.label}", title="Commodities", expand=False))
        show_calendar_output("commodities")
        return

    # Handle --trades flag
    if trades:
        print(Panel(f"[b]Regime:[/b] {regime.label}", title="Commodities", expand=False))
        show_trades_output("commodities", regime.label)
        return

    # Handle --delta flag
    if delta:
        from lox.cli_commands.shared.labs_utils import get_delta_metrics
        
        delta_days = parse_delta_period(delta)
        metric_keys = [
            "WTI:wti:$",
            "Gold:gold:$",
            "Copper:copper:$",
            "Broad Index:broad_index:$",
            "Pressure Score:commodity_pressure_score:",
            "WTI 20d%:wti_ret_20d_pct:%",
        ]
        metrics_for_delta, prev_regime = get_delta_metrics("commodities", snapshot_data, metric_keys, delta_days)
        show_delta_summary("commodities", regime.label, prev_regime, metrics_for_delta, delta_days)
        
        if prev_regime is None:
            console.print(f"\n[dim]No cached data from {delta_days}d ago. Run `lox labs commodities` daily to build history.[/dim]")
        return

    # Uniform regime panel
    score = 70 if regime.name == "energy_shock" else (65 if regime.name == "commodity_reflation" else (30 if regime.name == "commodity_disinflation" else 50))
    inp = state.inputs
    def _fmt(v, decimals=2, pct=False):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        return f"{v:+.{decimals}f}%" if pct else f"{v:.{decimals}f}"
    def _z(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        return f"{v:+.2f}"
    metrics = [
        {"name": "WTI", "value": f"${_fmt(inp.wti)}", "context": f"20d {_fmt(inp.wti_ret_20d_pct, 1, True)} z={_z(inp.z_wti_ret_20d)}"},
        {"name": "Gold", "value": f"${_fmt(inp.gold)}", "context": f"20d {_fmt(inp.gold_ret_20d_pct, 1, True)} z={_z(inp.z_gold_ret_20d)}"},
        {"name": "Copper", "value": f"${_fmt(inp.copper)}", "context": f"60d {_fmt(inp.copper_ret_60d_pct, 1, True)} z={_z(inp.z_copper_ret_60d)}"},
        {"name": "Broad index", "value": f"${_fmt(inp.broad_index)}", "context": f"60d {_fmt(inp.broad_ret_60d_pct, 1, True)} z={_z(inp.z_broad_ret_60d)}"},
        {"name": "Pressure score", "value": _fmt(inp.commodity_pressure_score, 2), "context": "z-score composite"},
        {"name": "Energy shock", "value": "Yes" if inp.energy_shock else "No", "context": "WTI spike"},
        {"name": "Metals impulse", "value": "Yes" if inp.metals_impulse else "No", "context": "metals strength"},
    ]
    asof = state.asof if hasattr(state, "asof") else "n/a"
    print(render_regime_panel(
        domain="Commodities",
        asof=asof,
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
            domain="commodities",
            snapshot=snapshot_data,
            regime_label=regime.label,
            regime_description=regime.description,
        )


def register(commod_app: typer.Typer) -> None:
    # Default callback so `lox labs commodities --llm` works without `snapshot`
    @commod_app.callback(invoke_without_command=True)
    def commodities_default(
        ctx: typer.Context,
        llm: bool = typer.Option(False, "--llm", help="Get PhD-level LLM analysis with real-time data"),
        features: bool = typer.Option(False, "--features", help="Export ML-ready feature vector (JSON)"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
        delta: str = typer.Option("", "--delta", help="Show changes vs N days ago (e.g., 7d, 1w, 1m)"),
        alert: bool = typer.Option(False, "--alert", help="Only output if regime is extreme (for cron/monitoring)"),
        calendar: bool = typer.Option(False, "--calendar", help="Show upcoming events that could shift this regime"),
        trades: bool = typer.Option(False, "--trades", help="Show quick trade expressions for current regime"),
    ):
        """Commodities regime (oil/gold/copper/broad index)"""
        if ctx.invoked_subcommand is None:
            _run_commodities_snapshot(llm=llm, features=features, json_out=json_out, delta=delta, alert=alert, calendar=calendar, trades=trades)

    @commod_app.command("snapshot")
    def snapshot(
        start: str = typer.Option("2011-01-01", "--start"),
        refresh: bool = typer.Option(False, "--refresh"),
        llm: bool = typer.Option(False, "--llm", help="Get PhD-level LLM analysis with real-time data"),
        features: bool = typer.Option(False, "--features", help="Export ML-ready feature vector (JSON)"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
        delta: str = typer.Option("", "--delta", help="Show changes vs N days ago (e.g., 7d, 1w, 1m)"),
        alert: bool = typer.Option(False, "--alert", help="Only output if regime is extreme (for cron/monitoring)"),
        calendar: bool = typer.Option(False, "--calendar", help="Show upcoming events that could shift this regime"),
        trades: bool = typer.Option(False, "--trades", help="Show quick trade expressions for current regime"),
    ):
        """Commodities snapshot: oil/gold/copper/broad index."""
        _run_commodities_snapshot(start=start, refresh=refresh, llm=llm, features=features, json_out=json_out, delta=delta, alert=alert, calendar=calendar, trades=trades)

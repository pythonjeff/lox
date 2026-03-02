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
    ticker: str = "",
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

    def _mtg_ctx():
        v = inp.mortgage_30y
        if not isinstance(v, (int, float)):
            return "30-year fixed rate"
        if v > 7.5:
            return "crisis affordability — housing frozen"
        if v > 7.0:
            return "severe headwind"
        if v > 6.5:
            return "elevated — affordability strained"
        if v > 5.5:
            return "above normal"
        if v > 4.5:
            return "neutral — manageable"
        return "low — housing tailwind"

    def _ust_ctx():
        v = inp.ust_10y
        if not isinstance(v, (int, float)):
            return "10Y benchmark"
        if v > 5.0:
            return "very high — driving mortgage costs"
        if v > 4.5:
            return "elevated — upward pressure on mortgages"
        if v > 3.5:
            return "moderate"
        return "low — supportive of housing"

    def _spread_ctx():
        v = inp.mortgage_spread
        if not isinstance(v, (int, float)):
            return "mortgage-Treasury spread"
        if v > 2.5:
            return "very wide — MBS stress / bank pullback"
        if v > 2.0:
            return "wide — credit risk premium"
        if v > 1.5:
            return "normal range"
        return "tight — strong MBS demand"

    def _z_spread_ctx():
        v = inp.z_mortgage_spread
        if not isinstance(v, (int, float)):
            return "vs history"
        if v > 1.5:
            return f"z={v:+.2f} — spread extreme vs history"
        if v > 0.5:
            return f"z={v:+.2f} — wider than usual"
        if v < -1.5:
            return f"z={v:+.2f} — unusually tight"
        if v < -0.5:
            return f"z={v:+.2f} — tighter than usual"
        return f"z={v:+.2f} — normal range"

    def _mbs_ctx():
        v = inp.z_mbs_rel_ret_60d
        if not isinstance(v, (int, float)):
            return "MBS vs Treasuries"
        if v > 1.0:
            return f"z={v:+.2f} — MBS outperforming (demand)"
        if v < -1.0:
            return f"z={v:+.2f} — MBS underperforming (stress)"
        return f"z={v:+.2f} — inline with Treasuries"

    def _builder_ctx():
        v = inp.z_homebuilder_rel_ret_60d
        if not isinstance(v, (int, float)):
            return "homebuilders vs market"
        if v > 1.0:
            return f"z={v:+.2f} — builders outperforming (optimism)"
        if v < -1.0:
            return f"z={v:+.2f} — builders lagging (housing fear)"
        return f"z={v:+.2f} — inline with market"

    def _pressure_ctx():
        v = inp.housing_pressure_score
        if not isinstance(v, (int, float)):
            return "composite"
        if v > 1.5:
            return "high pressure — housing stress"
        if v > 0.5:
            return "elevated — headwinds building"
        if v > -0.5:
            return "neutral"
        if v > -1.5:
            return "easing — conditions improving"
        return "strong tailwind — housing supportive"

    metrics = [
        {"name": "Mortgage 30y", "value": _v(inp.mortgage_30y), "context": _mtg_ctx()},
        {"name": "UST 10y", "value": _v(inp.ust_10y), "context": _ust_ctx()},
        {"name": "Mortgage spread", "value": _v(inp.mortgage_spread), "context": _spread_ctx()},
        {"name": "Z mortgage spread", "value": _v(inp.z_mortgage_spread), "context": _z_spread_ctx()},
        {"name": "Z MBS rel 60d", "value": _v(inp.z_mbs_rel_ret_60d), "context": _mbs_ctx()},
        {"name": "Z Homebuilders 60d", "value": _v(inp.z_homebuilder_rel_ret_60d), "context": _builder_ctx()},
        {"name": "Pressure score", "value": _v(inp.housing_pressure_score), "context": _pressure_ctx()},
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
            ticker=ticker,
        )


def register(housing_app: typer.Typer) -> None:
    # Default callback so `lox labs housing --llm` works without `snapshot`
    @housing_app.callback(invoke_without_command=True)
    def housing_default(
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
        """Housing / MBS regime (mortgage spreads + housing proxies)"""
        if ctx.invoked_subcommand is None:
            run_housing_snapshot(llm=llm, ticker=ticker, features=features, json_out=json_out, delta=delta, alert=alert, calendar=calendar, trades=trades)

    @housing_app.command("snapshot")
    def snapshot(
        start: str = typer.Option("2011-01-01", "--start"),
        refresh: bool = typer.Option(False, "--refresh"),
        features: bool = typer.Option(False, "--features", help="Export ML-ready feature vector (JSON)"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
        delta: str = typer.Option("", "--delta", help="Show changes vs N days ago (e.g., 7d, 1w, 1m)"),
        llm: bool = typer.Option(False, "--llm", help="Chat with LLM analyst"),
        ticker: str = typer.Option("", "--ticker", "-t", help="Ticker for focused chat (used with --llm)"),
        alert: bool = typer.Option(False, "--alert", help="Only output if regime is extreme (for cron/monitoring)"),
        calendar: bool = typer.Option(False, "--calendar", help="Show upcoming events that could shift this regime"),
        trades: bool = typer.Option(False, "--trades", help="Show quick trade expressions for current regime"),
    ):
        """Housing / MBS snapshot: mortgage spread stress + market proxies."""
        run_housing_snapshot(
            start=start, refresh=refresh, features=features, json_out=json_out,
            delta=delta, llm=llm, ticker=ticker, alert=alert, calendar=calendar, trades=trades,
        )


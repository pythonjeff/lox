from __future__ import annotations

import json

import typer
from rich import print
from rich.panel import Panel

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings
from lox.funding.features import funding_feature_vector
from lox.funding.models import FundingInputs
from lox.funding.regime import classify_funding_regime
from lox.funding.signals import FUNDING_FRED_SERIES, build_funding_state


def run_funding_snapshot(
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
    """
    Shared implementation for funding snapshot/outlook commands.
    Kept as a top-level helper so other CLI aliases (e.g. `lox labs liquidity ...`) can call it safely.
    """
    from rich.console import Console
    from lox.cli_commands.shared.labs_utils import (
        handle_output_flags, parse_delta_period, show_delta_summary,
        show_alert_output, show_calendar_output, show_trades_output,
    )

    console = Console()
    settings = load_settings()
    state = build_funding_state(settings=settings, start_date=start, refresh=refresh)
    fi = state.inputs

    def _fmt_pct(x: object) -> str:
        return f"{float(x):.2f}%" if isinstance(x, (int, float)) else "n/a"

    def _fmt_bps(x: object) -> str:
        return f"{float(x):+.1f}bp" if isinstance(x, (int, float)) else "n/a"

    def _fmt_ratio(x: object) -> str:
        return f"{100.0*float(x):.0f}%" if isinstance(x, (int, float)) else "n/a"

    # Classify from derived inputs only (no hard-coded levels).
    regime = classify_funding_regime(
        FundingInputs(
            spread_corridor_bps=fi.spread_corridor_bps,
            spike_5d_bps=fi.spike_5d_bps,
            persistence_20d=fi.persistence_20d,
            vol_20d_bps=fi.vol_20d_bps,
            tight_threshold_bps=fi.tight_threshold_bps,
            stress_threshold_bps=fi.stress_threshold_bps,
            persistence_tight=fi.persistence_tight,
            persistence_stress=fi.persistence_stress,
            vol_tight_bps=fi.vol_tight_bps,
            vol_stress_bps=fi.vol_stress_bps,
        )
    )

    # Build snapshot and features
    snapshot_data = {
        "sofr": fi.sofr,
        "tgcr": fi.tgcr,
        "bgcr": fi.bgcr,
        "effr": fi.effr,
        "iorb": fi.iorb,
        "obfr": fi.obfr,
        "spread_corridor_bps": fi.spread_corridor_bps,
        "spread_sofr_effr_bps": fi.spread_sofr_effr_bps,
        "spread_bgcr_tgcr_bps": fi.spread_bgcr_tgcr_bps,
        "spike_5d_bps": fi.spike_5d_bps,
        "persistence_20d": fi.persistence_20d,
        "vol_20d_bps": fi.vol_20d_bps,
        "regime": regime.label or regime.name,
    }

    # Use existing feature vector for ML features
    vec = funding_feature_vector(state)
    feature_dict = vec.features

    # Handle --features and --json flags (standardized output)
    if handle_output_flags(
        domain="funding",
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
        show_alert_output("funding", regime.label or regime.name, snapshot_data, regime.description)
        return

    # Handle --calendar flag
    if calendar:
        print(Panel.fit(f"[b]Regime:[/b] {regime.label or regime.name}", title="US Funding", border_style="cyan"))
        show_calendar_output("funding")
        return

    # Handle --trades flag
    if trades:
        print(Panel.fit(f"[b]Regime:[/b] {regime.label or regime.name}", title="US Funding", border_style="cyan"))
        show_trades_output("funding", regime.label or regime.name)
        return

    # Handle --delta flag
    if delta:
        from lox.cli_commands.shared.labs_utils import get_delta_metrics
        
        delta_days = parse_delta_period(delta)
        metric_keys = [
            "SOFR:sofr:%",
            "EFFR:effr:%",
            "Corridor Spread:spread_corridor_bps:bp",
            "Spike 5d:spike_5d_bps:bp",
            "Vol 20d:vol_20d_bps:bp",
            "Persistence:persistence_20d:%",
        ]
        metrics_for_delta, prev_regime = get_delta_metrics("funding", snapshot_data, metric_keys, delta_days)
        show_delta_summary("funding", regime.label or regime.name, prev_regime, metrics_for_delta, delta_days)
        
        if prev_regime is None:
            console.print(f"\n[dim]No cached data from {delta_days}d ago. Run `lox labs funding` daily to build history.[/dim]")
        return

    # Uniform regime panel
    score = 80 if "stress" in regime.name else (60 if "tightening" in regime.name else 40)
    corridor_name = fi.spread_corridor_name or "SOFR-EFFR"
    metrics = [
        {"name": "SOFR", "value": _fmt_pct(fi.sofr), "context": "secured"},
        {"name": "EFFR", "value": _fmt_pct(fi.effr), "context": "unsecured"},
        {"name": f"Corridor ({corridor_name})", "value": _fmt_bps(fi.spread_corridor_bps), "context": "spread"},
        {"name": "SOFR–EFFR", "value": _fmt_bps(fi.spread_sofr_effr_bps), "context": "basis"},
        {"name": "Spike 5d", "value": _fmt_bps(fi.spike_5d_bps), "context": "max 5d"},
        {"name": "Vol 20d", "value": _fmt_bps(fi.vol_20d_bps), "context": "std"},
        {"name": "Persistence 20d", "value": _fmt_ratio(fi.persistence_20d), "context": ">stress"},
    ]
    print(render_regime_panel(
        domain="Funding",
        asof=state.asof,
        regime_label=regime.label or regime.name,
        score=score,
        percentile=None,
        description=regime.description,
        metrics=metrics,
    ))

    if llm:
        from lox.llm.core.analyst import llm_analyze_regime
        from rich.markdown import Markdown
        
        print("\n[bold cyan]Generating LLM analysis...[/bold cyan]\n")
        
        analysis = llm_analyze_regime(
            settings=settings,
            domain="funding",
            snapshot=snapshot_data,
            regime_label=regime.label or regime.name,
            regime_description=regime.description,
        )
        
        print(Panel(Markdown(analysis), title="Analysis", expand=False))


def funding_snapshot(**kwargs) -> None:
    """Entry point for `lox regime funding` (no subcommand)."""
    run_funding_snapshot(**kwargs)


def register(funding_app: typer.Typer) -> None:
    # Default callback so `lox labs funding --llm` works without `snapshot`
    @funding_app.callback(invoke_without_command=True)
    def funding_default(
        ctx: typer.Context,
        llm: bool = typer.Option(False, "--llm", help="Get PhD-level LLM analysis with real-time data"),
        features: bool = typer.Option(False, "--features", help="Export ML-ready feature vector (JSON)"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
        delta: str = typer.Option("", "--delta", help="Show changes vs N days ago (e.g., 7d, 1w, 1m)"),
        alert: bool = typer.Option(False, "--alert", help="Only output if regime is extreme (for cron/monitoring)"),
        calendar: bool = typer.Option(False, "--calendar", help="Show upcoming events that could shift this regime"),
        trades: bool = typer.Option(False, "--trades", help="Show quick trade expressions for current regime"),
    ):
        """Short-term funding markets (SOFR, repo spreads) — price of money in daily markets"""
        if ctx.invoked_subcommand is None:
            run_funding_snapshot(llm=llm, features=features, json_out=json_out, delta=delta, alert=alert, calendar=calendar, trades=trades)

    @funding_app.command("snapshot")
    def funding_snapshot(
        start: str = typer.Option("2011-01-01", "--start", help="Start date YYYY-MM-DD"),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh FRED downloads"),
        features: bool = typer.Option(False, "--features", help="Export ML-ready feature vector (JSON)"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
        delta: str = typer.Option("", "--delta", help="Show changes vs N days ago (e.g., 7d, 1w, 1m)"),
        llm: bool = typer.Option(False, "--llm", help="Get PhD-level LLM analysis with real-time data"),
        alert: bool = typer.Option(False, "--alert", help="Only output if regime is extreme (for cron/monitoring)"),
        calendar: bool = typer.Option(False, "--calendar", help="Show upcoming events that could shift this regime"),
        trades: bool = typer.Option(False, "--trades", help="Show quick trade expressions for current regime"),
    ):
        """
        Funding regime snapshot (secured rates MVP).

        Series:
        - SOFR, TGCR, BGCR (core)
        - EFFR (DFF) anchor
        - IORB/IOER (optional; preferred corridor anchor)
        - OBFR (optional cross-check)
        """
        run_funding_snapshot(start=start, refresh=bool(refresh), features=bool(features), json_out=bool(json_out), delta=delta, llm=bool(llm), alert=alert, calendar=calendar, trades=trades)

    @funding_app.command("outlook")
    def funding_outlook(
        start: str = typer.Option("2011-01-01", "--start", help="Start date YYYY-MM-DD"),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh FRED downloads"),
        features: bool = typer.Option(False, "--features", help="Export ML-ready feature vector (JSON)"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
        delta: str = typer.Option("", "--delta", help="Show changes vs N days ago (e.g., 7d, 1w, 1m)"),
        llm: bool = typer.Option(False, "--llm", help="Get PhD-level LLM analysis with real-time data"),
        alert: bool = typer.Option(False, "--alert", help="Only output if regime is extreme (for cron/monitoring)"),
        calendar: bool = typer.Option(False, "--calendar", help="Show upcoming events that could shift this regime"),
        trades: bool = typer.Option(False, "--trades", help="Show quick trade expressions for current regime"),
    ):
        """Alias for `snapshot` (back-compat UX)."""
        run_funding_snapshot(start=start, refresh=bool(refresh), features=bool(features), json_out=bool(json_out), delta=delta, llm=bool(llm), alert=alert, calendar=calendar, trades=trades)

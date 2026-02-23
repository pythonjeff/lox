from __future__ import annotations

import typer
from rich import print
from rich.panel import Panel

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings
from lox.rates.regime import MOMENTUM_THRESHOLD_PCT, classify_rates_regime
from lox.rates.signals import RATES_FRED_SERIES, build_rates_state

# ── Breakeven / real yield interpretation thresholds (%) ──────────────────
BE_ELEVATED = 2.8
BE_ABOVE_TARGET = 2.3
BE_NEAR_TARGET = 1.8
REAL_YIELD_RESTRICTIVE = 2.5
REAL_YIELD_POSITIVE = 1.5
REAL_YIELD_MILD = 0.5


def _fmt_pct(x: object) -> str:
    return f"{float(x):.2f}%" if isinstance(x, (int, float)) else "n/a"

def _fmt_bps(x: object) -> str:
    return f"{float(x) * 100.0:+.0f}bp" if isinstance(x, (int, float)) else "n/a"

def _fmt_chg(x: object) -> str:
    return f"{float(x):+.2f}%" if isinstance(x, (int, float)) else "n/a"

def _fmt_z(x: object) -> str:
    return f"{float(x):+.2f}" if isinstance(x, (int, float)) else "n/a"


def _momentum_context(d2y: float | None, d_long: float | None, long_label: str = "30Y") -> str:
    """One-line interpretation of front vs back end movement."""
    if d2y is None or d_long is None:
        return ""
    thresh = MOMENTUM_THRESHOLD_PCT
    if d2y <= thresh and d_long > thresh:
        return "bear steepener"
    if d2y >= -thresh and d_long < -thresh:
        return "bull flattener"
    if d2y > thresh and d_long > thresh:
        return "bear flattener" if d2y > d_long else "parallel selloff"
    if d2y < -thresh and d_long < -thresh:
        return "bull steepener" if d2y < d_long else "parallel rally"
    return "quiet"


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

    # ── Full snapshot for LLM / delta / json ─────────────────────────────
    snapshot_data = {
        "ust_3m": ri.ust_3m,
        "ust_2y": ri.ust_2y,
        "ust_5y": ri.ust_5y,
        "ust_10y": ri.ust_10y,
        "ust_30y": ri.ust_30y,
        "curve_2s10s": ri.curve_2s10s,
        "curve_2s30s": ri.curve_2s30s,
        "curve_5s30s": ri.curve_5s30s,
        "ust_2y_chg_20d": ri.ust_2y_chg_20d,
        "ust_10y_chg_20d": ri.ust_10y_chg_20d,
        "ust_30y_chg_20d": ri.ust_30y_chg_20d,
        "curve_2s30s_chg_20d": ri.curve_2s30s_chg_20d,
        "z_curve_2s10s": ri.z_curve_2s10s,
        "z_curve_2s30s": ri.z_curve_2s30s,
        "z_ust_10y_chg_20d": ri.z_ust_10y_chg_20d,
        "z_ust_2y_chg_20d": ri.z_ust_2y_chg_20d,
        "z_ust_30y_chg_20d": ri.z_ust_30y_chg_20d,
        "real_yield_10y": ri.real_yield_10y,
        "breakeven_10y": ri.breakeven_10y,
        "real_yield_10y_chg_20d": ri.real_yield_10y_chg_20d,
        "regime": regime.display_label,
    }

    feature_dict = {
        "ust_2y": ri.ust_2y,
        "ust_5y": ri.ust_5y,
        "ust_10y": ri.ust_10y,
        "ust_30y": ri.ust_30y,
        "curve_2s10s": ri.curve_2s10s,
        "curve_2s30s": ri.curve_2s30s,
        "curve_5s30s": ri.curve_5s30s,
        "ust_2y_chg_20d": ri.ust_2y_chg_20d,
        "ust_10y_chg_20d": ri.ust_10y_chg_20d,
        "ust_30y_chg_20d": ri.ust_30y_chg_20d,
        "z_curve_2s10s": ri.z_curve_2s10s,
        "z_curve_2s30s": ri.z_curve_2s30s,
        "z_ust_10y_chg_20d": ri.z_ust_10y_chg_20d,
        "z_ust_2y_chg_20d": ri.z_ust_2y_chg_20d,
        "z_ust_30y_chg_20d": ri.z_ust_30y_chg_20d,
        "real_yield_10y": ri.real_yield_10y,
        "breakeven_10y": ri.breakeven_10y,
    }

    # Handle --features and --json flags
    if handle_output_flags(
        domain="rates",
        snapshot=snapshot_data,
        features=feature_dict,
        regime=regime.display_label,
        regime_description=regime.description,
        asof=state.asof,
        output_json=json_out,
        output_features=features,
    ):
        return

    if alert:
        show_alert_output("rates", regime.display_label, snapshot_data, regime.description)
        return

    if calendar:
        print(Panel.fit(f"[b]Regime:[/b] {regime.display_label}", title="US Rates", border_style="magenta"))
        show_calendar_output("rates")
        return

    if trades:
        print(Panel.fit(f"[b]Regime:[/b] {regime.display_label}", title="US Rates", border_style="magenta"))
        show_trades_output("rates", regime.display_label)
        return

    if delta:
        from lox.cli_commands.shared.labs_utils import get_delta_metrics

        delta_days = parse_delta_period(delta)
        metric_keys = [
            "UST 2Y:ust_2y:%",
            "UST 10Y:ust_10y:%",
            "UST 30Y:ust_30y:%",
            "Curve 2s10s:curve_2s10s:",
            "Curve 2s30s:curve_2s30s:",
            "2Y Δ20d:ust_2y_chg_20d:%",
            "10Y Δ20d:ust_10y_chg_20d:%",
            "30Y Δ20d:ust_30y_chg_20d:%",
            "Real yield 10Y:real_yield_10y:%",
            "Breakeven 10Y:breakeven_10y:%",
        ]
        metrics_for_delta, prev_regime = get_delta_metrics("rates", snapshot_data, metric_keys, delta_days)
        show_delta_summary("rates", regime.display_label, prev_regime, metrics_for_delta, delta_days)

        if prev_regime is None:
            console.print(f"\n[dim]No cached data from {delta_days}d ago. Run `lox regime rates` daily to build history.[/dim]")
        return

    # ── Sectioned metrics for the panel ──────────────────────────────────
    d_long = ri.ust_30y_chg_20d if ri.ust_30y_chg_20d is not None else ri.ust_10y_chg_20d
    curve_move = _momentum_context(ri.ust_2y_chg_20d, d_long)

    def _be_ctx() -> str:
        be = ri.breakeven_10y
        if be is None:
            return "market inflation expectations"
        if be > BE_ELEVATED:
            return "elevated inflation priced in"
        if be > BE_ABOVE_TARGET:
            return "above target"
        if be > BE_NEAR_TARGET:
            return "near target"
        return "below target — disinflation priced"

    def _real_ctx() -> str:
        ry = ri.real_yield_10y
        if ry is None:
            return "TIPS-implied real rate"
        if ry > REAL_YIELD_RESTRICTIVE:
            return "restrictive — tight financial conditions"
        if ry > REAL_YIELD_POSITIVE:
            return "positive — normal tightening"
        if ry > REAL_YIELD_MILD:
            return "mildly positive"
        if ry > 0:
            return "near zero — accommodative"
        return "negative — very accommodative"

    metrics = [
        # ── Curve Levels ──
        {"name": "─── Curve Levels ───", "value": "", "context": ""},
        {"name": "3M", "value": _fmt_pct(ri.ust_3m), "context": "T-bill anchor"},
        {"name": "2Y", "value": _fmt_pct(ri.ust_2y), "context": "Fed expectations"},
        {"name": "5Y", "value": _fmt_pct(ri.ust_5y), "context": "belly"},
        {"name": "10Y", "value": _fmt_pct(ri.ust_10y), "context": "benchmark"},
        {"name": "30Y", "value": _fmt_pct(ri.ust_30y), "context": "long end"},
        # ── Curve Shape ──
        {"name": "─── Curve Shape ───", "value": "", "context": ""},
        {"name": "2s10s", "value": _fmt_bps(ri.curve_2s10s), "context": f"z={_fmt_z(ri.z_curve_2s10s)}"},
        {"name": "2s30s", "value": _fmt_bps(ri.curve_2s30s), "context": f"z={_fmt_z(ri.z_curve_2s30s)}"},
        {"name": "5s30s", "value": _fmt_bps(ri.curve_5s30s), "context": "long-end steepness"},
        # ── Momentum ──
        {"name": "─── Momentum (20d) ───", "value": "", "context": ""},
        {"name": "2Y Δ20d", "value": _fmt_chg(ri.ust_2y_chg_20d), "context": f"z={_fmt_z(ri.z_ust_2y_chg_20d)}"},
        {"name": "10Y Δ20d", "value": _fmt_chg(ri.ust_10y_chg_20d), "context": f"z={_fmt_z(ri.z_ust_10y_chg_20d)}"},
        {"name": "30Y Δ20d", "value": _fmt_chg(ri.ust_30y_chg_20d), "context": f"z={_fmt_z(ri.z_ust_30y_chg_20d)}"},
        {"name": "Curve move", "value": curve_move or "—", "context": "front vs back end"},
        # ── Real Yields ──
        {"name": "─── Real Yields ───", "value": "", "context": ""},
        {"name": "10Y Real", "value": _fmt_pct(ri.real_yield_10y), "context": _real_ctx()},
        {"name": "10Y Breakeven", "value": _fmt_pct(ri.breakeven_10y), "context": _be_ctx()},
        {"name": "Real Δ20d", "value": _fmt_chg(ri.real_yield_10y_chg_20d), "context": "real rate movement"},
    ]

    print(render_regime_panel(
        domain="Rates",
        asof=state.asof,
        regime_label=regime.display_label,
        score=regime.score,
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
            regime_label=regime.display_label,
            regime_description=regime.description,
            ticker=ticker,
        )


def rates_snapshot(**kwargs) -> None:
    """Entry point for `lox regime rates` (no subcommand)."""
    _run_rates_snapshot(**kwargs)


def register(rates_app: typer.Typer) -> None:
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
        """Rates / yield curve regime (full curve dynamics, steepener/flattener, real yields)"""
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
        """Rates / yield curve regime snapshot (full curve + steepener/flattener dynamics)."""
        _run_rates_snapshot(start=start, refresh=refresh, llm=llm, ticker=ticker, features=features, json_out=json_out, delta=delta, alert=alert, calendar=calendar, trades=trades)

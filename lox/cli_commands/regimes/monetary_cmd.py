from __future__ import annotations

import typer
from rich import print
from rich.panel import Panel

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings
from lox.monetary.models import MonetaryInputs
from lox.monetary.regime import classify_monetary_regime
from lox.monetary.signals import build_monetary_page_data
from lox.utils.formatting import fmt_usd_from_millions


def _ctx_z(z: float | None, *, low: float = -0.75, high: float = 0.75) -> str:
    if not isinstance(z, (int, float)):
        return "n/a"
    v = float(z)
    if v <= low:
        return f"{v:+.2f} → Lower vs recent history (not a scarcity signal by itself)"
    if v >= high:
        return f"{v:+.2f} → Higher vs recent history"
    return f"{v:+.2f} → Around recent average"


def _ctx_qt(z_qt: float | None) -> str:
    if not isinstance(z_qt, (int, float)):
        return "n/a"
    v = float(z_qt)
    if v <= -0.75:
        return f"{v:+.2f} → Fast shrink (QT pressure)"
    if v >= 0.75:
        return f"{v:+.2f} → Balance sheet expanding (liquidity support / QT offset)"
    return f"{v:+.2f} → Normal pace"


def _run_monetary_snapshot(
    lookback_years: int = 5,
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
    """Shared implementation for monetary snapshot."""
    from rich.console import Console
    from lox.cli_commands.shared.labs_utils import (
        handle_output_flags, parse_delta_period, show_delta_summary,
        show_alert_output, show_calendar_output, show_trades_output,
    )

    console = Console()
    settings = load_settings()
    d = build_monetary_page_data(settings=settings, lookback_years=lookback_years, refresh=refresh)

    effr = d.get("effr")
    effr_disp = f"{float(effr):.2f}%" if isinstance(effr, (int, float)) else "n/a"

    res = d.get("reserves") if isinstance(d.get("reserves"), dict) else {}
    res_level = fmt_usd_from_millions(res.get("level"))
    res_chg_13w = fmt_usd_from_millions(res.get("chg_13w"))
    res_z = res.get("z_level")
    res_pct_gdp = res.get("pct_gdp")
    res_pct_gdp_disp = f"{float(res_pct_gdp):.1f}%" if isinstance(res_pct_gdp, (int, float)) else "n/a"

    fa = d.get("fed_assets") if isinstance(d.get("fed_assets"), dict) else {}
    fa_level = fmt_usd_from_millions(fa.get("level"))
    fa_chg_13w = fmt_usd_from_millions(fa.get("chg_13w"))
    fa_z = fa.get("z_chg_13w")

    rrp = d.get("on_rrp") if isinstance(d.get("on_rrp"), dict) else {}
    rrp_level = fmt_usd_from_millions(rrp.get("level"))
    rrp_chg_13w = fmt_usd_from_millions(rrp.get("chg_13w"))
    rrp_z = rrp.get("z_level")

    regime = classify_monetary_regime(
        MonetaryInputs(
            z_total_reserves=float(res_z) if isinstance(res_z, (int, float)) else None,
            z_on_rrp=float(rrp_z) if isinstance(rrp_z, (int, float)) else None,
            z_fed_assets_chg_13w=float(fa_z) if isinstance(fa_z, (int, float)) else None,
        )
    )

    # Build snapshot and features
    snapshot_data = {
        "effr": effr,
        "reserves_level": res.get("level"),
        "reserves_chg_13w": res.get("chg_13w"),
        "reserves_z_level": res_z,
        "reserves_pct_gdp": res_pct_gdp,
        "fed_assets_level": fa.get("level"),
        "fed_assets_chg_13w": fa.get("chg_13w"),
        "fed_assets_z_chg_13w": fa_z,
        "on_rrp_level": rrp.get("level"),
        "on_rrp_chg_13w": rrp.get("chg_13w"),
        "on_rrp_z_level": rrp_z,
        "regime": regime.label or regime.name,
    }

    feature_dict = {
        "effr": effr,
        "reserves_level_millions": res.get("level"),
        "reserves_z_level": res_z,
        "reserves_pct_gdp": res_pct_gdp,
        "fed_assets_level_millions": fa.get("level"),
        "fed_assets_z_chg_13w": fa_z,
        "on_rrp_level_millions": rrp.get("level"),
        "on_rrp_z_level": rrp_z,
    }

    # Handle --features and --json flags
    if handle_output_flags(
        domain="monetary",
        snapshot=snapshot_data,
        features=feature_dict,
        regime=regime.label or regime.name,
        regime_description=regime.description,
        asof=d.get("asof"),
        output_json=json_out,
        output_features=features,
    ):
        return

    # Handle --alert flag (silent unless extreme)
    if alert:
        show_alert_output("monetary", regime.label or regime.name, snapshot_data, regime.description)
        return

    # Handle --calendar flag
    if calendar:
        print(Panel.fit(f"[b]Regime:[/b] {regime.label or regime.name}", title="US Monetary", border_style="cyan"))
        show_calendar_output("monetary")
        return

    # Handle --trades flag
    if trades:
        print(Panel.fit(f"[b]Regime:[/b] {regime.label or regime.name}", title="US Monetary", border_style="cyan"))
        show_trades_output("monetary", regime.label or regime.name)
        return

    # Handle --delta flag
    if delta:
        from lox.cli_commands.shared.labs_utils import get_delta_metrics
        
        delta_days = parse_delta_period(delta)
        metric_keys = [
            "EFFR:effr:%",
            "Reserves z:reserves_z_level:",
            "Reserves % GDP:reserves_pct_gdp:%",
            "Fed Assets z:fed_assets_z_chg_13w:",
            "ON RRP z:on_rrp_z_level:",
        ]
        metrics_for_delta, prev_regime = get_delta_metrics("monetary", snapshot_data, metric_keys, delta_days)
        show_delta_summary("monetary", regime.label or regime.name, prev_regime, metrics_for_delta, delta_days)
        
        if prev_regime is None:
            console.print(f"\n[dim]No cached data from {delta_days}d ago. Run `lox labs monetary` daily to build history.[/dim]")
        return

    # Uniform regime panel
    score = 70 if "qt_biting" in regime.name or "scarcity" in regime.name else (30 if "abundant" in regime.name else 50)
    metrics = [
        {"name": "EFFR", "value": effr_disp, "context": "policy rate"},
        {"name": "Reserves level", "value": res_level, "context": _ctx_z(float(res_z)) if isinstance(res_z, (int, float)) else "n/a"},
        {"name": "Reserves % GDP", "value": res_pct_gdp_disp, "context": "anchor"},
        {"name": "Reserves Δ13w", "value": res_chg_13w, "context": "change"},
        {"name": "Fed assets Δ13w", "value": fa_chg_13w, "context": _ctx_qt(float(fa_z)) if isinstance(fa_z, (int, float)) else "n/a"},
        {"name": "ON RRP level", "value": rrp_level, "context": _ctx_z(float(rrp_z)) if isinstance(rrp_z, (int, float)) else "n/a"},
    ]
    print(render_regime_panel(
        domain="Monetary",
        asof=d.get("asof", "n/a"),
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
            domain="monetary",
            snapshot=snapshot_data,
            regime_label=regime.label or regime.name,
            regime_description=regime.description,
            ticker=ticker,
        )


def register(monetary_app: typer.Typer) -> None:
    # Default callback so `lox labs monetary --llm` works without `snapshot`
    @monetary_app.callback(invoke_without_command=True)
    def monetary_default(
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
        """Fed aggregate liquidity (reserves, balance sheet, RRP) — quantity of money in system"""
        if ctx.invoked_subcommand is None:
            _run_monetary_snapshot(llm=llm, ticker=ticker, features=features, json_out=json_out, delta=delta, alert=alert, calendar=calendar, trades=trades)

    @monetary_app.command("snapshot")
    def monetary_snapshot(
        lookback_years: int = typer.Option(5, "--lookback-years", help="How many years of history to load."),
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
        """Monetary regime snapshot (Fed balance sheet, reserves, RRP)."""
        _run_monetary_snapshot(lookback_years=lookback_years, refresh=refresh, llm=llm, ticker=ticker, features=features, json_out=json_out, delta=delta, alert=alert, calendar=calendar, trades=trades)

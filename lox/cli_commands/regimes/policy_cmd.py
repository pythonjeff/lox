"""CLI command for the Policy / geopolitical uncertainty regime.

Displays EPU index reading, policy news pulse, cross-regime signals,
and the 3-layer classifier output.

Author: Lox Capital Research
"""
from __future__ import annotations

from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table as RichTable

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings


# ─────────────────────────────────────────────────────────────────────────────
# Sparkline helpers
# ─────────────────────────────────────────────────────────────────────────────

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float] | tuple[float, ...]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo if hi != lo else 1.0
    return "".join(
        _SPARK_CHARS[min(len(_SPARK_CHARS) - 1, int((v - lo) / span * (len(_SPARK_CHARS) - 1)))]
        for v in values
    )


def _trend_arrow(values: list[float] | tuple[float, ...]) -> str:
    """Compare first half vs second half to determine trend direction."""
    if len(values) < 6:
        return ""
    mid = len(values) // 2
    old_avg = sum(values[:mid]) / mid
    new_avg = sum(values[mid:]) / (len(values) - mid)
    delta = new_avg - old_avg
    threshold = (max(values) - min(values)) * 0.1 if max(values) != min(values) else 0.01
    if abs(delta) < threshold:
        return "[dim]→ stable[/dim]"
    if delta > 0:
        return "[red]↑ rising[/red]"
    return "[green]↓ falling[/green]"


# ─────────────────────────────────────────────────────────────────────────────
# Block 1: EPU Sparkline
# ─────────────────────────────────────────────────────────────────────────────

def _show_epu_sparkline(console: Console, inputs) -> None:
    """Display EPU 90-day sparkline with trend and Δ30d."""
    series = inputs.epu_series_90d
    if not series:
        return

    parts: list[str] = [f"  EPU  {_sparkline(series)}"]
    if inputs.epu_level is not None:
        parts.append(f"  {inputs.epu_level:.0f}")
    if inputs.epu_1y_percentile is not None:
        parts.append(f"  ({inputs.epu_1y_percentile:.0f}th pctl)")
    parts.append(f"  {_trend_arrow(series)}")
    if inputs.epu_30d_change is not None:
        sign = "+" if inputs.epu_30d_change > 0 else ""
        chg_color = "red" if inputs.epu_30d_change > 0 else "green"
        parts.append(f"  [{chg_color}]Δ30d: {sign}{inputs.epu_30d_change:.0f}[/{chg_color}]")

    console.print()
    console.print("[dim]─── Policy Uncertainty Index (90d daily) ──────────────────────[/dim]")
    console.print("".join(parts))


# ─────────────────────────────────────────────────────────────────────────────
# Block 2: News Pulse
# ─────────────────────────────────────────────────────────────────────────────

def _show_news_pulse(console: Console, inputs) -> None:
    """Display top policy headlines table."""
    headlines = inputs.news_top_headlines
    if not headlines:
        return

    table = RichTable(
        title="Policy News Pulse",
        show_header=True,
        header_style="bold yellow",
        box=None,
        padding=(0, 1),
    )
    table.add_column("Date", style="dim", min_width=10)
    table.add_column("Source", min_width=8)
    table.add_column("Headline", min_width=40)

    for h in headlines:
        table.add_row(
            h.get("date", ""),
            h.get("source", ""),
            h.get("title", "")[:100],
        )

    # Volume context
    vol_parts: list[str] = []
    if inputs.news_article_count_7d is not None:
        vol_parts.append(f"7d: {inputs.news_article_count_7d} articles")
    if inputs.news_article_count_30d is not None:
        vol_parts.append(f"30d: {inputs.news_article_count_30d}")
    if inputs.news_article_count_7d is not None and inputs.news_article_count_30d is not None:
        weekly_avg = inputs.news_article_count_30d / 4.28
        if weekly_avg > 0:
            ratio = inputs.news_article_count_7d / weekly_avg
            if ratio > 2.0:
                vol_parts.append("[red]SPIKE[/red]")
            elif ratio > 1.5:
                vol_parts.append("[yellow]elevated[/yellow]")
            else:
                vol_parts.append("[dim]normal[/dim]")

    console.print()
    console.print(table)
    if vol_parts:
        console.print(f"  Volume: {' | '.join(vol_parts)}")


# ─────────────────────────────────────────────────────────────────────────────
# Block 3: Cross-Regime Signals
# ─────────────────────────────────────────────────────────────────────────────

def _show_cross_regime_signals(console: Console, policy_score: float) -> None:
    """Show cross-regime confirmation/divergence for policy."""
    lines: list[str] = []

    try:
        from lox.data.regime_history import get_score_series

        for domain, display in [("inflation", "Inflation"), ("commodities", "Commodities"), ("volatility", "Vol")]:
            series = get_score_series(domain)
            if not series:
                continue
            latest = series[-1]
            sc = latest.get("score")
            lb = latest.get("label", "")
            if not isinstance(sc, (int, float)):
                continue
            short_lb = lb.split("(")[0].strip() if "(" in lb else lb

            if domain == "inflation":
                if sc >= 50 and policy_score >= 55:
                    lines.append(
                        f"  {display} score {sc:.0f} ({short_lb}) + policy elevated → "
                        f"[yellow]trade war pass-through risk — tariffs feeding CPI[/yellow]"
                    )
                elif sc >= 50 and policy_score < 40:
                    lines.append(
                        f"  {display} score {sc:.0f} ({short_lb}) but policy calm → "
                        f"[dim]inflation is demand-driven, not policy-driven[/dim]"
                    )
                elif sc < 40 and policy_score >= 55:
                    lines.append(
                        f"  {display} score {sc:.0f} ({short_lb}) + policy stress → "
                        f"[dim]policy shock hasn't passed through to prices yet[/dim]"
                    )
                else:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [dim]neutral[/dim]")

            elif domain == "commodities":
                if sc >= 50 and policy_score >= 55:
                    lines.append(
                        f"  {display} score {sc:.0f} ({short_lb}) + policy stress → "
                        f"[red]supply disruption + policy risk = commodity squeeze[/red]"
                    )
                elif sc >= 50 and policy_score < 40:
                    lines.append(
                        f"  {display} score {sc:.0f} ({short_lb}) but policy calm → "
                        f"[dim]commodity stress is supply-side, not geopolitical[/dim]"
                    )
                else:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [dim]neutral[/dim]")

            elif domain == "volatility":
                if sc > 60 and policy_score >= 55:
                    lines.append(
                        f"  {display} score {sc:.0f} ({short_lb}) + policy stress → "
                        f"[red]market pricing geopolitical tail risk[/red]"
                    )
                elif sc < 35 and policy_score >= 55:
                    lines.append(
                        f"  {display} score {sc:.0f} ({short_lb}) but policy elevated → "
                        f"[yellow]complacency — market ignoring policy risk[/yellow]"
                    )
                elif sc > 60 and policy_score < 40:
                    lines.append(
                        f"  {display} score {sc:.0f} ({short_lb}) + policy calm → "
                        f"[dim]vol spike is not policy-driven[/dim]"
                    )
                else:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [dim]neutral[/dim]")

    except Exception:
        pass

    if lines:
        console.print()
        console.print("[dim]─── Cross-Regime Signals ──────────────────────────────────────[/dim]")
        for ln in lines:
            console.print(ln)


# ─────────────────────────────────────────────────────────────────────────────
# Core entry point
# ─────────────────────────────────────────────────────────────────────────────

def policy_snapshot(
    *,
    llm: bool = False,
    ticker: str = "",
    refresh: bool = False,
    features: bool = False,
    json_out: bool = False,
    delta: str = "",
    alert: bool = False,
    calendar: bool = False,
    trades: bool = False,
) -> None:
    """Entry point for ``lox regime policy``."""
    settings = load_settings()
    console = Console()

    from lox.altdata.policy_market import compute_policy_inputs

    inputs = compute_policy_inputs(settings=settings, refresh=refresh)

    if inputs.error:
        console.print(f"[yellow]⚠  {inputs.error}[/yellow]")
        return

    from lox.policy.regime import classify_policy_regime

    # Fetch oil disruption score for Hormuz cross-signal
    oil_disruption: float | None = None
    try:
        from lox.data.regime_history import get_score_series
        oil_series = get_score_series("oil")
        if oil_series:
            oil_disruption = oil_series[-1].get("score")
    except Exception:
        pass

    # Fetch cross-regime scores for Layer 3
    inflation_score: float | None = None
    commodities_score: float | None = None
    volatility_score: float | None = None
    try:
        from lox.data.regime_history import get_score_series
        for domain, var_name in [("inflation", "inflation_score"),
                                  ("commodities", "commodities_score"),
                                  ("volatility", "volatility_score")]:
            series = get_score_series(domain)
            if series:
                s = series[-1].get("score")
                if isinstance(s, (int, float)):
                    if var_name == "inflation_score":
                        inflation_score = s
                    elif var_name == "commodities_score":
                        commodities_score = s
                    elif var_name == "volatility_score":
                        volatility_score = s
    except Exception:
        pass

    result = classify_policy_regime(
        epu_level=inputs.epu_level,
        epu_1y_percentile=inputs.epu_1y_percentile,
        epu_30d_change=inputs.epu_30d_change,
        news_article_count_7d=inputs.news_article_count_7d,
        news_article_count_30d=inputs.news_article_count_30d,
        import_price_yoy=inputs.import_price_yoy,
        import_price_mom_accel=inputs.import_price_mom_accel,
        vix_level=inputs.vix_level,
        dxy_20d_chg=inputs.dxy_20d_chg,
        oil_disruption_score=oil_disruption,
        inflation_score=inflation_score,
        commodities_score=commodities_score,
        volatility_score=volatility_score,
    )

    # ── Build snapshot for delta / export ─────────────────────────────────
    snapshot_data = {
        "epu_level": inputs.epu_level,
        "epu_1y_percentile": inputs.epu_1y_percentile,
        "epu_30d_change": inputs.epu_30d_change,
        "news_7d": inputs.news_article_count_7d,
        "news_30d": inputs.news_article_count_30d,
        "import_price_yoy": inputs.import_price_yoy,
        "import_price_mom_accel": inputs.import_price_mom_accel,
        "vix_level": inputs.vix_level,
        "dxy_20d_chg": inputs.dxy_20d_chg,
    }

    from lox.cli_commands.shared.labs_utils import (
        handle_output_flags, parse_delta_period, show_delta_summary,
        get_delta_metrics, save_snapshot,
    )

    feature_dict = result.to_feature_dict()
    save_snapshot("policy", snapshot_data, result.label)

    # ── Handle --features and --json flags ────────────────────────────────
    if handle_output_flags(
        domain="policy",
        snapshot=snapshot_data,
        features=feature_dict,
        regime=result.label,
        regime_description=result.description,
        asof=inputs.asof,
        output_json=json_out,
        output_features=features,
    ):
        return

    # ── Handle --alert flag ───────────────────────────────────────────────
    if alert:
        if result.score < 65:
            return  # suppress output if not extreme
        console.print(
            f"[red bold]⚠ POLICY ALERT:[/red bold] {result.label} "
            f"(score {result.score:.0f}) — {result.description}"
        )
        return

    # ── Handle --delta flag ───────────────────────────────────────────────
    if delta:
        delta_days = parse_delta_period(delta)
        metric_keys = [
            "EPU Level:epu_level:",
            "EPU %ile:epu_1y_percentile:%",
            "EPU Δ30d:epu_30d_change:",
            "News 7d:news_7d:",
            "News 30d:news_30d:",
            "Import Px YoY:import_price_yoy:%",
        ]
        metrics_for_delta, prev_regime = get_delta_metrics(
            "policy", snapshot_data, metric_keys, delta_days,
        )
        show_delta_summary(
            "policy", result.label, prev_regime, metrics_for_delta, delta_days,
        )
        if prev_regime is None:
            console.print(
                f"\n[dim]No cached data from {delta_days}d ago. "
                f"Run `lox regime policy` daily to build history.[/dim]"
            )
        return

    # ── Build metrics for panel ───────────────────────────────────────────
    def _v(x, fmt="{:.1f}"):
        return fmt.format(x) if x is not None else "n/a"

    metrics = [
        {
            "name": "EPU Index",
            "value": _v(inputs.epu_level, "{:.0f}"),
            "context": _epu_context(inputs.epu_level),
        },
        {
            "name": "EPU Percentile (1Y)",
            "value": f"{inputs.epu_1y_percentile:.0f}th" if inputs.epu_1y_percentile is not None else "n/a",
            "context": _pctl_context(inputs.epu_1y_percentile),
        },
        {
            "name": "EPU Δ30d",
            "value": f"{inputs.epu_30d_change:+.0f}" if inputs.epu_30d_change is not None else "n/a",
            "context": _momentum_context(inputs.epu_30d_change),
        },
        {
            "name": "─── Trade Friction ───",
            "value": "",
            "context": "",
        },
        {
            "name": "Import Px YoY",
            "value": f"{inputs.import_price_yoy:+.1f}%" if inputs.import_price_yoy is not None else "n/a",
            "context": _import_context(inputs.import_price_yoy),
        },
        {
            "name": "Import Accel",
            "value": f"{inputs.import_price_mom_accel:+.2f}" if inputs.import_price_mom_accel is not None else "n/a",
            "context": "accelerating" if inputs.import_price_mom_accel is not None and inputs.import_price_mom_accel > 0.5 else (
                "decelerating" if inputs.import_price_mom_accel is not None and inputs.import_price_mom_accel < -0.5 else "stable"
            ),
        },
        {
            "name": "─── News Volume ───",
            "value": "",
            "context": "",
        },
        {
            "name": "Policy News (7d)",
            "value": str(inputs.news_article_count_7d) if inputs.news_article_count_7d is not None else "n/a",
            "context": _news_context(inputs.news_article_count_7d),
        },
        {
            "name": "Policy News (30d)",
            "value": str(inputs.news_article_count_30d) if inputs.news_article_count_30d is not None else "n/a",
            "context": "",
        },
    ]

    from lox.regimes.trend import get_domain_trend
    trend = get_domain_trend("policy", result.score, result.label)

    print(render_regime_panel(
        domain="Policy",
        asof=inputs.asof,
        regime_label=result.label,
        score=result.score,
        percentile=None,
        description=result.description,
        metrics=metrics,
        trend=trend,
    ))

    # ── Block 1: EPU Sparkline ────────────────────────────────────────────
    _show_epu_sparkline(console, inputs)

    # ── Block 2: News Pulse ───────────────────────────────────────────────
    _show_news_pulse(console, inputs)

    # ── Block 3: Cross-Regime Signals ─────────────────────────────────────
    _show_cross_regime_signals(console, result.score)

    # ── Handle --calendar flag ────────────────────────────────────────────
    if calendar:
        _show_policy_calendar(console)

    # ── Handle --trades flag ──────────────────────────────────────────────
    if trades:
        _show_trade_expressions(console, result)

    # ── LLM chat ──────────────────────────────────────────────────────────
    if llm:
        from lox.cli_commands.shared.regime_chat import start_regime_chat

        start_regime_chat(
            settings=settings,
            domain="policy",
            snapshot=snapshot_data,
            regime_label=result.label,
            regime_description=result.description,
            ticker=ticker,
            console=console,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Context helpers
# ─────────────────────────────────────────────────────────────────────────────

def _epu_context(level: float | None) -> str:
    if level is None:
        return "—"
    if level > 400:
        return "crisis-level"
    if level > 300:
        return "very elevated"
    if level > 200:
        return "elevated"
    if level > 150:
        return "above average"
    if level > 100:
        return "normal"
    return "very calm"


def _pctl_context(pctl: float | None) -> str:
    if pctl is None:
        return "—"
    if pctl > 90:
        return "extreme"
    if pctl > 75:
        return "high"
    if pctl > 50:
        return "above median"
    if pctl > 25:
        return "below median"
    return "low"


def _momentum_context(chg: float | None) -> str:
    if chg is None:
        return "—"
    if chg > 100:
        return "rapid escalation"
    if chg > 50:
        return "escalating"
    if chg > 25:
        return "rising"
    if chg < -100:
        return "rapid de-escalation"
    if chg < -50:
        return "de-escalating"
    if chg < -25:
        return "declining"
    return "stable"


def _import_context(yoy: float | None) -> str:
    if yoy is None:
        return "—"
    if yoy > 10:
        return "severe pass-through"
    if yoy > 5:
        return "significant"
    if yoy > 2:
        return "moderate"
    if yoy > 0:
        return "mild"
    return "benign"


def _news_context(count: int | None) -> str:
    if count is None:
        return "—"
    if count >= 30:
        return "crisis volume"
    if count >= 20:
        return "very high"
    if count >= 15:
        return "high"
    if count >= 10:
        return "elevated"
    if count >= 5:
        return "normal"
    return "quiet"


# ─────────────────────────────────────────────────────────────────────────────
# Calendar & trade helpers
# ─────────────────────────────────────────────────────────────────────────────

def _show_policy_calendar(console: Console) -> None:
    """Show upcoming policy-relevant events."""
    console.print()
    console.print("[dim]─── Policy Calendar ───────────────────────────────────────────[/dim]")
    console.print("  [dim]Trade policy events are ad-hoc. Watch for:[/dim]")
    console.print("  • USTR tariff announcements & comment deadlines")
    console.print("  • Executive orders on trade/economic security")
    console.print("  • G7/G20 summits & bilateral trade meetings")
    console.print("  • Congressional trade legislation markups")
    console.print("  • FRED EPU releases (daily, ~2d lag)")


def _show_trade_expressions(console: Console, result) -> None:
    """Show quick trade ideas for the current policy regime."""
    console.print()
    console.print("[dim]─── Trade Expressions ─────────────────────────────────────────[/dim]")

    if result.score >= 65:
        console.print("  [bold]Policy Stress / Crisis[/bold]")
        console.print("  • LONG GLD calls — gold as geopolitical hedge")
        console.print("  • SHORT EEM puts — EM exposed to trade friction")
        console.print("  • LONG XLE equity — energy benefits from supply disruption")
        console.print("  • LONG TIP — TIPS hedge tariff-driven inflation")
        console.print("  • SHORT XLY puts — consumer discretionary under pressure")
    elif result.score >= 50:
        console.print("  [bold]Elevated Uncertainty[/bold]")
        console.print("  • LONG GLD calls (starter) — optionality on escalation")
        console.print("  • REDUCE EM equity exposure — trim before clarity")
        console.print("  • LONG DXY via UUP — safe-haven demand")
    elif result.score >= 35:
        console.print("  [bold]Moderate Uncertainty[/bold]")
        console.print("  • Monitor for escalation signals")
        console.print("  • Consider cheap tail hedges (GLD, VIX calls)")
    else:
        console.print("  [bold]Policy Calm / Low Uncertainty[/bold]")
        console.print("  • Favor risk-on: LONG QQQ, EEM, reduce hedges")
        console.print("  • Policy tailwind for cyclicals and trade-sensitive names")

"""CLI command for the Policy / Geopolitical Uncertainty regime."""
from __future__ import annotations

from rich import print
from rich.panel import Panel

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings


# ── Interpretation thresholds ─────────────────────────────────────────────

# EPU level context (historical median ~110, mean ~130)
EPU_CRISIS = 400
EPU_HIGH = 300
EPU_ELEVATED = 200
EPU_MODERATE = 150
EPU_NORMAL = 100

# News article count (7d)
NEWS_CRISIS = 30
NEWS_ELEVATED = 15
NEWS_MODERATE = 10
NEWS_NORMAL = 5

# Import price YoY thresholds
IMP_HIGH = 5.0
IMP_MODERATE = 2.0
IMP_MILD = 0.0


# ── Formatting helpers ────────────────────────────────────────────────────

def _fmt(x: object, fmt: str = "{:.1f}") -> str:
    return fmt.format(float(x)) if isinstance(x, (int, float)) else "n/a"

def _fmt_pct(x: object) -> str:
    return f"{float(x):+.1f}%" if isinstance(x, (int, float)) else "n/a"

def _fmt_int(x: object) -> str:
    return f"{int(x)}" if isinstance(x, (int, float)) else "n/a"

def _fmt_chg(x: object) -> str:
    return f"{float(x):+.0f}" if isinstance(x, (int, float)) else "n/a"


# ── Context helpers ───────────────────────────────────────────────────────

def _epu_level_ctx(val) -> str:
    if not isinstance(val, (int, float)):
        return "Economic Policy Uncertainty Index"
    v = float(val)
    if v >= EPU_CRISIS:
        return "crisis territory — COVID/trade-war-era levels"
    if v >= EPU_HIGH:
        return "policy crisis — significant market risk"
    if v >= EPU_ELEVATED:
        return "elevated — tariff/regulation headlines heavy"
    if v >= EPU_MODERATE:
        return "above normal — policy noise building"
    if v >= EPU_NORMAL:
        return "normal range — baseline uncertainty"
    return "low — policy calm, risk-on backdrop"


def _epu_pctl_ctx(val) -> str:
    if not isinstance(val, (int, float)):
        return "1Y percentile of EPU index"
    v = float(val)
    if v >= 90:
        return "extreme — top decile vs trailing year"
    if v >= 75:
        return "high — upper quartile"
    if v >= 50:
        return "above median"
    if v >= 25:
        return "below median — relatively calm"
    return "low — bottom quartile, complacent"


def _epu_chg_ctx(val) -> str:
    if not isinstance(val, (int, float)):
        return "30-day change in EPU"
    v = float(val)
    if v > 100:
        return "rapid escalation — policy shock unfolding"
    if v > 50:
        return "escalating — uncertainty building fast"
    if v > 25:
        return "rising — watch for follow-through"
    if v > -25:
        return "stable — no major directional shift"
    if v > -50:
        return "easing — de-escalation underway"
    return "rapid de-escalation — risk clearing"


def _news_7d_ctx(val) -> str:
    if not isinstance(val, (int, float)):
        return "policy keyword articles (7 days)"
    v = int(val)
    if v >= NEWS_CRISIS:
        return "crisis-level media saturation"
    if v >= NEWS_ELEVATED:
        return "elevated — policy dominating news cycle"
    if v >= NEWS_MODERATE:
        return "above average — active headline risk"
    if v >= NEWS_NORMAL:
        return "normal flow"
    return "quiet — minimal policy headlines"


def _news_30d_ctx(val) -> str:
    if not isinstance(val, (int, float)):
        return "policy keyword articles (30 days)"
    v = int(val)
    if v >= 60:
        return "persistent high volume — sustained policy theme"
    if v >= 30:
        return "elevated run — multi-week focus"
    return "normal baseline"


def _import_yoy_ctx(val) -> str:
    if not isinstance(val, (int, float)):
        return "Import Price Index YoY"
    v = float(val)
    if v > 10:
        return "extreme — tariff/supply pass-through evident"
    if v > IMP_HIGH:
        return "elevated — cost-push pressure building"
    if v > IMP_MODERATE:
        return "mild upward — watch for acceleration"
    if v > IMP_MILD:
        return "negligible — no import cost pressure"
    if v > -2:
        return "declining — deflationary, trade friction easing"
    return "sharply negative — disinflation / strong dollar effect"


def _import_accel_ctx(val) -> str:
    if not isinstance(val, (int, float)):
        return "MoM acceleration (2nd derivative)"
    v = float(val)
    if v > 1.0:
        return "accelerating — tariff escalation passing through"
    if v > 0.5:
        return "mild acceleration"
    if v > -0.5:
        return "stable momentum"
    if v > -1.0:
        return "decelerating"
    return "sharply decelerating — pressure easing"


# ── Tail risk / headwind / tailwind assessment ────────────────────────────

def _build_winds(inputs) -> list[dict]:
    """Build tail-risk, headwind, and tailwind bullets from PolicyInputs."""
    winds: list[dict] = []

    epu = inputs.epu_level
    epu_chg = inputs.epu_30d_change
    news_7d = inputs.news_article_count_7d
    news_30d = inputs.news_article_count_30d
    imp_yoy = inputs.import_price_yoy
    imp_accel = inputs.import_price_mom_accel

    # Tail risks
    if isinstance(epu, (int, float)) and epu >= EPU_CRISIS:
        winds.append({"type": "TAIL RISK", "text": "EPU at crisis levels — market-wide risk repricing, safe-haven rotation likely", "color": "red"})
    if isinstance(epu_chg, (int, float)) and epu_chg > 100:
        winds.append({"type": "TAIL RISK", "text": f"EPU spike Δ{epu_chg:+.0f} in 30d — shock unfolding, vol expansion expected", "color": "red"})
    if isinstance(news_7d, (int, float)) and news_7d >= NEWS_CRISIS:
        winds.append({"type": "TAIL RISK", "text": f"{news_7d} policy articles in 7d — media saturation, headline-driven markets", "color": "red"})
    if isinstance(imp_yoy, (int, float)) and imp_yoy > 10:
        winds.append({"type": "TAIL RISK", "text": f"Import prices {imp_yoy:+.1f}% YoY — tariff pass-through accelerating, margin compression risk", "color": "red"})

    # Headwinds
    if isinstance(epu, (int, float)) and EPU_ELEVATED <= epu < EPU_CRISIS:
        winds.append({"type": "HEADWIND", "text": f"EPU {epu:.0f} — elevated uncertainty weighs on capex, M&A, risk appetite", "color": "yellow"})
    if isinstance(news_7d, (int, float)) and NEWS_MODERATE <= news_7d < NEWS_CRISIS:
        winds.append({"type": "HEADWIND", "text": f"{news_7d} policy articles (7d) — active headline risk, sector rotation possible", "color": "yellow"})
    if isinstance(imp_yoy, (int, float)) and IMP_MODERATE < imp_yoy <= 10:
        winds.append({"type": "HEADWIND", "text": f"Import prices {imp_yoy:+.1f}% YoY — cost-push building, watch for margin guidance cuts", "color": "yellow"})
    if isinstance(imp_accel, (int, float)) and imp_accel > 0.5:
        winds.append({"type": "HEADWIND", "text": "Import price acceleration — tariff escalation likely passing through to consumer", "color": "yellow"})

    # Tailwinds
    if isinstance(epu, (int, float)) and epu < EPU_NORMAL:
        winds.append({"type": "TAILWIND", "text": "Low policy uncertainty — supportive for capex, M&A activity, and risk appetite", "color": "green"})
    if isinstance(epu_chg, (int, float)) and epu_chg < -50:
        winds.append({"type": "TAILWIND", "text": f"EPU declining {epu_chg:+.0f} (30d) — de-escalation in progress, risk-on backdrop", "color": "green"})
    if isinstance(news_7d, (int, float)) and news_7d < NEWS_NORMAL:
        winds.append({"type": "TAILWIND", "text": "Low policy news volume — headline risk minimal", "color": "green"})
    if isinstance(imp_yoy, (int, float)) and imp_yoy < -2:
        winds.append({"type": "TAILWIND", "text": f"Import prices {imp_yoy:+.1f}% YoY — disinflation tailwind, margin expansion potential", "color": "green"})

    if not winds:
        winds.append({"type": "NEUTRAL", "text": "Policy uncertainty in normal range — no strong directional pressure", "color": "dim"})

    return winds


# ── Core implementation ───────────────────────────────────────────────────

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
    """Entry point for `lox regime policy`."""
    from rich.console import Console
    from lox.cli_commands.shared.labs_utils import (
        handle_output_flags, parse_delta_period, show_delta_summary,
        get_delta_metrics, save_snapshot,
        show_alert_output, show_calendar_output, show_trades_output,
    )

    settings = load_settings()
    console = Console()

    from lox.altdata.policy_market import compute_policy_inputs
    inputs = compute_policy_inputs(settings=settings, refresh=refresh)

    if inputs.error:
        console.print(f"[red]Policy data error:[/red] {inputs.error}")
        return

    # Fetch cross-regime scores for Layer 3 (from cached history)
    inflation_score = None
    commodities_score = None
    volatility_score = None
    oil_disruption = None
    try:
        from lox.data.regime_history import get_score_series
        for domain, target in [("inflation", "inflation"), ("commodities", "commodities"), ("volatility", "volatility"), ("oil", "oil")]:
            series = get_score_series(domain)
            if series:
                sc = series[-1].get("score")
                if domain == "inflation":
                    inflation_score = sc
                elif domain == "commodities":
                    commodities_score = sc
                elif domain == "volatility":
                    volatility_score = sc
                elif domain == "oil":
                    oil_disruption = sc
    except Exception:
        pass

    from lox.policy.regime import classify_policy_regime
    regime = classify_policy_regime(
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

    # ── Build snapshot dict ───────────────────────────────────────────────
    snapshot_data = {
        "epu_level": inputs.epu_level,
        "epu_1y_percentile": inputs.epu_1y_percentile,
        "epu_30d_change": inputs.epu_30d_change,
        "news_article_count_7d": inputs.news_article_count_7d,
        "news_article_count_30d": inputs.news_article_count_30d,
        "import_price_yoy": inputs.import_price_yoy,
        "import_price_mom_accel": inputs.import_price_mom_accel,
        "vix_level": inputs.vix_level,
        "dxy_20d_chg": inputs.dxy_20d_chg,
        "regime": regime.label,
    }

    feature_dict = {
        "epu_level": inputs.epu_level,
        "epu_1y_percentile": inputs.epu_1y_percentile,
        "epu_30d_change": inputs.epu_30d_change,
        "news_article_count_7d": inputs.news_article_count_7d,
        "news_article_count_30d": inputs.news_article_count_30d,
        "import_price_yoy": inputs.import_price_yoy,
        "import_price_mom_accel": inputs.import_price_mom_accel,
    }

    save_snapshot("policy", snapshot_data, regime.label)

    if handle_output_flags(
        domain="policy",
        snapshot=snapshot_data,
        features=feature_dict,
        regime=regime.label,
        regime_description=regime.description,
        asof=inputs.asof,
        output_json=json_out,
        output_features=features,
    ):
        return

    if alert:
        show_alert_output("policy", regime.label, snapshot_data, regime.description)
        return

    if calendar:
        print(Panel.fit(f"[b]Regime:[/b] {regime.label}", title="Policy Uncertainty", border_style="cyan"))
        show_calendar_output("policy")
        return

    if trades:
        print(Panel.fit(f"[b]Regime:[/b] {regime.label}", title="Policy Uncertainty", border_style="cyan"))
        show_trades_output("policy", regime.label)
        return

    if delta:
        delta_days = parse_delta_period(delta)
        metric_keys = [
            "EPU Level:epu_level:",
            "EPU %ile:epu_1y_percentile:",
            "EPU 30d Chg:epu_30d_change:",
            "News 7d:news_article_count_7d:",
            "News 30d:news_article_count_30d:",
            "Import Px YoY:import_price_yoy:%",
            "Import Accel:import_price_mom_accel:",
        ]
        metrics_for_delta, prev_regime = get_delta_metrics("policy", snapshot_data, metric_keys, delta_days)
        show_delta_summary("policy", regime.label, prev_regime, metrics_for_delta, delta_days)
        if prev_regime is None:
            console.print(f"\n[dim]No cached data from {delta_days}d ago. Run `lox regime policy` daily to build history.[/dim]")
        return

    # ── Metrics for panel ─────────────────────────────────────────────────
    metrics = [
        # ── EPU Index ──
        {"name": "─── EPU Index ───", "value": "", "context": ""},
        {"name": "EPU Level", "value": _fmt(inputs.epu_level, "{:.0f}"), "context": _epu_level_ctx(inputs.epu_level)},
        {"name": "1Y Percentile", "value": _fmt(inputs.epu_1y_percentile, "{:.0f}th"), "context": _epu_pctl_ctx(inputs.epu_1y_percentile)},
        {"name": "30d Change", "value": _fmt_chg(inputs.epu_30d_change), "context": _epu_chg_ctx(inputs.epu_30d_change)},
        # ── News Flow ──
        {"name": "─── News Flow ───", "value": "", "context": ""},
        {"name": "Articles (7d)", "value": _fmt_int(inputs.news_article_count_7d), "context": _news_7d_ctx(inputs.news_article_count_7d)},
        {"name": "Articles (30d)", "value": _fmt_int(inputs.news_article_count_30d), "context": _news_30d_ctx(inputs.news_article_count_30d)},
        # ── Import Prices ──
        {"name": "─── Import Prices ───", "value": "", "context": ""},
        {"name": "Import Px YoY", "value": _fmt_pct(inputs.import_price_yoy), "context": _import_yoy_ctx(inputs.import_price_yoy)},
        {"name": "MoM Accel", "value": _fmt(inputs.import_price_mom_accel, "{:+.2f}"), "context": _import_accel_ctx(inputs.import_price_mom_accel)},
        # ── Cross-Signal Inputs ──
        {"name": "─── Cross-Signals ───", "value": "", "context": ""},
        {"name": "VIX", "value": _fmt(inputs.vix_level, "{:.1f}"), "context": "amplifier input — confirms risk when policy elevated"},
        {"name": "DXY 20d Chg", "value": _fmt_pct(inputs.dxy_20d_chg), "context": "safe-haven demand signal"},
    ]

    # ── Trend ─────────────────────────────────────────────────────────────
    from lox.regimes.trend import get_domain_trend
    trend = get_domain_trend("policy", regime.score, regime.label)

    print(render_regime_panel(
        domain="Policy Uncertainty",
        asof=inputs.asof,
        regime_label=regime.label,
        score=regime.score,
        percentile=None,
        description=regime.description,
        metrics=metrics,
        trend=trend,
    ))

    # ── Tail Risks / Headwinds / Tailwinds ────────────────────────────────
    winds = _build_winds(inputs)
    if winds:
        console.print()
        console.print("[dim]─── Tail Risks / Headwinds / Tailwinds ────────────────────────[/dim]")
        for w in winds:
            color = w["color"]
            wtype = w["type"]
            text = w["text"]
            if wtype == "TAIL RISK":
                console.print(f"  [bold {color}]⚠ {wtype}[/bold {color}]: [{color}]{text}[/{color}]")
            elif wtype == "HEADWIND":
                console.print(f"  [{color}]▼ {wtype}[/{color}]: [{color}]{text}[/{color}]")
            elif wtype == "TAILWIND":
                console.print(f"  [{color}]▲ {wtype}[/{color}]: [{color}]{text}[/{color}]")
            else:
                console.print(f"  [{color}]● {wtype}[/{color}]: [{color}]{text}[/{color}]")

    # ── Top Headlines ────────────────────────────────────────────────────
    if inputs.news_top_headlines:
        console.print()
        console.print("[dim]─── Top Policy Headlines ──────────────────────────────────────[/dim]")
        for h in inputs.news_top_headlines:
            title = h.get("title", "")
            source = h.get("source", "")
            date = h.get("date", "")
            console.print(f"  [dim]{date}[/dim]  {title}  [dim]({source})[/dim]")

    # ── Cross-regime signals ──────────────────────────────────────────────
    _show_cross_regime_signals(console, inputs, regime)

    if llm:
        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis
        print_llm_regime_analysis(
            settings=settings,
            domain="policy",
            snapshot=snapshot_data,
            regime_label=regime.label,
            regime_description=regime.description,
            ticker=ticker,
        )


def _show_cross_regime_signals(console, inputs, regime) -> None:
    """Show cross-regime confirmation/divergence signals."""
    lines: list[str] = []

    try:
        from lox.data.regime_history import get_score_series
        policy_score = regime.score

        for domain, display in [("inflation", "Inflation"), ("volatility", "Volatility"), ("commodities", "Commodities"), ("credit", "Credit"), ("growth", "Growth")]:
            series = get_score_series(domain)
            if not series:
                continue
            latest = series[-1]
            sc = latest.get("score")
            lb = latest.get("label", "")
            if not isinstance(sc, (int, float)):
                continue
            short_lb = lb.split("(")[0].strip() if "(" in lb else lb

            is_policy_high = policy_score >= 55
            is_policy_low = policy_score < 35

            if domain == "inflation" and is_policy_high and sc >= 50:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]trade war cascade — tariff-driven inflation + uncertainty = stagflation risk[/yellow]")
            elif domain == "volatility" and is_policy_high and sc < 35:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]complacency — policy stress elevated but vol subdued, surprise risk[/yellow]")
            elif domain == "volatility" and is_policy_high and sc >= 55:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [red]confirmed — policy stress + vol elevated = risk-off[/red]")
            elif domain == "commodities" and is_policy_high and sc >= 50:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]supply chain pressure — policy + commodity stress = cost-push inflation[/yellow]")
            elif domain == "credit" and is_policy_high and sc >= 50:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]funding stress — policy uncertainty tightening credit conditions[/yellow]")
            elif domain == "growth" and is_policy_high and sc >= 55:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [red]growth drag — policy uncertainty suppressing activity[/red]")
            elif is_policy_low:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [dim]policy calm, neutral cross-signal[/dim]")
            else:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [dim]neutral[/dim]")
    except Exception:
        pass

    if lines:
        console.print()
        console.print("[dim]─── Cross-Regime Signals ──────────────────────────────────────[/dim]")
        for ln in lines:
            console.print(ln)

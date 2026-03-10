"""CLI command for the USD regime."""
from __future__ import annotations

from rich import print
from rich.panel import Panel

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings


# ── Interpretation thresholds ─────────────────────────────────────────────

# DXY-style broad index level context (DTWEXBGS trades ~100-130 range)
IDX_VERY_HIGH = 120
IDX_HIGH = 112
IDX_NEUTRAL_HI = 105
IDX_NEUTRAL_LO = 98
IDX_LOW = 92

# 20-day change %
CHG20_SURGE = 2.0
CHG20_STRONG = 1.0
CHG20_WEAK = -1.0
CHG20_PLUNGE = -2.0

# 60-day change %
CHG60_SURGE = 4.0
CHG60_STRONG = 2.0
CHG60_WEAK = -2.0
CHG60_PLUNGE = -4.0

# Realized vol (annualized %)
RVOL_HIGH = 10.0
RVOL_ELEVATED = 7.0
RVOL_NORMAL = 5.0

# 200d MA distance %
MA_EXTENDED = 3.0
MA_ABOVE = 1.0
MA_BELOW = -1.0
MA_BREAKDOWN = -3.0


# ── Formatting helpers ────────────────────────────────────────────────────

def _fmt(x: object, fmt: str = "{:.2f}") -> str:
    return fmt.format(float(x)) if isinstance(x, (int, float)) else "n/a"

def _fmt_pct(x: object) -> str:
    return f"{float(x):+.2f}%" if isinstance(x, (int, float)) else "n/a"

def _fmt_z(x: object) -> str:
    return f"{float(x):+.2f}" if isinstance(x, (int, float)) else "n/a"


# ── Context helpers ───────────────────────────────────────────────────────

def _level_ctx(val) -> str:
    if not isinstance(val, (int, float)):
        return "trade-weighted broad USD index"
    v = float(val)
    if v >= IDX_VERY_HIGH:
        return "historically strong — EM/commodity headwind"
    if v >= IDX_HIGH:
        return "elevated — multinational earnings pressure"
    if v >= IDX_NEUTRAL_HI:
        return "firm — upper neutral range"
    if v >= IDX_NEUTRAL_LO:
        return "neutral — no strong directional pressure"
    if v >= IDX_LOW:
        return "soft — tailwind for EM & commodities"
    return "historically weak — potential confidence concern"


def _chg20_ctx(val) -> str:
    if not isinstance(val, (int, float)):
        return "20-day change"
    v = float(val)
    if v >= CHG20_SURGE:
        return "surging — rapid strength, EM pain likely"
    if v >= CHG20_STRONG:
        return "strengthening — building momentum"
    if v > CHG20_WEAK:
        return "range-bound — no clear trend"
    if v > CHG20_PLUNGE:
        return "weakening — risk-on, commodity tailwind"
    return "plunging — potential disorderly move"


def _chg60_ctx(val) -> str:
    if not isinstance(val, (int, float)):
        return "60-day change"
    v = float(val)
    if v >= CHG60_SURGE:
        return "major rally — sustained strength"
    if v >= CHG60_STRONG:
        return "firm trend — grinding higher"
    if v > CHG60_WEAK:
        return "trendless — consolidating"
    if v > CHG60_PLUNGE:
        return "weakening trend — persistent softness"
    return "sharp decline — possible regime shift"


def _yoy_ctx(val) -> str:
    if not isinstance(val, (int, float)):
        return "year-over-year change"
    v = float(val)
    if v > 8.0:
        return "extreme appreciation — EM crisis watch"
    if v > 4.0:
        return "strong YoY — sustained headwind"
    if v > 1.0:
        return "modestly higher vs year ago"
    if v > -1.0:
        return "flat YoY — neutral"
    if v > -4.0:
        return "modestly lower — mild tailwind"
    return "sharp YoY decline — structural shift"


def _z_level_ctx(val) -> str:
    if not isinstance(val, (int, float)):
        return "vs 3-year history"
    v = float(val)
    if v > 2.0:
        return f"z={v:+.2f} — extreme vs history"
    if v > 1.0:
        return f"z={v:+.2f} — elevated vs history"
    if v < -2.0:
        return f"z={v:+.2f} — deeply depressed vs history"
    if v < -1.0:
        return f"z={v:+.2f} — weak vs history"
    return f"z={v:+.2f} — normal range"


def _z_mom_ctx(val) -> str:
    if not isinstance(val, (int, float)):
        return "60-day momentum vs history"
    v = float(val)
    direction = "strengthening" if v > 0 else "weakening"
    if abs(v) > 2.0:
        return f"z={v:+.2f} — extreme {direction} vs history"
    if abs(v) > 1.0:
        return f"z={v:+.2f} — notable {direction}"
    return f"z={v:+.2f} — normal range"


def _strength_ctx(val) -> str:
    if not isinstance(val, (int, float)):
        return "composite strength score"
    v = float(val)
    if v >= 2.0:
        return "extreme strength — surge territory"
    if v >= 1.0:
        return "above average — USD strength regime"
    if v > -1.0:
        return "neutral band — no strong signal"
    if v > -2.0:
        return "below average — USD weakness regime"
    return "extreme weakness — plunge territory"


def _ma_dist_ctx(val) -> str:
    if not isinstance(val, (int, float)):
        return "distance from 200d moving average"
    v = float(val)
    if v > MA_EXTENDED:
        return "well above 200d MA — extended, mean-revert risk"
    if v > MA_ABOVE:
        return "above 200d MA — uptrend intact"
    if v > MA_BELOW:
        return "near 200d MA — pivoting"
    if v > MA_BREAKDOWN:
        return "below 200d MA — downtrend developing"
    return "far below 200d MA — deep downtrend"


def _rvol_ctx(val) -> str:
    if not isinstance(val, (int, float)):
        return "90-day realized volatility"
    v = float(val)
    if v > RVOL_HIGH:
        return "high vol — disorderly FX moves possible"
    if v > RVOL_ELEVATED:
        return "elevated vol — larger daily swings"
    if v > RVOL_NORMAL:
        return "normal vol range"
    return "low vol — compressed, breakout risk"


# ── Tail risk / headwind / tailwind assessment ────────────────────────────

def _build_winds(ui) -> list[dict]:
    """Build tail-risk, headwind, and tailwind bullets from UsdInputs."""
    winds: list[dict] = []

    score = ui.usd_strength_score
    z_level = ui.z_usd_level
    chg20 = ui.usd_chg_20d_pct
    chg60 = ui.usd_chg_60d_pct
    yoy = ui.usd_yoy_chg_pct
    ma_dist = ui.usd_200d_ma_dist_pct
    rvol = ui.usd_90d_rvol
    level = ui.usd_index_broad

    # Tail risks
    if isinstance(score, (int, float)) and score >= 2.0:
        winds.append({"type": "TAIL RISK", "text": "Dollar surge — EM funding stress, commodity crash risk, US earnings translation headwind", "color": "red"})
    if isinstance(score, (int, float)) and score <= -2.0:
        winds.append({"type": "TAIL RISK", "text": "Dollar plunge — potential confidence crisis, imported inflation spike", "color": "red"})
    if isinstance(rvol, (int, float)) and rvol > RVOL_HIGH:
        winds.append({"type": "TAIL RISK", "text": f"FX vol {rvol:.1f}% — disorderly currency moves, carry unwinds possible", "color": "red"})
    if isinstance(ma_dist, (int, float)) and ma_dist > MA_EXTENDED and isinstance(z_level, (int, float)) and z_level > 1.5:
        winds.append({"type": "TAIL RISK", "text": "USD extended above 200d MA at extreme z-score — mean-reversion risk", "color": "red"})

    # Headwinds (USD strong → equity headwinds)
    if isinstance(score, (int, float)) and 1.0 <= score < 2.0:
        winds.append({"type": "HEADWIND", "text": "Strong dollar — multinational revenue translation drag (est. 2-4% EPS headwind per 10% DXY move)", "color": "yellow"})
    if isinstance(chg20, (int, float)) and chg20 >= CHG20_STRONG:
        winds.append({"type": "HEADWIND", "text": f"Short-term momentum {chg20:+.2f}% (20d) — fast moves create dislocation risk", "color": "yellow"})
    if isinstance(chg60, (int, float)) and chg60 >= CHG60_STRONG:
        winds.append({"type": "HEADWIND", "text": f"Sustained strength {chg60:+.2f}% (60d) — commodities & EM under pressure", "color": "yellow"})
    if isinstance(yoy, (int, float)) and yoy > 4.0:
        winds.append({"type": "HEADWIND", "text": f"YoY appreciation {yoy:+.1f}% — persistent earnings drag for S&P multinationals (~40% foreign rev)", "color": "yellow"})

    # Tailwinds (USD weak → equity tailwinds)
    if isinstance(score, (int, float)) and -2.0 < score <= -1.0:
        winds.append({"type": "TAILWIND", "text": "Weak dollar — EM capital inflows, commodity tailwind, US multinational boost", "color": "green"})
    if isinstance(chg20, (int, float)) and chg20 <= CHG20_WEAK:
        winds.append({"type": "TAILWIND", "text": f"Short-term weakness {chg20:+.2f}% (20d) — risk-on conditions, EM rally", "color": "green"})
    if isinstance(chg60, (int, float)) and chg60 <= CHG60_WEAK:
        winds.append({"type": "TAILWIND", "text": f"Sustained decline {chg60:+.2f}% (60d) — commodities bid, carry trade favorable", "color": "green"})
    if isinstance(yoy, (int, float)) and yoy < -4.0:
        winds.append({"type": "TAILWIND", "text": f"YoY depreciation {yoy:+.1f}% — structural tailwind, EM outperformance likely", "color": "green"})
    if isinstance(rvol, (int, float)) and rvol < RVOL_NORMAL:
        winds.append({"type": "TAILWIND", "text": f"Low FX vol {rvol:.1f}% — carry strategies attractive, less hedging cost", "color": "green"})

    # Neutral / mixed
    if not winds:
        winds.append({"type": "NEUTRAL", "text": "Dollar in neutral range — limited directional pressure on equity/commodity complex", "color": "dim"})

    return winds


# ── Core implementation ───────────────────────────────────────────────────

def usd_snapshot(
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
    """Entry point for `lox regime usd`."""
    from rich.console import Console
    from lox.cli_commands.shared.labs_utils import (
        handle_output_flags, parse_delta_period, show_delta_summary,
        get_delta_metrics, save_snapshot,
        show_alert_output, show_calendar_output, show_trades_output,
    )

    settings = load_settings()
    console = Console()

    from lox.usd.signals import build_usd_state
    state = build_usd_state(settings=settings, refresh=refresh)
    ui = state.inputs

    from lox.usd.regime import classify_usd_regime
    regime = classify_usd_regime(ui)

    # ── Build snapshot dict ───────────────────────────────────────────────
    snapshot_data = {
        "usd_index_broad": ui.usd_index_broad,
        "usd_chg_20d_pct": ui.usd_chg_20d_pct,
        "usd_chg_60d_pct": ui.usd_chg_60d_pct,
        "usd_yoy_chg_pct": ui.usd_yoy_chg_pct,
        "z_usd_level": ui.z_usd_level,
        "z_usd_chg_60d": ui.z_usd_chg_60d,
        "usd_strength_score": ui.usd_strength_score,
        "usd_200d_ma_dist_pct": ui.usd_200d_ma_dist_pct,
        "usd_90d_rvol": ui.usd_90d_rvol,
        "is_usd_strong": ui.is_usd_strong,
        "is_usd_weak": ui.is_usd_weak,
        "regime": regime.label,
    }

    feature_dict = {
        "usd_index_broad": ui.usd_index_broad,
        "usd_chg_20d_pct": ui.usd_chg_20d_pct,
        "usd_chg_60d_pct": ui.usd_chg_60d_pct,
        "usd_yoy_chg_pct": ui.usd_yoy_chg_pct,
        "z_usd_level": ui.z_usd_level,
        "z_usd_chg_60d": ui.z_usd_chg_60d,
        "usd_strength_score": ui.usd_strength_score,
        "usd_200d_ma_dist_pct": ui.usd_200d_ma_dist_pct,
        "usd_90d_rvol": ui.usd_90d_rvol,
    }

    save_snapshot("usd", snapshot_data, regime.label)

    if handle_output_flags(
        domain="usd",
        snapshot=snapshot_data,
        features=feature_dict,
        regime=regime.label,
        regime_description=regime.description,
        asof=state.asof,
        output_json=json_out,
        output_features=features,
    ):
        return

    if alert:
        show_alert_output("usd", regime.label, snapshot_data, regime.description)
        return

    if calendar:
        print(Panel.fit(f"[b]Regime:[/b] {regime.label}", title="US Dollar", border_style="cyan"))
        show_calendar_output("usd")
        return

    if trades:
        print(Panel.fit(f"[b]Regime:[/b] {regime.label}", title="US Dollar", border_style="cyan"))
        show_trades_output("usd", regime.label)
        return

    if delta:
        delta_days = parse_delta_period(delta)
        metric_keys = [
            "USD Index:usd_index_broad:",
            "20d Chg:usd_chg_20d_pct:%",
            "60d Chg:usd_chg_60d_pct:%",
            "YoY Chg:usd_yoy_chg_pct:%",
            "Z Level:z_usd_level:",
            "Strength:usd_strength_score:",
            "200d MA Dist:usd_200d_ma_dist_pct:%",
            "90d RVol:usd_90d_rvol:%",
        ]
        metrics_for_delta, prev_regime = get_delta_metrics("usd", snapshot_data, metric_keys, delta_days)
        show_delta_summary("usd", regime.label, prev_regime, metrics_for_delta, delta_days)
        if prev_regime is None:
            console.print(f"\n[dim]No cached data from {delta_days}d ago. Run `lox regime usd` daily to build history.[/dim]")
        return

    # ── Metrics for panel ─────────────────────────────────────────────────
    metrics = [
        # ── USD Level ──
        {"name": "─── USD Level ───", "value": "", "context": ""},
        {"name": "Broad Index", "value": _fmt(ui.usd_index_broad, "{:.2f}"), "context": _level_ctx(ui.usd_index_broad)},
        {"name": "200d MA Dist", "value": _fmt_pct(ui.usd_200d_ma_dist_pct), "context": _ma_dist_ctx(ui.usd_200d_ma_dist_pct)},
        {"name": "Z Level (3Y)", "value": _fmt_z(ui.z_usd_level), "context": _z_level_ctx(ui.z_usd_level)},
        # ── Momentum ──
        {"name": "─── Momentum ───", "value": "", "context": ""},
        {"name": "20d Change", "value": _fmt_pct(ui.usd_chg_20d_pct), "context": _chg20_ctx(ui.usd_chg_20d_pct)},
        {"name": "60d Change", "value": _fmt_pct(ui.usd_chg_60d_pct), "context": _chg60_ctx(ui.usd_chg_60d_pct)},
        {"name": "YoY Change", "value": _fmt_pct(ui.usd_yoy_chg_pct), "context": _yoy_ctx(ui.usd_yoy_chg_pct)},
        {"name": "Z Momentum (60d)", "value": _fmt_z(ui.z_usd_chg_60d), "context": _z_mom_ctx(ui.z_usd_chg_60d)},
        # ── Volatility ──
        {"name": "─── Volatility ───", "value": "", "context": ""},
        {"name": "90d RVol", "value": f"{float(ui.usd_90d_rvol):.1f}%" if isinstance(ui.usd_90d_rvol, (int, float)) else "n/a", "context": _rvol_ctx(ui.usd_90d_rvol)},
        # ── Composite ──
        {"name": "─── Composite Score ───", "value": "", "context": ""},
        {"name": "Strength Score", "value": _fmt_z(ui.usd_strength_score), "context": _strength_ctx(ui.usd_strength_score)},
    ]

    # ── Trend ─────────────────────────────────────────────────────────────
    from lox.regimes.trend import get_domain_trend
    trend = get_domain_trend("usd", regime.score, regime.label)

    print(render_regime_panel(
        domain="US Dollar",
        asof=state.asof,
        regime_label=regime.label,
        score=regime.score,
        percentile=None,
        description=regime.description,
        metrics=metrics,
        trend=trend,
    ))

    # ── Tail Risks / Headwinds / Tailwinds ────────────────────────────────
    winds = _build_winds(ui)
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

    # ── Cross-regime signals ──────────────────────────────────────────────
    _show_cross_regime_signals(console, ui)

    if llm:
        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis
        print_llm_regime_analysis(
            settings=settings,
            domain="usd",
            snapshot=snapshot_data,
            regime_label=regime.label,
            regime_description=regime.description,
            ticker=ticker,
        )


def _show_cross_regime_signals(console, ui) -> None:
    """Show cross-regime confirmation/divergence signals."""
    lines: list[str] = []

    try:
        from lox.data.regime_history import get_score_series
        for domain, display in [("rates", "Rates"), ("growth", "Growth"), ("commodities", "Commodities"), ("credit", "Credit")]:
            series = get_score_series(domain)
            if not series:
                continue
            latest = series[-1]
            sc = latest.get("score")
            lb = latest.get("label", "")
            if not isinstance(sc, (int, float)):
                continue
            short_lb = lb.split("(")[0].strip() if "(" in lb else lb

            score = ui.usd_strength_score
            is_strong = isinstance(score, (int, float)) and score > 1.0
            is_weak = isinstance(score, (int, float)) and score < -1.0

            if domain == "rates" and is_strong and sc > 55:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) + strong USD → [yellow]double tightening — EM/risk assets pressured[/yellow]")
            elif domain == "rates" and is_weak and sc < 40:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) + weak USD → [green]double easing — supportive for risk[/green]")
            elif domain == "growth" and is_strong and sc > 60:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]growth deterioration + strong USD = risk-off cocktail[/yellow]")
            elif domain == "commodities" and is_strong and sc < 40:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [dim]commodities soft, consistent with strong USD[/dim]")
            elif domain == "commodities" and is_weak:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [green]weak USD → commodity reflation likely[/green]")
            elif domain == "credit" and is_strong and sc > 50:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]strong USD + credit stress = EM debt risk[/yellow]")
            else:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [dim]neutral[/dim]")
    except Exception:
        pass

    if lines:
        console.print()
        console.print("[dim]─── Cross-Regime Signals ──────────────────────────────────────[/dim]")
        for ln in lines:
            console.print(ln)

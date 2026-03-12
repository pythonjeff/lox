"""CLI command for the Positioning & Flow regime."""
from __future__ import annotations

from rich import print
from rich.panel import Panel
from rich.table import Table

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings


# ── Formatting helpers ────────────────────────────────────────────────────

def _fmt(x: object, fmt: str = "{:.1f}") -> str:
    return fmt.format(float(x)) if isinstance(x, (int, float)) else "n/a"

def _fmt_sign(x: object, fmt: str = "{:+.1f}") -> str:
    return fmt.format(float(x)) if isinstance(x, (int, float)) else "n/a"

def _fmt_pct(x: object) -> str:
    return f"{float(x):.1f}%" if isinstance(x, (int, float)) else "n/a"

def _fmt_z(x: object) -> str:
    return f"{float(x):+.2f}" if isinstance(x, (int, float)) else "n/a"


# ── Context helpers ───────────────────────────────────────────────────────

def _vix_term_ctx(slope) -> str:
    if not isinstance(slope, (int, float)):
        return "VIX3M / VIX ratio"
    if slope < 0.85:
        return "deep backwardation — near-term panic, hedging demand extreme"
    if slope < 0.95:
        return "mild backwardation — caution, near-term stress elevated"
    if slope < 1.0:
        return "slight backwardation — market on edge"
    if slope < 1.10:
        return "normal contango — no unusual stress"
    if slope < 1.15:
        return "moderate contango — complacent, low near-term worry"
    return "steep contango — extreme complacency, potential contrarian signal"


def _pcr_ctx(pcr) -> str:
    if not isinstance(pcr, (int, float)):
        return "OI-weighted equity put/call ratio"
    if pcr > 1.2:
        return "heavy put buying — institutional hedging or fear"
    if pcr > 1.0:
        return "elevated protection — above-average put demand"
    if pcr > 0.8:
        return "neutral — balanced options flow"
    if pcr > 0.6:
        return "call-skewed — bullish positioning"
    return "extreme call bias — speculative froth"


def _aaii_ctx(bull) -> str:
    if not isinstance(bull, (int, float)):
        return "AAII Investor Sentiment Survey"
    if bull > 55:
        return "extreme bullishness — contrarian caution, crowded long"
    if bull > 45:
        return "above-average optimism"
    if bull > 35:
        return "neutral range"
    if bull > 25:
        return "below-average — some pessimism"
    if bull > 15:
        return "bearish — contrarian bullish signal historically"
    return "extreme bearishness — contrarian buy signal"


def _gex_ctx(gex_bn) -> str:
    if not isinstance(gex_bn, (int, float)):
        return "aggregate dealer gamma exposure"
    if gex_bn > 5.0:
        return "very positive — dealers long gamma, market pinning/stabilizing"
    if gex_bn > 2.0:
        return "positive — supportive, vol-suppressing"
    if gex_bn > 0:
        return "mildly positive — mild stabilization"
    if gex_bn > -2.0:
        return "negative — dealers short gamma, vol-amplifying"
    if gex_bn > -5.0:
        return "significantly negative — mechanical selling risk, wider moves"
    return "deeply negative — extreme vol amplification, tail risk elevated"


def _skew_ctx(skew) -> str:
    if not isinstance(skew, (int, float)):
        return "25-delta risk reversal (put IV - call IV)"
    if skew > 8:
        return "very steep — extreme downside protection demand"
    if skew > 5:
        return "elevated — meaningful put premium, hedging demand"
    if skew > 2:
        return "normal — standard demand for downside protection"
    if skew > 0:
        return "mild — balanced skew"
    return "inverted — calls richer than puts, unusual complacency"


def _cot_z_ctx(z) -> str:
    if not isinstance(z, (int, float)):
        return "1Y z-score of net speculative positioning"
    if z > 2.0:
        return "extreme crowded long — contrarian risk"
    if z > 1.0:
        return "crowded long — above-average bullish positioning"
    if z > -1.0:
        return "neutral range"
    if z > -2.0:
        return "crowded short — above-average bearish positioning"
    return "extreme crowded short — contrarian bullish"


# ── Tail risk / headwind / tailwind assessment ────────────────────────────

def _build_winds(inputs, regime) -> list[dict]:
    """Build tail-risk, headwind, and tailwind bullets."""
    winds: list[dict] = []

    slope = inputs.vix_term_slope
    pcr = inputs.put_call_ratio
    gex = inputs.gex_total
    aaii = inputs.aaii_bull_pct
    cot_z = inputs.cot_z_score

    # Tail risks
    if isinstance(slope, (int, float)) and slope < 0.85:
        winds.append({"type": "TAIL RISK", "text": f"VIX term deep backwardation ({slope:.2f}x) — near-term panic, hedging demand extreme", "color": "red"})

    if isinstance(gex, (int, float)) and gex < -5.0:
        winds.append({"type": "TAIL RISK", "text": f"GEX deeply negative ({gex:+.1f}bn) — dealers short gamma, mechanical selling risk, flash crash potential", "color": "red"})

    if isinstance(pcr, (int, float)) and pcr > 1.5:
        winds.append({"type": "TAIL RISK", "text": f"P/C ratio {pcr:.2f} — capitulation-level put buying", "color": "red"})

    if cot_z:
        es_z = cot_z.get("ES")
        if isinstance(es_z, (int, float)) and es_z < -2.0:
            winds.append({"type": "TAIL RISK", "text": f"S&P futures net spec z={es_z:+.1f} — extreme short crowding, squeeze risk", "color": "red"})

    # Headwinds
    if isinstance(gex, (int, float)) and -5.0 <= gex < 0:
        winds.append({"type": "HEADWIND", "text": f"GEX negative ({gex:+.1f}bn) — vol-amplifying, expect wider daily ranges", "color": "yellow"})

    if isinstance(slope, (int, float)) and 0.85 <= slope < 0.95:
        winds.append({"type": "HEADWIND", "text": f"VIX term in backwardation ({slope:.2f}x) — near-term stress, hedging elevated", "color": "yellow"})

    if isinstance(pcr, (int, float)) and 1.0 <= pcr <= 1.5:
        winds.append({"type": "HEADWIND", "text": f"P/C ratio elevated ({pcr:.2f}) — meaningful put protection demand", "color": "yellow"})

    # Tailwinds
    if isinstance(gex, (int, float)) and gex > 5.0:
        winds.append({"type": "TAILWIND", "text": f"GEX very positive ({gex:+.1f}bn) — dealers long gamma, market pinning, low vol", "color": "green"})

    if isinstance(slope, (int, float)) and slope > 1.10:
        winds.append({"type": "TAILWIND", "text": f"VIX term steep contango ({slope:.2f}x) — no near-term fear, risk-on backdrop", "color": "green"})

    if isinstance(aaii, (int, float)) and aaii < 20:
        winds.append({"type": "TAILWIND", "text": f"AAII bull {aaii:.0f}% — extreme bearishness is historically contrarian bullish", "color": "green"})

    if cot_z:
        es_z = cot_z.get("ES")
        if isinstance(es_z, (int, float)) and es_z > 2.0:
            winds.append({"type": "HEADWIND", "text": f"S&P futures net spec z={es_z:+.1f} — extreme long crowding, contrarian risk", "color": "yellow"})

    if not winds:
        winds.append({"type": "NEUTRAL", "text": "Positioning in normal range — no strong directional signal", "color": "dim"})

    return winds


# ── Core implementation ───────────────────────────────────────────────────

def positioning_snapshot(
    *,
    llm: bool = False,
    ticker: str = "SPY",
    refresh: bool = False,
    features: bool = False,
    json_out: bool = False,
    delta: str = "",
    alert: bool = False,
    calendar: bool = False,
    trades: bool = False,
) -> None:
    """Entry point for `lox regime positioning`."""
    from rich.console import Console
    from lox.cli_commands.shared.labs_utils import (
        handle_output_flags, parse_delta_period, show_delta_summary,
        get_delta_metrics, save_snapshot,
        show_alert_output, show_calendar_output, show_trades_output,
    )

    settings = load_settings()
    console = Console()

    from lox.positioning.data import compute_positioning_inputs
    inputs = compute_positioning_inputs(settings=settings, ticker=ticker or "SPY", refresh=refresh)

    if inputs.error:
        console.print(f"[red]Positioning data error:[/red] {inputs.error}")
        return

    # Fetch cross-regime scores for Layer 3
    vol_score = None
    credit_score = None
    try:
        from lox.data.regime_history import get_score_series
        for domain, target in [("volatility", "vol"), ("credit", "credit")]:
            series = get_score_series(domain)
            if series:
                sc = series[-1].get("score")
                if domain == "volatility":
                    vol_score = sc
                elif domain == "credit":
                    credit_score = sc
    except Exception:
        pass

    from lox.positioning.regime import classify_positioning
    regime = classify_positioning(
        vix_term_slope=inputs.vix_term_slope,
        put_call_ratio=inputs.put_call_ratio,
        aaii_bull_pct=inputs.aaii_bull_pct,
        cot_net_spec=inputs.cot_net_spec,
        cot_z_score=inputs.cot_z_score,
        gex_total=inputs.gex_total,
        gex_flip_level=inputs.gex_flip_level,
        skew_25d=inputs.skew_25d,
        short_interest_pct=inputs.short_interest_pct,
        vol_score=vol_score,
        credit_score=credit_score,
    )

    # ── Build snapshot dict ───────────────────────────────────────────────
    snapshot_data = {
        "vix_term_slope": inputs.vix_term_slope,
        "put_call_ratio": inputs.put_call_ratio,
        "aaii_bull_pct": inputs.aaii_bull_pct,
        "gex_total_bn": inputs.gex_total,
        "gex_flip_level": inputs.gex_flip_level,
        "skew_25d": inputs.skew_25d,
        "cot_z_es": (inputs.cot_z_score or {}).get("ES"),
        "regime": regime.label,
    }

    feature_dict = {
        "vix_term_slope": inputs.vix_term_slope,
        "put_call_ratio": inputs.put_call_ratio,
        "aaii_bull_pct": inputs.aaii_bull_pct,
        "gex_total_bn": inputs.gex_total,
        "skew_25d": inputs.skew_25d,
        "cot_z_es": (inputs.cot_z_score or {}).get("ES"),
    }

    save_snapshot("positioning", snapshot_data, regime.label)

    if handle_output_flags(
        domain="positioning",
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
        show_alert_output("positioning", regime.label, snapshot_data, regime.description)
        return

    if calendar:
        print(Panel.fit(f"[b]Regime:[/b] {regime.label}", title="Positioning & Flow", border_style="cyan"))
        show_calendar_output("positioning")
        return

    if trades:
        print(Panel.fit(f"[b]Regime:[/b] {regime.label}", title="Positioning & Flow", border_style="cyan"))
        show_trades_output("positioning", regime.label)
        return

    if delta:
        delta_days = parse_delta_period(delta)
        metric_keys = [
            "VIX Term Slope:vix_term_slope:",
            "P/C Ratio:put_call_ratio:",
            "GEX ($bn):gex_total_bn:",
            "Skew 25d:skew_25d:",
            "COT z (ES):cot_z_es:",
            "AAII Bull:aaii_bull_pct:%",
        ]
        metrics_for_delta, prev_regime = get_delta_metrics("positioning", snapshot_data, metric_keys, delta_days)
        show_delta_summary("positioning", regime.label, prev_regime, metrics_for_delta, delta_days)
        if prev_regime is None:
            console.print(f"\n[dim]No cached data from {delta_days}d ago. Run `lox regime positioning` daily to build history.[/dim]")
        return

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 1: Regime Header
    # ═══════════════════════════════════════════════════════════════════════
    metrics = [
        {"name": "─── VIX Term Structure ───", "value": "", "context": ""},
        {"name": "VIX Level", "value": _fmt(inputs.vix_level), "context": "CBOE Volatility Index"},
        {"name": "VIX3M Level", "value": _fmt(inputs.vix3m_level), "context": "3-month VIX"},
        {"name": "Term Slope (3M/VIX)", "value": _fmt(inputs.vix_term_slope, "{:.3f}"), "context": _vix_term_ctx(inputs.vix_term_slope)},
        {"name": "─── Options Flow ───", "value": "", "context": ""},
        {"name": "Put/Call Ratio (OI)", "value": _fmt(inputs.put_call_ratio, "{:.2f}"), "context": _pcr_ctx(inputs.put_call_ratio)},
        {"name": "25d Skew (vol pts)", "value": _fmt_sign(inputs.skew_25d, "{:+.1f}"), "context": _skew_ctx(inputs.skew_25d)},
        {"name": "─── Sentiment ───", "value": "", "context": ""},
        {"name": "AAII Bull %", "value": _fmt_pct(inputs.aaii_bull_pct), "context": _aaii_ctx(inputs.aaii_bull_pct)},
    ]

    from lox.regimes.trend import get_domain_trend
    trend = get_domain_trend("positioning", regime.score, regime.label)

    print(render_regime_panel(
        domain="Positioning & Flow",
        asof=inputs.asof,
        regime_label=regime.label,
        score=regime.score,
        percentile=None,
        description=regime.description,
        metrics=metrics,
        trend=trend,
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 2: Dealer Gamma Exposure (GEX)
    # ═══════════════════════════════════════════════════════════════════════
    if inputs.gex_total is not None:
        console.print()
        gex_lines: list[str] = []
        gex_lines.append(f"  [b]GEX Total:[/b]  {inputs.gex_total:+.2f} $bn")
        if inputs.gex_spot is not None:
            gex_lines.append(f"  [b]Spot:[/b]       {inputs.gex_spot:.2f}")
        if inputs.gex_flip_level is not None:
            gex_lines.append(f"  [b]Flip Level:[/b] {inputs.gex_flip_level:.0f}")
            if inputs.gex_spot is not None:
                dist = ((inputs.gex_flip_level - inputs.gex_spot) / inputs.gex_spot) * 100
                gex_lines.append(f"  [b]Distance:[/b]   {dist:+.1f}% from spot")

        gex_lines.append("")
        if inputs.gex_total > 2.0:
            gex_lines.append("  [green]Dealers LONG gamma — vol suppressed, market pinning around strikes[/green]")
        elif inputs.gex_total > 0:
            gex_lines.append("  [dim]Dealers mildly long gamma — moderate stabilization[/dim]")
        elif inputs.gex_total > -2.0:
            gex_lines.append("  [yellow]Dealers SHORT gamma — vol amplified, expect wider daily ranges[/yellow]")
        else:
            gex_lines.append("  [red]Dealers deeply short gamma — mechanical selling, tail risk elevated[/red]")

        panel = Panel.fit(
            "\n".join(gex_lines),
            title="[bold]Dealer Gamma Exposure (GEX)[/bold]",
            border_style="cyan",
        )
        console.print(panel)

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 3: COT Speculative Positioning
    # ═══════════════════════════════════════════════════════════════════════
    if inputs.cot_net_spec:
        console.print()
        cot_table = Table(box=None, padding=(0, 2))
        cot_table.add_column("Asset", style="bold")
        cot_table.add_column("Net Spec", justify="right")
        cot_table.add_column("Z-Score", justify="right")
        cot_table.add_column("Report Date", style="dim")
        cot_table.add_column("Context")

        from lox.positioning.data import COT_SYMBOLS
        for code in COT_SYMBOLS:
            net = inputs.cot_net_spec.get(code)
            if net is None:
                continue
            z = (inputs.cot_z_score or {}).get(code)
            dt = (inputs.cot_dates or {}).get(code, "")
            ctx = _cot_z_ctx(z)

            # Color based on z-score
            z_str = _fmt_z(z)
            if isinstance(z, (int, float)):
                if abs(z) > 1.5:
                    z_str = f"[red]{z_str}[/red]" if z < 0 else f"[yellow]{z_str}[/yellow]"

            cot_table.add_row(
                code,
                f"{net:+,.0f}",
                z_str,
                dt,
                f"[dim]{ctx}[/dim]",
            )

        panel = Panel.fit(
            cot_table,
            title="[bold]CFTC COT — Net Speculative Positioning[/bold]",
            border_style="cyan",
        )
        console.print(panel)

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 4: Short Interest
    # ═══════════════════════════════════════════════════════════════════════
    if inputs.short_interest_pct:
        console.print()
        si_lines: list[str] = []
        for tk, si in sorted(inputs.short_interest_pct.items()):
            ctx = ""
            if si > 15:
                ctx = "[red]heavy short interest — squeeze potential[/red]"
            elif si > 10:
                ctx = "[yellow]elevated — bears positioned[/yellow]"
            elif si > 5:
                ctx = "[dim]moderate[/dim]"
            else:
                ctx = "[dim]low — no crowded shorts[/dim]"
            si_lines.append(f"  {tk:6s}  SI: {si:.1f}%  {ctx}")

        panel = Panel.fit(
            "\n".join(si_lines),
            title="[bold]Short Interest[/bold]",
            border_style="cyan",
        )
        console.print(panel)

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 5: Tail Risks / Headwinds / Tailwinds
    # ═══════════════════════════════════════════════════════════════════════
    winds = _build_winds(inputs, regime)
    if winds:
        console.print()
        console.print("[dim]─── Positioning Signals ────────────────────────────────────────[/dim]")
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
    _show_cross_regime_signals(console, inputs, regime)

    if llm:
        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis
        print_llm_regime_analysis(
            settings=settings,
            domain="positioning",
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
        pos_score = regime.score

        for domain, display in [("volatility", "Volatility"), ("credit", "Credit"), ("growth", "Growth"), ("rates", "Rates")]:
            series = get_score_series(domain)
            if not series:
                continue
            latest = series[-1]
            sc = latest.get("score")
            lb = latest.get("label", "")
            if not isinstance(sc, (int, float)):
                continue
            short_lb = lb.split("(")[0].strip() if "(" in lb else lb

            is_pos_high = pos_score >= 55
            is_pos_low = pos_score < 35

            if domain == "volatility":
                if is_pos_high and sc >= 55:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [red]confirmed — panic positioning + vol elevated = risk-off[/red]")
                elif is_pos_high and sc < 30:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]divergence — positioning fearful but vol calm, possible overreaction[/yellow]")
                elif is_pos_low and sc >= 55:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]divergence — complacent positioning but vol elevated, surprise risk[/yellow]")
                else:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [dim]aligned[/dim]")
            elif domain == "credit":
                if is_pos_high and sc >= 50:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [red]confirmed — positioning + credit stress = full risk-off[/red]")
                elif is_pos_low and sc >= 55:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]divergence — complacent positioning, credit stress building[/yellow]")
                else:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [dim]neutral[/dim]")
            elif domain == "growth":
                if is_pos_high and sc >= 55:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [red]growth slowing + defensive positioning = recession trade[/red]")
                elif is_pos_low and sc < 35:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [green]growth healthy + complacent positioning = risk-on[/green]")
                else:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [dim]neutral[/dim]")
            elif domain == "rates":
                if is_pos_high and sc >= 55:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]rates stress + defensive positioning = duration risk[/yellow]")
                else:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [dim]neutral[/dim]")
    except Exception:
        pass

    if lines:
        console.print()
        console.print("[dim]─── Cross-Regime Signals ──────────────────────────────────────[/dim]")
        for ln in lines:
            console.print(ln)

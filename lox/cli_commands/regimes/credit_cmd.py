"""CLI command for the Credit regime."""
from __future__ import annotations

from rich import print
from rich.panel import Panel

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings


# ─────────────────────────────────────────────────────────────────────────────
# Sparkline / trend helpers (reused from fiscal pattern)
# ─────────────────────────────────────────────────────────────────────────────

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo if hi != lo else 1.0
    return "".join(
        _SPARK_CHARS[min(len(_SPARK_CHARS) - 1, int((v - lo) / span * (len(_SPARK_CHARS) - 1)))]
        for v in values
    )


def _trend_label(values: list[float], higher_is_worse: bool = True) -> str:
    if len(values) < 4:
        return ""
    mid = len(values) // 2
    old_avg = sum(values[:mid]) / mid
    new_avg = sum(values[mid:]) / (len(values) - mid)
    delta = new_avg - old_avg
    threshold = (max(values) - min(values)) * 0.1 if max(values) != min(values) else 0.01
    if abs(delta) < threshold:
        return "[dim]stable[/dim]"
    if higher_is_worse:
        return "[red]widening[/red]" if delta > 0 else "[green]tightening[/green]"
    return "[green]improving[/green]" if delta > 0 else "[red]weakening[/red]"


def _extract_last_n(df, n: int = 30) -> list[float]:
    """Extract last N daily values (in bps) from a FRED DataFrame."""
    if df is None or df.empty:
        return []
    s = df.sort_values("date")["value"].dropna().tail(n)
    return [float(v) * 100 for v in s]


# ─────────────────────────────────────────────────────────────────────────────
# Context helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hy_context(hy_oas_val):
    if hy_oas_val is None:
        return "ICE BofA HY"
    if hy_oas_val > 700:
        return "crisis-level spreads"
    if hy_oas_val > 500:
        return "stressed — risk-off"
    if hy_oas_val > 400:
        return "widening — caution warranted"
    if hy_oas_val > 300:
        return "slightly elevated, watch momentum"
    if hy_oas_val > 250:
        return "normal range"
    return "very tight — risk appetite high"


def _velocity_5d_context(hy_5d_chg):
    if hy_5d_chg is None:
        return "5-day velocity"
    if hy_5d_chg > 50:
        return "rapid widening — crisis speed"
    if hy_5d_chg > 25:
        return "widening sharply"
    if hy_5d_chg > 10:
        return "moderate widening"
    if hy_5d_chg > 0:
        return "drifting wider"
    if hy_5d_chg > -15:
        return "stable to tightening"
    return "rapid tightening — risk-on"


def _velocity_30d_context(hy_30d_chg):
    if hy_30d_chg is None:
        return "30-day trend"
    if hy_30d_chg > 100:
        return "crisis acceleration"
    if hy_30d_chg > 50:
        return "sharp widening trend"
    if hy_30d_chg > 20:
        return "widening — stress building"
    if hy_30d_chg > 0:
        return "mild widening"
    if hy_30d_chg > -30:
        return "stable to tightening"
    return "compressing — risk appetite returning"


def _percentile_context(hy_1y_pctl, hy_90d_pctl):
    pctl = hy_1y_pctl if hy_1y_pctl is not None else hy_90d_pctl
    if pctl is None:
        return "vs history"
    if pctl > 90:
        return "extreme — near 1Y wides"
    if pctl > 75:
        return "elevated vs 1Y range"
    if pctl > 50:
        return "above median, leaning wide"
    if pctl > 25:
        return "below median, leaning tight"
    return "near 1Y tights — complacent?"


def _bbb_context(bbb_oas_val):
    if bbb_oas_val is None:
        return "IG benchmark"
    if bbb_oas_val > 300:
        return "stressed — IG contagion"
    if bbb_oas_val > 200:
        return "elevated — watch for downgrades"
    if bbb_oas_val > 150:
        return "slightly wide"
    if bbb_oas_val > 100:
        return "normal range"
    return "very tight — strong IG demand"


def _aaa_context(aaa_oas_val):
    if aaa_oas_val is None:
        return "flight-to-quality gauge"
    if aaa_oas_val > 100:
        return "wide — even safe havens repricing"
    if aaa_oas_val > 60:
        return "slightly elevated"
    if aaa_oas_val > 30:
        return "normal — no flight-to-quality"
    return "compressed — heavy quality bid"


def _vix_context(vix_val):
    if vix_val is None:
        return "cross-market vol"
    if vix_val > 30:
        return "panic — confirms systemic stress"
    if vix_val > 25:
        return "fear — credit widening confirmed"
    if vix_val > 20:
        return "elevated, watch for contagion"
    if vix_val > 15:
        return "calm — spread moves may be idiosyncratic"
    return "complacent — low vol, risk-on"


def _ccc_context(ccc_oas_val):
    if ccc_oas_val is None:
        return "weakest public credits"
    if ccc_oas_val > 1500:
        return "distress — default risk priced in"
    if ccc_oas_val > 1000:
        return "stressed — weakest names under pressure"
    if ccc_oas_val > 800:
        return "elevated — watching for deterioration"
    if ccc_oas_val > 600:
        return "normal range for lowest tier"
    return "tight — risk appetite reaching for yield"


def _bb_context(bb_oas_val):
    if bb_oas_val is None:
        return "best HY tier"
    if bb_oas_val > 400:
        return "stressed — even best HY repricing"
    if bb_oas_val > 300:
        return "elevated — risk-off creeping into HY"
    if bb_oas_val > 200:
        return "slightly wide"
    if bb_oas_val > 150:
        return "normal range"
    return "tight — strong fallen-angel demand"


def _b_context(single_b_oas_val):
    if single_b_oas_val is None:
        return "middle HY tier"
    if single_b_oas_val > 600:
        return "stressed — middle tier under pressure"
    if single_b_oas_val > 450:
        return "elevated — credit differentiation rising"
    if single_b_oas_val > 300:
        return "normal range"
    if single_b_oas_val > 200:
        return "tight — reach for yield in play"
    return "very tight — risk heavily underpriced"


def _ccc_bb_context(ccc_bb):
    if ccc_bb is None:
        return "quality tier dispersion"
    if ccc_bb > 1200:
        return "distress — weakest credits blowing out"
    if ccc_bb > 900:
        return "stress emerging in lowest tiers"
    if ccc_bb > 600:
        return "normal discrimination"
    if ccc_bb > 400:
        return "compressed — risk being underpriced?"
    return "very tight — worst credits priced like BB"


def _delinq_context(cc_delinq_val):
    if cc_delinq_val is None:
        return "consumer stress signal"
    if cc_delinq_val > 5.0:
        return "distress — consumer balance sheets cracking"
    if cc_delinq_val > 3.5:
        return "elevated — consumer stress building"
    if cc_delinq_val > 2.5:
        return "slightly elevated, monitoring"
    return "healthy — consumer credit solid"


def _sloos_context(sloos_val):
    if sloos_val is None:
        return "bank lending standards"
    if sloos_val > 40:
        return "crisis tightening — private credit only option"
    if sloos_val > 20:
        return "tight — borrowers pushed to shadow lenders"
    if sloos_val > 0:
        return "neutral — banks slightly cautious"
    if sloos_val > -10:
        return "neutral — lending standards stable"
    return "easing — banks competing for loans"


def _shadow_context(sigs):
    if sigs >= 3:
        return "stress — multiple shadow signals active"
    if sigs >= 2:
        return "warning — hidden stress building"
    if sigs >= 1:
        return "one signal active, watch for confirmation"
    return "clear — no hidden stress detected"


# ─────────────────────────────────────────────────────────────────────────────
# Credit curve & cross-regime display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _curve_signal(delta_bp):
    if delta_bp is None:
        return "—"
    if delta_bp > 30:
        return "[red]steepening stress[/red]"
    if delta_bp > 15:
        return "[yellow]widening — accelerating[/yellow]"
    if delta_bp > 5:
        return "[yellow]widening[/yellow]"
    if delta_bp > -5:
        return "[dim]stable[/dim]"
    if delta_bp > -15:
        return "[green]tightening[/green]"
    return "[green]compressing[/green]"


def _show_spread_sparklines(console, hy_df, bbb_df, ccc_df, bb_df, hy_oas_val, bbb_oas_val, ccc_bb):
    """Show 30-day sparklines for key spread series."""
    hy_vals = _extract_last_n(hy_df, 30)
    bbb_vals = _extract_last_n(bbb_df, 30)

    ccc_vals = _extract_last_n(ccc_df, 30)
    bb_vals = _extract_last_n(bb_df, 30)
    ccc_bb_vals = []
    if len(ccc_vals) == len(bb_vals) and len(ccc_vals) > 0:
        ccc_bb_vals = [c - b for c, b in zip(ccc_vals, bb_vals)]

    lines: list[str] = []
    if len(hy_vals) >= 10:
        lines.append(
            f"  HY OAS   {_sparkline(hy_vals)}  "
            f"{hy_oas_val:.0f}bp  {_trend_label(hy_vals)}"
            if hy_oas_val is not None
            else f"  HY OAS   {_sparkline(hy_vals)}  {_trend_label(hy_vals)}"
        )
    if len(ccc_bb_vals) >= 10:
        lines.append(
            f"  CCC-BB   {_sparkline(ccc_bb_vals)}  "
            f"{ccc_bb:.0f}bp  {_trend_label(ccc_bb_vals)}"
            if ccc_bb is not None
            else f"  CCC-BB   {_sparkline(ccc_bb_vals)}  {_trend_label(ccc_bb_vals)}"
        )
    if len(bbb_vals) >= 10:
        lines.append(
            f"  BBB      {_sparkline(bbb_vals)}  "
            f"{bbb_oas_val:.0f}bp  {_trend_label(bbb_vals)}"
            if bbb_oas_val is not None
            else f"  BBB      {_sparkline(bbb_vals)}  {_trend_label(bbb_vals)}"
        )
    if lines:
        console.print()
        console.print("[dim]─── Spread Velocity (30d) ─────────────────────────────────────[/dim]")
        for ln in lines:
            console.print(ln)


def _show_credit_curve(console, tier_data: dict[str, dict]):
    """
    Show credit quality curve with 30d shift per tier.
    tier_data: {tier_name: {"now": float|None, "30d_ago": float|None}}
    """
    from rich.table import Table as RichTable

    has_data = any(
        isinstance(v.get("now"), (int, float))
        for v in tier_data.values()
    )
    if not has_data:
        return

    ct = RichTable(
        title="Credit Curve (30d shift)",
        show_header=True,
        header_style="bold yellow",
        box=None,
        padding=(0, 1),
    )
    ct.add_column("Tier")
    ct.add_column("Now", justify="right")
    ct.add_column("30d ago", justify="right")
    ct.add_column("Δ", justify="right")
    ct.add_column("Signal", style="dim")

    for tier, vals in tier_data.items():
        now = vals.get("now")
        ago = vals.get("30d_ago")
        if not isinstance(now, (int, float)):
            continue
        now_s = f"{now:.0f}bp"
        ago_s = f"{ago:.0f}bp" if isinstance(ago, (int, float)) else "—"
        delta = now - ago if isinstance(ago, (int, float)) else None
        delta_s = f"{delta:+.0f}bp" if delta is not None else "—"
        ct.add_row(tier, now_s, ago_s, delta_s, _curve_signal(delta))

    # CCC-BB summary
    ccc_now = tier_data.get("CCC", {}).get("now")
    bb_now = tier_data.get("BB", {}).get("now")
    ccc_ago = tier_data.get("CCC", {}).get("30d_ago")
    bb_ago = tier_data.get("BB", {}).get("30d_ago")

    console.print()
    console.print(ct)

    if isinstance(ccc_now, (int, float)) and isinstance(bb_now, (int, float)):
        gap_now = ccc_now - bb_now
        gap_ago = (ccc_ago - bb_ago) if isinstance(ccc_ago, (int, float)) and isinstance(bb_ago, (int, float)) else None
        if gap_ago is not None:
            gap_d = gap_now - gap_ago
            if gap_d > 20:
                interp = "[red]Quality curve steepening → lower quality under pressure[/red]"
            elif gap_d > 5:
                interp = "[yellow]Mild steepening — watch for acceleration[/yellow]"
            elif gap_d < -20:
                interp = "[green]Quality curve compressing → risk appetite broadening[/green]"
            elif gap_d < -5:
                interp = "[green]Mild compression — risk-on reaching for yield[/green]"
            else:
                interp = "[dim]Quality curve stable[/dim]"
            console.print(f"  CCC-BB gap: {gap_now:.0f}bp → was {gap_ago:.0f}bp ({gap_d:+.0f}bp)")
            console.print(f"  {interp}")


def _show_cross_regime_signals(console, vix_val, hy_oas_val, hy_30d_chg, credit_score):
    """Show cross-regime confirmation/divergence signals."""
    lines: list[str] = []

    # VIX vs credit
    if isinstance(vix_val, (int, float)) and isinstance(hy_30d_chg, (int, float)):
        if vix_val < 20 and hy_30d_chg > 15:
            lines.append(f"  VIX {vix_val:.1f} (calm) vs HY widening → [yellow]possible divergence[/yellow]")
        elif vix_val > 25 and hy_30d_chg > 15:
            lines.append(f"  VIX {vix_val:.1f} + HY widening → [red]risk-off confirmed across assets[/red]")
        elif vix_val < 18 and isinstance(hy_oas_val, (int, float)) and hy_oas_val < 280:
            lines.append(f"  VIX {vix_val:.1f} + HY tight → [green]broad risk-on, consistent[/green]")
        elif isinstance(vix_val, (int, float)):
            lines.append(f"  VIX {vix_val:.1f} — [dim]no strong vol/credit divergence[/dim]")

    # Pull latest scores from other regimes
    try:
        from lox.data.regime_history import get_score_series
        for domain, display in [("rates", "Rates"), ("growth", "Growth"), ("volatility", "Vol")]:
            series = get_score_series(domain)
            if not series:
                continue
            latest = series[-1]
            sc = latest.get("score")
            lb = latest.get("label", "")
            if not isinstance(sc, (int, float)):
                continue
            short_lb = lb.split("(")[0].strip() if "(" in lb else lb
            if domain == "rates" and sc > 60 and credit_score > 50:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) + credit widening → [yellow]classic risk-off setup[/yellow]")
            elif domain == "growth" and sc > 65 and credit_score > 50:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]macro deterioration confirming credit stress[/yellow]")
            elif domain == "growth" and sc < 40 and credit_score > 50:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [dim]no macro deterioration confirming[/dim]")
            elif domain == "volatility" and sc > 60 and credit_score > 50:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]vol elevated, confirms credit stress[/yellow]")
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
# Core implementation
# ─────────────────────────────────────────────────────────────────────────────

def credit_snapshot(
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
    """Entry point for `lox regime credit`."""
    from rich.console import Console
    from lox.cli_commands.shared.labs_utils import (
        handle_output_flags, parse_delta_period, show_delta_summary,
        get_delta_metrics, save_snapshot,
        show_alert_output, show_calendar_output, show_trades_output,
    )

    settings = load_settings()
    console = Console()
    from lox.data.fred import FredClient

    fred = FredClient(api_key=settings.FRED_API_KEY)

    hy_oas_val = None
    bbb_oas_val = None
    aaa_oas_val = None
    hy_5d_chg = None
    hy_30d_chg = None
    bbb_30d_chg = None
    hy_90d_pctl = None
    hy_1y_pctl = None
    vix_val = None
    asof = "—"

    # Keep raw DataFrames for sparklines / curve analysis
    hy_df = None
    bbb_df = None
    ccc_df = None
    bb_df = None
    b_df = None

    # HY OAS + velocity + percentiles
    try:
        hy_df = fred.fetch_series("BAMLH0A0HYM2", start_date="2020-01-01", refresh=refresh)
        if hy_df is not None and not hy_df.empty:
            hy_df = hy_df.sort_values("date")
            hy_oas_val = float(hy_df["value"].iloc[-1]) * 100
            asof = str(hy_df["date"].iloc[-1].date())
            if len(hy_df) >= 5:
                hy_5d_chg = (float(hy_df["value"].iloc[-1]) - float(hy_df["value"].iloc[-5])) * 100
            if len(hy_df) >= 22:
                hy_30d_chg = (float(hy_df["value"].iloc[-1]) - float(hy_df["value"].iloc[-22])) * 100
            if len(hy_df) >= 63:
                recent = hy_df["value"].iloc[-63:]
                hy_90d_pctl = float((recent <= hy_df["value"].iloc[-1]).mean() * 100)
            if len(hy_df) >= 252:
                recent_1y = hy_df["value"].iloc[-252:]
                hy_1y_pctl = float((recent_1y <= hy_df["value"].iloc[-1]).mean() * 100)
    except Exception:
        pass

    # BBB OAS + velocity
    try:
        bbb_df = fred.fetch_series("BAMLC0A4CBBB", start_date="2020-01-01", refresh=refresh)
        if bbb_df is not None and not bbb_df.empty:
            bbb_df = bbb_df.sort_values("date")
            bbb_oas_val = float(bbb_df["value"].iloc[-1]) * 100
            if len(bbb_df) >= 22:
                bbb_30d_chg = (float(bbb_df["value"].iloc[-1]) - float(bbb_df["value"].iloc[-22])) * 100
    except Exception:
        pass

    # AAA OAS
    aaa_df = None
    try:
        aaa_df = fred.fetch_series("BAMLC0A1CAAA", start_date="2020-01-01", refresh=refresh)
        if aaa_df is not None and not aaa_df.empty:
            aaa_oas_val = float(aaa_df.sort_values("date")["value"].iloc[-1]) * 100
    except Exception:
        pass

    # VIX for cross-market confirmation
    try:
        from lox.macro.signals import build_macro_state
        macro = build_macro_state(settings=settings, start_date="2020-01-01", refresh=refresh)
        vix_val = macro.inputs.vix
    except Exception:
        pass

    # ── Shadow credit data ────────────────────────────────────────────────
    ccc_oas_val = None
    bb_oas_val = None
    single_b_oas_val = None
    cc_delinq_val = None
    sloos_val = None

    try:
        ccc_df = fred.fetch_series("BAMLH0A3HYC", start_date="2020-01-01", refresh=refresh)
        if ccc_df is not None and not ccc_df.empty:
            ccc_df = ccc_df.sort_values("date")
            ccc_oas_val = float(ccc_df["value"].iloc[-1]) * 100
    except Exception:
        pass

    try:
        bb_df = fred.fetch_series("BAMLH0A1HYBB", start_date="2020-01-01", refresh=refresh)
        if bb_df is not None and not bb_df.empty:
            bb_df = bb_df.sort_values("date")
            bb_oas_val = float(bb_df["value"].iloc[-1]) * 100
    except Exception:
        pass

    try:
        b_df = fred.fetch_series("BAMLH0A2HYB", start_date="2020-01-01", refresh=refresh)
        if b_df is not None and not b_df.empty:
            b_df = b_df.sort_values("date")
            single_b_oas_val = float(b_df["value"].iloc[-1]) * 100
    except Exception:
        pass

    try:
        delinq_df = fred.fetch_series("DRCCLACBS", start_date="2020-01-01", refresh=refresh)
        if delinq_df is not None and not delinq_df.empty:
            cc_delinq_val = float(delinq_df.sort_values("date")["value"].iloc[-1])
    except Exception:
        pass

    try:
        sloos_df = fred.fetch_series("DRTSCLCC", start_date="2020-01-01", refresh=refresh)
        if sloos_df is not None and not sloos_df.empty:
            sloos_val = float(sloos_df.sort_values("date")["value"].iloc[-1])
    except Exception:
        pass

    from lox.credit.regime import classify_credit
    result = classify_credit(
        hy_oas=hy_oas_val,
        bbb_oas=bbb_oas_val,
        aaa_oas=aaa_oas_val,
        hy_oas_30d_chg=hy_30d_chg,
        hy_oas_90d_percentile=hy_90d_pctl,
        hy_oas_1y_percentile=hy_1y_pctl,
        hy_oas_5d_chg=hy_5d_chg,
        bbb_oas_30d_chg=bbb_30d_chg,
        vix=vix_val,
        ccc_oas=ccc_oas_val,
        bb_oas=bb_oas_val,
        single_b_oas=single_b_oas_val,
        cc_delinquency_rate=cc_delinq_val,
        sloos_tightening=sloos_val,
    )

    ccc_bb = (ccc_oas_val - bb_oas_val) if ccc_oas_val is not None and bb_oas_val is not None else None

    # ── Build snapshot and feature dicts ──────────────────────────────────
    snapshot_data = {
        "hy_oas": hy_oas_val,
        "bbb_oas": bbb_oas_val,
        "aaa_oas": aaa_oas_val,
        "hy_5d_chg": hy_5d_chg,
        "hy_30d_chg": hy_30d_chg,
        "bbb_30d_chg": bbb_30d_chg,
        "hy_1y_pctl": hy_1y_pctl,
        "hy_90d_pctl": hy_90d_pctl,
        "vix": vix_val,
        "ccc_oas": ccc_oas_val,
        "bb_oas": bb_oas_val,
        "single_b_oas": single_b_oas_val,
        "ccc_bb_spread": ccc_bb,
        "cc_delinquency": cc_delinq_val,
        "sloos_tightening": sloos_val,
        "regime": result.label,
    }

    feature_dict = {
        "hy_oas": hy_oas_val,
        "bbb_oas": bbb_oas_val,
        "aaa_oas": aaa_oas_val,
        "hy_5d_chg": hy_5d_chg,
        "hy_30d_chg": hy_30d_chg,
        "bbb_30d_chg": bbb_30d_chg,
        "hy_1y_pctl": hy_1y_pctl,
        "vix": vix_val,
        "ccc_oas": ccc_oas_val,
        "bb_oas": bb_oas_val,
        "single_b_oas": single_b_oas_val,
        "ccc_bb_spread": ccc_bb,
        "cc_delinquency": cc_delinq_val,
        "sloos_tightening": sloos_val,
    }

    # Save snapshot for delta tracking
    save_snapshot("credit", snapshot_data, result.label)

    # Handle --features and --json flags
    if handle_output_flags(
        domain="credit",
        snapshot=snapshot_data,
        features=feature_dict,
        regime=result.label,
        regime_description=result.description,
        asof=asof,
        output_json=json_out,
        output_features=features,
    ):
        return

    # Handle --alert flag
    if alert:
        show_alert_output("credit", result.label, snapshot_data, result.description)
        return

    # Handle --calendar flag
    if calendar:
        print(Panel.fit(f"[b]Regime:[/b] {result.label}", title="Credit / Spreads", border_style="cyan"))
        show_calendar_output("credit")
        return

    # Handle --trades flag
    if trades:
        print(Panel.fit(f"[b]Regime:[/b] {result.label}", title="Credit / Spreads", border_style="cyan"))
        show_trades_output("credit", result.label)
        return

    # Handle --delta flag
    if delta:
        delta_days = parse_delta_period(delta)
        metric_keys = [
            "HY OAS:hy_oas:bp",
            "BBB OAS:bbb_oas:bp",
            "AAA OAS:aaa_oas:bp",
            "HY 5d Chg:hy_5d_chg:bp",
            "HY 30d Chg:hy_30d_chg:bp",
            "CCC OAS:ccc_oas:bp",
            "BB OAS:bb_oas:bp",
            "CCC-BB Spread:ccc_bb_spread:bp",
            "CC Delinquency:cc_delinquency:%",
            "SLOOS:sloos_tightening:%",
        ]
        metrics_for_delta, prev_regime = get_delta_metrics("credit", snapshot_data, metric_keys, delta_days)
        show_delta_summary("credit", result.label, prev_regime, metrics_for_delta, delta_days)
        if prev_regime is None:
            console.print(f"\n[dim]No cached data from {delta_days}d ago. Run `lox regime credit` daily to build history.[/dim]")
        return

    # ── Build metrics and render panel ────────────────────────────────────
    def _v(x, fmt="{:.0f}bp"):
        return fmt.format(x) if x is not None else "n/a"

    shadow_sigs = int(result.metrics.get("Shadow Sigs", 0))

    metrics = [
        {"name": "HY OAS", "value": _v(hy_oas_val), "context": _hy_context(hy_oas_val)},
        {"name": "5d Change", "value": _v(hy_5d_chg, "{:+.0f}bp"), "context": _velocity_5d_context(hy_5d_chg)},
        {"name": "30d Change", "value": _v(hy_30d_chg, "{:+.0f}bp"), "context": _velocity_30d_context(hy_30d_chg)},
        {"name": "1Y Percentile", "value": f"{hy_1y_pctl:.0f}th" if hy_1y_pctl is not None else ("n/a" if hy_90d_pctl is None else f"{hy_90d_pctl:.0f}th (90d)"), "context": _percentile_context(hy_1y_pctl, hy_90d_pctl)},
        {"name": "BBB OAS", "value": _v(bbb_oas_val), "context": _bbb_context(bbb_oas_val)},
        {"name": "AAA OAS", "value": _v(aaa_oas_val), "context": _aaa_context(aaa_oas_val)},
        {"name": "VIX", "value": f"{vix_val:.1f}" if vix_val is not None else "n/a", "context": _vix_context(vix_val)},
        {"name": "─── Shadow Credit ───", "value": "", "context": ""},
        {"name": "CCC OAS", "value": _v(ccc_oas_val), "context": _ccc_context(ccc_oas_val)},
        {"name": "BB OAS", "value": _v(bb_oas_val), "context": _bb_context(bb_oas_val)},
        {"name": "B OAS", "value": _v(single_b_oas_val), "context": _b_context(single_b_oas_val)},
        {"name": "CCC-BB Spread", "value": _v(ccc_bb), "context": _ccc_bb_context(ccc_bb)},
        {"name": "CC Delinquency", "value": f"{cc_delinq_val:.1f}%" if cc_delinq_val is not None else "n/a", "context": _delinq_context(cc_delinq_val)},
        {"name": "SLOOS Tightening", "value": f"{sloos_val:+.0f}%" if sloos_val is not None else "n/a", "context": _sloos_context(sloos_val)},
        {"name": "Shadow Signals", "value": str(shadow_sigs), "context": _shadow_context(shadow_sigs)},
    ]

    from lox.regimes.trend import get_domain_trend
    trend = get_domain_trend("credit", result.score, result.label)

    print(render_regime_panel(
        domain="Credit",
        asof=asof,
        regime_label=result.label,
        score=result.score,
        percentile=None,
        description=result.description,
        metrics=metrics,
        trend=trend,
    ))

    # ── Block 1: Spread velocity sparklines ───────────────────────────────
    _show_spread_sparklines(console, hy_df, bbb_df, ccc_df, bb_df, hy_oas_val, bbb_oas_val, ccc_bb)

    # ── Block 2: Credit quality curve ─────────────────────────────────────
    def _val_30d_ago(df):
        if df is None or df.empty or len(df) < 22:
            return None
        return float(df.sort_values("date")["value"].iloc[-22]) * 100

    tier_data = {
        "AAA": {"now": aaa_oas_val, "30d_ago": _val_30d_ago(aaa_df)},
        "BBB": {"now": bbb_oas_val, "30d_ago": _val_30d_ago(bbb_df)},
        "BB": {"now": bb_oas_val, "30d_ago": _val_30d_ago(bb_df)},
        "B": {"now": single_b_oas_val, "30d_ago": _val_30d_ago(b_df)},
        "CCC": {"now": ccc_oas_val, "30d_ago": _val_30d_ago(ccc_df)},
    }
    _show_credit_curve(console, tier_data)

    # ── Block 3: Cross-regime signals ─────────────────────────────────────
    _show_cross_regime_signals(console, vix_val, hy_oas_val, hy_30d_chg, result.score)

    if llm:
        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis

        print_llm_regime_analysis(
            settings=settings,
            domain="credit",
            snapshot=snapshot_data,
            regime_label=result.label,
            regime_description=result.description,
            ticker=ticker,
        )

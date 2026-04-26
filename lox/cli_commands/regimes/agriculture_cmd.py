"""
CLI command for the Agriculture & Food Inflation regime.

Tracks crop prices (corn, wheat, soybeans), input costs (natural gas,
fertilizer equities, diesel), and food-inflation pressure signals.
"""
from __future__ import annotations

import logging

from rich import print as rprint

logger = logging.getLogger(__name__)

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
        return "[red]rising[/red]" if delta > 0 else "[green]falling[/green]"
    return "[green]rising[/green]" if delta > 0 else "[red]falling[/red]"


def _z_color(z: float | None) -> str:
    if z is None:
        return "dim"
    az = abs(z)
    if az > 2.0:
        return "bold red" if z > 0 else "bold green"
    if az > 1.0:
        return "red" if z > 0 else "green"
    if az > 0.5:
        return "yellow" if z > 0 else "cyan"
    return "dim"


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:+.1f}%"


def _fmt_z(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:+.2f}"


def _fmt_dollar(v: float | None, decimals: int = 2) -> str:
    if v is None:
        return "—"
    return f"${v:.{decimals}f}"


def _analog_ctx(pct: float | None) -> str:
    if pct is None:
        return ""
    if pct > 90:
        return "near 2022 peak"
    if pct > 70:
        return "elevated vs 2022"
    if pct > 50:
        return "halfway to 2022 peak"
    if pct > 30:
        return "well below 2022"
    return "far below 2022"


def _input_score_ctx(score: float | None) -> str:
    if score is None:
        return "no data"
    if score > 2.0:
        return "shock — extreme input cost pressure"
    if score > 1.25:
        return "elevated — building cost pressure"
    if score > 0.5:
        return "above average"
    if score > -0.5:
        return "normal range"
    if score > -1.25:
        return "below average — easing"
    return "depressed — input costs falling"


def _crop_score_ctx(score: float | None) -> str:
    if score is None:
        return "no data"
    if score > 1.5:
        return "surging — strong upside momentum"
    if score > 1.0:
        return "rising — above-trend momentum"
    if score > 0.5:
        return "mildly bullish"
    if score > -0.5:
        return "range-bound"
    if score > -1.0:
        return "soft — below-trend"
    return "weak — downtrend"


def _show_sparklines(console, df):
    """Show 30d sparklines for key series."""
    import pandas as pd

    lines: list[str] = []
    series_config = [
        ("CORN", "Corn", True, "$"),
        ("WEAT", "Wheat", True, "$"),
        ("NATGAS", "Nat Gas", True, "$"),
        ("FERT_BASKET", "Fert Bskt", True, ""),
        ("CPI_FOOD_YOY", "CPI Food", True, ""),
        ("BEEF_PRICE", "Beef", True, "$"),
        ("EGG_PRICE", "Eggs", True, "$"),
    ]

    for col, label, higher_is_worse, prefix in series_config:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna().tail(30).tolist()
        if len(vals) < 10:
            continue
        cur = f"{prefix}{vals[-1]:.2f}" if prefix else f"{vals[-1]:.1f}"
        lines.append(
            f"  {label:<10} {_sparkline(vals)}  {cur}  {_trend_label(vals, higher_is_worse=higher_is_worse)}"
        )

    if lines:
        console.print()
        console.print("[dim]─── Price Trends (30d) ─────────────────────────────────────────[/dim]")
        for ln in lines:
            console.print(ln)


def _cot_label(z: float | None) -> str:
    if z is None:
        return ""
    if z > 2.0:
        return "[bold red]extreme long[/bold red]"
    if z > 1.0:
        return "[yellow]crowded long[/yellow]"
    if z > 0.5:
        return "[dim]modestly long[/dim]"
    if z > -0.5:
        return "[dim]neutral[/dim]"
    if z > -1.0:
        return "[dim]modestly short[/dim]"
    if z > -2.0:
        return "[cyan]crowded short[/cyan]"
    return "[bold cyan]extreme short[/bold cyan]"


def _show_cot_positioning(console, inputs):
    """Show CFTC Commitments of Traders net speculative positioning."""
    lines: list[str] = []
    for label, net, z in [
        ("Corn", inputs.cot_corn_net, inputs.cot_corn_z),
        ("Wheat", inputs.cot_wheat_net, inputs.cot_wheat_z),
        ("Soybeans", inputs.cot_soybeans_net, inputs.cot_soybeans_z),
    ]:
        if net is None:
            continue
        net_str = f"{net:+,.0f}" if abs(net) >= 1000 else f"{net:+.0f}"
        zc = _z_color(z)
        z_str = f"[{zc}]z: {z:+.2f}[/{zc}]" if z is not None else ""
        pos_label = _cot_label(z)
        lines.append(f"  {label:<10} {net_str:>12} contracts  {z_str}  {pos_label}")

    if lines:
        console.print()
        date_str = f" ({inputs.cot_date})" if inputs.cot_date else ""
        console.print(f"[dim]─── CFTC Positioning (Managed Money){date_str} ─────────────────────[/dim]")
        for ln in lines:
            console.print(ln)


def _show_wasde(console, inputs):
    """Show WASDE supply/demand balances (stocks-to-use)."""
    from lox.data.usda import stu_context

    lines: list[str] = []
    for label, stu, es, comm in [
        ("Corn", inputs.wasde_corn_stu_pct, inputs.wasde_corn_ending_stocks, "corn"),
        ("Wheat", inputs.wasde_wheat_stu_pct, inputs.wasde_wheat_ending_stocks, "wheat"),
        ("Soybeans", inputs.wasde_soy_stu_pct, inputs.wasde_soy_ending_stocks, "soybeans"),
    ]:
        if stu is None:
            continue
        ctx = stu_context(comm, stu)
        color = "red" if "tight" in ctx else ("cyan" if "comfortable" in ctx or "burdensome" in ctx else "dim")
        lines.append(f"  {label} S/U: [{color}]{stu:.1f}%[/{color}]  ({ctx})")

    if lines:
        console.print()
        my_str = f" MY{inputs.wasde_market_year}/{inputs.wasde_market_year + 1 - 2000}" if inputs.wasde_market_year else ""
        console.print(f"[dim]─── WASDE Supply/Demand{my_str} ─────────────────────────────────[/dim]")
        for ln in lines:
            console.print(ln)
    elif not inputs.wasde_corn_stu_pct:
        console.print()
        console.print("[dim]─── WASDE Supply/Demand ────────────────────────────────────────[/dim]")
        console.print("  [dim]Set USDA_FAS_API_KEY in .env for stocks-to-use data (free at apps.fas.usda.gov/opendata/register)[/dim]")


def _show_seasonal(console, inputs):
    """Show seasonal z-score context for key series."""
    lines: list[str] = []
    for label, sz, rz in [
        ("Corn 20d", inputs.sz_corn_ret_20d, inputs.z_corn_ret_20d),
        ("Wheat 20d", inputs.sz_wheat_ret_20d, inputs.z_wheat_ret_20d),
        ("Nat Gas 20d", inputs.sz_natgas_ret_20d, inputs.z_natgas_ret_20d),
    ]:
        if sz is None:
            continue
        szc = _z_color(sz)
        ctx = ""
        if rz is not None:
            diff = abs(rz) - abs(sz)
            if diff > 0.5:
                ctx = " — move is unusual even for this season"
            elif diff < -0.5:
                ctx = " — normal seasonal pattern"
        lines.append(f"  {label:<12} seasonal z: [{szc}]{sz:+.2f}[/{szc}]{ctx}")

    if lines:
        console.print()
        console.print("[dim]─── Seasonal Context ──────────────────────────────────────────[/dim]")
        for ln in lines:
            console.print(ln)


def _pp_diff(diff: float | None, threshold: float = 5) -> str:
    """Format a percentage-point difference with color."""
    if diff is None:
        return ""
    color = "green" if diff > threshold else ("red" if diff < -threshold else "dim")
    return f"[{color}]{diff:+.0f}pp vs avg[/{color}]"


def _condition_bar(ge: float) -> str:
    """Build a G/E condition bar."""
    bar_pct = min(100, max(0, ge))
    bar_len = int(bar_pct / 5)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    color = "green" if ge >= 65 else ("yellow" if ge >= 55 else "red")
    return f"[{color}]{bar} {ge:.0f}% G/E[/{color}]"


CROP_ETFS = {"corn": "CORN", "soy": "SOYB", "wheat": "WEAT"}


def _planting_read(vs_avg: float | None, ticker: str) -> str:
    """One-line planting pace interpretation."""
    if vs_avg is None:
        return ""
    if vs_avg < -10:
        return f"[red]significantly late → bullish {ticker}[/red]"
    if vs_avg < -5:
        return f"[yellow]late planting → yield risk[/yellow]"
    if vs_avg > 5:
        return f"[green]ahead → eases supply risk[/green]"
    return ""


def _condition_read(ge: float | None, vs_avg: float | None, ticker: str) -> str:
    """One-line condition trading read."""
    if ge is None:
        return ""
    parts = []
    if ge < 45:
        parts.append(f"[red]yield stress → bullish {ticker}[/red]")
    elif ge < 55:
        parts.append(f"[yellow]below avg → watch {ticker}[/yellow]")
    elif ge > 70:
        parts.append(f"[green]strong crop → bearish supply premium[/green]")
    if vs_avg is not None and vs_avg < -5 and ge < 60:
        parts.append("[yellow]deteriorating[/yellow]")
    return "  ".join(parts)


def _pp_ahead(diff: float | None) -> str:
    """Format pp difference as directional language."""
    if diff is None:
        return ""
    if abs(diff) <= 2:
        return "[dim]on pace[/dim]"
    if diff > 0:
        return f"[green]+{diff:.0f}pp ahead[/green]"
    return f"[red]{diff:+.0f}pp behind[/red]"


def _show_crop_reports(console, inputs):
    """Show NASS crop report data — plantings, progress, conditions with ETF trading reads."""
    lines: list[str] = []

    # ── Prospective Plantings ─────────────────────────────────────────
    planting_lines = []
    for label, ticker, acres, yoy in [
        ("Corn", "CORN", inputs.corn_planted_acres_m, inputs.corn_planted_yoy_pct),
        ("Soybeans", "SOYB", inputs.soy_planted_acres_m, inputs.soy_planted_yoy_pct),
        ("Wheat", "WEAT", inputs.wheat_planted_acres_m, inputs.wheat_planted_yoy_pct),
    ]:
        if acres is None:
            continue
        yoy_str = ""
        if yoy is not None:
            color = "red" if yoy < -2 else ("green" if yoy > 2 else "dim")
            yoy_str = f"  [{color}]{yoy:+.1f}% YoY[/{color}]"
        planting_lines.append(f"  {ticker:<5} {acres:.1f}M acres{yoy_str}")

    if planting_lines:
        lines.append("  [bold]Prospective Plantings[/bold]")
        lines.extend(planting_lines)

    # ── Crop Progress (ETF-focused) ───────────────────────────────────
    progress_lines = []
    for ticker, pct_pl, vs_avg, pct_em, em_vs_avg in [
        ("CORN", inputs.corn_pct_planted, inputs.corn_pct_planted_vs_avg,
         inputs.corn_pct_emerged, inputs.corn_pct_emerged_vs_avg),
        ("SOYB", inputs.soy_pct_planted, inputs.soy_pct_planted_vs_avg,
         inputs.soy_pct_emerged, None),
        ("WEAT", inputs.wheat_pct_planted, inputs.wheat_pct_planted_vs_avg,
         None, None),
    ]:
        if pct_pl is None:
            continue
        parts = [f"planted {pct_pl:.0f}%  {_pp_ahead(vs_avg)}"]
        if ticker != "WEAT":
            if pct_em is not None:
                em_pace = _pp_ahead(em_vs_avg) if em_vs_avg is not None else ""
                parts.append(f"emerged {pct_em:.0f}%  {em_pace}")
            else:
                parts.append("emerged —")
        read = _planting_read(vs_avg, ticker)
        read_str = f"   — {read}" if read else ""
        progress_lines.append(f"  {ticker:<5} {'   '.join(parts)}{read_str}")

    if progress_lines:
        wk = inputs.crop_report_week or "?"
        lines.append(f"  [bold]Crop Progress[/bold]  (wk ending {wk})")
        lines.extend(progress_lines)

    # ── Crop Condition (with trading verdict) ─────────────────────────
    condition_lines = []
    for ticker, ge, vs_avg in [
        ("CORN", inputs.corn_condition_ge, inputs.corn_condition_ge_vs_avg),
        ("SOYB", inputs.soy_condition_ge, inputs.soy_condition_ge_vs_avg),
        ("WEAT", inputs.wheat_condition_ge, inputs.wheat_condition_ge_vs_avg),
    ]:
        if ge is None:
            continue
        bar = _condition_bar(ge)
        avg_str = ""
        if vs_avg is not None:
            avg_str = f"  {_pp_diff(vs_avg, threshold=3)}"
        read = _condition_read(ge, vs_avg, ticker)
        read_str = f"   ← {read}" if read else ""
        condition_lines.append(f"  {ticker:<5} {bar}{avg_str}{read_str}")

    if condition_lines:
        lines.append(f"  [bold]Crop Condition[/bold]")
        lines.extend(condition_lines)
        if inputs.crop_condition_composite is not None:
            comp_color = "green" if inputs.crop_condition_composite >= 65 else (
                "yellow" if inputs.crop_condition_composite >= 55 else "red"
            )
            lines.append(f"  Composite: [{comp_color}]{inputs.crop_condition_composite:.0f}% G/E[/{comp_color}]")

    # ── Growth Stages (silking, etc.) ─────────────────────────────────
    growth_lines = []
    if inputs.corn_pct_silking is not None:
        silk_str = f"  Corn silking: {inputs.corn_pct_silking:.0f}%"
        # Add trading context for silking behind average
        if hasattr(inputs, "corn_pct_silking_vs_avg") and inputs.corn_pct_silking_vs_avg is not None:
            if inputs.corn_pct_silking_vs_avg < -5:
                silk_str += f"  [yellow]({inputs.corn_pct_silking_vs_avg:+.0f}pp vs avg) — pollination risk → bullish CORN[/yellow]"
            else:
                silk_str += f"  ({inputs.corn_pct_silking_vs_avg:+.0f}pp vs avg)"
        growth_lines.append(silk_str)

    if growth_lines:
        lines.append(f"  [bold]Growth Stages[/bold]")
        lines.extend(growth_lines)

    # ── Planting delay warning ────────────────────────────────────────
    if inputs.planting_delay_count is not None and inputs.planting_delay_count >= 2:
        lines.append(f"  [yellow]⚠ {inputs.planting_delay_count} crops behind planting schedule — aggregate supply risk[/yellow]")

    if lines:
        console.print()
        console.print("[dim]─── USDA Crop Reports (NASS) ──────────────────────────────────[/dim]")
        for ln in lines:
            console.print(ln)
    else:
        console.print()
        console.print("[dim]─── USDA Crop Reports ─────────────────────────────────────────[/dim]")
        try:
            from lox.config import load_settings
            key = load_settings().usda_nass_api_key
        except Exception:
            key = None
        if not key:
            console.print("  [dim]Set USDA_NASS_API_KEY in .env for crop reports (free at quickstats.nass.usda.gov/api)[/dim]")
        else:
            from datetime import date as _d
            today = _d.today()
            if today.month <= 3 and today.day < 31:
                console.print("  [dim]Prospective Plantings releases March 31 — crop reports will populate then[/dim]")
            elif today.month < 4 or (today.month == 4 and today.day < 7):
                console.print("  [dim]Crop Progress reports begin first week of April[/dim]")
            else:
                console.print("  [dim]No crop report data available yet for this season[/dim]")


def _show_usda_calendar(console):
    """Show upcoming USDA report calendar."""
    try:
        from lox.data.usda_nass import upcoming_usda_reports
        reports = upcoming_usda_reports(days_ahead=21)
    except Exception:
        return

    if not reports:
        return

    from datetime import date as _date

    console.print()
    console.print("[dim]─── USDA Calendar (next 3 weeks) ──────────────────────────────[/dim]")
    today = _date.today()
    for r in reports[:8]:
        days_away = (r.date - today).days
        if days_away == 0:
            when = "[bold yellow]TODAY[/bold yellow]"
        elif days_away == 1:
            when = "[yellow]tomorrow[/yellow]"
        elif days_away <= 3:
            when = f"[yellow]{r.date.strftime('%a %b %d')}[/yellow]"
        else:
            when = f"[dim]{r.date.strftime('%a %b %d')}[/dim]"

        impact_icon = "●" if r.impact == "high" else "○"
        impact_color = "red" if r.impact == "high" else "dim"
        console.print(f"  [{impact_color}]{impact_icon}[/{impact_color}] {when:<28} {r.name}")


def _show_signals(console, inputs):
    """Show actionable cross-signals."""
    lines: list[str] = []

    # COT + fundamentals cross-signal
    if inputs.cot_corn_z is not None and inputs.wasde_corn_stu_pct is not None:
        if inputs.cot_corn_z > 1.0 and inputs.wasde_corn_stu_pct < 10:
            lines.append(
                f"  Specs long corn (z: {inputs.cot_corn_z:+.1f}) + tight S/U ({inputs.wasde_corn_stu_pct:.1f}%) "
                "→ [green]fundamentals support positioning[/green]"
            )
        elif inputs.cot_corn_z > 1.5 and inputs.wasde_corn_stu_pct > 15:
            lines.append(
                f"  Specs crowded long corn (z: {inputs.cot_corn_z:+.1f}) but S/U comfortable ({inputs.wasde_corn_stu_pct:.1f}%) "
                "→ [yellow]positioning vulnerable to correction[/yellow]"
            )
        elif inputs.cot_corn_z < -1.0 and inputs.wasde_corn_stu_pct < 10:
            lines.append(
                f"  Specs short corn (z: {inputs.cot_corn_z:+.1f}) + tight S/U ({inputs.wasde_corn_stu_pct:.1f}%) "
                "→ [green]short squeeze risk — fundamentals bullish[/green]"
            )

    # Acreage + supply cross-signals
    if inputs.corn_planted_yoy_pct is not None and inputs.corn_planted_yoy_pct < -3:
        if inputs.wasde_corn_stu_pct is not None and inputs.wasde_corn_stu_pct < 12:
            lines.append(
                f"  Corn acres down {abs(inputs.corn_planted_yoy_pct):.1f}% + tight S/U ({inputs.wasde_corn_stu_pct:.1f}%) "
                "→ [red]supply squeeze risk[/red]"
            )
        elif inputs.cot_corn_z is not None and inputs.cot_corn_z < -0.5:
            lines.append(
                f"  Corn acres down {abs(inputs.corn_planted_yoy_pct):.1f}% + specs short (z: {inputs.cot_corn_z:+.1f}) "
                "→ [green]contrarian bullish setup[/green]"
            )

    # Per-crop ETF condition + positioning cross-signals
    # Wheat condition poor + COT neutral/short → bullish WEAT
    if inputs.wheat_condition_ge is not None and inputs.wheat_condition_ge < 45:
        cot_neutral_or_short = inputs.cot_wheat_z is None or inputs.cot_wheat_z < 0.5
        cot_desc = f"specs neutral (z: {inputs.cot_wheat_z:+.2f})" if inputs.cot_wheat_z is not None else "specs data n/a"
        if cot_neutral_or_short:
            vs = f" ({inputs.wheat_condition_ge_vs_avg:+.0f}pp vs avg)" if inputs.wheat_condition_ge_vs_avg is not None else ""
            lines.append(
                f"  Wheat G/E {inputs.wheat_condition_ge:.0f}%{vs} + {cot_desc}\n"
                f"  → [green]bullish WEAT — supply stress not yet priced, limited spec crowding[/green]"
            )

    # Soybean planting well ahead + COT crowded long → bearish SOYB
    if inputs.soy_pct_planted_vs_avg is not None and inputs.soy_pct_planted_vs_avg > 5:
        if inputs.cot_soybeans_z is not None and inputs.cot_soybeans_z > 1.0:
            lines.append(
                f"  Soybean planting +{inputs.soy_pct_planted_vs_avg:.0f}pp ahead + specs crowded long (z: {inputs.cot_soybeans_z:+.2f})\n"
                f"  → [yellow]bearish SOYB — supply building + crowded[/yellow]"
            )

    # Any crop condition < 45% → weather premium building
    for crop, ticker, ge in [
        ("Corn", "CORN", inputs.corn_condition_ge),
        ("Soybean", "SOYB", inputs.soy_condition_ge),
        ("Wheat", "WEAT", inputs.wheat_condition_ge),
    ]:
        if ge is not None and ge < 45:
            lines.append(f"  {crop} G/E {ge:.0f}% → [yellow]{ticker} weather premium building[/yellow]")

    # Multi-crop condition signals
    weak_crops = []
    for label, ge in [("Corn", inputs.corn_condition_ge), ("Soy", inputs.soy_condition_ge), ("Wheat", inputs.wheat_condition_ge)]:
        if ge is not None and ge < 55:
            weak_crops.append(f"{label} {ge:.0f}%")
    if weak_crops and inputs.crop_momentum_score is not None and inputs.crop_momentum_score < 0.5:
        lines.append(
            f"  Poor crop condition ({', '.join(weak_crops)} G/E) not yet priced in "
            "→ [yellow]watch for weather premium[/yellow]"
        )

    if inputs.crop_condition_composite is not None and inputs.crop_condition_composite < 55:
        lines.append(
            f"  Composite crop condition at {inputs.crop_condition_composite:.0f}% G/E "
            "→ [red]broad yield stress across crops[/red]"
        )
        # Composite < 55 + momentum not reflecting it → bullish DBA
        if inputs.crop_momentum_score is not None and inputs.crop_momentum_score < 0:
            lines.append(
                f"  Composite {inputs.crop_condition_composite:.0f}% G/E + crop momentum negative "
                "→ [green]broad supply stress not priced → bullish DBA[/green]"
            )

    if inputs.planting_delay_count is not None and inputs.planting_delay_count >= 2:
        lines.append(
            f"  {inputs.planting_delay_count} crops behind planting schedule (>5pp vs avg) "
            "→ [yellow]aggregate supply risk building[/yellow]"
        )
    elif inputs.corn_pct_planted_vs_avg is not None and inputs.corn_pct_planted_vs_avg < -10:
        lines.append(
            f"  Corn planting significantly behind avg ({inputs.corn_pct_planted_vs_avg:+.0f}pp) "
            "→ [yellow]yield risk building — weather premium[/yellow]"
        )

    if inputs.cost_pass_through_lag:
        div = inputs.fert_corn_divergence
        div_str = f" (divergence z: {div:+.1f})" if div is not None else ""
        lines.append(f"  Fertilizer costs leading crop prices{div_str} → [yellow]cost pass-through building[/yellow]")

    if inputs.natgas_corn_ratio_z is not None and inputs.natgas_corn_ratio_z > 1.5:
        lines.append(f"  Nat gas / corn ratio elevated (z: {inputs.natgas_corn_ratio_z:+.1f}) → [yellow]corn underpriced vs energy input[/yellow]")
    elif inputs.natgas_corn_ratio_z is not None and inputs.natgas_corn_ratio_z < -1.5:
        lines.append(f"  Nat gas / corn ratio compressed (z: {inputs.natgas_corn_ratio_z:+.1f}) → [cyan]input costs cheap vs crops[/cyan]")

    if inputs.input_cost_score is not None and inputs.crop_momentum_score is not None:
        if inputs.input_cost_score > 1.0 and inputs.crop_momentum_score < 0:
            lines.append("  Input costs rising + crop prices flat/falling → [yellow]divergence — watch for catch-up[/yellow]")
        elif inputs.input_cost_score < -1.0 and inputs.crop_momentum_score > 1.0:
            lines.append("  Input costs falling + crop prices rising → [cyan]margin expansion for producers[/cyan]")

    # Seasonal vs rolling divergence
    if inputs.sz_corn_ret_20d is not None and inputs.z_corn_ret_20d is not None:
        if inputs.z_corn_ret_20d > 1.0 and inputs.sz_corn_ret_20d < 0.3:
            lines.append("  Corn rally looks strong rolling z but [dim]normal seasonally[/dim] — discount the signal")
        elif inputs.z_corn_ret_20d > 0.5 and inputs.sz_corn_ret_20d > 1.5:
            lines.append("  Corn rally is [bold yellow]unusual even for this season[/bold yellow] — higher conviction signal")

    # Food CPI signals
    if inputs.food_accel_flag and inputs.cpi_food_3m_ann is not None:
        lines.append(
            f"  Food CPI accelerating: 3m ann {inputs.cpi_food_3m_ann:+.1f}% vs YoY {inputs.cpi_food_yoy:+.1f}% "
            "→ [red]inflation momentum building[/red]"
        )
    if inputs.grocery_shock and inputs.grocery_restaurant_gap is not None:
        lines.append(
            f"  Grocery prices outpacing restaurants by {inputs.grocery_restaurant_gap:+.1f}pp "
            "→ [yellow]consumer wallet pressure[/yellow]"
        )
    if inputs.protein_spike and inputs.protein_z is not None:
        lines.append(
            f"  Protein composite z: {inputs.protein_z:+.1f} "
            "→ [red]protein spike — headline/political risk[/red]"
        )
    if inputs.farm_to_retail_spread is not None and inputs.farm_to_retail_spread < -2:
        lines.append(
            f"  PPI > CPI food by {abs(inputs.farm_to_retail_spread):.1f}pp "
            "→ [yellow]pipeline pressure — retail pass-through coming[/yellow]"
        )

    # Cross-regime signals
    try:
        from lox.data.regime_history import get_score_series
        for domain, display in [("inflation", "Inflation"), ("consumer", "Consumer")]:
            series = get_score_series(domain)
            if not series:
                continue
            sc = series[-1].get("score")
            if not isinstance(sc, (int, float)):
                continue
            if domain == "inflation" and sc > 55 and inputs.food_inflation_score is not None and inputs.food_inflation_score > 1.0:
                lines.append(f"  Inflation regime {sc:.0f} + food inflation elevated → [red]reinforcing inflation impulse[/red]")
            elif domain == "consumer" and sc > 55 and inputs.crop_surge:
                lines.append(f"  Consumer stress {sc:.0f} + crop surge → [red]food inflation hitting consumers[/red]")
    except Exception:
        pass

    if lines:
        console.print()
        console.print("[dim]─── Signals ───────────────────────────────────────────────────[/dim]")
        for ln in lines:
            console.print(ln)


def _show_2022_analog(console, inputs):
    """Show comparison to 2022 peaks."""
    lines: list[str] = []

    for label, pct, peak_label in [
        ("Corn", inputs.corn_pct_of_2022_peak, "$29.00"),
        ("Nat Gas", inputs.natgas_pct_of_2022_peak, "$8.81"),
    ]:
        if pct is None:
            continue
        bar_len = min(20, int(pct / 5))
        bar = "█" * bar_len + "░" * (20 - bar_len)
        color = "red" if pct > 70 else ("yellow" if pct > 40 else "dim")
        lines.append(f"  {label:<8} [{color}]{bar} {pct:.0f}%[/{color}]  (peak: {peak_label})")

    if lines:
        console.print()
        console.print("[dim]─── 2022 Analog ───────────────────────────────────────────────[/dim]")
        for ln in lines:
            console.print(ln)


def _yoy_arrow(yoy: float | None, median_threshold: float = 2.5) -> str:
    """Return a directional arrow for YoY values."""
    if yoy is None:
        return ""
    if yoy > median_threshold:
        return "[red]▲[/red]"
    if yoy < -median_threshold:
        return "[green]▼[/green]"
    return "[dim]─[/dim]"


def _show_food_inflation_dashboard(console, inputs):
    """Panel A: Food Inflation Dashboard — consumer CPI breakdown."""
    if inputs.cpi_food_yoy is None:
        return

    lines: list[str] = []

    # Headline CPI food
    accel_tag = ""
    if inputs.cpi_food_3m_ann is not None:
        if inputs.cpi_food_accel is not None and inputs.cpi_food_accel > 0.5:
            accel_tag = "  [red]accelerating ▲[/red]"
        elif inputs.cpi_food_accel is not None and inputs.cpi_food_accel < -0.5:
            accel_tag = "  [green]decelerating ▼[/green]"
        else:
            accel_tag = "  [dim]stable[/dim]"
        lines.append(
            f"  CPI Food (All)      {_fmt_pct(inputs.cpi_food_yoy):>8} YoY"
            f"    3m ann: {_fmt_pct(inputs.cpi_food_3m_ann)}{accel_tag}"
        )
    else:
        lines.append(f"  CPI Food (All)      {_fmt_pct(inputs.cpi_food_yoy):>8} YoY")

    if inputs.cpi_food_home_yoy is not None:
        lines.append(f"  Food at Home        {_fmt_pct(inputs.cpi_food_home_yoy):>8} YoY")
    if inputs.cpi_food_away_yoy is not None:
        lines.append(f"  Food Away           {_fmt_pct(inputs.cpi_food_away_yoy):>8} YoY")

    # Sub-categories
    lines.append("")
    for label, yoy in [
        ("Cereals & Bakery", inputs.cpi_cereals_yoy),
        ("Meats/Poultry/Eggs", inputs.cpi_meats_yoy),
        ("Dairy", inputs.cpi_dairy_yoy),
        ("Fruits & Vegetables", inputs.cpi_fruits_veg_yoy),
    ]:
        if yoy is None:
            continue
        c = "red" if yoy > 4 else ("yellow" if yoy > 2 else ("green" if yoy < 0 else "dim"))
        arrow = _yoy_arrow(yoy)
        lines.append(f"  {label:<20} [{c}]{_fmt_pct(yoy):>8} YoY[/{c}]   {arrow}")

    # Breadth
    if inputs.food_breadth_count is not None:
        lines.append(f"\n  Breadth: {inputs.food_breadth_count}/5 categories above median")

    console.print()
    console.print("[dim]─── Food Inflation Dashboard ──────────────────────────────────[/dim]")
    for ln in lines:
        console.print(ln)


def _show_farm_to_retail(console, inputs):
    """Panel B: Farm-to-Retail Pipeline — PPI, proteins, softs."""
    has_pipeline = inputs.ppi_food_mfg_yoy is not None or inputs.beef_price is not None or inputs.sugar_price is not None
    if not has_pipeline:
        return

    lines: list[str] = []

    # Pipeline spread
    if inputs.ppi_food_mfg_yoy is not None:
        lines.append(f"  PPI Food Mfg        {_fmt_pct(inputs.ppi_food_mfg_yoy):>8} YoY    [dim](factory gate)[/dim]")
    if inputs.cpi_food_home_yoy is not None:
        lines.append(f"  CPI Food Home       {_fmt_pct(inputs.cpi_food_home_yoy):>8} YoY    [dim](retail shelf)[/dim]")
    if inputs.farm_to_retail_spread is not None:
        spread_c = "red" if inputs.farm_to_retail_spread > 2 else ("yellow" if inputs.farm_to_retail_spread > 0 else "cyan")
        lines.append(f"  Spread:             [{spread_c}]{inputs.farm_to_retail_spread:+.1f}pp[/{spread_c}]       [dim]retail {'>' if inputs.farm_to_retail_spread > 0 else '<'} wholesale[/dim]")

    # Proteins
    protein_lines = []
    for label, price, unit in [
        ("Ground Beef", inputs.beef_price, "/lb"),
        ("Chicken Breast", inputs.chicken_price, "/lb"),
        ("Eggs (dozen)", inputs.egg_price, "/dz"),
    ]:
        if price is None:
            continue
        protein_lines.append(f"  {label:<18} {_fmt_dollar(price)}{unit}")
    if protein_lines:
        lines.append("")
        lines.append("  [bold]Proteins:[/bold]")
        lines.extend(protein_lines)
        if inputs.protein_yoy_avg is not None:
            pc = "red" if inputs.protein_yoy_avg > 5 else ("yellow" if inputs.protein_yoy_avg > 2 else "dim")
            lines.append(f"  Composite YoY:     [{pc}]{_fmt_pct(inputs.protein_yoy_avg)}[/{pc}]")

    # Soft commodities
    soft_lines = []
    for label, ticker, price in [
        ("Sugar (CANE)", "CANE", inputs.sugar_price),
        ("Coffee (JO)", "JO", inputs.coffee_price),
        ("Cocoa (NIB)", "NIB", inputs.cocoa_price),
    ]:
        if price is None:
            continue
        soft_lines.append(f"  {label:<18} {_fmt_dollar(price)}")
    if soft_lines:
        lines.append("")
        lines.append("  [bold]Softs:[/bold]")
        lines.extend(soft_lines)
        if inputs.soft_ret_20d is not None:
            sc = "red" if inputs.soft_ret_20d > 3 else ("green" if inputs.soft_ret_20d < -3 else "dim")
            z_str = f"  z: {_fmt_z(inputs.soft_z)}" if inputs.soft_z is not None else ""
            lines.append(f"  Composite 20d:     [{sc}]{_fmt_pct(inputs.soft_ret_20d)}[/{sc}]{z_str}")

    if lines:
        console.print()
        console.print("[dim]─── Farm-to-Retail Pipeline ───────────────────────────────────[/dim]")
        for ln in lines:
            console.print(ln)


def agriculture_snapshot(
    *,
    llm: bool = False,
    ticker: str = "",
    refresh: bool = False,
    delta: str = "",
    alert: bool = False,
    json_out: bool = False,
) -> None:
    """Entry point for `lox regime ag`."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from lox.config import load_settings
    from lox.cli_commands.shared.labs_utils import save_snapshot

    console = Console()
    settings = load_settings()

    # ── Fetch data ─────────────────────────────────────────────────────
    console.print("[dim]Loading agriculture & input cost data...[/dim]")

    from lox.agriculture.signals import build_agriculture_state, build_agriculture_dataset
    from lox.agriculture.regime import classify_agriculture_regime

    state = build_agriculture_state(settings=settings, refresh=refresh)
    inputs = state.inputs
    regime = classify_agriculture_regime(inputs)

    # ── Fetch live quotes to overlay on FRED/Alpaca data ───────────────
    from lox.agriculture.signals import _fetch_live_quotes
    live = _fetch_live_quotes(settings)
    if live.get("CORN"):
        inputs.corn = live["CORN"]
    if live.get("WEAT"):
        inputs.wheat = live["WEAT"]

    # ── Build snapshot dict for persistence ────────────────────────────
    snapshot_data = {
        "corn": inputs.corn,
        "wheat": inputs.wheat,
        "soybeans": inputs.soybeans,
        "ag_broad": inputs.ag_broad,
        "natgas": inputs.natgas,
        "diesel": inputs.diesel,
        "fert_basket": inputs.fert_basket_level,
        "input_cost_score": inputs.input_cost_score,
        "crop_momentum_score": inputs.crop_momentum_score,
        "food_inflation_score": inputs.food_inflation_score,
        "fert_corn_divergence": inputs.fert_corn_divergence,
        "corn_ret_20d_pct": inputs.corn_ret_20d_pct,
        "natgas_ret_20d_pct": inputs.natgas_ret_20d_pct,
        "corn_pct_of_2022_peak": inputs.corn_pct_of_2022_peak,
        "natgas_pct_of_2022_peak": inputs.natgas_pct_of_2022_peak,
        "cot_corn_net": inputs.cot_corn_net,
        "cot_corn_z": inputs.cot_corn_z,
        "cot_wheat_net": inputs.cot_wheat_net,
        "cot_wheat_z": inputs.cot_wheat_z,
        "cot_soybeans_net": inputs.cot_soybeans_net,
        "cot_soybeans_z": inputs.cot_soybeans_z,
        "wasde_corn_stu_pct": inputs.wasde_corn_stu_pct,
        "wasde_wheat_stu_pct": inputs.wasde_wheat_stu_pct,
        "wasde_soy_stu_pct": inputs.wasde_soy_stu_pct,
        "sz_corn_ret_20d": inputs.sz_corn_ret_20d,
        "sz_wheat_ret_20d": inputs.sz_wheat_ret_20d,
        "sz_natgas_ret_20d": inputs.sz_natgas_ret_20d,
        "corn_planted_acres_m": inputs.corn_planted_acres_m,
        "corn_planted_yoy_pct": inputs.corn_planted_yoy_pct,
        "soy_planted_acres_m": inputs.soy_planted_acres_m,
        "soy_planted_yoy_pct": inputs.soy_planted_yoy_pct,
        "wheat_planted_acres_m": inputs.wheat_planted_acres_m,
        "wheat_planted_yoy_pct": inputs.wheat_planted_yoy_pct,
        "corn_pct_planted": inputs.corn_pct_planted,
        "corn_pct_planted_vs_avg": inputs.corn_pct_planted_vs_avg,
        "corn_pct_emerged": inputs.corn_pct_emerged,
        "corn_pct_silking": inputs.corn_pct_silking,
        "corn_condition_ge": inputs.corn_condition_ge,
        "corn_condition_ge_vs_avg": inputs.corn_condition_ge_vs_avg,
        "soy_pct_planted": inputs.soy_pct_planted,
        "soy_pct_planted_vs_avg": inputs.soy_pct_planted_vs_avg,
        "soy_pct_emerged": inputs.soy_pct_emerged,
        "soy_condition_ge": inputs.soy_condition_ge,
        "soy_condition_ge_vs_avg": inputs.soy_condition_ge_vs_avg,
        "wheat_pct_planted": inputs.wheat_pct_planted,
        "wheat_pct_planted_vs_avg": inputs.wheat_pct_planted_vs_avg,
        "wheat_condition_ge": inputs.wheat_condition_ge,
        "wheat_condition_ge_vs_avg": inputs.wheat_condition_ge_vs_avg,
        "crop_condition_composite": inputs.crop_condition_composite,
        "planting_delay_count": inputs.planting_delay_count,
        # Food CPI
        "cpi_food_yoy": inputs.cpi_food_yoy,
        "cpi_food_home_yoy": inputs.cpi_food_home_yoy,
        "cpi_food_away_yoy": inputs.cpi_food_away_yoy,
        "cpi_food_3m_ann": inputs.cpi_food_3m_ann,
        "ppi_food_mfg_yoy": inputs.ppi_food_mfg_yoy,
        "farm_to_retail_spread": inputs.farm_to_retail_spread,
        "food_breadth_count": inputs.food_breadth_count,
        "protein_yoy_avg": inputs.protein_yoy_avg,
        "protein_z": inputs.protein_z,
        "soft_ret_20d": inputs.soft_ret_20d,
        "soft_z": inputs.soft_z,
    }
    save_snapshot("agriculture", snapshot_data, regime.label or regime.name)

    # ── JSON output ────────────────────────────────────────────────────
    if json_out:
        import json
        out = {"regime": regime.name, "label": regime.label, **snapshot_data, "asof": state.asof}
        console.print(json.dumps(out, indent=2, default=str))
        return

    # ── Alert mode ─────────────────────────────────────────────────────
    if alert:
        is_extreme = (
            inputs.input_shock or inputs.crop_surge or inputs.cost_pass_through_lag
            or inputs.broad_food_inflation or inputs.food_accel_flag
        )
        if is_extreme:
            console.print(f"[bold red]AG ALERT[/bold red]: {regime.label}")
            if inputs.corn:
                console.print(f"  Corn {_fmt_dollar(inputs.corn)}  20d: {_fmt_pct(inputs.corn_ret_20d_pct)}")
            if inputs.natgas:
                console.print(f"  Nat Gas {_fmt_dollar(inputs.natgas)}  20d: {_fmt_pct(inputs.natgas_ret_20d_pct)}")
            if inputs.food_inflation_score is not None:
                console.print(f"  Food inflation score: {inputs.food_inflation_score:+.2f}")
            if inputs.cpi_food_yoy is not None:
                console.print(f"  CPI Food YoY: {_fmt_pct(inputs.cpi_food_yoy)}")
            if inputs.broad_food_inflation:
                console.print(f"  [red]Broad food inflation: {inputs.food_breadth_count}/5 breadth[/red]")
            if inputs.food_accel_flag:
                console.print(f"  [yellow]Food CPI accelerating: 3m ann {_fmt_pct(inputs.cpi_food_3m_ann)} vs YoY {_fmt_pct(inputs.cpi_food_yoy)}[/yellow]")
        return

    # ── Delta mode ─────────────────────────────────────────────────────
    if delta:
        from lox.cli_commands.shared.labs_utils import parse_delta_period, get_delta_metrics, show_delta_summary

        delta_days = parse_delta_period(delta)
        metric_keys = [
            "Corn:corn:$",
            "Wheat:wheat:$",
            "Nat Gas:natgas:$",
            "Diesel:diesel:$",
            "Input Cost Score:input_cost_score:",
            "Crop Momentum:crop_momentum_score:",
            "Food Inflation:food_inflation_score:",
            "Fert-Corn Div:fert_corn_divergence:",
            "CPI Food YoY:cpi_food_yoy:%",
            "PPI Food Mfg:ppi_food_mfg_yoy:%",
            "Pipeline Spread:farm_to_retail_spread:",
            "Protein YoY:protein_yoy_avg:%",
            "Corn G/E:corn_condition_ge:%",
            "Soy G/E:soy_condition_ge:%",
            "Wheat G/E:wheat_condition_ge:%",
            "Crop Composite:crop_condition_composite:%",
        ]
        metrics_for_delta, prev_regime = get_delta_metrics("agriculture", snapshot_data, metric_keys, delta_days)
        show_delta_summary("agriculture", regime.label or regime.name, prev_regime, metrics_for_delta, delta_days)
        if prev_regime is None:
            console.print(f"\n[dim]No cached data from {delta_days}d ago. Run `lox regime ag` daily to build history.[/dim]")
        return

    # ── Build main panel ───────────────────────────────────────────────
    # Price line
    price_parts = []
    if inputs.corn is not None:
        chg = f" ({inputs.corn_ret_20d_pct:+.1f}% 20d)" if inputs.corn_ret_20d_pct is not None else ""
        price_parts.append(f"Corn {_fmt_dollar(inputs.corn)}{chg}")
    if inputs.wheat is not None:
        chg = f" ({inputs.wheat_ret_20d_pct:+.1f}%)" if inputs.wheat_ret_20d_pct is not None else ""
        price_parts.append(f"Wheat {_fmt_dollar(inputs.wheat)}{chg}")
    if inputs.soybeans is not None:
        price_parts.append(f"Soy {_fmt_dollar(inputs.soybeans)}")
    if inputs.ag_broad is not None:
        price_parts.append(f"DBA {_fmt_dollar(inputs.ag_broad)}")
    price_line = " | ".join(price_parts) if price_parts else "No crop data"

    # Input cost line
    input_parts = []
    if inputs.natgas is not None:
        z = inputs.z_natgas_ret_20d
        zc = _z_color(z)
        input_parts.append(f"NatGas {_fmt_dollar(inputs.natgas)} [{zc}]z:{_fmt_z(z)}[/{zc}]")
    if inputs.fert_basket_level is not None:
        z = inputs.z_fert_basket_ret_20d
        zc = _z_color(z)
        input_parts.append(f"Fert [{zc}]z:{_fmt_z(z)}[/{zc}]")
    if inputs.diesel is not None:
        z = inputs.z_diesel_ret_60d
        zc = _z_color(z)
        input_parts.append(f"Diesel {_fmt_dollar(inputs.diesel)} [{zc}]z:{_fmt_z(z)}[/{zc}]")
    input_line = " | ".join(input_parts) if input_parts else "No input cost data"

    # Score summary
    score_parts = []
    if inputs.input_cost_score is not None:
        c = _z_color(inputs.input_cost_score)
        score_parts.append(f"Input Costs: [{c}]{inputs.input_cost_score:+.2f}[/{c}] ({_input_score_ctx(inputs.input_cost_score)})")
    if inputs.crop_momentum_score is not None:
        c = _z_color(inputs.crop_momentum_score)
        score_parts.append(f"Crop Momentum: [{c}]{inputs.crop_momentum_score:+.2f}[/{c}] ({_crop_score_ctx(inputs.crop_momentum_score)})")
    if inputs.food_inflation_score is not None:
        c = _z_color(inputs.food_inflation_score)
        score_parts.append(f"Food Inflation: [{c}]{inputs.food_inflation_score:+.2f}[/{c}]")

    # Regime color
    regime_color = "bold red" if regime.name in ("ag_input_shock", "food_inflation", "broad_food_inflation") else (
        "yellow" if regime.name in ("crop_surge", "cost_pass_through", "ag_cost_reflation") else (
            "cyan" if regime.name == "ag_disinflation" else "green"
        )
    )

    panel_lines = [
        price_line,
        f"[{regime_color}]{regime.label}[/{regime_color}]",
        "",
        input_line,
        "",
    ]
    panel_lines.extend(score_parts)

    panel_text = Text.from_markup("\n".join(panel_lines))
    asof_label = "live" if live.get("CORN") else state.asof
    panel = Panel(
        panel_text,
        title="[bold]Agriculture & Food Inflation[/bold]",
        subtitle=f"[dim]{asof_label}[/dim]",
        border_style="yellow",
        padding=(1, 2),
    )
    rprint(panel)

    # ── Sparklines ─────────────────────────────────────────────────────
    try:
        df = build_agriculture_dataset(settings=settings, refresh=False)
        _show_sparklines(console, df)
    except Exception:
        pass

    # ── Food Inflation Dashboard ──────────────────────────────────────
    _show_food_inflation_dashboard(console, inputs)

    # ── Farm-to-Retail Pipeline ───────────────────────────────────────
    _show_farm_to_retail(console, inputs)

    # ── CFTC Positioning ──────────────────────────────────────────────
    _show_cot_positioning(console, inputs)

    # ── WASDE Supply/Demand ───────────────────────────────────────────
    _show_wasde(console, inputs)

    # ── USDA Crop Reports (NASS) ─────────────────────────────────────
    _show_crop_reports(console, inputs)

    # ── Seasonal Context ──────────────────────────────────────────────
    _show_seasonal(console, inputs)

    # ── Signals ────────────────────────────────────────────────────────
    _show_signals(console, inputs)

    # ── 2022 analog ────────────────────────────────────────────────────
    _show_2022_analog(console, inputs)

    # ── USDA Calendar ────────────────────────────────────────────────
    _show_usda_calendar(console)

    # ── LLM analysis ───────────────────────────────────────────────────
    if llm:
        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis
        print_llm_regime_analysis(
            settings=settings,
            domain="agriculture",
            snapshot=snapshot_data,
            regime_label=regime.label,
            regime_description=(
                "Agriculture & food inflation regime tracking crop prices (corn, wheat, soybeans), "
                "input costs (natural gas, fertilizer, diesel), and food-inflation pressure. "
                "Natural gas is the primary cost driver for urea/ammonia fertilizer production."
            ),
            ticker=ticker,
        )

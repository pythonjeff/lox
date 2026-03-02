"""CLI command for the Funding / Liquidity regime."""
from __future__ import annotations

import json

import typer
from rich import print
from rich.panel import Panel

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings
from lox.funding.features import funding_feature_vector
from lox.funding.regime import classify_funding_regime
from lox.funding.signals import FUNDING_FRED_SERIES, build_funding_state, build_funding_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Sparkline / trend helpers
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


def _extract_col_last_n(df, col: str, n: int = 30) -> list[float]:
    """Extract last N values of a column from a DataFrame."""
    import pandas as pd
    if df is None or df.empty or col not in df.columns:
        return []
    s = pd.to_numeric(df[col], errors="coerce").dropna().tail(n)
    return [float(v) for v in s]


# ─────────────────────────────────────────────────────────────────────────────
# Context helpers
# ─────────────────────────────────────────────────────────────────────────────

def _corridor_ctx(v):
    if not isinstance(v, (int, float)):
        return "corridor spread"
    v = float(v)
    if abs(v) > 10:
        return "wide — funding stress"
    if abs(v) > 5:
        return "slightly wide — watch closely"
    return "normal — within corridor"


def _sofr_effr_ctx(v):
    if not isinstance(v, (int, float)):
        return "secured vs unsecured"
    v = float(v)
    if v < -5:
        return "repo richening — collateral scarce"
    if v < -2:
        return "secured slightly below unsecured"
    if v > 5:
        return "repo cheapening — reserves ample"
    return "normal basis"


def _spike_ctx(v):
    if not isinstance(v, (int, float)):
        return "5-day max spike"
    v = float(v)
    if v > 20:
        return "severe spike — funding stress event"
    if v > 10:
        return "notable spike — quarter-end?"
    if v > 5:
        return "mild spike"
    return "calm — no funding disruption"


def _vol_ctx(v):
    if not isinstance(v, (int, float)):
        return "funding volatility"
    v = float(v)
    if v > 10:
        return "high vol — unstable funding"
    if v > 5:
        return "elevated vol"
    return "low vol — stable funding"


def _persist_ctx(v):
    if not isinstance(v, (int, float)):
        return "stress persistence"
    v = float(v)
    if v > 0.5:
        return "sustained stress — not resolving"
    if v > 0.2:
        return "intermittent stress"
    return "no persistent stress"


def _rrp_ctx(v):
    if not isinstance(v, (int, float)):
        return "overnight RRP buffer"
    v = float(v) / 1000  # to trillions
    if v > 1.5:
        return "large buffer — ample excess liquidity"
    if v > 0.5:
        return "moderate buffer"
    if v > 0.1:
        return "shrinking — buffer nearly depleted"
    return "near zero — no excess liquidity"


def _reserves_ctx(v):
    if not isinstance(v, (int, float)):
        return "bank reserves"
    v = float(v) / 1000  # to trillions
    if v > 3.5:
        return "abundant — well above scarcity"
    if v > 3.0:
        return "comfortable"
    if v > 2.5:
        return "adequate — approaching scarcity zone"
    return "scarce — stress risk elevated"


def _tga_ctx(v):
    if v is None:
        return "Treasury cash flow"
    v = float(v) / 1000  # millions to billions
    if v > 50:
        return "large buildup — draining reserves"
    if v > 10:
        return "building — mild drain"
    if v > -10:
        return "stable"
    if v > -50:
        return "drawing down — adding reserves"
    return "large drawdown — reserve injection"


# ─────────────────────────────────────────────────────────────────────────────
# Spread velocity sparklines
# ─────────────────────────────────────────────────────────────────────────────

def _show_rate_sparklines(console, df, fi):
    """Show 30-day sparklines for key funding series."""
    corridor_vals = _extract_col_last_n(df, "CORRIDOR_SPREAD_BPS", 30)
    sofr_effr_vals = _extract_col_last_n(df, "SOFR_EFFR_BPS", 30)
    vol_vals = _extract_col_last_n(df, "VOL_20D_BPS", 30)
    spike_vals = _extract_col_last_n(df, "SPIKE_5D_BPS", 30)

    lines: list[str] = []
    corridor_name = fi.spread_corridor_name or "Corridor"
    if len(corridor_vals) >= 10:
        cur = f"{corridor_vals[-1]:+.1f}bp" if corridor_vals else ""
        lines.append(
            f"  {corridor_name:<12} {_sparkline(corridor_vals)}  {cur}  {_trend_label(corridor_vals)}"
        )
    if len(sofr_effr_vals) >= 10:
        cur = f"{sofr_effr_vals[-1]:+.1f}bp" if sofr_effr_vals else ""
        lines.append(
            f"  {'SOFR-EFFR':<12} {_sparkline(sofr_effr_vals)}  {cur}  {_trend_label(sofr_effr_vals)}"
        )
    if len(vol_vals) >= 10:
        cur = f"{vol_vals[-1]:.1f}bp" if vol_vals else ""
        lines.append(
            f"  {'Vol 20d':<12} {_sparkline(vol_vals)}  {cur}  {_trend_label(vol_vals)}"
        )
    if len(spike_vals) >= 10:
        cur = f"{spike_vals[-1]:+.1f}bp" if spike_vals else ""
        lines.append(
            f"  {'Spike 5d':<12} {_sparkline(spike_vals)}  {cur}  {_trend_label(spike_vals)}"
        )

    if lines:
        console.print()
        console.print("[dim]─── Funding Velocity (30d) ────────────────────────────────────[/dim]")
        for ln in lines:
            console.print(ln)


# ─────────────────────────────────────────────────────────────────────────────
# Rate corridor table
# ─────────────────────────────────────────────────────────────────────────────

def _corridor_shift_signal(delta_bp):
    if delta_bp is None:
        return "—"
    if delta_bp > 10:
        return "[red]surging — stress[/red]"
    if delta_bp > 5:
        return "[yellow]rising[/yellow]"
    if delta_bp > 2:
        return "[yellow]drifting higher[/yellow]"
    if delta_bp > -2:
        return "[dim]stable[/dim]"
    if delta_bp > -5:
        return "[green]easing[/green]"
    if delta_bp > -10:
        return "[green]falling[/green]"
    return "[green]plunging[/green]"


def _show_rate_corridor(console, df):
    """Show rate corridor with 30d shifts per component."""
    import pandas as pd
    from rich.table import Table as RichTable

    if df is None or df.empty:
        return

    rates = ["SOFR", "EFFR", "IORB", "TGCR", "BGCR", "OBFR"]
    rate_labels = {
        "SOFR": "SOFR (secured)",
        "EFFR": "EFFR (unsecured)",
        "IORB": "IORB (corridor top)",
        "TGCR": "TGCR (tri-party)",
        "BGCR": "BGCR (broad GC)",
        "OBFR": "OBFR (bank funding)",
    }

    has_data = False
    rows = []
    for rate in rates:
        if rate not in df.columns:
            continue
        col = pd.to_numeric(df[rate], errors="coerce").dropna()
        if col.empty:
            continue
        now = float(col.iloc[-1])
        ago = float(col.iloc[-22]) if len(col) >= 22 else None
        has_data = True
        delta = (now - ago) * 100 if ago is not None else None  # % -> bps
        rows.append((rate_labels.get(rate, rate), now, ago, delta))

    if not has_data:
        return

    ct = RichTable(
        title="Rate Corridor (30d shift)",
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 1),
    )
    ct.add_column("Rate")
    ct.add_column("Now", justify="right")
    ct.add_column("30d ago", justify="right")
    ct.add_column("Δ (bps)", justify="right")
    ct.add_column("Signal", style="dim")

    for label, now, ago, delta in rows:
        now_s = f"{now:.2f}%"
        ago_s = f"{ago:.2f}%" if ago is not None else "—"
        delta_s = f"{delta:+.0f}" if delta is not None else "—"
        ct.add_row(label, now_s, ago_s, delta_s, _corridor_shift_signal(delta))

    console.print()
    console.print(ct)

    # Corridor width interpretation
    if len(rows) >= 2:
        sofr_row = next((r for r in rows if "SOFR" in r[0]), None)
        effr_row = next((r for r in rows if "EFFR" in r[0]), None)
        iorb_row = next((r for r in rows if "IORB" in r[0]), None)
        if sofr_row and effr_row:
            basis_now = (sofr_row[1] - effr_row[1]) * 100
            if iorb_row:
                corridor_now = (sofr_row[1] - iorb_row[1]) * 100
                if corridor_now > 10:
                    interp = "[red]SOFR trading well above IORB — funding stress in secured markets[/red]"
                elif corridor_now > 5:
                    interp = "[yellow]Corridor widening — repo pressure building[/yellow]"
                elif corridor_now < -5:
                    interp = "[green]SOFR below IORB — excess reserves keeping rates pinned[/green]"
                else:
                    interp = "[dim]Corridor normal — rates within expected band[/dim]"
                console.print(f"  SOFR-IORB corridor: {corridor_now:+.1f}bp  {interp}")
            if abs(basis_now) > 5:
                if basis_now > 0:
                    console.print(f"  SOFR-EFFR basis: {basis_now:+.1f}bp — [yellow]secured > unsecured, collateral demand[/yellow]")
                else:
                    console.print(f"  SOFR-EFFR basis: {basis_now:+.1f}bp — [green]secured < unsecured, repo functioning[/green]")


# ─────────────────────────────────────────────────────────────────────────────
# Structural liquidity table
# ─────────────────────────────────────────────────────────────────────────────

def _structural_signal(name, value, chg_13w):
    """Interpret structural liquidity level + 13-week momentum."""
    if name == "on_rrp":
        v_tn = float(value) / 1_000_000 if value else 0  # millions -> trillions
        if v_tn < 0.05:
            level = "[red]depleted[/red]"
        elif v_tn < 0.2:
            level = "[yellow]low[/yellow]"
        elif v_tn < 0.5:
            level = "[dim]moderate[/dim]"
        else:
            level = "[green]ample[/green]"
        if chg_13w is not None:
            chg_bn = float(chg_13w) / 1000
            if chg_bn < -100:
                return f"{level} + [red]draining fast (-${abs(chg_bn):.0f}B/13w)[/red]"
            if chg_bn < -20:
                return f"{level} + [yellow]draining (-${abs(chg_bn):.0f}B/13w)[/yellow]"
        return level
    elif name == "reserves":
        v_tn = float(value) / 1_000_000 if value else 0
        if v_tn < 2.5:
            level = "[red]scarce[/red]"
        elif v_tn < 3.0:
            level = "[yellow]adequate[/yellow]"
        elif v_tn < 3.5:
            level = "[dim]comfortable[/dim]"
        else:
            level = "[green]abundant[/green]"
        if chg_13w is not None:
            chg_bn = float(chg_13w) / 1000
            if chg_bn < -100:
                return f"{level} + [red]declining fast (-${abs(chg_bn):.0f}B/13w)[/red]"
            if chg_bn < -30:
                return f"{level} + [yellow]declining (-${abs(chg_bn):.0f}B/13w)[/yellow]"
            if chg_bn > 50:
                return f"{level} + [green]growing (+${chg_bn:.0f}B/13w)[/green]"
        return level
    elif name == "fed_assets":
        if chg_13w is not None:
            chg_bn = float(chg_13w) / 1000
            if chg_bn < -20:
                return f"[yellow]QT active (-${abs(chg_bn):.0f}B/13w)[/yellow]"
            if chg_bn > 20:
                return f"[green]expanding (+${chg_bn:.0f}B/13w)[/green]"
            return f"[dim]flat ({chg_bn:+.0f}B/13w)[/dim]"
        return "[dim]—[/dim]"
    return "[dim]—[/dim]"


def _show_structural_liquidity(console, fi):
    """Show structural liquidity dashboard: reserves, RRP, TGA, Fed assets."""
    from rich.table import Table as RichTable

    def _fmt_tn(v):
        return f"${float(v) / 1_000_000:.2f}T" if isinstance(v, (int, float)) else "n/a"

    def _fmt_bn(v):
        return f"${float(v) / 1000:.0f}B" if isinstance(v, (int, float)) else "n/a"

    rows = []
    if fi.bank_reserves_usd_bn is not None:
        rows.append((
            "Bank Reserves",
            _fmt_tn(fi.bank_reserves_usd_bn),
            _structural_signal("reserves", fi.bank_reserves_usd_bn, fi.bank_reserves_chg_13w),
        ))
    if fi.on_rrp_usd_bn is not None:
        rows.append((
            "ON RRP",
            _fmt_bn(fi.on_rrp_usd_bn),
            _structural_signal("on_rrp", fi.on_rrp_usd_bn, fi.on_rrp_chg_13w),
        ))
    if fi.fed_assets_usd_bn is not None:
        rows.append((
            "Fed Balance Sheet",
            _fmt_tn(fi.fed_assets_usd_bn),
            _structural_signal("fed_assets", fi.fed_assets_usd_bn, fi.fed_assets_chg_13w),
        ))
    if fi.tga_usd_bn is not None:
        tga_ctx = _tga_ctx(fi.tga_chg_4w)
        tga_chg = f"  4w Δ: ${float(fi.tga_chg_4w)/1000:+.0f}B" if fi.tga_chg_4w is not None else ""
        rows.append((
            "TGA (Treasury)",
            _fmt_bn(fi.tga_usd_bn),
            f"{tga_ctx}{tga_chg}",
        ))

    if not rows:
        return

    st = RichTable(
        title="Structural Liquidity",
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 1),
    )
    st.add_column("Component", min_width=18)
    st.add_column("Level", justify="right", min_width=10)
    st.add_column("Signal", min_width=30)

    for name, val, signal in rows:
        st.add_row(name, val, signal)

    console.print()
    console.print(st)

    # Scarcity warning
    reserves_tn = float(fi.bank_reserves_usd_bn) / 1_000_000 if fi.bank_reserves_usd_bn else 0
    rrp_bn = float(fi.on_rrp_usd_bn) / 1000 if fi.on_rrp_usd_bn else 0
    if reserves_tn < 3.0 and rrp_bn < 100:
        console.print("  [bold red]⚠ Reserve scarcity zone: RRP depleted + reserves below $3T — funding stress likely[/bold red]")
    elif reserves_tn < 3.0:
        console.print("  [yellow]⚠ Reserves approaching scarcity — watch for quarter-end stress[/yellow]")
    elif rrp_bn < 50:
        console.print("  [yellow]⚠ RRP nearly depleted — excess liquidity buffer gone[/yellow]")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-regime signals
# ─────────────────────────────────────────────────────────────────────────────

def _show_cross_regime_signals(console, fi, funding_score):
    """Show cross-regime confirmation/divergence signals for funding."""
    lines: list[str] = []

    try:
        from lox.data.regime_history import get_score_series
        for domain, display in [("credit", "Credit"), ("volatility", "Vol"), ("rates", "Rates"), ("growth", "Growth")]:
            series = get_score_series(domain)
            if not series:
                continue
            latest = series[-1]
            sc = latest.get("score")
            lb = latest.get("label", "")
            if not isinstance(sc, (int, float)):
                continue
            short_lb = lb.split("(")[0].strip() if "(" in lb else lb

            if domain == "credit":
                if sc > 55 and funding_score > 60:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) + funding stress → [red]liquidity crunch confirmed across credit + funding[/red]")
                elif sc > 55 and funding_score < 45:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) but funding calm → [yellow]credit stress not yet in plumbing[/yellow]")
                elif sc < 35 and funding_score > 60:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]funding stressed but credit calm — plumbing issue, not solvency[/yellow]")
                else:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [dim]neutral[/dim]")
            elif domain == "volatility":
                if sc > 60 and funding_score > 60:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [red]vol + funding stress — risk-off cascade risk[/red]")
                elif sc > 60 and funding_score < 45:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]vol elevated, funding okay — equity-led, not systemic[/yellow]")
                elif sc < 30 and funding_score > 60:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]vol suppressed but funding stressed — hidden plumbing risk[/yellow]")
                else:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [dim]neutral[/dim]")
            elif domain == "rates":
                if sc > 60 and funding_score > 55:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]rates + funding stress — Treasury supply pressure[/yellow]")
                else:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [dim]neutral[/dim]")
            elif domain == "growth":
                if sc > 65 and funding_score > 55:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [red]growth slowing + funding tight — recession risk[/red]")
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

def run_funding_snapshot(
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
    """
    Shared implementation for funding snapshot/outlook commands.
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

    # Fetch full DataFrame for sparklines/corridor analysis
    try:
        funding_df = build_funding_dataset(settings=settings, start_date=start, refresh=refresh)
    except Exception:
        funding_df = None

    def _fmt_pct(x: object) -> str:
        return f"{float(x):.2f}%" if isinstance(x, (int, float)) else "n/a"

    def _fmt_bps(x: object) -> str:
        return f"{float(x):+.1f}bp" if isinstance(x, (int, float)) else "n/a"

    def _fmt_ratio(x: object) -> str:
        return f"{100.0*float(x):.0f}%" if isinstance(x, (int, float)) else "n/a"

    regime = classify_funding_regime(fi)

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

    vec = funding_feature_vector(state)
    feature_dict = vec.features

    # Handle --features and --json flags
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

    if alert:
        show_alert_output("funding", regime.label or regime.name, snapshot_data, regime.description)
        return

    if calendar:
        print(Panel.fit(f"[b]Regime:[/b] {regime.label or regime.name}", title="US Funding", border_style="cyan"))
        show_calendar_output("funding")
        return

    if trades:
        print(Panel.fit(f"[b]Regime:[/b] {regime.label or regime.name}", title="US Funding", border_style="cyan"))
        show_trades_output("funding", regime.label or regime.name)
        return

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

    # ── Main panel ────────────────────────────────────────────────────────
    score = regime.score
    corridor_name = fi.spread_corridor_name or "SOFR-EFFR"

    def _fmt_usd_bn(x: object, div: float = 1000.0) -> str:
        return f"${float(x) / div:,.0f}B" if isinstance(x, (int, float)) else "n/a"

    def _fmt_usd_tn(x: object, div: float = 1_000_000.0) -> str:
        return f"${float(x) / div:.1f}T" if isinstance(x, (int, float)) else "n/a"

    metrics = [
        {"name": "─── Rates ───", "value": "", "context": ""},
        {"name": "SOFR", "value": _fmt_pct(fi.sofr), "context": "secured overnight"},
        {"name": "EFFR", "value": _fmt_pct(fi.effr), "context": "unsecured overnight"},
        {"name": f"Corridor ({corridor_name})", "value": _fmt_bps(fi.spread_corridor_bps), "context": _corridor_ctx(fi.spread_corridor_bps)},
        {"name": "SOFR–EFFR", "value": _fmt_bps(fi.spread_sofr_effr_bps), "context": _sofr_effr_ctx(fi.spread_sofr_effr_bps)},
        {"name": "─── Stress ───", "value": "", "context": ""},
        {"name": "Spike 5d", "value": _fmt_bps(fi.spike_5d_bps), "context": _spike_ctx(fi.spike_5d_bps)},
        {"name": "Vol 20d", "value": _fmt_bps(fi.vol_20d_bps), "context": _vol_ctx(fi.vol_20d_bps)},
        {"name": "Persistence 20d", "value": _fmt_ratio(fi.persistence_20d), "context": _persist_ctx(fi.persistence_20d)},
        {"name": "─── Structural ───", "value": "", "context": ""},
        {"name": "ON RRP", "value": _fmt_usd_bn(fi.on_rrp_usd_bn), "context": _rrp_ctx(fi.on_rrp_usd_bn)},
        {"name": "Bank Reserves", "value": _fmt_usd_tn(fi.bank_reserves_usd_bn), "context": _reserves_ctx(fi.bank_reserves_usd_bn)},
        {"name": "TGA 4wk Δ", "value": _fmt_usd_bn(fi.tga_chg_4w) if fi.tga_chg_4w is not None else "n/a", "context": _tga_ctx(fi.tga_chg_4w)},
    ]

    from lox.regimes.trend import get_domain_trend
    trend = get_domain_trend("liquidity", score, regime.label or regime.name)

    print(render_regime_panel(
        domain="Funding",
        asof=state.asof,
        regime_label=regime.label or regime.name,
        score=score,
        percentile=None,
        description=regime.description,
        metrics=metrics,
        trend=trend,
    ))

    # ── Block 1: Spread velocity sparklines ───────────────────────────────
    _show_rate_sparklines(console, funding_df, fi)

    # ── Block 2: Rate corridor with 30d shifts ───────────────────────────
    _show_rate_corridor(console, funding_df)

    # ── Block 3: Structural liquidity dashboard ──────────────────────────
    _show_structural_liquidity(console, fi)

    # ── Block 4: Cross-regime signals ────────────────────────────────────
    _show_cross_regime_signals(console, fi, score)

    if llm:
        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis
        print_llm_regime_analysis(
            settings=settings,
            domain="funding",
            snapshot=snapshot_data,
            regime_label=regime.label or regime.name,
            regime_description=regime.description,
            ticker=ticker,
        )


def funding_snapshot(**kwargs) -> None:
    """Entry point for `lox regime funding` (no subcommand)."""
    run_funding_snapshot(**kwargs)


def register(funding_app: typer.Typer) -> None:
    @funding_app.callback(invoke_without_command=True)
    def funding_default(
        ctx: typer.Context,
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
        """Short-term funding markets (SOFR, repo spreads) — price of money in daily markets"""
        if ctx.invoked_subcommand is None:
            run_funding_snapshot(refresh=refresh, llm=llm, ticker=ticker, features=features, json_out=json_out, delta=delta, alert=alert, calendar=calendar, trades=trades)

    @funding_app.command("snapshot")
    def funding_snapshot(
        start: str = typer.Option("2011-01-01", "--start", help="Start date YYYY-MM-DD"),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh FRED downloads"),
        features: bool = typer.Option(False, "--features", help="Export ML-ready feature vector (JSON)"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
        delta: str = typer.Option("", "--delta", help="Show changes vs N days ago (e.g., 7d, 1w, 1m)"),
        llm: bool = typer.Option(False, "--llm", help="Chat with LLM analyst"),
        ticker: str = typer.Option("", "--ticker", "-t", help="Ticker for focused chat (used with --llm)"),
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
        run_funding_snapshot(start=start, refresh=bool(refresh), features=bool(features), json_out=bool(json_out), delta=delta, llm=bool(llm), ticker=ticker, alert=alert, calendar=calendar, trades=trades)

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

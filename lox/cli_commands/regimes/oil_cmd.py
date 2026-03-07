"""
CLI command for the Oil & Energy Chokepoint regime.

Deep-dive into crude oil prices (WTI, Brent), spreads, volatility,
and multi-chokepoint shipping traffic from IMF PortWatch.
Tracks Hormuz, Bab el-Mandeb, Suez, Malacca, Bosporus, and Cape of Good Hope.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from rich import print as rprint

logger = logging.getLogger(__name__)

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
        return "[red]rising[/red]" if delta > 0 else "[green]falling[/green]"
    return "[green]rising[/green]" if delta > 0 else "[red]falling[/red]"


# ─────────────────────────────────────────────────────────────────────────────
# Oil data fetching
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_oil_live(settings) -> dict:
    """
    Fetch live WTI/Brent from FMP when available.
    Uses CL=F/BZ=F (futures) or USO/BNO (ETFs) as proxies.
    Returns {wti, brent, asof} or empty dict on failure.
    """
    if not getattr(settings, "FMP_API_KEY", None):
        return {}
    import requests
    wti_syms = {"CL=F", "CLUSD", "USO"}
    brent_syms = {"BZ=F", "BZUSD", "BNO"}
    for symbols in [["CL=F", "BZ=F"], ["USO", "BNO"]]:
        try:
            url = "https://financialmodelingprep.com/api/v3/quote/" + ",".join(symbols)
            resp = requests.get(url, params={"apikey": settings.FMP_API_KEY}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                continue
            out = {}
            for item in data:
                sym = (item.get("symbol") or "").upper()
                price = item.get("price")
                if price is not None:
                    try:
                        p = float(price)
                        if sym in wti_syms:
                            out["wti"] = p
                        elif sym in brent_syms:
                            out["brent"] = p
                    except (ValueError, TypeError):
                        pass
            if out.get("wti") is not None:
                out["asof"] = "live"
                return out
        except Exception:
            continue
    return {}


def _fetch_oil_data(settings, refresh: bool = False) -> dict:
    """Fetch WTI and Brent crude data from FRED."""
    from lox.data.fred import FredClient
    import pandas as pd

    fred = FredClient(api_key=settings.FRED_API_KEY)
    result = {
        "wti_df": None, "brent_df": None,
        "wti": None, "brent": None,
        "wti_5d_chg": None, "wti_30d_chg": None, "wti_90d_chg": None,
        "brent_5d_chg": None, "brent_30d_chg": None, "brent_90d_chg": None,
        "brent_wti_spread": None, "brent_wti_spread_30d_ago": None,
        "wti_vol_20d": None, "brent_vol_20d": None,
        "wti_1y_pctl": None, "wti_1y_high": None, "wti_1y_low": None,
        "asof": "—",
    }

    for series_id, key in [("DCOILWTICO", "wti"), ("DCOILBRENTEU", "brent")]:
        try:
            df = fred.fetch_series(series_id, start_date="2020-01-01", refresh=refresh)
            if df is None or df.empty:
                continue
            df = df.sort_values("date").dropna(subset=["value"])
            result[f"{key}_df"] = df
            result[key] = float(df["value"].iloc[-1])
            if key == "wti":
                result["asof"] = str(df["date"].iloc[-1].date())

            vals = df["value"].astype(float)
            if len(vals) >= 5:
                result[f"{key}_5d_chg"] = float(vals.iloc[-1] - vals.iloc[-5])
            if len(vals) >= 22:
                result[f"{key}_30d_chg"] = float(vals.iloc[-1] - vals.iloc[-22])
            if len(vals) >= 63:
                result[f"{key}_90d_chg"] = float(vals.iloc[-1] - vals.iloc[-63])

            # 20-day realized volatility (annualized)
            if len(vals) >= 22:
                rets = vals.pct_change().dropna().tail(20)
                result[f"{key}_vol_20d"] = float(rets.std() * (252 ** 0.5) * 100)

            # 1-year percentile and range
            if key == "wti" and len(vals) >= 252:
                last_1y = vals.iloc[-252:]
                result["wti_1y_pctl"] = float((last_1y <= vals.iloc[-1]).mean() * 100)
                result["wti_1y_high"] = float(last_1y.max())
                result["wti_1y_low"] = float(last_1y.min())
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", series_id, e)

    # Brent-WTI spread
    if result["brent"] is not None and result["wti"] is not None:
        result["brent_wti_spread"] = result["brent"] - result["wti"]
    if result["brent_df"] is not None and result["wti_df"] is not None:
        try:
            bdf = result["brent_df"].set_index("date")["value"]
            wdf = result["wti_df"].set_index("date")["value"]
            spread = (bdf - wdf).dropna()
            if len(spread) >= 22:
                result["brent_wti_spread_30d_ago"] = float(spread.iloc[-22])
        except Exception:
            pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Multi-chokepoint shipping data (via lox.data.shipping)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_all_chokepoints(days: int = 365, marinetraffic_key: str | None = None):
    """Fetch all oil-relevant chokepoints and compute composite disruption."""
    from lox.data.shipping import fetch_oil_chokepoints, compute_composite_disruption, get_hormuz_compat
    chokepoints = fetch_oil_chokepoints(days=days, marinetraffic_key=marinetraffic_key)
    composite = compute_composite_disruption(chokepoints)
    hz = get_hormuz_compat(chokepoints)
    return chokepoints, composite, hz


# ─────────────────────────────────────────────────────────────────────────────
# Context helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wti_ctx(price):
    if price is None:
        return "WTI benchmark"
    if price > 100:
        return "crisis premium — inflation risk"
    if price > 85:
        return "elevated — cost-push pressure"
    if price > 70:
        return "normal range"
    if price > 55:
        return "soft — deflationary impulse"
    return "depressed — demand destruction"


def _brent_ctx(price):
    if price is None:
        return "Brent benchmark"
    if price > 105:
        return "crisis — global supply disruption"
    if price > 90:
        return "elevated — geopolitical premium"
    if price > 75:
        return "normal range"
    if price > 60:
        return "soft — demand weakness"
    return "depressed — oversupply"


def _spread_ctx(spread):
    if spread is None:
        return "Brent-WTI basis"
    if spread > 10:
        return "wide — global tightness vs US supply"
    if spread > 5:
        return "normal — Brent premium for geography"
    if spread > 2:
        return "narrow — US export parity"
    if spread > -1:
        return "tight — WTI at parity"
    return "inverted — US tighter than global (unusual)"


def _vol_ctx(vol):
    if vol is None:
        return "realized vol"
    if vol > 50:
        return "extreme — crisis volatility"
    if vol > 35:
        return "elevated — event risk"
    if vol > 25:
        return "above average"
    if vol > 15:
        return "normal range"
    return "suppressed — complacency?"


def _pctl_ctx(pctl):
    if pctl is None:
        return "vs 1Y history"
    if pctl > 90:
        return "near 1Y highs"
    if pctl > 75:
        return "elevated vs history"
    if pctl > 50:
        return "above median"
    if pctl > 25:
        return "below median — soft"
    return "near 1Y lows"


def _tanker_ctx(latest, avg_30d):
    if latest is None or avg_30d is None:
        return "tanker transits"
    ratio = latest / avg_30d if avg_30d > 0 else 1.0
    if ratio < 0.25:
        return "[bold red]strait effectively shut — blockade[/bold red]"
    if ratio < 0.5:
        return "[red]severe drop — blockade/disruption[/red]"
    if ratio < 0.7:
        return "[red]below average — rerouting or slowdown[/red]"
    if ratio < 0.95:
        return "slightly below norm"
    if ratio < 1.05:
        return "normal flow"
    if ratio < 1.2:
        return "above average — pre-sanctions buildup?"
    return "surge — unusual activity"


# ─────────────────────────────────────────────────────────────────────────────
# Display blocks
# ─────────────────────────────────────────────────────────────────────────────

def _show_oil_sparklines(console, oil):
    """Show 30d sparklines for WTI, Brent, and spread."""
    lines: list[str] = []

    for key, label in [("wti", "WTI"), ("brent", "Brent")]:
        df = oil.get(f"{key}_df")
        if df is not None and len(df) >= 10:
            vals = [float(v) for v in df["value"].dropna().tail(30)]
            if len(vals) >= 10:
                cur = f"${vals[-1]:.2f}"
                lines.append(
                    f"  {label:<8} {_sparkline(vals)}  {cur}  {_trend_label(vals, higher_is_worse=True)}"
                )

    # Spread sparkline
    if oil["wti_df"] is not None and oil["brent_df"] is not None:
        try:
            import pandas as pd
            bdf = oil["brent_df"].set_index("date")["value"]
            wdf = oil["wti_df"].set_index("date")["value"]
            spread = (bdf - wdf).dropna().tail(30)
            if len(spread) >= 10:
                vals = [float(v) for v in spread]
                lines.append(
                    f"  {'B-W Sprd':<8} {_sparkline(vals)}  ${vals[-1]:.2f}  {_trend_label(vals)}"
                )
        except Exception:
            pass

    if lines:
        console.print()
        console.print("[dim]─── Price Velocity (30d) ──────────────────────────────────────[/dim]")
        for ln in lines:
            console.print(ln)


def _show_chokepoint_sparklines(console, chokepoints):
    """Compact chokepoint sparklines — just trends and stress flags."""
    if not chokepoints:
        console.print("  [yellow]PortWatch data unavailable[/yellow]")
        return

    for key in ["hormuz", "bab_el_mandeb", "suez", "malacca", "bosporus", "cape"]:
        cd = chokepoints.get(key)
        if cd is None:
            continue
        vals = cd.tanker_values_30d
        if len(vals) < 10:
            continue

        avg_7 = cd.avg_7d_tankers or 0
        baseline = cd.baseline_tankers
        if baseline and baseline > 0:
            pct_vs = ((avg_7 / baseline) - 1) * 100
        else:
            avg_30 = cd.avg_30d_tankers or 0
            pct_vs = ((avg_7 / avg_30) - 1) * 100 if avg_30 > 0 else 0
        color = "green" if pct_vs >= -5 else ("yellow" if pct_vs >= -20 else "red")

        flag = ""
        if cd.disruption_score >= 30:
            flag = f"  [bold red]⚠[/bold red]"
        elif cd.disruption_score >= 15:
            flag = f"  [yellow]![/yellow]"

        console.print(
            f"  {cd.short:<13} {_sparkline(vals)}  "
            f"{avg_7:>3.0f}  [{color}]{pct_vs:+4.0f}%[/{color}]{flag}"
        )


def _show_signals(console, oil, chokepoints, composite):
    """Show only actionable signals — skip anything neutral."""
    if not chokepoints or oil.get("wti") is None:
        return

    lines: list[str] = []
    wti_30d_chg = oil.get("wti_30d_chg")

    hz = chokepoints.get("hormuz")
    if hz and hz.avg_7d_tankers and hz.baseline_tankers and hz.baseline_tankers > 0:
        ratio = hz.avg_7d_tankers / hz.baseline_tankers
        if ratio < 0.8 and wti_30d_chg is not None and wti_30d_chg > 5:
            lines.append("  Hormuz ↓ + WTI ↑ → [red]supply disruption confirmed[/red]")
        elif ratio < 0.8 and (wti_30d_chg is None or wti_30d_chg < 2):
            lines.append("  Hormuz ↓ but WTI flat → [yellow]rerouting absorbed[/yellow]")

    bab = chokepoints.get("bab_el_mandeb")
    suez = chokepoints.get("suez")
    if bab and bab.disruption_score >= 15 and suez and suez.disruption_score >= 15:
        lines.append(f"  Red Sea stressed: BaM {bab.disruption_score} + Suez {suez.disruption_score} → [yellow]shipping costs elevated[/yellow]")
    elif bab and bab.disruption_score >= 30:
        lines.append(f"  Bab el-Mandeb disrupted ({bab.disruption_score}) → [red]Red Sea risk[/red]")

    if composite.rerouting_detected:
        lines.append(f"  [yellow]⚡ Rerouting: Cape GH traffic up while Red Sea down[/yellow]")

    disrupted_count = sum(1 for k, cd in chokepoints.items()
                         if not cd.is_reroute_indicator and cd.disruption_score >= 15)
    if disrupted_count >= 3:
        lines.append(f"  [bold red]⚠ {disrupted_count} chokepoints disrupted — systemic supply risk[/bold red]")

    try:
        from lox.data.regime_history import get_score_series
        for domain, display in [("credit", "Credit"), ("volatility", "Vol"), ("growth", "Growth")]:
            series = get_score_series(domain)
            if not series:
                continue
            sc = series[-1].get("score")
            if not isinstance(sc, (int, float)):
                continue
            if domain == "growth" and sc > 60 and wti_30d_chg is not None and wti_30d_chg < -10:
                lines.append(f"  Growth {sc:.0f} + WTI falling → [yellow]demand destruction[/yellow]")
            elif domain == "credit" and sc > 55 and wti_30d_chg is not None and wti_30d_chg > 10:
                lines.append(f"  Credit {sc:.0f} + oil spike → [red]stagflationary[/red]")
            elif domain == "volatility" and sc > 55:
                lines.append(f"  Vol elevated ({sc:.0f})")
    except Exception:
        pass

    if lines:
        console.print()
        console.print("[dim]─── Signals ───────────────────────────────────────────────────[/dim]")
        for ln in lines:
            console.print(ln)


# ─────────────────────────────────────────────────────────────────────────────
# Core implementation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_disruption_score(hz: dict | None) -> tuple[int, str, dict]:
    """Legacy wrapper — kept for backward compatibility with snapshot delta logic."""
    details: dict = {}
    if hz is None:
        return 0, "Unknown", details
    # Extract from backward-compat hz dict (populated by get_hormuz_compat)
    details["baseline_tankers"] = hz.get("_baseline_tankers")
    details["current_tankers_7d"] = hz.get("n_tanker_7d_avg")
    details["baseline_cap"] = hz.get("_baseline_cap")
    details["current_cap_7d"] = hz.get("capacity_tanker_7d_avg")
    tanker_7d = hz.get("n_tanker_7d_avg", 0)
    baseline = hz.get("_baseline_tankers", 0)
    if baseline and baseline > 0 and tanker_7d:
        details["transit_pct_of_baseline"] = (tanker_7d / baseline) * 100
    return hz.get("_disruption_score", 0), hz.get("_disruption_label", "Unknown"), details


def oil_snapshot(
    *,
    llm: bool = False,
    ticker: str = "",
    refresh: bool = False,
    delta: str = "",
    alert: bool = False,
) -> None:
    """Entry point for `lox regime oil`."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from lox.config import load_settings
    from lox.cli_commands.shared.labs_utils import save_snapshot

    console = Console()
    settings = load_settings()

    # ── Fetch oil data ────────────────────────────────────────────────────
    oil = _fetch_oil_data(settings, refresh=refresh)
    live = _fetch_oil_live(settings)
    if live.get("wti") is not None:
        oil["wti"] = live["wti"]
        oil["asof"] = "live"
        if live.get("brent") is not None:
            oil["brent"] = live["brent"]
            oil["brent_wti_spread"] = live["brent"] - live["wti"]

    # ── Fetch all oil chokepoints ────────────────────────────────────────
    mt_key = getattr(settings, "MARINETRAFFIC_API_KEY", None)
    console.print("[dim]Fetching oil chokepoint transit data (Hormuz, Bab el-Mandeb, Suez, Malacca, Bosporus, Cape GH)...[/dim]")
    chokepoints, composite, hz = _fetch_all_chokepoints(days=365, marinetraffic_key=mt_key)

    # Use composite score as the primary disruption metric
    disruption_score = composite.score
    disruption_label = composite.label

    # Build backward-compat disruption_details from Hormuz data
    hz_cd = chokepoints.get("hormuz")
    disruption_details: dict = {}
    if hz_cd:
        disruption_details["baseline_tankers"] = hz_cd.baseline_tankers
        disruption_details["current_tankers_7d"] = hz_cd.avg_7d_tankers
        disruption_details["baseline_cap"] = hz_cd.baseline_cap_tanker
        disruption_details["current_cap_7d"] = hz_cd.avg_7d_cap_tanker
        if hz_cd.baseline_tankers and hz_cd.baseline_tankers > 0 and hz_cd.avg_7d_tankers:
            disruption_details["transit_pct_of_baseline"] = (hz_cd.avg_7d_tankers / hz_cd.baseline_tankers) * 100
        disruption_details["trajectory"] = hz_cd.trajectory
        disruption_details["trajectory_pct"] = hz_cd.trajectory_pct
        # Inject into hz dict for legacy compat
        if hz:
            hz["_baseline_tankers"] = hz_cd.baseline_tankers
            hz["_baseline_cap"] = hz_cd.baseline_cap_tanker
            hz["_disruption_score"] = hz_cd.disruption_score
            hz["_disruption_label"] = hz_cd.disruption_label

    # ── Build snapshot for persistence ───────────────────────────────────
    snapshot_data = {
        "wti": oil["wti"],
        "brent": oil["brent"],
        "brent_wti_spread": oil.get("brent_wti_spread"),
        "wti_5d_chg": oil.get("wti_5d_chg"),
        "wti_30d_chg": oil.get("wti_30d_chg"),
        "wti_vol_20d": oil.get("wti_vol_20d"),
        "wti_1y_pctl": oil.get("wti_1y_pctl"),
        "disruption_score": disruption_score,
        "disruption_label": disruption_label,
        "composite_disruption_score": composite.score,
        "rerouting_detected": composite.rerouting_detected,
    }
    for key in ["hormuz", "bab_el_mandeb", "suez", "malacca", "bosporus", "cape"]:
        cd = chokepoints.get(key)
        if cd:
            snapshot_data[f"{key}_tanker_7d_avg"] = cd.avg_7d_tankers
            snapshot_data[f"{key}_tanker_30d_avg"] = cd.avg_30d_tankers
            snapshot_data[f"{key}_disruption_score"] = cd.disruption_score
    if hz:
        snapshot_data["hormuz_total_7d_avg"] = hz.get("n_total_7d_avg")
        snapshot_data["hormuz_cap_tanker_7d_avg"] = hz.get("capacity_tanker_7d_avg")

    save_snapshot("oil", snapshot_data, disruption_label)

    # ── Handle --alert flag ───────────────────────────────────────────────
    if alert:
        if disruption_score >= 30 or (oil.get("wti_30d_chg") and oil["wti_30d_chg"] > 15):
            console.print(f"[bold red]OIL ALERT[/bold red]: {disruption_label} (score {disruption_score}/100)")
            if oil["wti"]:
                console.print(f"  WTI ${oil['wti']:.2f}  30d chg: ${oil.get('wti_30d_chg', 0):+.2f}")
            if hz:
                t7 = hz.get("n_tanker_7d_avg", 0)
                t30 = hz.get("n_tanker_30d_avg", 0)
                console.print(f"  Hormuz tankers: {t7:.0f}/day (7d avg) vs {t30:.0f}/day (30d avg)")
        return

    # ── Handle --delta flag ───────────────────────────────────────────────
    if delta:
        from lox.cli_commands.shared.labs_utils import parse_delta_period, get_delta_metrics, show_delta_summary

        delta_days = parse_delta_period(delta)
        metric_keys = [
            "WTI:wti:$",
            "Brent:brent:$",
            "B-W Spread:brent_wti_spread:$",
            "WTI Vol:wti_vol_20d:%",
            "Disruption Score:disruption_score:",
            "Hormuz Tankers (7d):hormuz_tanker_7d_avg:",
            "Hormuz Tankers (30d):hormuz_tanker_30d_avg:",
        ]
        metrics_for_delta, prev_regime = get_delta_metrics("oil", snapshot_data, metric_keys, delta_days)
        show_delta_summary("oil", disruption_label, prev_regime, metrics_for_delta, delta_days)
        if prev_regime is None:
            console.print(f"\n[dim]No cached data from {delta_days}d ago. Run `lox regime oil` daily to build history.[/dim]")
        return

    # ── Build compact panel ─────────────────────────────────────────────
    wti = oil["wti"]
    brent = oil["brent"]
    spread = oil.get("brent_wti_spread")

    # Oil sentiment
    if wti is not None:
        if wti > 90:
            sentiment = "[bold red]Supply Stress[/bold red]"
        elif wti > 75:
            sentiment = "[yellow]Elevated[/yellow]"
        elif wti > 60:
            sentiment = "[green]Normal[/green]"
        elif wti > 45:
            sentiment = "[cyan]Soft[/cyan]"
        else:
            sentiment = "[bold cyan]Demand Destruction[/bold cyan]"
    else:
        sentiment = "[dim]Unknown[/dim]"

    # Price line
    price_parts = []
    if wti is not None:
        chg_30 = oil.get("wti_30d_chg")
        chg_str = f" ({chg_30:+.0f} 30d)" if chg_30 is not None else ""
        price_parts.append(f"WTI ${wti:.2f}{chg_str}")
    if brent is not None:
        price_parts.append(f"Brent ${brent:.2f}")
    if spread is not None:
        price_parts.append(f"Sprd ${spread:.2f}")
    price_line = " | ".join(price_parts) if price_parts else "No data"

    # Context line
    ctx_parts = [sentiment]
    vol = oil.get("wti_vol_20d")
    if vol is not None:
        v_color = "red" if vol > 35 else ("yellow" if vol > 25 else "dim")
        ctx_parts.append(f"[{v_color}]Vol {vol:.0f}%[/{v_color}]")
    pctl = oil.get("wti_1y_pctl")
    if pctl is not None:
        ctx_parts.append(f"[dim]{pctl:.0f}th pctl[/dim]")

    # Disruption line
    d_color = "bold red" if disruption_score >= 50 else ("red" if disruption_score >= 30 else ("yellow" if disruption_score >= 15 else "green"))
    traj = composite.trajectory
    t_arrow = {"recovering": "↗", "worsening": "↘"}.get(traj, "→")
    disruption_line = f"Supply Disruption: [{d_color}]{disruption_score}/100 {disruption_label}[/{d_color}]  {t_arrow} {traj}"

    # Per-chokepoint mini scores
    cp_parts = []
    for key, label in [("hormuz", "Hz"), ("bab_el_mandeb", "BaM"), ("suez", "Sz"), ("malacca", "Ml"), ("bosporus", "Bs")]:
        sc = composite.per_chokepoint.get(key)
        if sc is not None:
            c = "red" if sc >= 30 else ("yellow" if sc >= 15 else "green")
            cp_parts.append(f"[{c}]{label}:{sc}[/{c}]")
    cp_line = "  ".join(cp_parts)

    # Build panel text
    panel_lines = [
        f"{price_line}",
        f"{'  '.join(ctx_parts)}",
        "",
        disruption_line,
        f"  {cp_line}",
    ]
    if composite.rerouting_detected:
        panel_lines.append("  [yellow]⚡ Rerouting detected — Cape GH absorbing Red Sea diversion[/yellow]")

    panel_text = Text.from_markup("\n".join(panel_lines))
    panel = Panel(
        panel_text,
        title="[bold]Oil & Energy Chokepoint[/bold]",
        subtitle=f"[dim]{oil['asof']}[/dim]",
        border_style="yellow",
        padding=(1, 2),
    )
    rprint(panel)

    # ── Chokepoint sparklines ─────────────────────────────────────────────
    console.print()
    any_asof = next((cd.asof for cd in chokepoints.values() if cd.asof), "?")
    console.print(f"[dim]─── Chokepoint Trends (30d tankers/day, AIS {any_asof}) ────────[/dim]")
    _show_chokepoint_sparklines(console, chokepoints)

    # ── Actionable signals only ───────────────────────────────────────────
    _show_signals(console, oil, chokepoints, composite)

    if llm:
        llm_data = {
            "wti": wti,
            "brent": brent,
            "brent_wti_spread": spread,
            "wti_5d_chg": oil.get("wti_5d_chg"),
            "wti_30d_chg": oil.get("wti_30d_chg"),
            "wti_90d_chg": oil.get("wti_90d_chg"),
            "wti_vol_20d": oil.get("wti_vol_20d"),
            "wti_1y_pctl": oil.get("wti_1y_pctl"),
            "composite_disruption_score": composite.score,
            "composite_disruption_label": composite.label,
            "rerouting_detected": composite.rerouting_detected,
        }
        for key in ["hormuz", "bab_el_mandeb", "suez", "malacca", "bosporus", "cape"]:
            cd = chokepoints.get(key)
            if cd:
                llm_data[f"{key}_tankers_7d"] = cd.avg_7d_tankers
                llm_data[f"{key}_disruption"] = cd.disruption_score

        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis
        print_llm_regime_analysis(
            settings=settings,
            domain="oil",
            snapshot=llm_data,
            regime_label=sentiment,
            regime_description="Oil price and multi-chokepoint shipping traffic analysis (Hormuz, Bab el-Mandeb, Suez, Malacca, Bosporus, Cape of Good Hope)",
            ticker=ticker,
        )

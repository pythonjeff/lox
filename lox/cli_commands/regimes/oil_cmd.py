"""
CLI command for the Oil & Energy Chokepoint regime.

Deep-dive into crude oil prices (WTI, Brent), spreads, volatility,
and Strait of Hormuz shipping traffic from IMF PortWatch.
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
# Strait of Hormuz data fetching (IMF PortWatch ArcGIS API)
# ─────────────────────────────────────────────────────────────────────────────

HORMUZ_PORTID = "chokepoint6"
PORTWATCH_BASE = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "Daily_Chokepoints_Data/FeatureServer/0/query"
)


def _fetch_hormuz_data(days: int = 120) -> dict | None:
    """
    Fetch Strait of Hormuz transit data from IMF PortWatch ArcGIS API.
    Returns dict with DataFrame and summary stats, or None on failure.
    """
    import requests
    import pandas as pd

    try:
        all_rows = []
        # Fetch recent data in DESC order, paginate if needed
        for offset in range(0, 2000, 1000):
            resp = requests.get(
                PORTWATCH_BASE,
                params={
                    "where": f"portid='{HORMUZ_PORTID}'",
                    "outFields": "date,n_tanker,n_total,n_container,n_dry_bulk,"
                                 "capacity_tanker,capacity,portname",
                    "outSR": "4326",
                    "f": "json",
                    "resultRecordCount": "1000",
                    "resultOffset": str(offset),
                    "orderByFields": "date DESC",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                break
            features = data.get("features", [])
            if not features:
                break
            all_rows.extend(f["attributes"] for f in features)
            if len(features) < 1000:
                break

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"], unit="ms")
        df = df.sort_values("date").reset_index(drop=True)

        # Trim to requested window
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df["date"] >= pd.Timestamp(cutoff)].reset_index(drop=True)
        if df.empty:
            return None

        result = {"df": df, "asof": str(df["date"].iloc[-1].date())}

        for col in ["n_tanker", "n_total", "capacity_tanker", "capacity"]:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if not s.empty:
                    result[f"{col}_latest"] = float(s.iloc[-1])
                    result[f"{col}_7d_avg"] = float(s.tail(7).mean())
                    result[f"{col}_30d_avg"] = float(s.tail(30).mean())
                    if len(s) >= 60:
                        result[f"{col}_60d_avg"] = float(s.tail(60).mean())
                    result[f"{col}_values"] = [float(v) for v in s.tail(90)]

        return result
    except Exception as e:
        logger.warning("Failed to fetch Hormuz data: %s", e)
        return None


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


def _show_hormuz_panel(console, hz, disruption_details=None):
    """Show Strait of Hormuz shipping traffic dashboard."""
    from rich.table import Table as RichTable

    if hz is None:
        console.print("\n[dim]─── Strait of Hormuz ──────────────────────────────────────────[/dim]")
        console.print("  [yellow]IMF PortWatch data unavailable — check connectivity[/yellow]")
        return

    baseline_tankers = (disruption_details or {}).get("baseline_tankers")

    console.print()
    console.print("[dim]─── Strait of Hormuz (IMF PortWatch) ─────────────────────────[/dim]")
    console.print(f"  [dim]AIS data as of: {hz.get('asof', '?')}[/dim]")

    # Sparklines for tanker transits and total transits
    for col, label in [("n_tanker", "Tankers"), ("n_total", "All Ships")]:
        vals = hz.get(f"{col}_values", [])
        if len(vals) >= 10:
            avg_7 = hz.get(f"{col}_7d_avg", 0)
            # Compare 7d avg vs pre-conflict baseline when available
            if col == "n_tanker" and baseline_tankers and baseline_tankers > 0:
                pct_vs = ((avg_7 / baseline_tankers) - 1) * 100
                ref_label = "pre-conflict"
            else:
                avg_30 = hz.get(f"{col}_30d_avg", 0)
                pct_vs = ((avg_7 / avg_30) - 1) * 100 if avg_30 > 0 else 0
                ref_label = "30d avg"
            color = "green" if pct_vs >= -5 else ("yellow" if pct_vs >= -20 else "red")
            console.print(
                f"  {label:<10} {_sparkline(vals[-30:])}  "
                f"{avg_7:.0f}/day (7d)  "
                f"[{color}]{pct_vs:+.0f}% vs {ref_label}[/{color}]"
            )

    # Capacity sparkline (tanker)
    cap_vals = hz.get("capacity_tanker_values", [])
    if len(cap_vals) >= 10:
        avg_7_cap = hz.get("capacity_tanker_7d_avg", 0)
        baseline_cap = (disruption_details or {}).get("baseline_cap")
        if baseline_cap and baseline_cap > 0:
            pct = ((avg_7_cap / baseline_cap) - 1) * 100
            ref_label = "pre-conflict"
        else:
            avg_30_cap = hz.get("capacity_tanker_30d_avg", 0)
            pct = ((avg_7_cap / avg_30_cap) - 1) * 100 if avg_30_cap > 0 else 0
            ref_label = "30d avg"
        color = "green" if pct >= -5 else ("yellow" if pct >= -20 else "red")
        console.print(
            f"  {'Tnkr Cap':<10} {_sparkline(cap_vals[-30:])}  "
            f"{avg_7_cap/1e6:.1f}M DWT (7d)  "
            f"[{color}]{pct:+.0f}% vs {ref_label}[/{color}]"
        )

    # Stats table
    st = RichTable(
        title="Transit Summary",
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 1),
    )
    st.add_column("Metric", min_width=18)
    st.add_column("Today", justify="right")
    st.add_column("7d Avg", justify="right")
    st.add_column("30d Avg", justify="right")
    st.add_column("Baseline", justify="right")
    st.add_column("Signal")

    b_tankers = (disruption_details or {}).get("baseline_tankers")
    b_cap = (disruption_details or {}).get("baseline_cap")

    for col, label in [("n_tanker", "Tanker Transits"), ("n_total", "Total Transits")]:
        latest = hz.get(f"{col}_latest")
        avg7 = hz.get(f"{col}_7d_avg")
        avg30 = hz.get(f"{col}_30d_avg")
        if avg7 is None:
            continue
        baseline_val = b_tankers if "tanker" in col else None
        # Use Today for Signal when available — 7d avg lags acute events (e.g. invasion yesterday)
        ref = baseline_val or avg30
        if "tanker" in col and ref and ref > 0:
            use_for_signal = latest if latest is not None else avg7
            ctx = _tanker_ctx(use_for_signal, ref)
        else:
            ctx = ""
        st.add_row(
            label,
            f"{latest:.0f}" if latest is not None else "—",
            f"{avg7:.0f}",
            f"{avg30:.0f}" if avg30 is not None else "—",
            f"{baseline_val:.0f}" if baseline_val is not None else "—",
            ctx,
        )

    for col, label in [("capacity_tanker", "Tanker Capacity"), ("capacity", "Total Capacity")]:
        latest = hz.get(f"{col}_latest")
        avg7 = hz.get(f"{col}_7d_avg")
        avg30 = hz.get(f"{col}_30d_avg")
        if avg7 is None:
            continue
        baseline_c = b_cap if "tanker" in col else None
        st.add_row(
            label,
            f"{latest/1e6:.1f}M" if latest is not None else "—",
            f"{avg7/1e6:.1f}M",
            f"{avg30/1e6:.1f}M" if avg30 is not None else "—",
            f"{baseline_c/1e6:.1f}M" if baseline_c is not None else "—",
            "",
        )

    console.print()
    console.print(st)

    # Disruption detection + trajectory (use 7d avg vs baseline, not single-day)
    tanker_7d = hz.get("n_tanker_7d_avg", 0)
    tanker_30d = hz.get("n_tanker_30d_avg", 0)
    tanker_60d = hz.get("n_tanker_60d_avg")
    baseline_t = (disruption_details or {}).get("baseline_tankers", 0)

    ref_val = baseline_t if baseline_t > 0 else tanker_30d
    if ref_val > 0 and tanker_7d > 0:
        ratio_vs_baseline = tanker_7d / ref_val

        if ratio_vs_baseline < 0.5:
            console.print(f"  [bold red]⚠ SEVERE DISRUPTION: Tanker traffic at {ratio_vs_baseline*100:.0f}% of baseline — active blockade[/bold red]")
        elif ratio_vs_baseline < 0.7:
            console.print(f"  [red]⚠ DISRUPTION: Tanker traffic at {ratio_vs_baseline*100:.0f}% of baseline — significant rerouting[/red]")
        elif ratio_vs_baseline < 0.85:
            console.print(f"  [yellow]⚠ Below normal: Tanker traffic at {ratio_vs_baseline*100:.0f}% of baseline — supply constrained[/yellow]")

        # Trajectory: 7d vs 30d vs 60d
        trajectory = (disruption_details or {}).get("trajectory", "")
        if trajectory == "worsening":
            console.print("  [bold red]↘ WORSENING: 7d traffic declining — disruption escalating[/bold red]")
            console.print("    [red]Oil upside risk: supply shortfall deepening → expect further price pressure[/red]")
        elif trajectory == "recovering":
            console.print("  [green]↗ RECOVERING: 7d traffic improving — disruption easing[/green]")
            console.print("    [green]Oil risk: supply gradually returning → geopolitical premium may fade[/green]")
        elif ratio_vs_baseline < 0.9:
            console.print("  [yellow]→ PERSISTING: Traffic below baseline, no strong recovery yet[/yellow]")
            console.print("    [yellow]Oil risk: sustained supply deficit → prices stay supported[/yellow]")


def _show_oil_hormuz_cross(console, oil, hz):
    """Cross-reference oil prices with Hormuz traffic."""
    if hz is None or oil.get("wti") is None:
        return

    lines: list[str] = []

    tanker_latest = hz.get("n_tanker_latest", 0)
    tanker_30d = hz.get("n_tanker_30d_avg", 0)
    wti = oil["wti"]
    wti_30d_chg = oil.get("wti_30d_chg")

    if tanker_30d > 0:
        ratio = tanker_latest / tanker_30d
        if ratio < 0.8 and wti_30d_chg is not None and wti_30d_chg > 5:
            lines.append(
                "  Hormuz traffic ↓ + WTI ↑ → [red]supply disruption confirmed — geopolitical premium building[/red]"
            )
        elif ratio < 0.8 and (wti_30d_chg is None or wti_30d_chg < 2):
            lines.append(
                "  Hormuz traffic ↓ but WTI flat → [yellow]rerouting absorbed, watch for delayed impact[/yellow]"
            )
        elif ratio > 1.1 and wti_30d_chg is not None and wti_30d_chg < -5:
            lines.append(
                "  Hormuz traffic ↑ + WTI ↓ → [green]supply flowing freely — demand weakness driving prices[/green]"
            )

    # Cross-regime signals
    try:
        from lox.data.regime_history import get_score_series
        for domain, display in [("credit", "Credit"), ("volatility", "Vol"), ("growth", "Growth")]:
            series = get_score_series(domain)
            if not series:
                continue
            latest = series[-1]
            sc = latest.get("score")
            lb = latest.get("label", "")
            if not isinstance(sc, (int, float)):
                continue
            short_lb = lb.split("(")[0].strip() if "(" in lb else lb

            if domain == "growth" and sc > 60 and wti_30d_chg is not None and wti_30d_chg < -10:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) + WTI falling → [yellow]demand destruction — recessionary signal[/yellow]")
            elif domain == "growth" and sc < 40 and wti_30d_chg is not None and wti_30d_chg > 10:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) + WTI rising → [yellow]supply-driven — not demand pull[/yellow]")
            elif domain == "volatility" and sc > 55:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]elevated vol environment[/yellow]")
            elif domain == "credit" and sc > 55 and wti_30d_chg is not None and wti_30d_chg > 10:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) + oil spike → [red]stagflationary setup[/red]")
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

def _compute_disruption_score(hz: dict | None) -> tuple[int, str, dict]:
    """
    Compute a 0-100 disruption severity score for Hormuz.
    0 = normal flow, 100 = complete blockade.
    Compares against pre-conflict baseline (oldest available data).
    Returns (score, label, details).
    """
    details: dict = {}
    if hz is None:
        return 0, "Unknown", details

    df = hz.get("df")
    if df is None or df.empty:
        return 0, "No data", details

    import pandas as pd

    tanker_s = pd.to_numeric(df["n_tanker"], errors="coerce").dropna()
    cap_s = pd.to_numeric(df.get("capacity_tanker", pd.Series(dtype=float)), errors="coerce").dropna()

    if len(tanker_s) < 14:
        return 0, "Insufficient data", details

    # Pre-conflict baseline: use oldest 30 days as proxy for normal levels
    baseline_tankers = float(tanker_s.head(30).mean())
    baseline_cap = float(cap_s.head(30).mean()) if len(cap_s) >= 30 else None

    # Current state: 7d average (more stable than single day)
    current_tankers = float(tanker_s.tail(7).mean())
    current_cap = float(cap_s.tail(7).mean()) if len(cap_s) >= 7 else None

    details["baseline_tankers"] = baseline_tankers
    details["current_tankers_7d"] = current_tankers
    details["baseline_cap"] = baseline_cap
    details["current_cap_7d"] = current_cap

    # Transit deficit vs baseline
    transit_ratio = current_tankers / baseline_tankers if baseline_tankers > 0 else 1.0
    transit_deficit = max(0, 1.0 - transit_ratio)
    details["transit_pct_of_baseline"] = transit_ratio * 100

    # Capacity deficit vs baseline
    cap_deficit = 0.0
    if baseline_cap and baseline_cap > 0 and current_cap:
        cap_ratio = current_cap / baseline_cap
        cap_deficit = max(0, 1.0 - cap_ratio)
        details["cap_pct_of_baseline"] = cap_ratio * 100

    # Weighted score
    if baseline_cap and baseline_cap > 0:
        raw = transit_deficit * 0.4 + cap_deficit * 0.6
    else:
        raw = transit_deficit
    score = int(min(100, raw * 100))

    # Trajectory: 7d vs 30d (is it getting better or worse?)
    avg_30 = float(tanker_s.tail(30).mean())
    if avg_30 > 0:
        trajectory = (current_tankers - avg_30) / avg_30 * 100
        details["trajectory_pct"] = trajectory
        if trajectory > 10:
            details["trajectory"] = "recovering"
        elif trajectory < -10:
            details["trajectory"] = "worsening"
        else:
            details["trajectory"] = "stable"

    if score >= 70:
        label = "Severe Disruption"
    elif score >= 50:
        label = "Major Disruption"
    elif score >= 30:
        label = "Moderate Disruption"
    elif score >= 15:
        label = "Mild Disruption"
    elif score >= 5:
        label = "Minor Disruption"
    else:
        label = "Normal Flow"

    return score, label, details


def oil_snapshot(
    *,
    llm: bool = False,
    ticker: str = "",
    refresh: bool = False,
    delta: str = "",
    alert: bool = False,
) -> None:
    """Entry point for `lox regime oil`."""
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table as RichTable
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

    # ── Fetch Hormuz data ─────────────────────────────────────────────────
    console.print("[dim]Fetching Strait of Hormuz transit data from IMF PortWatch...[/dim]")
    hz = _fetch_hormuz_data(days=365)

    # ── Disruption score ──────────────────────────────────────────────────
    disruption_score, disruption_label, disruption_details = _compute_disruption_score(hz)

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
    }
    if hz:
        snapshot_data["hormuz_tanker_7d_avg"] = hz.get("n_tanker_7d_avg")
        snapshot_data["hormuz_tanker_30d_avg"] = hz.get("n_tanker_30d_avg")
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

    # ── Build main panel ──────────────────────────────────────────────────
    wti = oil["wti"]
    brent = oil["brent"]
    spread = oil.get("brent_wti_spread")

    def _v(x, fmt="{:.2f}"):
        return fmt.format(x) if x is not None else "n/a"

    def _chg(x, fmt="{:+.2f}"):
        return fmt.format(x) if x is not None else "n/a"

    # Headline
    headline_parts = []
    if wti is not None:
        headline_parts.append(f"WTI ${wti:.2f}")
    if brent is not None:
        headline_parts.append(f"Brent ${brent:.2f}")
    if spread is not None:
        headline_parts.append(f"Spread ${spread:.2f}")
    headline = " | ".join(headline_parts) if headline_parts else "No data"

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

    # Hormuz status (use disruption score)
    hz_status = ""
    if disruption_score >= 30:
        hz_status = f" | [bold red]⚠ Hormuz DISRUPTED ({disruption_score}/100)[/bold red]"
    elif disruption_score >= 15:
        hz_status = f" | [yellow]Hormuz below normal ({disruption_score}/100)[/yellow]"
    elif disruption_score >= 5:
        hz_status = f" | [yellow]Hormuz minor disruption ({disruption_score}/100)[/yellow]"
    elif hz is not None:
        hz_status = " | [green]Hormuz normal flow[/green]"

    metrics = RichTable(box=None, padding=(0, 2), show_header=True, header_style="dim")
    metrics.add_column("Metric", min_width=22)
    metrics.add_column("Value", justify="right", min_width=10)
    metrics.add_column("Context", min_width=30)

    metrics.add_row("─── Crude Oil ───", "", "")
    metrics.add_row("WTI Crude", f"${_v(wti)}", _wti_ctx(wti))
    metrics.add_row("Brent Crude", f"${_v(brent)}", _brent_ctx(brent))
    metrics.add_row("Brent-WTI Spread", f"${_v(spread)}", _spread_ctx(spread))

    metrics.add_row("─── Velocity ───", "", "")
    metrics.add_row("WTI 5d", f"${_chg(oil.get('wti_5d_chg'))}", "5-day change")
    metrics.add_row("WTI 30d", f"${_chg(oil.get('wti_30d_chg'))}", "30-day change")
    metrics.add_row("WTI 90d", f"${_chg(oil.get('wti_90d_chg'))}", "90-day change")

    metrics.add_row("─── Risk ───", "", "")
    metrics.add_row("WTI Vol (20d ann.)", f"{_v(oil.get('wti_vol_20d'), '{:.0f}')}%", _vol_ctx(oil.get("wti_vol_20d")))
    metrics.add_row("Brent Vol (20d ann.)", f"{_v(oil.get('brent_vol_20d'), '{:.0f}')}%", _vol_ctx(oil.get("brent_vol_20d")))
    if oil.get("wti_1y_pctl") is not None:
        metrics.add_row("WTI 1Y Percentile", f"{oil['wti_1y_pctl']:.0f}th", _pctl_ctx(oil["wti_1y_pctl"]))
    if oil.get("wti_1y_high") is not None and oil.get("wti_1y_low") is not None:
        metrics.add_row("WTI 1Y Range", f"${oil['wti_1y_low']:.0f}–${oil['wti_1y_high']:.0f}", f"current ${wti:.0f}" if wti else "")

    # Disruption score in panel
    metrics.add_row("─── Hormuz Disruption ───", "", "")
    d_color = "bold red" if disruption_score >= 50 else ("red" if disruption_score >= 30 else ("yellow" if disruption_score >= 15 else "green"))
    d_bar_len = int(disruption_score / 2)
    d_bar = "█" * d_bar_len + "░" * (50 - d_bar_len)
    metrics.add_row("Disruption Score", f"[{d_color}]{disruption_score}/100[/{d_color}]", f"[{d_color}]{disruption_label}[/{d_color}]")
    metrics.add_row("Severity", f"[{d_color}]{d_bar}[/{d_color}]", "")

    # Baseline comparison
    baseline_t = disruption_details.get("baseline_tankers")
    current_t = disruption_details.get("current_tankers_7d")
    pct_baseline = disruption_details.get("transit_pct_of_baseline")
    trajectory = disruption_details.get("trajectory", "")
    traj_pct = disruption_details.get("trajectory_pct")
    if baseline_t and current_t and pct_baseline is not None:
        metrics.add_row(
            "vs Pre-Conflict",
            f"{pct_baseline:.0f}%",
            f"{current_t:.0f}/day vs {baseline_t:.0f}/day baseline",
        )
    if trajectory and traj_pct is not None:
        t_color = "green" if trajectory == "recovering" else ("red" if trajectory == "worsening" else "yellow")
        t_arrow = "↗" if trajectory == "recovering" else ("↘" if trajectory == "worsening" else "→")
        metrics.add_row(
            "Trajectory",
            f"[{t_color}]{t_arrow} {trajectory.upper()}[/{t_color}]",
            f"7d avg {traj_pct:+.0f}% vs 30d avg",
        )

    parts = Group(
        Text.from_markup(f"As of: {oil['asof']}\n"),
        Text.from_markup(f"{sentiment}  {headline}{hz_status}\n"),
        metrics,
    )

    panel = Panel(
        parts,
        title="[bold]Oil & Energy Chokepoint[/bold]",
        border_style="yellow",
        padding=(1, 2),
    )
    rprint(panel)

    # ── Block 1: Price sparklines ─────────────────────────────────────────
    _show_oil_sparklines(console, oil)

    # ── Block 2: Strait of Hormuz dashboard ──────────────────────────────
    _show_hormuz_panel(console, hz, disruption_details)

    # ── Block 3: Cross-regime signals ────────────────────────────────────
    _show_oil_hormuz_cross(console, oil, hz)

    if llm:
        snapshot_data = {
            "wti": wti,
            "brent": brent,
            "brent_wti_spread": spread,
            "wti_5d_chg": oil.get("wti_5d_chg"),
            "wti_30d_chg": oil.get("wti_30d_chg"),
            "wti_90d_chg": oil.get("wti_90d_chg"),
            "wti_vol_20d": oil.get("wti_vol_20d"),
            "wti_1y_pctl": oil.get("wti_1y_pctl"),
        }
        if hz:
            snapshot_data["hormuz_tanker_latest"] = hz.get("n_tanker_latest")
            snapshot_data["hormuz_tanker_30d_avg"] = hz.get("n_tanker_30d_avg")
            snapshot_data["hormuz_total_latest"] = hz.get("n_total_latest")

        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis
        print_llm_regime_analysis(
            settings=settings,
            domain="oil",
            snapshot=snapshot_data,
            regime_label=sentiment,
            regime_description="Oil price and Strait of Hormuz shipping traffic analysis",
            ticker=ticker,
        )

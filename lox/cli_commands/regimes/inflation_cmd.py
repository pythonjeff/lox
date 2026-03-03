"""CLI command for the Inflation regime (split from Macro)."""
from __future__ import annotations

from rich import print
from rich.console import Console

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings


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
        return "[red]accelerating[/red]" if delta > 0 else "[green]decelerating[/green]"
    return "[green]rising[/green]" if delta > 0 else "[red]falling[/red]"


def _extract_monthly_last_n(df, col: str, n: int = 12) -> list[float]:
    """Extract last N monthly values from a daily DataFrame (forward-filled monthly series)."""
    import pandas as pd
    if df is None or df.empty or col not in df.columns:
        return []
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return []
    # Take last row per month to get unique monthly values
    df_sub = df[["date", col]].dropna(subset=[col]).copy()
    df_sub["month"] = pd.to_datetime(df_sub["date"]).dt.to_period("M")
    monthly = df_sub.groupby("month", as_index=False).last().tail(n)
    return [float(v) for v in monthly[col]]


def _extract_daily_last_n(df, col: str, n: int = 30) -> list[float]:
    """Extract last N daily values from a DataFrame."""
    import pandas as pd
    if df is None or df.empty or col not in df.columns:
        return []
    s = pd.to_numeric(df[col], errors="coerce").dropna().tail(n)
    return [float(v) for v in s]


def _show_inflation_sparklines(console, macro_df) -> None:
    """Show 12-month sparklines for CPI, Core CPI, Breakeven, momentum."""
    lines: list[str] = []
    cpi_vals = _extract_monthly_last_n(macro_df, "CPI_YOY", 12)
    core_vals = _extract_monthly_last_n(macro_df, "CORE_CPI_YOY", 12)
    be5_vals = _extract_daily_last_n(macro_df, "T5YIE", 30)
    cpi3m_vals = _extract_monthly_last_n(macro_df, "CPI_3M_ANN", 12)
    if len(cpi_vals) >= 6:
        cur = f"{cpi_vals[-1]:.1f}%" if cpi_vals else ""
        lines.append(f"  {'CPI YoY':<10} {_sparkline(cpi_vals)}  {cur}  {_trend_label(cpi_vals)}")
    if len(core_vals) >= 6:
        cur = f"{core_vals[-1]:.1f}%" if core_vals else ""
        lines.append(f"  {'Core CPI':<10} {_sparkline(core_vals)}  {cur}  {_trend_label(core_vals)}")
    if len(be5_vals) >= 10:
        cur = f"{be5_vals[-1]:.2f}%" if be5_vals else ""
        lines.append(f"  {'5Y BE':<10} {_sparkline(be5_vals)}  {cur}  {_trend_label(be5_vals, higher_is_worse=True)}")
    if len(cpi3m_vals) >= 6:
        cur = f"{cpi3m_vals[-1]:.1f}%" if cpi3m_vals else ""
        lines.append(f"  {'CPI 3m Ann':<10} {_sparkline(cpi3m_vals)}  {cur}  {_trend_label(cpi3m_vals)}")
    if lines:
        console.print()
        console.print("[dim]─── Inflation Velocity (12mo) ───────────────────────────────────[/dim]")
        for ln in lines:
            console.print(ln)


def _decomp_signal(delta_pct):
    if delta_pct is None:
        return "—"
    if delta_pct > 0.5:
        return "[red]accelerating[/red]"
    if delta_pct > 0.2:
        return "[yellow]rising[/yellow]"
    if delta_pct > -0.2:
        return "[dim]stable[/dim]"
    if delta_pct > -0.5:
        return "[green]cooling[/green]"
    return "[green]decelerating[/green]"


def _show_inflation_decomposition_table(console, fred, refresh: bool) -> None:
    """Show shelter/supercore/goods with 3mo shift."""
    from rich.table import Table as RichTable
    series_map = [
        ("CUSR0000SAH1", "Shelter CPI", 13),
        ("CUSR0000SASL2RS", "Supercore", 13),
        ("CUSR0000SACL1E", "Core Goods", 13),
    ]
    rows = []
    for sid, label, months in series_map:
        try:
            df = fred.fetch_series(sid, start_date="2020-01-01", refresh=refresh)
            if df is None or len(df) < months + 3:
                continue
            df = df.sort_values("date")
            yoy = (df["value"].iloc[-1] / df["value"].iloc[-months] - 1.0) * 100.0
            yoy_3mo = (df["value"].iloc[-4] / df["value"].iloc[-months - 3] - 1.0) * 100.0
            delta = yoy - yoy_3mo if yoy_3mo is not None else None
            rows.append((label, yoy, yoy_3mo, delta))
        except Exception:
            continue
    if not rows:
        return
    st = RichTable(
        title="Decomposition (3mo shift)",
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 1),
    )
    st.add_column("Component", min_width=14)
    st.add_column("Now", justify="right")
    st.add_column("3mo ago", justify="right")
    st.add_column("Δ", justify="right")
    st.add_column("Signal", style="dim")
    for label, now, ago, delta in rows:
        now_s = f"{now:.1f}%" if now is not None else "—"
        ago_s = f"{ago:.1f}%" if ago is not None else "—"
        delta_s = f"{delta:+.1f}pp" if delta is not None else "—"
        st.add_row(label, now_s, ago_s, delta_s, _decomp_signal(delta))
    console.print()
    console.print(st)


def _show_cross_regime_signals(console, infl_score: int | float, cpi_yoy, breakeven_5y) -> None:
    lines: list[str] = []
    try:
        from lox.data.regime_history import get_score_series
        for domain, display in [("credit", "Credit"), ("rates", "Rates"), ("growth", "Growth"), ("oil", "Oil")]:
            series = get_score_series(domain)
            if not series:
                continue
            latest = series[-1]
            sc = latest.get("score")
            lb = latest.get("label", "")
            if not isinstance(sc, (int, float)):
                continue
            short_lb = lb.split("(")[0].strip() if "(" in lb else lb
            if domain == "rates" and sc > 60 and infl_score > 55:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) + inflation elevated → [red]stagflation setup — Fed trapped[/red]")
            elif domain == "rates" and sc < 40 and infl_score > 55:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]inflation hot but rates easing — watch for pivot[/yellow]")
            elif domain == "growth" and sc > 65 and infl_score > 55:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) + inflation → [red]stagflation risk[/red]")
            elif domain == "growth" and sc < 40 and infl_score > 55:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]growth weak + inflation sticky — worst of both[/yellow]")
            elif domain == "oil" and infl_score > 50:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [yellow]oil + inflation — supply-side pressure[/yellow]")
            else:
                lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [dim]neutral[/dim]")
    except Exception:
        pass
    if breakeven_5y is not None and cpi_yoy is not None:
        gap = cpi_yoy - breakeven_5y
        if gap > 1.0:
            lines.append(f"  CPI-BE gap {gap:.1f}pp → [yellow]realized above expectations — TIPS cheap vs realized[/yellow]")
        elif gap < -1.0:
            lines.append(f"  CPI-BE gap {gap:.1f}pp → [green]expectations above realized — market pricing inflation risk[/green]")
    if lines:
        console.print()
        console.print("[dim]─── Cross-Regime Signals ──────────────────────────────────────[/dim]")
        for ln in lines:
            console.print(ln)


def _yoy_from_index(df, months: int = 13):
    """Compute YoY % from a monthly index-level FRED series."""
    if df is None or len(df) < months:
        return None
    df = df.sort_values("date")
    return (df["value"].iloc[-1] / df["value"].iloc[-months] - 1.0) * 100.0


def _fetch_yoy(fred, series_id: str, *, start_date: str, refresh: bool, months: int = 13):
    """Fetch a FRED series and compute YoY %. Returns None on failure."""
    try:
        df = fred.fetch_series(series_id, start_date=start_date, refresh=refresh)
        return _yoy_from_index(df, months=months)
    except Exception:
        return None


def inflation_snapshot(*, llm: bool = False, ticker: str = "", refresh: bool = False) -> None:
    """Entry point for `lox regime inflation`."""
    settings = load_settings()

    from lox.data.fred import FredClient
    from lox.macro.signals import build_macro_state

    macro_state = build_macro_state(settings=settings, start_date="2011-01-01", refresh=refresh)
    inp = macro_state.inputs

    fred = FredClient(api_key=settings.FRED_API_KEY)
    start = "2011-01-01"

    cpi_yoy = inp.cpi_yoy
    core_cpi_yoy = inp.core_cpi_yoy
    median_cpi_yoy = inp.median_cpi_yoy
    cpi_3m_ann = inp.cpi_3m_annualized
    cpi_6m_ann = inp.cpi_6m_annualized
    breakeven_5y = inp.breakeven_5y
    breakeven_10y = inp.breakeven_10y
    breakeven_5y5y = inp.breakeven_5y5y

    # ── Additional FRED series ──────────────────────────────────────────
    core_pce_yoy = _fetch_yoy(fred, "PCEPILFE", start_date=start, refresh=refresh)
    ppi_yoy = _fetch_yoy(fred, "PPIFIS", start_date=start, refresh=refresh)

    trimmed_mean_pce_yoy = None
    try:
        tm_df = fred.fetch_series("PCETRIM12M159SFRBDAL", start_date=start, refresh=refresh)
        if tm_df is not None and len(tm_df) >= 1:
            trimmed_mean_pce_yoy = float(tm_df.sort_values("date")["value"].iloc[-1])
    except Exception:
        pass

    oil_price_yoy_pct = None
    try:
        oil_df = fred.fetch_series("DCOILWTICO", start_date=start, refresh=refresh)
        if oil_df is not None and len(oil_df) >= 252:
            oil_df = oil_df.sort_values("date").dropna(subset=["value"])
            if len(oil_df) >= 252:
                oil_price_yoy_pct = (oil_df["value"].iloc[-1] / oil_df["value"].iloc[-252] - 1.0) * 100.0
    except Exception:
        pass

    # ── Import prices (tariff / FX pass-through) ─────────────────────────
    import_price_yoy = _fetch_yoy(fred, "IR", start_date=start, refresh=refresh)

    # ── Layer 3: Decomposition series ───────────────────────────────────
    shelter_cpi_yoy = _fetch_yoy(fred, "CUSR0000SAH1", start_date=start, refresh=refresh)
    supercore_yoy = _fetch_yoy(fred, "CUSR0000SASL2RS", start_date=start, refresh=refresh)
    core_goods_yoy = _fetch_yoy(fred, "CUSR0000SACL1E", start_date=start, refresh=refresh)

    # ── Classify ────────────────────────────────────────────────────────
    from lox.inflation.regime import classify_inflation

    result = classify_inflation(
        cpi_yoy=cpi_yoy,
        core_pce_yoy=core_pce_yoy,
        breakeven_5y=breakeven_5y,
        ppi_yoy=ppi_yoy,
        core_cpi_yoy=core_cpi_yoy,
        trimmed_mean_pce_yoy=trimmed_mean_pce_yoy,
        median_cpi_yoy=median_cpi_yoy,
        cpi_3m_ann=cpi_3m_ann,
        cpi_6m_ann=cpi_6m_ann,
        breakeven_5y5y=breakeven_5y5y,
        breakeven_10y=breakeven_10y,
        oil_price_yoy_pct=oil_price_yoy_pct,
        import_price_yoy=import_price_yoy,
        shelter_cpi_yoy=shelter_cpi_yoy,
        supercore_yoy=supercore_yoy,
        core_goods_yoy=core_goods_yoy,
    )

    # ── Build sectioned metrics table ───────────────────────────────────
    def _v(x, fmt="{:.1f}%"):
        return fmt.format(x) if x is not None else "n/a"

    # Momentum context
    momentum_ctx = "n/a"
    if cpi_3m_ann is not None and cpi_yoy is not None:
        spread = cpi_3m_ann - cpi_yoy
        if spread > 0.5:
            momentum_ctx = "re-accelerating"
        elif spread > 0:
            momentum_ctx = "slightly accelerating"
        elif spread > -0.5:
            momentum_ctx = "slightly decelerating"
        else:
            momentum_ctx = "decelerating"

    # Shelter-supercore divergence context
    def _shelter_context():
        if shelter_cpi_yoy is None:
            return "~36% of CPI, 12mo lagged rents"
        if shelter_cpi_yoy > 5.0:
            return "hot — but lags actual rents by ~12mo"
        if shelter_cpi_yoy > 3.5:
            return "elevated, still catching up to prior rents"
        if shelter_cpi_yoy > 2.0:
            return "normalizing toward target"
        return "cooling"

    def _supercore_context():
        if supercore_yoy is None:
            return "services ex shelter — demand/wage signal"
        if supercore_yoy > 4.0:
            return "hot — sticky demand-driven pressure"
        if supercore_yoy > 3.0:
            return "elevated — wage/demand driven"
        if supercore_yoy > 2.0:
            return "consistent with ~2% target"
        return "cooling — demand pressure fading"

    def _import_context():
        if import_price_yoy is None:
            return "tariff/FX pass-through → PPI → CPI"
        if import_price_yoy > 10:
            return "surging — tariff/FX cost-push"
        if import_price_yoy > 5:
            return "rising — tariff/trade pressure building"
        if import_price_yoy > 2:
            return "mild import inflation"
        if import_price_yoy > -2:
            return "stable — no tariff pass-through yet"
        return "falling — trade channel deflationary"

    def _goods_context():
        if core_goods_yoy is None:
            return "tradeable, supply-chain driven"
        if core_goods_yoy < -0.5:
            return "deflating — typical post-supply-shock"
        if core_goods_yoy < 0.5:
            return "flat — normalized"
        if core_goods_yoy < 2.0:
            return "mild goods inflation"
        return "elevated — unusual for goods"

    # ── Build macro dataset for sparklines ───────────────────────────────────
    from lox.macro.signals import build_macro_dataset
    try:
        macro_df = build_macro_dataset(settings=settings, start_date="2011-01-01", refresh=refresh)
    except Exception:
        macro_df = None

    metrics = [
        # ── Headline ──
        {"name": "─── Headline ───", "value": "", "context": ""},
        {"name": "CPI YoY", "value": _v(cpi_yoy), "context": "headline"},
        {"name": "Core CPI YoY", "value": _v(core_cpi_yoy), "context": "ex food & energy"},
        {"name": "Core PCE YoY", "value": _v(core_pce_yoy), "context": "Fed target 2%"},
        # ── Decomposition ──
        {"name": "─── Decomposition ───", "value": "", "context": ""},
        {"name": "Shelter CPI YoY", "value": _v(shelter_cpi_yoy), "context": _shelter_context()},
        {"name": "Supercore YoY", "value": _v(supercore_yoy), "context": _supercore_context()},
        {"name": "Core Goods YoY", "value": _v(core_goods_yoy), "context": _goods_context()},
        # ── Momentum ──
        {"name": "─── Pipeline & Momentum ───", "value": "", "context": ""},
        {"name": "Import Prices YoY", "value": _v(import_price_yoy, "{:+.1f}%"), "context": _import_context()},
        {"name": "PPI YoY", "value": _v(ppi_yoy, "{:+.1f}%"), "context": "domestic producer costs"},
        {"name": "CPI 3m Ann", "value": _v(cpi_3m_ann), "context": momentum_ctx},
        # ── Expectations ──
        {"name": "─── Expectations ───", "value": "", "context": ""},
        {"name": "5Y Breakeven", "value": _v(breakeven_5y, "{:.2f}%"), "context": "market pricing"},
        {"name": "5Y5Y Forward", "value": _v(breakeven_5y5y, "{:.2f}%"), "context": "expectations anchor"},
        # ── Breadth ──
        {"name": "─── Breadth ───", "value": "", "context": ""},
        {"name": "Trimmed Mean PCE", "value": _v(trimmed_mean_pce_yoy), "context": "Dallas Fed (noise filter)"},
        {"name": "Median CPI YoY", "value": _v(median_cpi_yoy), "context": "Cleveland Fed (breadth)"},
    ]

    from lox.regimes.trend import get_domain_trend
    trend = get_domain_trend("inflation", result.score, result.label)

    print(render_regime_panel(
        domain="Inflation",
        asof=macro_state.asof,
        regime_label=result.label,
        score=result.score,
        percentile=None,
        description=result.description,
        metrics=metrics,
        trend=trend,
    ))

    # ── Block 1: Inflation velocity sparklines ──────────────────────────────
    console = Console()
    if macro_df is not None:
        _show_inflation_sparklines(console, macro_df)

    # ── Block 2: Decomposition table (3mo shift) ────────────────────────────
    _show_inflation_decomposition_table(console, fred, refresh)

    # ── Block 3: Cross-regime signals ───────────────────────────────────────
    _show_cross_regime_signals(console, result.score, cpi_yoy, breakeven_5y)

    if llm:
        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis

        snapshot = {
            "cpi_yoy": cpi_yoy, "core_pce_yoy": core_pce_yoy, "core_cpi_yoy": core_cpi_yoy,
            "trimmed_mean_pce_yoy": trimmed_mean_pce_yoy, "median_cpi_yoy": median_cpi_yoy,
            "cpi_3m_ann": cpi_3m_ann, "cpi_6m_ann": cpi_6m_ann,
            "breakeven_5y": breakeven_5y, "breakeven_5y5y": breakeven_5y5y,
            "ppi_yoy": ppi_yoy, "oil_price_yoy_pct": oil_price_yoy_pct,
            "import_price_yoy": import_price_yoy,
            "shelter_cpi_yoy": shelter_cpi_yoy, "supercore_yoy": supercore_yoy,
            "core_goods_yoy": core_goods_yoy,
        }
        print_llm_regime_analysis(
            settings=settings,
            domain="inflation",
            snapshot=snapshot,
            regime_label=result.label,
            regime_description=result.description,
            ticker=ticker,
        )

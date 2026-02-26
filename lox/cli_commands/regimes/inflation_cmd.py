"""CLI command for the Inflation regime (split from Macro)."""
from __future__ import annotations

import typer
from rich import print

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings


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


def inflation_snapshot(*, llm: bool = False, refresh: bool = False) -> None:
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

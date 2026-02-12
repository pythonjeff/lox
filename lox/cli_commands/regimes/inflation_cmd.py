"""CLI command for the Inflation regime (split from Macro)."""
from __future__ import annotations

import typer
from rich import print

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings


def inflation_snapshot(*, llm: bool = False) -> None:
    """Entry point for `lox regime inflation`."""
    settings = load_settings()

    from lox.macro.signals import build_macro_state
    macro_state = build_macro_state(settings=settings, start_date="2011-01-01")
    inp = macro_state.inputs

    cpi_yoy = inp.cpi_yoy
    core_cpi_yoy = inp.core_cpi_yoy
    median_cpi_yoy = inp.median_cpi_yoy
    cpi_3m_ann = inp.cpi_3m_annualized
    cpi_6m_ann = inp.cpi_6m_annualized
    breakeven_5y = inp.breakeven_5y
    breakeven_10y = inp.breakeven_10y
    breakeven_5y5y = inp.breakeven_5y5y

    # Oil YoY % for supply pipeline signal
    oil_price_yoy_pct = None
    try:
        from lox.data.fred import FredClient
        fred = FredClient(api_key=settings.FRED_API_KEY)
        oil_df = fred.fetch_series("DCOILWTICO", start_date="2011-01-01")
        if oil_df is not None and len(oil_df) >= 252:
            oil_df = oil_df.sort_values("date").dropna(subset=["value"])
            if len(oil_df) >= 252:
                oil_price_yoy_pct = (oil_df["value"].iloc[-1] / oil_df["value"].iloc[-252] - 1.0) * 100.0
    except Exception:
        pass

    # Core PCE from FRED
    core_pce_yoy = None
    try:
        from lox.data.fred import FredClient
        fred = FredClient(api_key=settings.FRED_API_KEY)
        pce_df = fred.fetch_series("PCEPILFE", start_date="2011-01-01")
        if pce_df is not None and len(pce_df) >= 13:
            pce_df = pce_df.sort_values("date")
            core_pce_yoy = (pce_df["value"].iloc[-1] / pce_df["value"].iloc[-13] - 1.0) * 100.0
    except Exception:
        pass

    # PPI from FRED
    ppi_yoy = None
    try:
        from lox.data.fred import FredClient
        fred = FredClient(api_key=settings.FRED_API_KEY)
        ppi_df = fred.fetch_series("PPIFIS", start_date="2011-01-01")
        if ppi_df is not None and len(ppi_df) >= 13:
            ppi_df = ppi_df.sort_values("date")
            ppi_yoy = (ppi_df["value"].iloc[-1] / ppi_df["value"].iloc[-13] - 1.0) * 100.0
    except Exception:
        pass

    # Trimmed Mean PCE from FRED (Dallas Fed)
    trimmed_mean_pce_yoy = None
    try:
        from lox.data.fred import FredClient
        fred = FredClient(api_key=settings.FRED_API_KEY)
        tm_df = fred.fetch_series("PCETRIM12M159SFRBDAL", start_date="2011-01-01")
        if tm_df is not None and len(tm_df) >= 1:
            tm_df = tm_df.sort_values("date")
            trimmed_mean_pce_yoy = float(tm_df["value"].iloc[-1])
    except Exception:
        pass

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
    )

    def _v(x, fmt="{:.1f}%"):
        return fmt.format(x) if x is not None else "n/a"

    # Momentum context string
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

    metrics = [
        {"name": "CPI YoY", "value": _v(cpi_yoy), "context": "headline"},
        {"name": "Core PCE YoY", "value": _v(core_pce_yoy), "context": "Fed target 2%"},
        {"name": "Core CPI YoY", "value": _v(core_cpi_yoy), "context": "ex food & energy"},
        {"name": "Trimmed Mean PCE", "value": _v(trimmed_mean_pce_yoy), "context": "Dallas Fed"},
        {"name": "Median CPI YoY", "value": _v(median_cpi_yoy), "context": "Cleveland Fed"},
        {"name": "CPI 3m Ann", "value": _v(cpi_3m_ann), "context": momentum_ctx},
        {"name": "5Y Breakeven", "value": _v(breakeven_5y, "{:.2f}%"), "context": "market pricing"},
        {"name": "5Y5Y Forward", "value": _v(breakeven_5y5y, "{:.2f}%"), "context": "expectations anchor"},
        {"name": "PPI YoY", "value": _v(ppi_yoy, "{:+.1f}%"), "context": "leading CPI"},
    ]

    print(render_regime_panel(
        domain="Inflation",
        asof=macro_state.asof,
        regime_label=result.label,
        score=result.score,
        percentile=None,
        description=result.description,
        metrics=metrics,
    ))

    if llm:
        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis

        snapshot = {
            "cpi_yoy": cpi_yoy, "core_pce_yoy": core_pce_yoy, "core_cpi_yoy": core_cpi_yoy,
            "trimmed_mean_pce_yoy": trimmed_mean_pce_yoy, "median_cpi_yoy": median_cpi_yoy,
            "cpi_3m_ann": cpi_3m_ann, "cpi_6m_ann": cpi_6m_ann,
            "breakeven_5y": breakeven_5y, "breakeven_5y5y": breakeven_5y5y,
            "ppi_yoy": ppi_yoy, "oil_price_yoy_pct": oil_price_yoy_pct,
        }
        print_llm_regime_analysis(
            settings=settings,
            domain="inflation",
            snapshot=snapshot,
            regime_label=result.label,
            regime_description=result.description,
        )

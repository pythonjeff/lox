"""CLI command for the Growth regime (split from Macro)."""
from __future__ import annotations

import typer
from rich import print

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings


def growth_snapshot(*, llm: bool = False) -> None:
    """Entry point for `lox regime growth`."""
    settings = load_settings()

    # Build macro state (shared data source)
    from lox.macro.signals import build_macro_state
    macro_state = build_macro_state(settings=settings, start_date="2011-01-01")
    inp = macro_state.inputs

    # Derive inputs
    payrolls_3m_level = None
    if inp.payrolls_3m_annualized is not None:
        payrolls_3m_level = inp.payrolls_3m_annualized * 157_000 / 100 / 12

    claims_4wk = inp.initial_claims_4w
    unemployment_rate = inp.unemployment_rate
    payrolls_mom = inp.payrolls_mom

    # Compute 13-week claims average for momentum signal
    claims_13wk = None
    try:
        from lox.data.fred import FredClient
        fred = FredClient(api_key=settings.FRED_API_KEY)
        claims_df = fred.fetch_series("ICSA", start_date="2011-01-01")
        if claims_df is not None and len(claims_df) >= 13:
            claims_df = claims_df.sort_values("date")
            claims_13wk = float(claims_df["value"].tail(13).mean())
    except Exception:
        pass

    # ISM from Trading Economics
    ism_val = None
    try:
        from lox.altdata.trading_economics import get_ism_manufacturing
        ism_val = get_ism_manufacturing()
    except Exception:
        pass

    # INDPRO from FRED
    indpro_yoy = None
    try:
        from lox.data.fred import FredClient
        fred = FredClient(api_key=settings.FRED_API_KEY)
        indpro_df = fred.fetch_series("INDPRO", start_date="2011-01-01")
        if indpro_df is not None and len(indpro_df) >= 13:
            indpro_df = indpro_df.sort_values("date")
            indpro_yoy = (indpro_df["value"].iloc[-1] / indpro_df["value"].iloc[-13] - 1.0) * 100.0
    except Exception:
        pass

    # Conference Board LEI from FRED (USSLIND) — discontinued mid-2024, use if fresh
    lei_yoy = None
    try:
        import pandas as _pd
        from lox.data.fred import FredClient
        fred = FredClient(api_key=settings.FRED_API_KEY)
        lei_df = fred.fetch_series("USSLIND", start_date="2011-01-01")
        if lei_df is not None and len(lei_df) >= 13:
            lei_df = lei_df.sort_values("date")
            # Only use if data is within last 6 months (LEI was discontinued)
            latest_date = _pd.to_datetime(lei_df["date"].iloc[-1])
            if (_pd.Timestamp.now() - latest_date).days < 180:
                lei_yoy = (lei_df["value"].iloc[-1] / lei_df["value"].iloc[-13] - 1.0) * 100.0
    except Exception:
        pass

    from lox.growth.regime import classify_growth
    result = classify_growth(
        payrolls_3m_ann=payrolls_3m_level,
        ism=ism_val,
        claims_4wk=claims_4wk,
        indpro_yoy=indpro_yoy,
        payrolls_mom=payrolls_mom,
        unemployment_rate=unemployment_rate,
        claims_13wk=claims_13wk,
        lei_yoy=lei_yoy,
    )

    def _v(x, fmt="{:.1f}"):
        return fmt.format(x) if x is not None else "n/a"

    metrics = [
        {"name": "Payrolls 3m ann", "value": f"{payrolls_3m_level:+,.0f}K" if payrolls_3m_level else "n/a", "context": "jobs/mo"},
        {"name": "Unemployment", "value": _v(unemployment_rate, "{:.1f}%"), "context": "UNRATE"},
        {"name": "ISM Mfg PMI", "value": _v(ism_val), "context": ">50 = expansion"},
        {"name": "Initial Claims 4wk", "value": f"{claims_4wk:,.0f}" if claims_4wk else "n/a", "context": "weekly avg"},
        {"name": "Industrial Prod YoY", "value": _v(indpro_yoy, "{:+.1f}%"), "context": "INDPRO"},
        {"name": "LEI YoY", "value": _v(lei_yoy, "{:+.1f}%") if lei_yoy is not None else "n/a", "context": "leading indicator"},
    ]

    print(render_regime_panel(
        domain="Growth",
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
            "payrolls_3m_ann": payrolls_3m_level, "ism": ism_val, "claims_4wk": claims_4wk,
            "indpro_yoy": indpro_yoy, "unemployment_rate": unemployment_rate,
            "lei_yoy": lei_yoy,
        }
        print_llm_regime_analysis(
            settings=settings,
            domain="growth",
            snapshot=snapshot,
            regime_label=result.label,
            regime_description=result.description,
        )

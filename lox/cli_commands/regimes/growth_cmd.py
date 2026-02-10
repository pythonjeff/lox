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

    from lox.growth.regime import classify_growth
    result = classify_growth(
        payrolls_3m_ann=payrolls_3m_level,
        ism=ism_val,
        claims_4wk=claims_4wk,
        indpro_yoy=indpro_yoy,
    )

    def _v(x, fmt="{:.1f}"):
        return fmt.format(x) if x is not None else "n/a"

    metrics = [
        {"name": "Payrolls 3m ann", "value": f"{payrolls_3m_level:+,.0f}K" if payrolls_3m_level else "n/a", "context": "jobs/mo"},
        {"name": "ISM Mfg PMI", "value": _v(ism_val), "context": ">50 = expansion"},
        {"name": "Initial Claims 4wk", "value": f"{claims_4wk:,.0f}" if claims_4wk else "n/a", "context": "weekly avg"},
        {"name": "Industrial Prod YoY", "value": _v(indpro_yoy, "{:+.1f}%"), "context": "INDPRO"},
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
        from lox.llm.core.analyst import llm_analyze_regime
        from rich.markdown import Markdown
        from rich.panel import Panel

        print("\n[bold cyan]Generating LLM analysis...[/bold cyan]\n")
        analysis = llm_analyze_regime(
            settings=settings,
            domain="growth",
            snapshot={"payrolls_3m_ann": payrolls_3m_level, "ism": ism_val, "claims_4wk": claims_4wk, "indpro_yoy": indpro_yoy},
            regime_label=result.label,
            regime_description=result.description,
        )
        print(Panel(Markdown(analysis), title="Analysis", expand=False))

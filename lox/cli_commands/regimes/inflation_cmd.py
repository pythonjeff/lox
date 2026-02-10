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
    breakeven_5y = inp.breakeven_5y

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

    from lox.inflation.regime import classify_inflation
    result = classify_inflation(
        cpi_yoy=cpi_yoy,
        core_pce_yoy=core_pce_yoy,
        breakeven_5y=breakeven_5y,
        ppi_yoy=ppi_yoy,
    )

    def _v(x, fmt="{:.1f}%"):
        return fmt.format(x) if x is not None else "n/a"

    metrics = [
        {"name": "CPI YoY", "value": _v(cpi_yoy), "context": "headline"},
        {"name": "Core PCE YoY", "value": _v(core_pce_yoy), "context": "Fed target 2%"},
        {"name": "5Y Breakeven", "value": _v(breakeven_5y, "{:.2f}%"), "context": "market pricing"},
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
        from lox.llm.core.analyst import llm_analyze_regime
        from rich.markdown import Markdown
        from rich.panel import Panel

        print("\n[bold cyan]Generating LLM analysis...[/bold cyan]\n")
        analysis = llm_analyze_regime(
            settings=settings,
            domain="inflation",
            snapshot={"cpi_yoy": cpi_yoy, "core_pce_yoy": core_pce_yoy, "breakeven_5y": breakeven_5y, "ppi_yoy": ppi_yoy},
            regime_label=result.label,
            regime_description=result.description,
        )
        print(Panel(Markdown(analysis), title="Analysis", expand=False))

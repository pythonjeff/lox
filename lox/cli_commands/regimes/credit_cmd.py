"""CLI command for the Credit regime (NEW)."""
from __future__ import annotations

from rich import print

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings


def credit_snapshot(*, llm: bool = False) -> None:
    """Entry point for `lox regime credit`."""
    settings = load_settings()
    from lox.data.fred import FredClient

    fred = FredClient(api_key=settings.FRED_API_KEY)

    hy_oas_val = None
    bbb_oas_val = None
    aaa_oas_val = None
    hy_30d_chg = None
    hy_90d_pctl = None
    asof = "—"

    try:
        hy_df = fred.fetch_series("BAMLH0A0HYM2", start_date="2020-01-01")
        if hy_df is not None and not hy_df.empty:
            hy_df = hy_df.sort_values("date")
            hy_oas_val = float(hy_df["value"].iloc[-1]) * 100  # pct pts → bps
            asof = str(hy_df["date"].iloc[-1].date())
            if len(hy_df) >= 22:
                hy_30d_chg = (float(hy_df["value"].iloc[-1]) - float(hy_df["value"].iloc[-22])) * 100
            if len(hy_df) >= 63:
                recent = hy_df["value"].iloc[-63:]
                hy_90d_pctl = float((recent <= hy_df["value"].iloc[-1]).mean() * 100)
    except Exception:
        pass

    try:
        bbb_df = fred.fetch_series("BAMLC0A4CBBB", start_date="2020-01-01")
        if bbb_df is not None and not bbb_df.empty:
            bbb_oas_val = float(bbb_df.sort_values("date")["value"].iloc[-1]) * 100
    except Exception:
        pass

    try:
        aaa_df = fred.fetch_series("BAMLC0A1CAAA", start_date="2020-01-01")
        if aaa_df is not None and not aaa_df.empty:
            aaa_oas_val = float(aaa_df.sort_values("date")["value"].iloc[-1]) * 100
    except Exception:
        pass

    from lox.credit.regime import classify_credit
    result = classify_credit(
        hy_oas=hy_oas_val,
        bbb_oas=bbb_oas_val,
        aaa_oas=aaa_oas_val,
        hy_oas_30d_chg=hy_30d_chg,
        hy_oas_90d_percentile=hy_90d_pctl,
    )

    def _v(x, fmt="{:.0f}bp"):
        return fmt.format(x) if x is not None else "n/a"

    metrics = [
        {"name": "HY OAS", "value": _v(hy_oas_val), "context": "ICE BofA HY"},
        {"name": "30d Change", "value": _v(hy_30d_chg, "{:+.0f}bp"), "context": "widening > 0"},
        {"name": "90d Percentile", "value": f"{hy_90d_pctl:.0f}th" if hy_90d_pctl else "n/a", "context": "vs recent"},
        {"name": "BBB OAS", "value": _v(bbb_oas_val), "context": "IG benchmark"},
        {"name": "AAA OAS", "value": _v(aaa_oas_val), "context": "flight-to-quality"},
    ]

    print(render_regime_panel(
        domain="Credit",
        asof=asof,
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
            domain="credit",
            snapshot={"hy_oas": hy_oas_val, "bbb_oas": bbb_oas_val, "aaa_oas": aaa_oas_val, "hy_30d_chg": hy_30d_chg},
            regime_label=result.label,
            regime_description=result.description,
        )
        print(Panel(Markdown(analysis), title="Analysis", expand=False))

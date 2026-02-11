"""CLI command for the Credit regime."""
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
    hy_5d_chg = None
    hy_30d_chg = None
    bbb_30d_chg = None
    hy_90d_pctl = None
    hy_1y_pctl = None
    vix_val = None
    asof = "—"

    # HY OAS + velocity + percentiles
    try:
        hy_df = fred.fetch_series("BAMLH0A0HYM2", start_date="2020-01-01")
        if hy_df is not None and not hy_df.empty:
            hy_df = hy_df.sort_values("date")
            hy_oas_val = float(hy_df["value"].iloc[-1]) * 100  # pct pts → bps
            asof = str(hy_df["date"].iloc[-1].date())
            if len(hy_df) >= 5:
                hy_5d_chg = (float(hy_df["value"].iloc[-1]) - float(hy_df["value"].iloc[-5])) * 100
            if len(hy_df) >= 22:
                hy_30d_chg = (float(hy_df["value"].iloc[-1]) - float(hy_df["value"].iloc[-22])) * 100
            if len(hy_df) >= 63:
                recent = hy_df["value"].iloc[-63:]
                hy_90d_pctl = float((recent <= hy_df["value"].iloc[-1]).mean() * 100)
            if len(hy_df) >= 252:
                recent_1y = hy_df["value"].iloc[-252:]
                hy_1y_pctl = float((recent_1y <= hy_df["value"].iloc[-1]).mean() * 100)
    except Exception:
        pass

    # BBB OAS + velocity
    try:
        bbb_df = fred.fetch_series("BAMLC0A4CBBB", start_date="2020-01-01")
        if bbb_df is not None and not bbb_df.empty:
            bbb_df = bbb_df.sort_values("date")
            bbb_oas_val = float(bbb_df["value"].iloc[-1]) * 100
            if len(bbb_df) >= 22:
                bbb_30d_chg = (float(bbb_df["value"].iloc[-1]) - float(bbb_df["value"].iloc[-22])) * 100
    except Exception:
        pass

    # AAA OAS
    try:
        aaa_df = fred.fetch_series("BAMLC0A1CAAA", start_date="2020-01-01")
        if aaa_df is not None and not aaa_df.empty:
            aaa_oas_val = float(aaa_df.sort_values("date")["value"].iloc[-1]) * 100
    except Exception:
        pass

    # VIX for cross-market confirmation
    try:
        from lox.macro.signals import build_macro_state
        macro = build_macro_state(settings=settings, start_date="2020-01-01")
        vix_val = macro.inputs.vix
    except Exception:
        pass

    from lox.credit.regime import classify_credit
    result = classify_credit(
        hy_oas=hy_oas_val,
        bbb_oas=bbb_oas_val,
        aaa_oas=aaa_oas_val,
        hy_oas_30d_chg=hy_30d_chg,
        hy_oas_90d_percentile=hy_90d_pctl,
        hy_oas_1y_percentile=hy_1y_pctl,
        hy_oas_5d_chg=hy_5d_chg,
        bbb_oas_30d_chg=bbb_30d_chg,
        vix=vix_val,
    )

    def _v(x, fmt="{:.0f}bp"):
        return fmt.format(x) if x is not None else "n/a"

    metrics = [
        {"name": "HY OAS", "value": _v(hy_oas_val), "context": "ICE BofA HY"},
        {"name": "5d Change", "value": _v(hy_5d_chg, "{:+.0f}bp"), "context": "velocity"},
        {"name": "30d Change", "value": _v(hy_30d_chg, "{:+.0f}bp"), "context": "widening > 0"},
        {"name": "1Y Percentile", "value": f"{hy_1y_pctl:.0f}th" if hy_1y_pctl is not None else ("n/a" if hy_90d_pctl is None else f"{hy_90d_pctl:.0f}th (90d)"), "context": "vs history"},
        {"name": "BBB OAS", "value": _v(bbb_oas_val), "context": "IG benchmark"},
        {"name": "AAA OAS", "value": _v(aaa_oas_val), "context": "flight-to-quality"},
        {"name": "VIX", "value": f"{vix_val:.1f}" if vix_val is not None else "n/a", "context": "cross-market"},
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
        snapshot = {
            "hy_oas": hy_oas_val, "bbb_oas": bbb_oas_val, "aaa_oas": aaa_oas_val,
            "hy_5d_chg": hy_5d_chg, "hy_30d_chg": hy_30d_chg,
            "bbb_30d_chg": bbb_30d_chg, "vix": vix_val,
            "hy_1y_pctl": hy_1y_pctl,
        }
        analysis = llm_analyze_regime(
            settings=settings,
            domain="credit",
            snapshot=snapshot,
            regime_label=result.label,
            regime_description=result.description,
        )
        print(Panel(Markdown(analysis), title="Analysis", expand=False))

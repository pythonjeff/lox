"""CLI command for the Consumer regime (NEW — absorbs Housing)."""
from __future__ import annotations

from rich import print

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings


def consumer_snapshot(*, llm: bool = False) -> None:
    """Entry point for `lox regime consumer`."""
    settings = load_settings()
    from lox.data.fred import FredClient

    fred = FredClient(api_key=settings.FRED_API_KEY)

    michigan_sent = None
    michigan_exp = None
    retail_mom = None
    cc_debt_yoy = None
    mortgage_30y = None
    asof = "—"

    # Michigan Sentiment from FRED
    try:
        sent_df = fred.fetch_series("UMCSENT", start_date="2011-01-01")
        if sent_df is not None and not sent_df.empty:
            sent_df = sent_df.sort_values("date")
            michigan_sent = float(sent_df["value"].iloc[-1])
            asof = str(sent_df["date"].iloc[-1].date())
    except Exception:
        pass

    # Michigan Expectations from TE
    try:
        from lox.altdata.trading_economics import get_michigan_expectations
        michigan_exp = get_michigan_expectations()
    except Exception:
        pass

    # Retail Sales MoM from FRED
    try:
        rs_df = fred.fetch_series("RSXFS", start_date="2011-01-01")
        if rs_df is not None and len(rs_df) >= 2:
            rs_df = rs_df.sort_values("date")
            retail_mom = (rs_df["value"].iloc[-1] / rs_df["value"].iloc[-2] - 1.0) * 100.0
    except Exception:
        pass

    # Consumer credit YoY
    try:
        cc_df = fred.fetch_series("TOTALSL", start_date="2011-01-01")
        if cc_df is not None and len(cc_df) >= 13:
            cc_df = cc_df.sort_values("date")
            cc_debt_yoy = (cc_df["value"].iloc[-1] / cc_df["value"].iloc[-13] - 1.0) * 100.0
    except Exception:
        pass

    # Mortgage rate
    try:
        mtg_df = fred.fetch_series("MORTGAGE30US", start_date="2011-01-01")
        if mtg_df is not None and not mtg_df.empty:
            mortgage_30y = float(mtg_df.sort_values("date")["value"].iloc[-1])
    except Exception:
        pass

    from lox.consumer.regime import classify_consumer
    result = classify_consumer(
        michigan_sentiment=michigan_sent,
        michigan_expectations=michigan_exp,
        retail_sales_control_mom=retail_mom,
        personal_spending_mom=None,
        credit_card_debt_yoy_chg=cc_debt_yoy,
        mortgage_30y=mortgage_30y,
    )

    def _v(x, fmt="{:.1f}"):
        return fmt.format(x) if x is not None else "n/a"

    metrics = [
        {"name": "Michigan Sentiment", "value": _v(michigan_sent, "{:.0f}"), "context": "long-run avg ~85"},
        {"name": "Michigan Expectations", "value": _v(michigan_exp, "{:.0f}"), "context": "leads sentiment"},
        {"name": "Retail Sales MoM", "value": _v(retail_mom, "{:+.1f}%"), "context": "control group"},
        {"name": "Consumer Credit YoY", "value": _v(cc_debt_yoy, "{:+.1f}%"), "context": "TOTALSL"},
        {"name": "30Y Mortgage", "value": _v(mortgage_30y, "{:.2f}%"), "context": "housing drag"},
    ]

    print(render_regime_panel(
        domain="Consumer",
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
            domain="consumer",
            snapshot={"michigan": michigan_sent, "retail_mom": retail_mom, "mortgage_30y": mortgage_30y},
            regime_label=result.label,
            regime_description=result.description,
        )
        print(Panel(Markdown(analysis), title="Analysis", expand=False))

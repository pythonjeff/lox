"""CLI command for the Consumer regime (NEW — absorbs Housing)."""
from __future__ import annotations

from rich import print

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings


def consumer_snapshot(*, llm: bool = False, ticker: str = "", refresh: bool = False) -> None:
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
        sent_df = fred.fetch_series("UMCSENT", start_date="2011-01-01", refresh=refresh)
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
        rs_df = fred.fetch_series("RSXFS", start_date="2011-01-01", refresh=refresh)
        if rs_df is not None and len(rs_df) >= 2:
            rs_df = rs_df.sort_values("date")
            retail_mom = (rs_df["value"].iloc[-1] / rs_df["value"].iloc[-2] - 1.0) * 100.0
    except Exception:
        pass

    # Consumer credit YoY
    try:
        cc_df = fred.fetch_series("TOTALSL", start_date="2011-01-01", refresh=refresh)
        if cc_df is not None and len(cc_df) >= 13:
            cc_df = cc_df.sort_values("date")
            cc_debt_yoy = (cc_df["value"].iloc[-1] / cc_df["value"].iloc[-13] - 1.0) * 100.0
    except Exception:
        pass

    # Mortgage rate
    try:
        mtg_df = fred.fetch_series("MORTGAGE30US", start_date="2011-01-01", refresh=refresh)
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

    def _michigan_ctx():
        if michigan_sent is None:
            return "long-run avg ~85"
        v = michigan_sent
        if v > 95:
            return "optimistic — above long-run average"
        if v > 80:
            return "healthy — near historical norm"
        if v > 65:
            return "below average — consumers cautious"
        if v > 55:
            return "pessimistic — confidence eroding"
        return "deeply depressed — recessionary mood"

    def _michigan_exp_ctx():
        if michigan_exp is None:
            return "forward-looking expectations"
        v = michigan_exp
        if v > 85:
            return "optimistic outlook"
        if v > 70:
            return "moderate expectations"
        if v > 55:
            return "pessimistic — expect deterioration"
        return "deeply pessimistic — recessionary fear"

    def _retail_ctx():
        if retail_mom is None:
            return "control group MoM"
        v = retail_mom
        if v > 1.0:
            return "strong spending — consumers aggressive"
        if v > 0.3:
            return "healthy spending growth"
        if v > -0.3:
            return "flat — consumers treading water"
        if v > -1.0:
            return "pulling back — demand softening"
        return "sharp decline — demand destruction"

    def _credit_ctx():
        if cc_debt_yoy is None:
            return "consumer credit growth"
        v = cc_debt_yoy
        if v > 8:
            return "rapid leverage — unsustainable pace"
        if v > 5:
            return "brisk borrowing — watch delinquencies"
        if v > 2:
            return "normal credit growth"
        if v > 0:
            return "slow growth — consumers cautious"
        return "deleveraging — credit contraction"

    def _mortgage_ctx():
        if mortgage_30y is None:
            return "30Y mortgage rate"
        v = mortgage_30y
        if v > 7.5:
            return "crisis affordability — housing frozen"
        if v > 7.0:
            return "severe affordability headwind"
        if v > 6.5:
            return "elevated — housing drag"
        if v > 5.5:
            return "above normal — moderate drag"
        if v > 4.5:
            return "neutral — manageable"
        return "low — housing tailwind"

    metrics = [
        {"name": "Michigan Sentiment", "value": _v(michigan_sent, "{:.0f}"), "context": _michigan_ctx()},
        {"name": "Michigan Expectations", "value": _v(michigan_exp, "{:.0f}"), "context": _michigan_exp_ctx()},
        {"name": "Retail Sales MoM", "value": _v(retail_mom, "{:+.1f}%"), "context": _retail_ctx()},
        {"name": "Consumer Credit YoY", "value": _v(cc_debt_yoy, "{:+.1f}%"), "context": _credit_ctx()},
        {"name": "30Y Mortgage", "value": _v(mortgage_30y, "{:.2f}%"), "context": _mortgage_ctx()},
    ]

    from lox.regimes.trend import get_domain_trend
    trend = get_domain_trend("consumer", result.score, result.label)

    print(render_regime_panel(
        domain="Consumer",
        asof=asof,
        regime_label=result.label,
        score=result.score,
        percentile=None,
        description=result.description,
        metrics=metrics,
        trend=trend,
    ))

    if llm:
        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis
        print_llm_regime_analysis(
            settings=settings,
            domain="consumer",
            snapshot={"michigan": michigan_sent, "retail_mom": retail_mom, "mortgage_30y": mortgage_30y},
            regime_label=result.label,
            regime_description=result.description,
            ticker=ticker,
        )

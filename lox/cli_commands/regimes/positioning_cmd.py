"""CLI command for the Positioning regime (NEW)."""
from __future__ import annotations

from rich import print

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings


def positioning_snapshot(*, llm: bool = False, ticker: str = "", refresh: bool = False) -> None:
    """Entry point for `lox regime positioning`."""
    settings = load_settings()

    vix_term_slope = None
    put_call = None
    aaii_bull = None
    asof = "—"

    # VIX term structure from volatility state
    try:
        from lox.volatility.signals import build_volatility_state
        vol_state = build_volatility_state(settings=settings, start_date="2020-01-01", refresh=refresh)
        asof = vol_state.asof
        inp = vol_state.inputs
        if inp.vix is not None and inp.vix_term_spread is not None and inp.vix > 0:
            vix3m = inp.vix - inp.vix_term_spread
            vix_term_slope = vix3m / inp.vix
    except Exception:
        pass

    # AAII Sentiment from Trading Economics
    try:
        from lox.altdata.trading_economics import get_aaii_bullish_sentiment
        aaii_bull = get_aaii_bullish_sentiment()
    except Exception:
        pass

    from lox.positioning.regime import classify_positioning
    result = classify_positioning(
        vix_term_slope=vix_term_slope,
        put_call_ratio=put_call,
        aaii_bull_pct=aaii_bull,
    )

    def _v(x, fmt="{:.2f}"):
        return fmt.format(x) if x is not None else "n/a"

    metrics = [
        {"name": "VIX Term Slope", "value": _v(vix_term_slope, "{:.2f}x"), "context": ">1 contango, <1 backwardation"},
        {"name": "Put/Call Ratio", "value": _v(put_call), "context": "equity options"},
        {"name": "AAII Bullish %", "value": _v(aaii_bull, "{:.0f}%"), "context": "contrarian signal"},
    ]

    print(render_regime_panel(
        domain="Positioning",
        asof=asof,
        regime_label=result.label,
        score=result.score,
        percentile=None,
        description=result.description,
        metrics=metrics,
    ))

    if llm:
        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis
        print_llm_regime_analysis(
            settings=settings,
            domain="positioning",
            snapshot={"vix_term_slope": vix_term_slope, "put_call": put_call, "aaii_bull": aaii_bull},
            regime_label=result.label,
            regime_description=result.description,
            ticker=ticker,
        )

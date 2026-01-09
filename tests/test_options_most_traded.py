from __future__ import annotations

from datetime import date

from ai_options_trader.data.alpaca import OptionCandidate
from ai_options_trader.options.most_traded import most_traded_options


def _c(
    symbol: str,
    *,
    volume: int | None = None,
    oi: int | None = None,
    bid: float | None = None,
    ask: float | None = None,
    delta: float | None = None,
    gamma: float | None = None,
    theta: float | None = None,
    vega: float | None = None,
    iv: float | None = None,
) -> OptionCandidate:
    return OptionCandidate(
        symbol=symbol,
        opt_type="",
        expiry=date(1970, 1, 1),
        strike=0.0,
        dte_days=0,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        iv=iv,
        oi=oi,
        volume=volume,
        bid=bid,
        ask=ask,
        last=None,
    )


def test_most_traded_filters_by_dte_and_type_and_ranks_by_volume():
    today = date(2026, 1, 7)
    # 30D out, call
    c1 = _c("SPY260206C00600000", volume=5000, oi=10000)
    # 10D out, call
    c2 = _c("SPY260117C00600000", volume=9000, oi=2000)
    # 100D out, call (excluded)
    c3 = _c("SPY260417C00600000", volume=20000, oi=1)
    # 20D out, put
    p1 = _c("SPY260127P00600000", volume=8000, oi=3000)

    ranked = most_traded_options(
        [c1, c2, c3, p1],
        ticker="SPY",
        min_dte_days=0,
        max_dte_days=90,
        want="call",
        top=10,
        today=today,
    )
    assert [x.symbol for x in ranked] == [c2.symbol, c1.symbol]


def test_most_traded_sort_by_open_interest_tiebreaks_volume():
    today = date(2026, 1, 7)
    a = _c("SPY260117C00600000", volume=1000, oi=5000)
    b = _c("SPY260117C00605000", volume=9000, oi=2000)
    ranked = most_traded_options(
        [a, b],
        ticker="SPY",
        min_dte_days=0,
        max_dte_days=90,
        want="both",
        top=10,
        sort="open_interest",
        today=today,
    )
    assert [x.symbol for x in ranked] == [a.symbol, b.symbol]


def test_most_traded_volume_override_takes_precedence():
    today = date(2026, 1, 7)
    a = _c("SPY260117C00600000", volume=None, oi=None)
    b = _c("SPY260117C00605000", volume=None, oi=None)
    vol_map = {a.symbol: 10, b.symbol: 200}
    ranked = most_traded_options(
        [a, b],
        ticker="SPY",
        min_dte_days=0,
        max_dte_days=90,
        want="both",
        top=10,
        sort="volume",
        volume_by_symbol=vol_map,
        today=today,
    )
    assert [x.symbol for x in ranked] == [b.symbol, a.symbol]


def test_most_traded_sort_by_abs_delta():
    today = date(2026, 1, 7)
    a = _c("SPY260117C00600000", delta=0.10, volume=1)
    b = _c("SPY260117P00600000", delta=-0.55, volume=1)
    ranked = most_traded_options(
        [a, b],
        ticker="SPY",
        min_dte_days=0,
        max_dte_days=90,
        want="both",
        top=10,
        sort="abs_delta",
        today=today,
    )
    assert [x.symbol for x in ranked] == [b.symbol, a.symbol]


def test_most_traded_sort_by_iv():
    today = date(2026, 1, 7)
    a = _c("SPY260117C00600000", iv=0.10, volume=1)
    b = _c("SPY260117C00605000", iv=0.35, volume=1)
    ranked = most_traded_options(
        [a, b],
        ticker="SPY",
        min_dte_days=0,
        max_dte_days=90,
        want="both",
        top=10,
        sort="iv",
        today=today,
    )
    assert [x.symbol for x in ranked] == [b.symbol, a.symbol]


def test_most_traded_sort_by_gamma():
    today = date(2026, 1, 7)
    a = _c("SPY260117C00600000", gamma=0.01, volume=1)
    b = _c("SPY260117C00605000", gamma=0.10, volume=1)
    ranked = most_traded_options(
        [a, b],
        ticker="SPY",
        min_dte_days=0,
        max_dte_days=90,
        want="both",
        top=10,
        sort="gamma",
        today=today,
    )
    assert [x.symbol for x in ranked] == [b.symbol, a.symbol]



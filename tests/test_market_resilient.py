from __future__ import annotations

import pandas as pd

import lox.data.market as market


def _frame(sym: str) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    return pd.DataFrame({sym: range(10)}, index=idx)


def test_resilient_fetch_skips_the_one_bad_ticker(monkeypatch):
    """A delisted name must not take the whole panel down with it."""
    def fake(*, settings, symbols, start, refresh=False):
        if "DEAD" in symbols:
            raise RuntimeError("FMP returned no historical data for DEAD.")
        return pd.concat([_frame(s) for s in symbols], axis=1)

    monkeypatch.setattr(market, "fetch_equity_daily_closes", fake)

    panel, failed = market.fetch_equity_daily_closes_resilient(
        settings=object(), symbols=["AAA", "DEAD", "BBB"], start="2024-01-01", chunk_size=3,
    )
    assert failed == ["DEAD"]
    assert sorted(panel.columns) == ["AAA", "BBB"]
    assert len(panel) == 10


def test_resilient_fetch_returns_empty_when_everything_fails(monkeypatch):
    def fake(*, settings, symbols, start, refresh=False):
        raise RuntimeError("no key")

    monkeypatch.setattr(market, "fetch_equity_daily_closes", fake)

    panel, failed = market.fetch_equity_daily_closes_resilient(
        settings=object(), symbols=["AAA", "BBB"], start="2024-01-01",
    )
    assert panel.empty
    assert sorted(failed) == ["AAA", "BBB"]


def test_resilient_fetch_dedupes_and_normalizes_symbols(monkeypatch):
    seen: list[list[str]] = []

    def fake(*, settings, symbols, start, refresh=False):
        seen.append(list(symbols))
        return pd.concat([_frame(s) for s in symbols], axis=1)

    monkeypatch.setattr(market, "fetch_equity_daily_closes", fake)

    panel, failed = market.fetch_equity_daily_closes_resilient(
        settings=object(), symbols=[" aaa ", "AAA", "bbb"], start="2024-01-01",
    )
    assert seen == [["AAA", "BBB"]]
    assert sorted(panel.columns) == ["AAA", "BBB"]
    assert failed == []


def test_resilient_fetch_handles_empty_input():
    panel, failed = market.fetch_equity_daily_closes_resilient(
        settings=object(), symbols=[], start="2024-01-01",
    )
    assert panel.empty
    assert failed == []


def test_resilient_fetch_chunks_large_lists(monkeypatch):
    calls: list[int] = []

    def fake(*, settings, symbols, start, refresh=False):
        calls.append(len(symbols))
        return pd.concat([_frame(s) for s in symbols], axis=1)

    monkeypatch.setattr(market, "fetch_equity_daily_closes", fake)

    symbols = [f"T{i}" for i in range(55)]
    panel, failed = market.fetch_equity_daily_closes_resilient(
        settings=object(), symbols=symbols, start="2024-01-01", chunk_size=25,
    )
    assert calls == [25, 25, 5]
    assert len(panel.columns) == 55

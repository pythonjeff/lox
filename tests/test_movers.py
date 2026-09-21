from __future__ import annotations

import numpy as np

from lox.suggest.movers import (
    _direction_from_momentum,
    build_structure,
    passes_liquidity,
    quote_movement_proxy,
)
from lox.suggest.signals.kinetics import compute_kinetics


def _quote(**kw) -> dict:
    base = {
        "symbol": "TEST",
        "price": 100.0,
        "previousClose": 100.0,
        "dayHigh": 101.0,
        "dayLow": 99.0,
        "yearHigh": 120.0,
        "yearLow": 80.0,
        "volume": 1_000_000,
        "avgVolume": 1_000_000,
        "changesPercentage": 0.5,
    }
    base.update(kw)
    return base


class _Mom:
    def __init__(self, signal: str, trend_quality: str = "RANGE_BOUND"):
        self.signal = signal
        self.trend_quality = trend_quality


# ── liquidity gate ───────────────────────────────────────────────────────────

def test_liquidity_gate_rejects_penny_and_thin_names():
    assert passes_liquidity(_quote(), min_price=5.0, min_dollar_volume=20_000_000)
    assert not passes_liquidity(_quote(price=2.0), min_price=5.0, min_dollar_volume=1_000)
    assert not passes_liquidity(
        _quote(avgVolume=1_000), min_price=5.0, min_dollar_volume=20_000_000,
    )


def test_liquidity_gate_tolerates_junk_values():
    assert not passes_liquidity(
        {"price": None, "avgVolume": "n/a"}, min_price=5.0, min_dollar_volume=1.0,
    )


# ── quote prefilter ──────────────────────────────────────────────────────────

def test_movement_proxy_prefers_the_wider_range():
    narrow = quote_movement_proxy(_quote(yearHigh=104.0, yearLow=96.0))
    wide = quote_movement_proxy(_quote(yearHigh=180.0, yearLow=40.0))
    assert wide > narrow


def test_movement_proxy_rewards_a_big_intraday_range():
    calm = quote_movement_proxy(_quote(dayHigh=100.5, dayLow=99.5))
    volatile = quote_movement_proxy(_quote(dayHigh=112.0, dayLow=94.0))
    assert volatile > calm


def test_movement_proxy_is_zero_without_a_price():
    assert quote_movement_proxy({"symbol": "X"}) == 0.0
    assert quote_movement_proxy(_quote(price=0)) == 0.0


# ── direction ────────────────────────────────────────────────────────────────

def test_direction_from_momentum_signals():
    assert _direction_from_momentum(_Mom("BREAKOUT")) == "LONG"
    assert _direction_from_momentum(_Mom("TRENDING_DOWN")) == "SHORT"
    assert _direction_from_momentum(_Mom("NEUTRAL", "STRONG_DOWN")) == "SHORT"
    assert _direction_from_momentum(_Mom("NEUTRAL", "RANGE_BOUND")) == "NEUTRAL"
    assert _direction_from_momentum(None) == "NEUTRAL"


def test_extended_up_in_a_strong_trend_is_not_auto_short():
    assert _direction_from_momentum(_Mom("EXTENDED_UP", "STRONG_UP")) == "LONG"
    assert _direction_from_momentum(_Mom("EXTENDED_UP", "RANGE_BOUND")) == "SHORT"


# ── structure mapping ────────────────────────────────────────────────────────

def _kinetics(vol: float, drift: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(drift, vol, 150))
    k = compute_kinetics(closes, ticker="T")
    assert k is not None
    return k


def test_trending_mover_gets_a_directional_structure():
    k = _kinetics(0.02, drift=0.012, seed=5)
    structure, kind, _ = build_structure(k, direction="LONG")
    assert kind == "DIRECTIONAL"
    assert "call" in structure


def test_short_direction_flips_to_puts():
    k = _kinetics(0.02, drift=-0.012, seed=5)
    _, kind, _ = build_structure(k, direction="SHORT")
    structure, _, _ = build_structure(k, direction="SHORT")
    assert kind == "DIRECTIONAL"
    assert "put" in structure


def test_imminent_earnings_overrides_everything():
    k = _kinetics(0.02, drift=0.012, seed=5)
    structure, kind, _ = build_structure(k, direction="LONG", days_to_earnings=3)
    assert kind == "EVENT"
    assert "Earnings in 3d" in structure


def test_far_off_earnings_does_not_override():
    k = _kinetics(0.02, drift=0.012, seed=5)
    _, kind, _ = build_structure(k, direction="LONG", days_to_earnings=45)
    assert kind == "DIRECTIONAL"


def test_choppy_name_with_expanding_vol_goes_long_vol():
    rng = np.random.default_rng(3)
    calm = rng.normal(0, 0.006, 90)
    loud = rng.normal(0, 0.035, 25)
    closes = 100.0 * np.cumprod(1.0 + np.concatenate([calm, loud]))
    k = compute_kinetics(closes, ticker="T")
    assert k is not None and k.vol_expansion > 1.15
    if k.character == "CHOPPY":
        _, kind, _ = build_structure(k, direction="NEUTRAL")
        assert kind == "LONG_VOL"


def test_structure_always_returns_three_parts():
    k = _kinetics(0.02, seed=9)
    for direction in ("LONG", "SHORT", "NEUTRAL"):
        structure, kind, notes = build_structure(k, direction=direction)
        assert structure and kind and notes


# ── end-to-end orchestration (hermetic: all network boundaries stubbed) ──────

def _panel(tickers: dict[str, tuple[float, float]]):
    """Build a close panel: {ticker: (daily_vol, drift)}."""
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=320, freq="B")
    data = {}
    for i, (t, (vol, drift)) in enumerate(tickers.items()):
        rng = np.random.default_rng(100 + i)
        data[t] = 100.0 * np.cumprod(1.0 + rng.normal(drift, vol, len(idx)))
    return pd.DataFrame(data, index=idx)


def test_run_movers_scan_end_to_end(monkeypatch):
    import lox.altdata.fmp as fmp
    import lox.data.market as market
    import lox.suggest.signals.catalyst as catalyst
    import lox.universe.sp500 as universe

    tickers = {
        "WILD": (0.040, 0.000),    # huge range
        "TREND": (0.022, 0.010),   # moves and goes somewhere
        "SLEEPY": (0.003, 0.000),  # barely moves
        "SPY": (0.009, 0.001),
    }
    panel = _panel(tickers)

    quotes = [
        _quote(symbol=t, price=float(panel[t].iloc[-1]),
               previousClose=float(panel[t].iloc[-2]),
               yearHigh=float(panel[t].max()), yearLow=float(panel[t].min()),
               dayHigh=float(panel[t].iloc[-1]) * 1.01,
               dayLow=float(panel[t].iloc[-1]) * 0.99,
               avgVolume=5_000_000, volume=6_000_000, name=f"{t} Corp")
        for t in tickers
    ]

    monkeypatch.setattr(universe, "build_scan_universe", lambda settings: list(tickers))
    monkeypatch.setattr(fmp, "fetch_batch_quotes_full", lambda **kw: quotes)
    monkeypatch.setattr(
        market, "fetch_equity_daily_closes_resilient",
        lambda **kw: (panel[[t for t in kw["symbols"] if t in panel.columns]], []),
    )
    monkeypatch.setattr(catalyst, "score_catalyst", lambda **kw: {})

    from lox.suggest.movers import run_movers_scan

    result = run_movers_scan(settings=object(), count=10, min_dollar_volume=1_000)

    assert result.universe_size == 4
    assert result.scored >= 3
    names = [c.ticker for c in result.candidates]
    assert names[0] in ("WILD", "TREND")
    assert names.index("WILD") < names.index("SLEEPY")

    for c in result.candidates:
        assert c.structure and c.handoff
        assert c.kinetics.sub_score >= 0

    trend = next(c for c in result.candidates if c.ticker == "TREND")
    assert trend.direction in ("LONG", "SHORT", "NEUTRAL")
    assert trend.kinetics.expected_move_21d_pct > 0


def test_run_movers_scan_respects_liquidity_gate(monkeypatch):
    import lox.altdata.fmp as fmp
    import lox.data.market as market
    import lox.suggest.signals.catalyst as catalyst
    import lox.universe.sp500 as universe

    panel = _panel({"THIN": (0.04, 0.0), "THICK": (0.04, 0.0)})
    quotes = [
        _quote(symbol="THIN", price=50.0, avgVolume=1_000),
        _quote(symbol="THICK", price=50.0, avgVolume=10_000_000),
    ]

    monkeypatch.setattr(universe, "build_scan_universe", lambda settings: ["THIN", "THICK"])
    monkeypatch.setattr(fmp, "fetch_batch_quotes_full", lambda **kw: quotes)
    monkeypatch.setattr(
        market, "fetch_equity_daily_closes_resilient",
        lambda **kw: (panel[[t for t in kw["symbols"] if t in panel.columns]], []),
    )
    monkeypatch.setattr(catalyst, "score_catalyst", lambda **kw: {})

    from lox.suggest.movers import run_movers_scan

    result = run_movers_scan(settings=object(), count=10, min_dollar_volume=20_000_000)
    assert [c.ticker for c in result.candidates] == ["THICK"]
    assert result.liquidity_survivors == 1


def test_run_movers_scan_survives_no_quotes(monkeypatch):
    import lox.altdata.fmp as fmp
    import lox.universe.sp500 as universe

    monkeypatch.setattr(universe, "build_scan_universe", lambda settings: ["AAA"])
    monkeypatch.setattr(fmp, "fetch_batch_quotes_full", lambda **kw: [])

    from lox.suggest.movers import run_movers_scan

    result = run_movers_scan(settings=object(), count=5)
    assert result.candidates == []
    assert result.universe_size == 1


def test_quiet_name_gets_no_trade():
    """A movement screen must not recommend a trade on a name that isn't moving."""
    rng = np.random.default_rng(21)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.003, 150))
    k = compute_kinetics(closes, ticker="SLEEPY")
    assert k is not None and k.character == "QUIET"

    for direction in ("LONG", "SHORT", "NEUTRAL"):
        structure, kind, _ = build_structure(k, direction=direction)
        assert kind == "WATCH"
        assert "No trade" in structure


def test_short_structure_matches_the_kind():
    from lox.suggest.movers import short_structure

    k = _kinetics(0.02, drift=0.012, seed=5)
    assert "call" in short_structure(k, kind="DIRECTIONAL", direction="LONG")
    assert "put" in short_structure(k, kind="DIRECTIONAL", direction="SHORT")
    assert short_structure(k, kind="LONG_VOL", direction="NEUTRAL").startswith("Long vol")
    assert short_structure(k, kind="SHORT_VOL", direction="NEUTRAL").startswith("Short vol")
    assert short_structure(k, kind="WATCH", direction="NEUTRAL").startswith("Watch")

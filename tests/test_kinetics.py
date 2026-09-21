from __future__ import annotations

import numpy as np
import pandas as pd

from lox.suggest.signals.kinetics import (
    _interp_curve,
    compute_kinetics,
    score_kinetics,
)


def _series(vol: float, drift: float = 0.0, n: int = 150, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.cumprod(1.0 + rng.normal(drift, vol, n))


def test_interp_curve_clamps_and_interpolates():
    curve = [(0.0, 0.0), (1.0, 10.0), (2.0, 20.0)]
    assert _interp_curve(-5.0, curve) == 0.0
    assert _interp_curve(99.0, curve) == 20.0
    assert _interp_curve(0.5, curve) == 5.0
    assert _interp_curve(1.5, curve) == 15.0


def test_returns_none_without_enough_history():
    assert compute_kinetics(np.array([100.0, 101.0, 102.0])) is None


def test_big_mover_outscores_quiet_name():
    quiet = compute_kinetics(_series(0.004, seed=1), ticker="QUIET")
    wild = compute_kinetics(_series(0.035, seed=1), ticker="WILD")
    assert quiet is not None and wild is not None
    assert wild.sub_score > quiet.sub_score
    assert wild.avg_abs_move_pct > quiet.avg_abs_move_pct
    assert wild.move_freq_pct > quiet.move_freq_pct


def test_move_frequency_counts_sessions_over_the_bar():
    # Alternating +3% / -3% moves: every session clears a 2% bar, none clears 5%.
    closes = [100.0]
    for i in range(80):
        closes.append(closes[-1] * (1.03 if i % 2 == 0 else 1 / 1.03))
    arr = np.array(closes)

    at_2pct = compute_kinetics(arr, move_threshold=0.02)
    at_5pct = compute_kinetics(arr, move_threshold=0.05)
    assert at_2pct is not None and at_5pct is not None
    assert at_2pct.move_freq_pct == 100.0
    assert at_5pct.move_freq_pct == 0.0


def test_one_gap_does_not_look_like_a_persistent_mover():
    """A name that sat still then gapped once should score below a steady mover."""
    flat = np.full(90, 100.0)
    # tiny jitter so stdev isn't exactly zero
    rng = np.random.default_rng(7)
    gapper = flat * (1.0 + rng.normal(0, 0.0015, 90))
    gapper[-20:] *= 1.30  # single 30% gap, then flat again

    steady = _series(0.025, seed=3)

    g = compute_kinetics(gapper, ticker="GAP")
    s = compute_kinetics(steady, ticker="STEADY")
    assert g is not None and s is not None
    assert g.persistence < s.persistence
    assert s.sub_score > g.sub_score


def test_trend_efficiency_separates_trend_from_chop():
    trending = compute_kinetics(_series(0.02, drift=0.012, seed=5))
    choppy = compute_kinetics(_series(0.02, drift=0.0, seed=5))
    assert trending is not None and choppy is not None
    assert trending.trend_efficiency > choppy.trend_efficiency
    assert trending.character == "TRENDER"


def test_vol_expansion_flags_a_name_waking_up():
    rng = np.random.default_rng(11)
    calm = rng.normal(0, 0.004, 80)
    loud = rng.normal(0, 0.030, 25)
    closes = 100.0 * np.cumprod(1.0 + np.concatenate([calm, loud]))

    k = compute_kinetics(closes)
    assert k is not None
    assert k.vol_expansion > 1.5
    assert k.rv_20d > k.rv_60d


def test_expected_move_scales_with_realized_vol():
    k = compute_kinetics(_series(0.02, seed=2))
    assert k is not None
    # 1-sigma monthly move should be roughly rv_20 * sqrt(21/252)
    expected = k.rv_20d * (21.0 / 252.0) ** 0.5 * 100.0
    assert abs(k.expected_move_21d_pct - expected) < 0.05
    assert k.expected_daily_move_pct < k.expected_move_21d_pct


def test_score_kinetics_skips_missing_and_short_columns():
    idx = pd.date_range("2024-01-01", periods=150, freq="B")
    panel = pd.DataFrame({
        "AAA": _series(0.02, seed=4),
        "SHORT": [np.nan] * 140 + list(_series(0.02, n=10, seed=4)),
    }, index=idx)

    out = score_kinetics(price_panel=panel, tickers=["AAA", "SHORT", "MISSING"])
    assert "AAA" in out
    assert "SHORT" not in out
    assert "MISSING" not in out


def test_score_kinetics_handles_empty_panel():
    assert score_kinetics(price_panel=pd.DataFrame(), tickers=["AAA"]) == {}


def test_gapper_needs_concentration_not_just_fat_tails():
    """A high-vol name with an outsized worst day is still a mover, not an event name."""
    rng = np.random.default_rng(31)
    returns = rng.normal(0, 0.035, 90)
    returns[40] = -0.17  # one bad day, ~5x the average, but the name moves daily anyway
    volatile = 100.0 * np.cumprod(1.0 + returns)

    k = compute_kinetics(volatile, ticker="VOLATILE")
    assert k is not None
    assert k.character != "GAPPER"
    assert k.move_freq_pct > 40


def test_gapper_detected_when_one_session_dominates_the_range():
    rng = np.random.default_rng(33)
    returns = rng.normal(0, 0.006, 90)
    returns[70] = 0.34  # the whole quarter's movement in a single print
    closes = 100.0 * np.cumprod(1.0 + returns)

    k = compute_kinetics(closes, ticker="EVENT")
    assert k is not None
    assert k.character == "GAPPER"

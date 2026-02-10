from __future__ import annotations

import numpy as np
import pandas as pd

from lox.options.moonshot import rank_moonshots, rank_moonshots_unconditional


def test_rank_moonshots_prefers_true_extreme_in_analogs():
    # Build a toy regime matrix where "asof" regime matches a cluster of past dates.
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    # Use constant regimes so the analog selector deterministically includes "early" dates;
    # we'll place our synthetic extreme within that window.
    regimes = pd.DataFrame({"usd_strength_score": 0.0, "rates_z_ust_10y": 0.0, "vol_z_vix": 0.0}, index=idx)

    # Price data: AAA has a single huge +40% forward move at one of the close analog dates;
    # BBB does not.
    px = pd.DataFrame(index=idx, data={"AAA": 100.0, "BBB": 100.0})
    # Simulate mild random walk
    rng = np.random.default_rng(0)
    px["AAA"] = 100.0 * (1.0 + pd.Series(rng.normal(0, 0.002, len(idx)), index=idx)).cumprod()
    px["BBB"] = 100.0 * (1.0 + pd.Series(rng.normal(0, 0.002, len(idx)), index=idx)).cumprod()

    # Force an extreme move in AAA around day 200 over horizon=7.
    event_day = idx[200]
    px.loc[event_day + pd.Timedelta(days=7), "AAA"] = px.loc[event_day, "AAA"] * 1.40

    ranked = rank_moonshots(
        px=px,
        regimes=regimes,
        asof=idx[-10],
        horizon_days=7,
        k_analogs=250,
        min_abs_extreme=0.25,
        min_samples=40,
        direction="bullish",
    )

    assert ranked, "Expected at least one moonshot candidate"
    # AAA should be the top pick due to the injected analog extreme.
    assert ranked[0].ticker == "AAA"
    assert ranked[0].direction == "bullish"
    assert ranked[0].best is not None and ranked[0].best >= 0.25


def test_rank_moonshots_unconditional_returns_candidates():
    idx = pd.date_range("2020-01-01", periods=260, freq="D")
    px = pd.DataFrame(index=idx, data={"AAA": 100.0, "BBB": 100.0})
    rng = np.random.default_rng(1)
    px["AAA"] = 100.0 * (1.0 + pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)).cumprod()
    px["BBB"] = 100.0 * (1.0 + pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)).cumprod()
    ranked = rank_moonshots_unconditional(px=px, asof=idx[-5], horizon_days=21, min_samples=100, direction="both")
    assert ranked


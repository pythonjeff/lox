"""Smoke tests for Fed balance sheet composition (lox.funding.fed_bs)."""
from __future__ import annotations

import pandas as pd

import lox.funding.fed_bs as fbs


class _FakeFred:
    """Returns plausible weekly Wednesday H.4.1 series with a known YTD trajectory."""

    def __init__(self, *args, **kwargs):
        pass

    def fetch_series(self, *, series_id, start_date, refresh):
        # 60 weekly Wednesdays starting Mar 2025 → through ~April 2026.
        # Spans a calendar year boundary so YTD logic is exercised.
        idx = pd.date_range("2025-03-05", periods=60, freq="W-WED")
        n = len(idx)

        # Baseline values ($ millions) modeled on real H.4.1 ballpark figures,
        # with directional trends that let signal logic be tested.
        if series_id == "WALCL":
            # Slowly declining (QT continuing): 7.5T → 7.3T
            return pd.DataFrame({"date": idx, "value": [7_500_000 - 3_500 * i for i in range(n)]})
        if series_id == "WSHOBL":  # bills — rising into year-end
            return pd.DataFrame({"date": idx, "value": [190_000 + 800 * i for i in range(n)]})
        if series_id == "WSHONBNL":  # notes/bonds — passive runoff
            return pd.DataFrame({"date": idx, "value": [4_400_000 - 2_000 * i for i in range(n)]})
        if series_id == "WSHONBIIL":  # TIPS — flat
            return pd.DataFrame({"date": idx, "value": [380_000] * n})
        if series_id == "WSHOFADSL":  # agency — flat residual
            return pd.DataFrame({"date": idx, "value": [2_000] * n})
        if series_id == "WSHOMCB":  # MBS runoff
            return pd.DataFrame({"date": idx, "value": [2_200_000 - 1_500 * i for i in range(n)]})
        if series_id == "WORAL":  # asset-side repo — zero (no SRF usage)
            return pd.DataFrame({"date": idx, "value": [0] * n})
        if series_id == "WLCFLPCL":  # discount window — calm
            return pd.DataFrame({"date": idx, "value": [1_500 + (i % 5) * 100 for i in range(n)]})
        if series_id == "H41RESPPALDKNWW":  # BTFP — winding down
            return pd.DataFrame({"date": idx, "value": [max(0, 80_000 - 1_000 * i) for i in range(n)]})
        raise ValueError(f"unexpected series {series_id}")


def _patch(monkeypatch):
    from lox.config import Settings
    monkeypatch.setattr(fbs, "FredClient", _FakeFred)
    monkeypatch.setattr(fbs, "load_settings", lambda: Settings(FRED_API_KEY="fake"))


def test_fetch_composition_returns_aligned_frame(monkeypatch):
    _patch(monkeypatch)
    df = fbs.fetch_fed_bs_composition()
    assert not df.empty
    # WALCL anchor plus all 8 child components
    for col in ("total_assets", "bills", "notes_bonds", "tips", "agency", "mbs",
                "repo", "primary_credit", "btfp"):
        assert col in df.columns
    # Weekly cadence — date column is monotonic
    assert df["date"].is_monotonic_increasing


def test_metrics_shape_and_signals(monkeypatch):
    _patch(monkeypatch)
    m = fbs.compute_fed_bs_metrics()
    assert isinstance(m["asof"], str)
    comps = m["components"]
    assert set(comps.keys()) == {
        "total_assets", "bills", "notes_bonds", "tips", "agency", "mbs",
        "repo", "primary_credit", "btfp",
    }

    total = comps["total_assets"]
    assert total["leg"] == "total"
    # QT trajectory: 13w delta should be negative
    assert total["delta_13w_m"] < 0
    # YTD also negative (crossed Jan 1 inside the synthetic series)
    assert total["delta_ytd_m"] is not None and total["delta_ytd_m"] < 0

    bills = comps["bills"]
    assert bills["leg"] == "purchase"
    assert bills["delta_13w_m"] > 0  # accumulating

    notes = comps["notes_bonds"]
    assert notes["delta_13w_m"] < 0  # running off

    btfp = comps["btfp"]
    assert btfp["leg"] == "lending"
    assert btfp["delta_13w_m"] < 0  # winding down


def test_no_api_key(monkeypatch):
    from lox.config import Settings
    monkeypatch.setattr(fbs, "load_settings", lambda: Settings(FRED_API_KEY=None))
    m = fbs.compute_fed_bs_metrics()
    assert m == {"asof": None, "components": None}

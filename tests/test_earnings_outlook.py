"""Smoke tests for the earnings outlook compute pipeline."""
from __future__ import annotations

from datetime import date, timedelta

from lox.cli_commands.research.ticker.compute import compute_earnings_outlook


def _mk_calendar(today: date) -> list[dict]:
    """3 past prints + 1 upcoming, all roughly quarterly."""
    return [
        {  # upcoming
            "date": (today + timedelta(days=18)).isoformat(),
            "fiscalDateEnding": (today + timedelta(days=10)).isoformat(),
            "epsEstimated": 2.10,
            "revenueEstimated": 90_000_000_000,
            "time": "amc",
            "eps": None,
            "revenue": None,
        },
        {  # 90 days ago — beat
            "date": (today - timedelta(days=90)).isoformat(),
            "fiscalDateEnding": (today - timedelta(days=98)).isoformat(),
            "eps": 2.05, "epsEstimated": 1.95,
            "revenue": 87_000_000_000, "revenueEstimated": 85_000_000_000,
        },
        {  # 180 days ago — miss
            "date": (today - timedelta(days=180)).isoformat(),
            "fiscalDateEnding": (today - timedelta(days=188)).isoformat(),
            "eps": 1.50, "epsEstimated": 1.60,
            "revenue": 80_000_000_000, "revenueEstimated": 82_000_000_000,
        },
        {  # 365 days ago — prior year of upcoming Q (for YoY)
            "date": (today - timedelta(days=365)).isoformat(),
            "fiscalDateEnding": (today + timedelta(days=10) - timedelta(days=365)).isoformat(),
            "eps": 1.80, "epsEstimated": 1.75,
            "revenue": 78_000_000_000, "revenueEstimated": 77_000_000_000,
        },
    ]


def _mk_price(today: date) -> dict:
    """Synthetic daily close series covering past earnings dates."""
    closes = []
    for i in range(400, -1, -1):
        d = today - timedelta(days=i)
        # Drift up with small daily noise — produces non-zero post-earnings moves
        closes.append({"date": d.isoformat(), "close": 100.0 + i * 0.05})
    return {"historical": closes}


def test_basic_pipeline():
    today = date.today()
    ed = {
        "earnings_calendar": _mk_calendar(today),
        "analyst_estimates": [
            {"date": f"{today.year + 1}-09-25", "estimatedEpsAvg": 9.50, "estimatedRevenueAvg": 420e9,
             "numberAnalystsEstimatedEps": 30, "numberAnalystEstimatedRevenue": 28},
            {"date": f"{today.year}-09-25", "estimatedEpsAvg": 8.70, "estimatedRevenueAvg": 395e9,
             "numberAnalystsEstimatedEps": 32, "numberAnalystEstimatedRevenue": 30},
            {"date": f"{today.year - 1}-09-25", "estimatedEpsAvg": 7.50, "estimatedRevenueAvg": 370e9,
             "numberAnalystsEstimatedEps": 35, "numberAnalystEstimatedRevenue": 33},
        ],
    }
    out = compute_earnings_outlook(
        earnings_data=ed,
        price_data=_mk_price(today),
        spot=150.0,
        implied_vol=0.30,
    )

    ne = out["next_earnings"]
    assert ne["dte"] == 18
    assert ne["consensus_eps"] == 2.10
    assert ne["prior_year_eps"] == 1.80  # YoY anchor matched

    hist = out["history"]
    assert len(hist) >= 2
    # Most recent first
    assert hist[0]["date"] > hist[1]["date"]
    # First has surprise calc
    assert hist[0]["surprise_pct"] is not None
    assert hist[0]["beat"] in (True, False)

    bs = out["beat_summary"]
    assert bs["n_quarters"] == len(hist)
    assert 0.0 <= bs["beat_rate"] <= 1.0

    im = out["implied_move"]
    assert im["vol_source"] == "iv"
    assert im["dte_to_earnings"] == 18
    assert im["move_1d_pct"] > 0
    assert im["move_to_earnings_pct"] > im["move_1d_pct"]

    fy = out["fy_estimates"]
    # Only future fiscal years, sorted asc, max 2
    assert len(fy) <= 2
    assert all(r["year"] >= today.year for r in fy)
    if len(fy) == 2:
        assert fy[0]["year"] <= fy[1]["year"]


def test_hv_fallback_when_no_iv():
    today = date.today()
    out = compute_earnings_outlook(
        earnings_data={"earnings_calendar": _mk_calendar(today)},
        price_data=_mk_price(today),
        spot=150.0,
        implied_vol=None,
        realized_vol_fallback=0.25,
    )
    im = out["implied_move"]
    assert im is not None
    assert im["vol_source"] == "hv"
    assert abs(im["vol"] - 0.25) < 1e-9


def test_empty_inputs_safe():
    out = compute_earnings_outlook(
        earnings_data={},
        price_data=None,
        spot=None,
        implied_vol=None,
    )
    assert out["next_earnings"] is None
    assert out["history"] == []
    assert out["beat_summary"] is None
    assert out["implied_move"] is None
    assert out["fy_estimates"] == []

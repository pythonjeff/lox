import pandas as pd

from ai_options_trader.fiscal.signals import _compute_auction_tail_and_dealer_take


def test_compute_auction_metrics_monthly_weighted_avg_and_pct_detection():
    # Two auctions in the same month; weight by total_accepted.
    df = pd.DataFrame(
        [
            {
                "auction_date": "2025-01-08",
                "security_type": "Note",
                "security_term": "10-Year",
                "high_yield": 4.25,
                "median_yield": 4.23,
                "total_accepted": 100.0,
                "primary_dealer_takedown": 20.0,
            },
            {
                "auction_date": "2025-01-22",
                "security_type": "Note",
                "security_term": "2-Year",
                "high_yield": 4.00,
                "median_yield": 3.98,
                "total_accepted": 300.0,
                "primary_dealer_takedown": 30.0,
            },
        ]
    )
    out = _compute_auction_tail_and_dealer_take(df)
    assert out.shape[0] == 1
    # tail_bps = (0.02*100)=2bp for each -> weighted avg 2bp
    assert abs(float(out.iloc[0]["AUCTION_TAIL_BPS"]) - 2.0) < 1e-6
    # dealer take pct = (20/100*100)=20%, (30/300*100)=10% -> weighted: (20*100 + 10*300)/400 = 12.5%
    assert abs(float(out.iloc[0]["DEALER_TAKE_PCT"]) - 12.5) < 1e-6


def test_compute_auction_metrics_excludes_bills():
    df = pd.DataFrame(
        [
            {
                "auction_date": "2025-01-08",
                "security_type": "Bill",
                "security_term": "13-Week",
                "high_discount_rate": 5.0,
                "median_discount_rate": 4.9,
                "total_accepted": 100.0,
                "primary_dealer_takedown": 50.0,
            }
        ]
    )
    out = _compute_auction_tail_and_dealer_take(df)
    assert out.empty


def test_compute_auction_metrics_uses_avg_med_yield_when_median_missing():
    # auctions_query schema uses avg_med_yield.
    df = pd.DataFrame(
        [
            {
                "auction_date": "2025-12-31",
                "security_type": "Note",
                "security_term": "10-Year",
                "high_yield": 4.25,
                "avg_med_yield": 4.23,
                "total_accepted": 100.0,
                "primary_dealer_accepted": 10.0,
            }
        ]
    )
    out = _compute_auction_tail_and_dealer_take(df)
    assert out.shape[0] == 1
    assert abs(float(out.iloc[0]["AUCTION_TAIL_BPS"]) - 2.0) < 1e-6



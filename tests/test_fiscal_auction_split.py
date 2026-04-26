"""
Phase A — auction-pillar split tests.

Covers:
  1. _compute_auction_demand_quality: weighted shares from raw bidder amounts.
  2. score_fiscal_regime: divergence flag fires when |clearing - quality| > 40.
  3. score_fiscal_regime: FPI is unchanged by adding the two weight=0.0 sub-scores
     (numerical equivalence vs the legacy 6-pillar composite).
"""
from __future__ import annotations

import pandas as pd

from lox.fiscal.models import FiscalInputs
from lox.fiscal.scoring import score_fiscal_regime
from lox.fiscal.signals import _compute_auction_demand_quality


# ─────────────────────────────────────────────────────────────────────────────
# 1. Demand-quality computation
# ─────────────────────────────────────────────────────────────────────────────


def test_demand_quality_weighted_shares_from_bidder_amounts():
    # Two coupon auctions, both with full bidder breakdown.
    # Auction A (small): indirect=60, direct=20, dealer=20, total=100
    # Auction B (large): indirect=80, direct=10, dealer=10, total=900
    # Weighted-by-total averages:
    #   indirect = (60 + 80*9) / (100+900)*... actually: sum(amt)/sum(total)*100
    #   indirect_amt sum = 60 + 720 = 780; total sum = 1000 → 78.0
    #   direct_amt sum   = 20 +  90 = 110; → 11.0
    #   dealer_amt sum   = 20 +  90 = 110; → 11.0
    df = pd.DataFrame(
        [
            {
                "auction_date": "2025-01-08",
                "security_type": "Note",
                "security_term": "10-Year",
                "indirect_bidder_accepted": 60.0,
                "direct_bidder_accepted": 20.0,
                "primary_dealer_accepted": 20.0,
                "total_accepted": 100.0,
            },
            {
                "auction_date": "2025-01-22",
                "security_type": "Note",
                "security_term": "2-Year",
                "indirect_bidder_accepted": 720.0,
                "direct_bidder_accepted": 90.0,
                "primary_dealer_accepted": 90.0,
                "total_accepted": 900.0,
            },
        ]
    )
    out = _compute_auction_demand_quality(df, n=6)
    assert out["n"] == 2
    assert abs(out["indirect_bid_share"] - 78.0) < 1e-6
    assert abs(out["direct_bid_share"] - 11.0) < 1e-6
    assert abs(out["primary_dealer_pct"] - 11.0) < 1e-6


def test_demand_quality_excludes_bills():
    df = pd.DataFrame(
        [
            {
                "auction_date": "2025-01-08",
                "security_type": "Bill",
                "security_term": "13-Week",
                "indirect_bidder_accepted": 50.0,
                "direct_bidder_accepted": 30.0,
                "primary_dealer_accepted": 20.0,
                "total_accepted": 100.0,
            }
        ]
    )
    out = _compute_auction_demand_quality(df, n=6)
    assert out["n"] == 0
    assert out["indirect_bid_share"] is None
    assert out["primary_dealer_pct"] is None


def test_demand_quality_caps_at_n_most_recent():
    # 8 auctions; n=3 should pick the three latest by auction_date.
    rows = []
    for i in range(8):
        rows.append({
            "auction_date": f"2025-01-{i+1:02d}",
            "security_type": "Note",
            "security_term": "5-Year",
            # Latest 3 (i=5,6,7) carry indirect=90; older ones indirect=10 → if
            # we accidentally include all 8, the share would be very different.
            "indirect_bidder_accepted": 90.0 if i >= 5 else 10.0,
            "direct_bidder_accepted": 5.0,
            "primary_dealer_accepted": 5.0,
            "total_accepted": 100.0,
        })
    df = pd.DataFrame(rows)
    out = _compute_auction_demand_quality(df, n=3)
    assert out["n"] == 3
    assert abs(out["indirect_bid_share"] - 90.0) < 1e-6


def test_demand_quality_returns_empty_when_no_bidder_columns():
    df = pd.DataFrame(
        [
            {
                "auction_date": "2025-01-08",
                "security_type": "Note",
                "security_term": "10-Year",
                "total_accepted": 100.0,
            }
        ]
    )
    out = _compute_auction_demand_quality(df, n=6)
    assert out["n"] == 0
    assert out["indirect_bid_share"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Divergence flag
# ─────────────────────────────────────────────────────────────────────────────


def test_divergence_flag_fires_when_clearing_clean_but_quality_decaying():
    # Clean clearing: tail ~0bp (z=-1), bid-to-cover healthy.
    # Decayed quality: indirect collapses to 40% (vs 65% baseline);
    #                  dealer takedown surges to 40% (vs 20% baseline).
    inputs = FiscalInputs(
        # Clearing-side (low scores)
        z_auction_tail_bps=-1.5,
        auction_tail_bps=0.5,
        bid_to_cover_avg=2.8,
        # Quality-side (high stress)
        indirect_bid_share=40.0,
        direct_bid_share=20.0,
        dealer_take_pct=40.0,
        auction_demand_window=6,
    )
    sc = score_fiscal_regime(inputs)

    by_name = {p.name: p for p in sc.sub_scores}
    clearing = by_name["Auction Clearing"].score
    quality = by_name["Auction Demand Quality"].score

    assert clearing < 30, f"expected clean clearing, got {clearing}"
    assert quality > 80, f"expected decayed quality, got {quality}"
    assert sc.divergence_flags.get("auction_clearing_vs_quality") is True


def test_divergence_flag_quiet_when_both_pillars_aligned():
    # Both pillars healthy: no divergence flag.
    inputs = FiscalInputs(
        z_auction_tail_bps=-0.5,
        auction_tail_bps=1.0,
        bid_to_cover_avg=2.6,
        indirect_bid_share=66.0,
        direct_bid_share=14.0,
        dealer_take_pct=20.0,
        auction_demand_window=6,
    )
    sc = score_fiscal_regime(inputs)
    assert sc.divergence_flags.get("auction_clearing_vs_quality") is False


def test_divergence_flag_suppressed_when_quality_inputs_missing():
    # No bidder data → quality score defaults to 50 (neutral); we should NOT
    # surface a misleading divergence flag.
    inputs = FiscalInputs(
        z_auction_tail_bps=2.5,  # very stressed clearing
        auction_tail_bps=6.0,
    )
    sc = score_fiscal_regime(inputs)
    assert "auction_clearing_vs_quality" not in sc.divergence_flags


# ─────────────────────────────────────────────────────────────────────────────
# 3. FPI numerical equivalence — adding weight=0.0 sub-scores must NOT shift FPI
# ─────────────────────────────────────────────────────────────────────────────


def test_fpi_unchanged_by_weight_zero_sub_scores():
    """
    FPI is the weighted average of the original 6 pillars. The new
    Auction Clearing and Auction Demand Quality sub-scores carry weight=0.0,
    so the composite must equal the manual weighted mean of the original 6.
    """
    inputs = FiscalInputs(
        z_deficit_12m=1.2,
        deficit_pct_receipts=0.18,
        deficit_trend_slope=-3000.0,
        interest_expense_yoy_accel=2.0,
        z_long_duration_issuance_share=0.8,
        wam_chg_12m=0.4,
        z_auction_tail_bps=1.5,
        z_dealer_take_pct=0.9,
        bid_to_cover_avg=2.3,
        auction_tail_bps=4.0,
        dealer_take_pct=28.0,
        foreign_holdings_chg_6m=-30_000.0,
        custody_holdings_chg_4w=-8_000.0,
        move_index_z=1.4,
        z_tga_chg_28d=0.7,
        # populate quality inputs so the new sub-scores are non-neutral
        indirect_bid_share=55.0,
        direct_bid_share=15.0,
        auction_demand_window=6,
    )
    sc = score_fiscal_regime(inputs)

    # Manual recompute over the 6 weighted pillars.
    weighted = [p for p in sc.sub_scores if p.weight > 0.0]
    assert len(weighted) == 6, "expected exactly 6 weighted pillars"
    expected_fpi = sum(p.score * p.weight for p in weighted) / sum(p.weight for p in weighted)
    assert abs(sc.fpi - expected_fpi) < 1e-9

    # And the two new sub-scores are present, unweighted.
    display_only = [p for p in sc.sub_scores if p.weight == 0.0]
    names = {p.name for p in display_only}
    assert names == {"Auction Clearing", "Auction Demand Quality"}

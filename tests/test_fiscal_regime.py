from lox.fiscal.models import FiscalInputs
from lox.fiscal.regime import classify_fiscal_regime


def test_fiscal_regime_benign_funding_default():
    r = classify_fiscal_regime(FiscalInputs())
    assert r.name == "benign_funding"


def test_fiscal_regime_heavy_funding_large_deficits():
    r = classify_fiscal_regime(FiscalInputs(z_deficit_12m=1.0))
    assert r.name == "heavy_funding"


def test_fiscal_regime_stress_building_large_deficits_long_tilt_and_dealer_rising():
    r = classify_fiscal_regime(
        FiscalInputs(
            z_deficit_12m=1.0,
            z_long_duration_issuance_share=0.9,
            z_dealer_take_pct=0.4,
        )
    )
    assert r.name == "stress_building"


def test_fiscal_regime_fiscal_dominance_risk_interest_accel_and_weak_auctions():
    r = classify_fiscal_regime(
        FiscalInputs(
            interest_expense_yoy_accel=2.0,
            z_auction_tail_bps=1.0,
        )
    )
    assert r.name == "fiscal_dominance_risk"



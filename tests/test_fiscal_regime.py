from lox.fiscal.models import FiscalInputs
from lox.fiscal.regime import classify_fiscal_regime


def test_fiscal_regime_default_moderate_support():
    """No inputs → moderate support (neutral)."""
    r = classify_fiscal_regime(FiscalInputs())
    assert r.name == "moderate_fiscal_support"


def test_fiscal_regime_contraction_small_deficits():
    """Small deficit (negative z) → fiscal contraction (private sector squeeze)."""
    r = classify_fiscal_regime(FiscalInputs(z_deficit_12m=-1.0))
    assert r.name == "fiscal_contraction"


def test_fiscal_regime_strong_stimulus_large_deficits():
    """Large deficit (positive z) with healthy auctions → strong fiscal stimulus."""
    r = classify_fiscal_regime(FiscalInputs(z_deficit_12m=1.0))
    assert r.name == "strong_fiscal_stimulus"


def test_fiscal_regime_fiscal_dominance_risk_interest_accel_and_weak_auctions():
    r = classify_fiscal_regime(
        FiscalInputs(
            interest_expense_yoy_accel=2.0,
            z_auction_tail_bps=1.0,
        )
    )
    assert r.name == "fiscal_dominance_risk"

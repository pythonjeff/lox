from lox.funding.models import FundingInputs
from lox.funding.regime import classify_funding_regime


def test_funding_regime_normal_default():
    r = classify_funding_regime(FundingInputs(spread_corridor_bps=0.0))
    assert r.name == "normal_funding"


def test_funding_regime_tightening_when_spread_above_tight():
    r = classify_funding_regime(
        FundingInputs(
            spread_corridor_bps=10.0,
            tight_threshold_bps=5.0,
            stress_threshold_bps=15.0,
            spike_5d_bps=10.0,
            vol_20d_bps=1.0,
            vol_tight_bps=2.0,
            vol_stress_bps=4.0,
            persistence_20d=0.0,
            persistence_tight=0.2,
            persistence_stress=0.4,
        )
    )
    assert r.name == "tightening_balance_sheet_constraint"


def test_funding_regime_stress_when_persistent():
    r = classify_funding_regime(
        FundingInputs(
            spread_corridor_bps=25.0,
            tight_threshold_bps=5.0,
            stress_threshold_bps=15.0,
            spike_5d_bps=25.0,
            vol_20d_bps=5.0,
            vol_tight_bps=2.0,
            vol_stress_bps=4.0,
            persistence_20d=0.8,
            persistence_tight=0.2,
            persistence_stress=0.4,
        )
    )
    assert r.name == "funding_stress"



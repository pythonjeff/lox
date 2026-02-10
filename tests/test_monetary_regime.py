from lox.monetary.models import MonetaryInputs
from lox.monetary.regime import classify_monetary_regime


def test_monetary_regime_abundant_reserves():
    r = classify_monetary_regime(MonetaryInputs(z_total_reserves=1.0))
    assert r.name == "abundant_reserves"


def test_monetary_regime_constrained_reserves_low_reserves_and_low_rrp():
    r = classify_monetary_regime(MonetaryInputs(z_total_reserves=-1.0, z_on_rrp=-1.0))
    assert r.name == "thinning_buffers_transitional"


def test_monetary_regime_qt_biting_when_assets_shrink_fast_and_buffers_low():
    r = classify_monetary_regime(
        MonetaryInputs(
            z_total_reserves=-1.0,
            z_on_rrp=-1.0,
            z_fed_assets_chg_13w=-1.0,
        )
    )
    assert r.name == "qt_biting"



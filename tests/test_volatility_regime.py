from ai_options_trader.volatility.models import VolatilityInputs
from ai_options_trader.volatility.regime import classify_volatility_regime


def test_volatility_regime_normal():
    r = classify_volatility_regime(VolatilityInputs(z_vix=0.0, z_vix_chg_5d=0.0, vol_pressure_score=0.0))
    assert r.name == "normal_vol"


def test_volatility_regime_elevated():
    r = classify_volatility_regime(VolatilityInputs(z_vix=1.2, z_vix_chg_5d=0.2, vol_pressure_score=1.0))
    assert r.name == "elevated_vol"


def test_volatility_regime_shock_on_momentum():
    r = classify_volatility_regime(VolatilityInputs(z_vix=0.5, z_vix_chg_5d=2.1, persist_20d=0.0))
    assert r.name == "vol_shock"



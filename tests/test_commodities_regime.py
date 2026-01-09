from ai_options_trader.commodities.models import CommoditiesInputs
from ai_options_trader.commodities.regime import classify_commodities_regime


def test_commodities_regime_neutral():
    r = classify_commodities_regime(CommoditiesInputs(commodity_pressure_score=0.0, energy_shock=False))
    assert r.name == "neutral"


def test_commodities_regime_reflation():
    r = classify_commodities_regime(CommoditiesInputs(commodity_pressure_score=1.3, energy_shock=False))
    assert r.name == "commodity_reflation"


def test_commodities_regime_energy_shock_overrides():
    r = classify_commodities_regime(CommoditiesInputs(commodity_pressure_score=-2.0, energy_shock=True))
    assert r.name == "energy_shock"



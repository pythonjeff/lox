from __future__ import annotations

from ai_options_trader.rates.models import RatesInputs
from ai_options_trader.rates.regime import classify_rates_regime


def test_rates_regime_inverted_curve():
    r = classify_rates_regime(RatesInputs(curve_2s10s=-0.25))
    assert r.name == "inverted_curve"


def test_rates_regime_rates_shock_up():
    r = classify_rates_regime(RatesInputs(curve_2s10s=0.5, z_ust_10y_chg_20d=2.0))
    assert r.name == "rates_shock_up"


def test_rates_regime_neutral():
    r = classify_rates_regime(RatesInputs(curve_2s10s=0.2, z_ust_10y_chg_20d=0.2))
    assert r.name == "neutral"



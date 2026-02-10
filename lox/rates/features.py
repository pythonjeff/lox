from __future__ import annotations

from dataclasses import dataclass

from lox.rates.models import RatesState
from lox.rates.regime import RatesRegime


@dataclass(frozen=True)
class RatesFeatureVector:
    asof: str
    features: dict[str, float]


def rates_feature_vector(state: RatesState, regime: RatesRegime) -> RatesFeatureVector:
    r = state.inputs
    f: dict[str, float] = {}

    def add(k: str, v: object):
        if isinstance(v, (int, float)):
            f[k] = float(v)

    add("rates.ust_2y", r.ust_2y)
    add("rates.ust_10y", r.ust_10y)
    add("rates.curve_2s10s", r.curve_2s10s)
    add("rates.ust_10y_chg_20d", r.ust_10y_chg_20d)
    add("rates.z_ust_10y", r.z_ust_10y)
    add("rates.z_ust_10y_chg_20d", r.z_ust_10y_chg_20d)
    add("rates.z_curve_2s10s", r.z_curve_2s10s)
    add("rates.z_curve_2s10s_chg_20d", r.z_curve_2s10s_chg_20d)

    # Regime one-hot (stable for ML)
    for name in ("neutral", "inverted_curve", "rates_shock_up", "rates_shock_down", "steep_curve"):
        f[f"rates.regime.{name}"] = 1.0 if regime.name == name else 0.0

    return RatesFeatureVector(asof=state.asof, features=f)



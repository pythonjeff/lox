from __future__ import annotations

from dataclasses import dataclass

from lox.rates.models import RatesState
from lox.rates.regime import RatesRegime


@dataclass(frozen=True)
class RatesFeatureVector:
    asof: str
    features: dict[str, float]


_REGIME_NAMES = (
    "neutral",
    "inverted_curve",
    "rates_shock_up",
    "rates_shock_down",
    "steep_curve",
    "bear_steepener",
    "bull_flattener",
    "bear_flattener",
    "bull_steepener",
)


def rates_feature_vector(state: RatesState, regime: RatesRegime) -> RatesFeatureVector:
    r = state.inputs
    f: dict[str, float] = {}

    def add(k: str, v: object) -> None:
        if isinstance(v, (int, float)):
            f[k] = float(v)

    # Levels
    add("rates.ust_2y", r.ust_2y)
    add("rates.ust_5y", r.ust_5y)
    add("rates.ust_10y", r.ust_10y)
    add("rates.ust_30y", r.ust_30y)
    add("rates.ust_3m", r.ust_3m)

    # Curve slopes
    add("rates.curve_2s10s", r.curve_2s10s)
    add("rates.curve_2s30s", r.curve_2s30s)
    add("rates.curve_5s30s", r.curve_5s30s)

    # Per-tenor momentum
    add("rates.ust_2y_chg_20d", r.ust_2y_chg_20d)
    add("rates.ust_10y_chg_20d", r.ust_10y_chg_20d)
    add("rates.ust_30y_chg_20d", r.ust_30y_chg_20d)
    add("rates.curve_2s10s_chg_20d", r.curve_2s10s_chg_20d)
    add("rates.curve_2s30s_chg_20d", r.curve_2s30s_chg_20d)

    # Real yields
    add("rates.real_yield_10y", r.real_yield_10y)
    add("rates.real_yield_5y", r.real_yield_5y)
    add("rates.breakeven_10y", r.breakeven_10y)
    add("rates.breakeven_5y", r.breakeven_5y)
    add("rates.real_yield_10y_chg_20d", r.real_yield_10y_chg_20d)

    # Z-scores
    add("rates.z_ust_10y", r.z_ust_10y)
    add("rates.z_ust_10y_chg_20d", r.z_ust_10y_chg_20d)
    add("rates.z_ust_2y_chg_20d", r.z_ust_2y_chg_20d)
    add("rates.z_ust_30y_chg_20d", r.z_ust_30y_chg_20d)
    add("rates.z_curve_2s10s", r.z_curve_2s10s)
    add("rates.z_curve_2s10s_chg_20d", r.z_curve_2s10s_chg_20d)
    add("rates.z_curve_2s30s", r.z_curve_2s30s)
    add("rates.z_curve_2s30s_chg_20d", r.z_curve_2s30s_chg_20d)

    # Regime one-hot
    for name in _REGIME_NAMES:
        f[f"rates.regime.{name}"] = 1.0 if regime.name == name else 0.0

    return RatesFeatureVector(asof=state.asof, features=f)

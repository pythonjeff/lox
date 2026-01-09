from __future__ import annotations

from ai_options_trader.commodities.models import CommoditiesState
from ai_options_trader.commodities.regime import CommoditiesRegime
from ai_options_trader.regimes.schema import RegimeVector, add_bool_feature, add_feature, add_one_hot


def commodities_feature_vector(state: CommoditiesState, regime: CommoditiesRegime) -> RegimeVector:
    f: dict[str, float] = {}
    ci = state.inputs

    add_feature(f, "commod.wti", ci.wti)
    add_feature(f, "commod.gold", ci.gold)
    add_feature(f, "commod.copper", ci.copper)
    add_feature(f, "commod.broad_index", ci.broad_index)
    add_feature(f, "commod.wti_ret_20d_pct", ci.wti_ret_20d_pct)
    add_feature(f, "commod.gold_ret_20d_pct", ci.gold_ret_20d_pct)
    add_feature(f, "commod.copper_ret_60d_pct", ci.copper_ret_60d_pct)
    add_feature(f, "commod.broad_ret_60d_pct", ci.broad_ret_60d_pct)
    add_feature(f, "commod.z_wti_ret_20d", ci.z_wti_ret_20d)
    add_feature(f, "commod.z_gold_ret_20d", ci.z_gold_ret_20d)
    add_feature(f, "commod.z_copper_ret_60d", ci.z_copper_ret_60d)
    add_feature(f, "commod.z_broad_ret_60d", ci.z_broad_ret_60d)
    add_feature(f, "commod.pressure_score", ci.commodity_pressure_score)
    add_bool_feature(f, "commod.energy_shock", ci.energy_shock)
    add_bool_feature(f, "commod.metals_impulse", ci.metals_impulse)

    add_one_hot(f, "commod.regime", regime.name, ("neutral", "commodity_reflation", "commodity_disinflation", "energy_shock"))

    return RegimeVector(asof=state.asof, features=f, notes="Commodities regime (oil/gold/broad) as scalar features + one-hot label.")



from __future__ import annotations

from ai_options_trader.macro.models import MacroState
from ai_options_trader.macro.regime import MacroRegime
from ai_options_trader.regimes.schema import (
    RegimeVector,
    add_bool_feature,
    add_feature,
    add_one_hot,
)


MACRO_REGIME_CHOICES = ("stagflation", "inflation", "reflation", "tightening", "disinflation_shock", "goldilocks")


def macro_feature_vector(macro_state: MacroState, macro_regime: MacroRegime) -> RegimeVector:
    """
    Convert the macro state + categorical regime into a scalar feature vector.

    Design:
    - Keep raw/derived scalars from MacroInputs
    - Encode the categorical regime label as one-hot floats
    - Add a couple simple "direction" scalars (infl_up / real_up) as 0/1
    """
    f: dict[str, float] = {}

    # --- Raw/derived inputs (already scalars) ---
    mi = macro_state.inputs
    add_feature(f, "macro.cpi_yoy", mi.cpi_yoy)
    add_feature(f, "macro.core_cpi_yoy", mi.core_cpi_yoy)
    add_feature(f, "macro.cpi_3m_annualized", mi.cpi_3m_annualized)
    add_feature(f, "macro.cpi_6m_annualized", mi.cpi_6m_annualized)
    add_feature(f, "macro.breakeven_5y", mi.breakeven_5y)
    add_feature(f, "macro.breakeven_10y", mi.breakeven_10y)
    add_feature(f, "macro.payrolls_yoy", mi.payrolls_yoy)
    add_feature(f, "macro.payrolls_3m_annualized", mi.payrolls_3m_annualized)
    add_feature(f, "macro.payrolls_mom", mi.payrolls_mom)
    add_feature(f, "macro.eff_fed_funds", mi.eff_fed_funds)
    add_feature(f, "macro.ust_2y", mi.ust_2y)
    add_feature(f, "macro.ust_10y", mi.ust_10y)
    add_feature(f, "macro.curve_2s10s", mi.curve_2s10s)
    add_feature(f, "macro.real_yield_proxy_10y", mi.real_yield_proxy_10y)
    add_feature(f, "macro.inflation_momentum_minus_be5y", mi.inflation_momentum_minus_be5y)
    add_feature(f, "macro.disconnect_score", mi.disconnect_score)

    # Transparent z-scored components (if present)
    if mi.components:
        add_feature(f, "macro.z_infl_mom_minus_be5y", mi.components.get("z_infl_mom_minus_be5y"))
        add_feature(f, "macro.z_real_yield_proxy_10y", mi.components.get("z_real_yield_proxy_10y"))

    # --- Simple directional flags ---
    # Prefer z-scores (relative to history) when available; otherwise fall back to raw thresholds at 0.
    z_infl = mi.components.get("z_infl_mom_minus_be5y") if mi.components else None
    z_real = mi.components.get("z_real_yield_proxy_10y") if mi.components else None
    if z_infl is not None and z_real is not None:
        add_bool_feature(f, "macro.infl_up", float(z_infl) > 0)
        add_bool_feature(f, "macro.real_yield_up", float(z_real) > 0)
    else:
        add_bool_feature(f, "macro.infl_up", (mi.inflation_momentum_minus_be5y is not None and mi.inflation_momentum_minus_be5y > 0))
        add_bool_feature(f, "macro.real_yield_up", (mi.real_yield_proxy_10y is not None and mi.real_yield_proxy_10y > 0))

    # --- Regime as one-hot floats ---
    add_one_hot(f, "macro.regime", macro_regime.name, MACRO_REGIME_CHOICES)

    return RegimeVector(
        asof=macro_state.asof,
        features=f,
        notes="Macro regime encoded as one-hot; raw/derived MacroInputs included as scalars.",
    )



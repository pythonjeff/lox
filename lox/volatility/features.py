from __future__ import annotations

from lox.regimes.schema import RegimeVector, add_feature, add_one_hot
from lox.volatility.models import VolatilityState
from lox.volatility.regime import VolatilityRegime


def volatility_feature_vector(state: VolatilityState, regime: VolatilityRegime) -> RegimeVector:
    f: dict[str, float] = {}
    vi = state.inputs

    add_feature(f, "vol.vix", vi.vix)
    add_feature(f, "vol.vix9d", vi.vix9d)
    add_feature(f, "vol.vix3m", vi.vix3m)
    add_feature(f, "vol.vix_chg_1d_pct", vi.vix_chg_1d_pct)
    add_feature(f, "vol.vix_chg_5d_pct", vi.vix_chg_5d_pct)
    add_feature(f, "vol.vix_term_spread", vi.vix_term_spread)
    add_feature(f, "vol.z_vix", vi.z_vix)
    add_feature(f, "vol.z_vix_chg_5d", vi.z_vix_chg_5d)
    add_feature(f, "vol.z_vix_term", vi.z_vix_term)
    add_feature(f, "vol.spike_20d_pct", vi.spike_20d_pct)
    add_feature(f, "vol.persist_20d", vi.persist_20d)
    add_feature(f, "vol.threshold_vix", vi.threshold_vix)
    add_feature(f, "vol.pressure_score", vi.vol_pressure_score)

    add_one_hot(f, "vol.regime", regime.name, ("normal_vol", "elevated_vol", "vol_shock"))

    return RegimeVector(asof=state.asof, features=f, notes="Volatility regime (VIX) as scalar features + one-hot label.")



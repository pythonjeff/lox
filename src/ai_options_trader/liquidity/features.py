from __future__ import annotations

from ai_options_trader.liquidity.models import LiquidityState
from ai_options_trader.regimes.schema import RegimeVector, add_bool_feature, add_feature


def liquidity_feature_vector(state: LiquidityState) -> RegimeVector:
    f: dict[str, float] = {}
    li = state.inputs

    add_feature(f, "liquidity.ust_10y", li.ust_10y)
    add_feature(f, "liquidity.ust_10y_chg_20d_bps", li.ust_10y_chg_20d_bps)
    add_feature(f, "liquidity.ust_10y_chg_60d_bps", li.ust_10y_chg_60d_bps)

    add_feature(f, "liquidity.hy_oas", li.hy_oas)
    add_feature(f, "liquidity.ig_oas", li.ig_oas)
    add_feature(f, "liquidity.baa10ym", li.baa10ym)

    add_feature(f, "liquidity.z_hy_oas", li.z_hy_oas)
    add_feature(f, "liquidity.z_ig_oas", li.z_ig_oas)
    add_feature(f, "liquidity.z_ust_10y_chg_20d", li.z_ust_10y_chg_20d)

    add_feature(f, "liquidity.tightness_score", li.liquidity_tightness_score)
    add_bool_feature(f, "liquidity.tight", li.is_liquidity_tight)

    if li.components:
        for k, v in li.components.items():
            add_feature(f, f"liquidity.component.{k}", v)

    return RegimeVector(asof=state.asof, features=f, notes="Liquidity regime (credit + rates) as scalar features.")



from __future__ import annotations

from lox.regimes.schema import RegimeVector, add_bool_feature, add_feature
from lox.usd.models import UsdState


def usd_feature_vector(state: UsdState) -> RegimeVector:
    f: dict[str, float] = {}
    ui = state.inputs

    add_feature(f, "usd.index_broad", ui.usd_index_broad)
    add_feature(f, "usd.chg_20d_pct", ui.usd_chg_20d_pct)
    add_feature(f, "usd.chg_60d_pct", ui.usd_chg_60d_pct)
    add_feature(f, "usd.z_level", ui.z_usd_level)
    add_feature(f, "usd.z_chg_60d", ui.z_usd_chg_60d)
    add_feature(f, "usd.strength_score", ui.usd_strength_score)

    add_bool_feature(f, "usd.strong", ui.is_usd_strong)
    add_bool_feature(f, "usd.weak", ui.is_usd_weak)

    if ui.components:
        for k, v in ui.components.items():
            add_feature(f, f"usd.component.{k}", v)

    return RegimeVector(asof=state.asof, features=f, notes="USD regime (DTWEXBGS) as scalar features.")



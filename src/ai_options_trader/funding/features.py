from __future__ import annotations

from ai_options_trader.funding.models import FundingState
from ai_options_trader.regimes.schema import RegimeVector, add_feature


def funding_feature_vector(state: FundingState) -> RegimeVector:
    f: dict[str, float] = {}
    fi = state.inputs

    add_feature(f, "funding.effr", fi.effr)
    add_feature(f, "funding.sofr", fi.sofr)
    add_feature(f, "funding.tgcr", fi.tgcr)
    add_feature(f, "funding.bgcr", fi.bgcr)
    add_feature(f, "funding.obfr", fi.obfr)
    add_feature(f, "funding.iorb", fi.iorb)

    add_feature(f, "funding.corridor_spread_bps", fi.spread_corridor_bps)
    add_feature(f, "funding.sofr_effr_bps", fi.spread_sofr_effr_bps)
    add_feature(f, "funding.bgcr_tgcr_bps", fi.spread_bgcr_tgcr_bps)

    add_feature(f, "funding.spike_5d_bps", fi.spike_5d_bps)
    add_feature(f, "funding.persistence_20d", fi.persistence_20d)
    add_feature(f, "funding.vol_20d_bps", fi.vol_20d_bps)

    # Baseline context (useful features but not required)
    add_feature(f, "funding.baseline_median_bps", fi.baseline_median_bps)
    add_feature(f, "funding.baseline_std_bps", fi.baseline_std_bps)
    add_feature(f, "funding.tight_threshold_bps", fi.tight_threshold_bps)
    add_feature(f, "funding.stress_threshold_bps", fi.stress_threshold_bps)
    add_feature(f, "funding.vol_baseline_bps", fi.vol_baseline_bps)
    add_feature(f, "funding.vol_tight_bps", fi.vol_tight_bps)
    add_feature(f, "funding.vol_stress_bps", fi.vol_stress_bps)
    add_feature(f, "funding.persistence_baseline", fi.persistence_baseline)
    add_feature(f, "funding.persistence_tight", fi.persistence_tight)
    add_feature(f, "funding.persistence_stress", fi.persistence_stress)

    if fi.components:
        for k, v in fi.components.items():
            add_feature(f, f"funding.component.{k}", v)

    return RegimeVector(asof=state.asof, features=f, notes="Funding regime (secured rates) as scalar features.")



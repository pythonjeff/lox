from __future__ import annotations

from typing import Iterable

from ai_options_trader.regimes.schema import RegimeVector, add_bool_feature, add_feature
from ai_options_trader.tariff.models import TariffRegimeState


def tariff_feature_vector(tariff_states: Iterable[TariffRegimeState], asof: str | None = None) -> RegimeVector:
    """
    Convert one or more tariff regime states into a scalar feature vector.

    Notes on schema stability:
    - Per-basket keys are stable *given a fixed basket set*.
    - We also emit basket-agnostic aggregates (mean/max/active_count) for ML models
      that don't want dynamic feature names.
    """
    f: dict[str, float] = {}
    scores: list[float] = []
    actives: list[float] = []

    inferred_asof: str | None = None
    for s in tariff_states:
        inferred_asof = inferred_asof or s.asof
        b = (s.basket or "unknown").strip()

        inp = s.inputs
        add_feature(f, f"tariff.{b}.z_cost_pressure", inp.z_cost_pressure)
        add_feature(f, f"tariff.{b}.equity_denial_beta", inp.equity_denial_beta)
        add_feature(f, f"tariff.{b}.z_earnings_fragility", inp.z_earnings_fragility)
        add_feature(f, f"tariff.{b}.score", inp.tariff_regime_score)
        add_bool_feature(f, f"tariff.{b}.active", inp.is_tariff_regime)

        # Components are already scalar-ish; flatten them
        if inp.components:
            for k, v in inp.components.items():
                add_feature(f, f"tariff.{b}.component.{k}", v)

        if inp.tariff_regime_score is not None:
            try:
                scores.append(float(inp.tariff_regime_score))
            except Exception:
                pass
        actives.append(1.0 if bool(inp.is_tariff_regime) else 0.0)

    if scores:
        f["tariff.score_mean"] = float(sum(scores) / len(scores))
        f["tariff.score_max"] = float(max(scores))
        f["tariff.score_min"] = float(min(scores))
    if actives:
        f["tariff.active_count"] = float(sum(actives))
        f["tariff.active_frac"] = float(sum(actives) / len(actives))

    return RegimeVector(
        asof=asof or inferred_asof or "",
        features=f,
        notes="Tariff regime per-basket features + basket-agnostic aggregates.",
    )



from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TariffRegimeSpec:
    # weights for composite
    w_cost: float = 0.45
    w_denial: float = 0.35
    w_fragility: float = 0.20

    # regime threshold (score above -> regime ON)
    threshold: float = 0.75

    # rolling windows
    denial_window_days: int = 252
    z_window_days: int = 252

    # cost momentum horizons (in trading days for daily series)
    # we will use these for daily cost proxies; monthly proxies will be transformed
    cost_short_days: int = 21     # ~1 month
    cost_long_days: int = 126     # ~6 months


DEFAULT_TARIFF_SPEC = TariffRegimeSpec()

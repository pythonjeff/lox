from __future__ import annotations

import numpy as np
import pandas as pd

from lox.tariff.signals import build_tariff_regime_state
from lox.tariff.theory import TariffRegimeSpec


def _sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", "2021-12-31", freq="D")

    cost = 100 + np.cumsum(rng.normal(0, 0.5, len(idx)))
    cost_df = pd.DataFrame({"COST_PROXY_V0": cost}, index=idx)

    # Benchmark + one universe ticker so rel returns/beta are meaningful
    xly = 100 + np.cumsum(rng.normal(0, 1.0, len(idx)))
    aaa = 80 + np.cumsum(rng.normal(0, 1.2, len(idx)))
    px = pd.DataFrame({"XLY": xly, "AAA": aaa}, index=idx)

    return cost_df, px


def test_build_tariff_regime_state_accepts_new_kwargs() -> None:
    cost_df, px = _sample_inputs()
    spec = TariffRegimeSpec(denial_window_days=20, z_window_days=20, cost_short_days=5, cost_long_days=15)

    state = build_tariff_regime_state(
        cost_df=cost_df,
        equity_prices=px,
        universe=["AAA"],
        benchmark="XLY",
        start_date="2020-01-01",
        spec=spec,
    )

    assert state.asof == "2021-12-31"
    assert state.inputs.tariff_regime_score is not None


def test_build_tariff_regime_state_accepts_alias_kwargs() -> None:
    cost_df, px = _sample_inputs()
    spec = TariffRegimeSpec(denial_window_days=20, z_window_days=20, cost_short_days=5, cost_long_days=15)

    state = build_tariff_regime_state(
        cost_proxies=cost_df,
        equity_prices_df=px,
        universe=["AAA"],
        benchmark="XLY",
        start_date="2020-01-01",
        spec=spec,
    )

    assert state.asof == "2021-12-31"
    assert state.inputs.tariff_regime_score is not None



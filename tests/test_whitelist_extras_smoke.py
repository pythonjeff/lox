from __future__ import annotations

import numpy as np
import pandas as pd

from ai_options_trader.portfolio.panel import build_macro_panel_dataset


def test_whitelist_extras_smoke():
    # Tiny synthetic dataset just to ensure the code paths work.
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    tickers = ["AAA", "BBB", "CCC"]

    prices = pd.DataFrame(
        {
            "AAA": 100 + np.cumsum(np.random.RandomState(0).normal(0, 1, size=len(dates))),
            "BBB": 90 + np.cumsum(np.random.RandomState(1).normal(0, 1, size=len(dates))),
            "CCC": 110 + np.cumsum(np.random.RandomState(2).normal(0, 1, size=len(dates))),
        },
        index=dates,
    ).abs()

    # Provide only the features needed for USD + each extra to resolve.
    Xr = pd.DataFrame(
        {
            "usd_strength_score": np.random.RandomState(3).normal(0, 1, size=len(dates)),
            "macro_disconnect_score": np.random.RandomState(4).normal(0, 1, size=len(dates)),
            "funding_tightness_score": np.random.RandomState(5).normal(0, 1, size=len(dates)),
            "rates_z_ust_10y_chg_20d": np.random.RandomState(6).normal(0, 1, size=len(dates)),
            "vol_pressure_score": np.random.RandomState(7).normal(0, 1, size=len(dates)),
            "commod_pressure_score": np.random.RandomState(8).normal(0, 1, size=len(dates)),
            "fiscal_pressure_score": np.random.RandomState(9).normal(0, 1, size=len(dates)),
        },
        index=dates,
    )

    extras = ["none", "macro", "funding", "rates", "vol", "commod", "fiscal"]
    for ex in extras:
        ds = build_macro_panel_dataset(
            regime_features=Xr,
            prices=prices,
            tickers=tickers,
            horizon_days=10,
            interaction_mode="whitelist",
            whitelist_extra=ex,
        )
        assert not ds.X.empty
        assert not ds.y.empty


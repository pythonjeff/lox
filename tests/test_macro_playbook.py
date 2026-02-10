from __future__ import annotations

import pandas as pd

from lox.ideas.macro_playbook import rank_macro_playbook


def test_macro_playbook_ranks_by_regime_conditioned_forward_returns():
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    # Two regimes: feature=0 for first half, feature=1 for second half
    X = pd.DataFrame({"feat": ([0.0] * 200) + ([1.0] * 200)}, index=idx)
    # Prices: asset A rallies after regime=1, asset B sells off after regime=1
    px = pd.DataFrame(index=idx)
    px["A"] = 100.0 + (X["feat"].cumsum() * 0.5).values
    px["B"] = 100.0 - (X["feat"].cumsum() * 0.5).values

    ideas = rank_macro_playbook(features=X, prices=px, tickers=["A", "B"], horizon_days=20, k=120, min_matches=50)
    assert len(ideas) == 2
    # In the second regime, A should be bullish and B bearish (by exp_return sign)
    d = {i.ticker: i for i in ideas}
    assert d["A"].direction == "bullish"
    assert d["B"].direction == "bearish"



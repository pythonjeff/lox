import pandas as pd

from lox.portfolio.panel_eval import walk_forward_panel_eval


def test_panel_eval_smoke_no_crash():
    # Tiny synthetic panel: should return a non-ok status but not crash.
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=50, freq="B"), ["A", "B", "C"]],
        names=["date", "ticker"],
    )
    X = pd.DataFrame({"f1": 0.0, "f2": 1.0}, index=idx)
    y = pd.Series(0.0, index=idx)
    res = walk_forward_panel_eval(X=X, y=y, horizon_days=10, min_train_days=10, step_days=5, top_k=1)
    assert res.status in {"single_class_train", "no_valid_folds", "insufficient_dates", "empty_after_dropna", "empty"}



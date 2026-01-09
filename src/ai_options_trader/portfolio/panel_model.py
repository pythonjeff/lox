from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TickerForecast:
    ticker: str
    prob_up: float | None
    exp_return: float | None
    notes: str = ""


def _make_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)),
        ]
    )


def _make_regressor() -> Pipeline:
    return Pipeline(steps=[("scaler", StandardScaler()), ("reg", Ridge(alpha=5.0))])


def fit_latest_with_models(
    *,
    X: pd.DataFrame,  # MultiIndex(date,ticker)
    y: pd.Series,
    min_train_rows: int = 1500,
) -> tuple[list[TickerForecast], dict[str, object], Pipeline | None, Pipeline | None, pd.DataFrame | None]:
    """
    Fit the latest-date panel and return fitted (clf, reg) + the latest cross-section feature frame (Xte).

    This is used for "explain" output and keeps the same leak-avoidance as the main path
    (train on dates strictly before the latest date).
    """
    if X.empty:
        return [], {"status": "empty"}, None, None, None
    latest_date = pd.to_datetime(X.index.get_level_values(0).max())

    # Split by date
    dates = X.index.get_level_values(0)
    train_mask = dates < latest_date
    test_mask = dates == latest_date
    Xtr = X.loc[train_mask]
    ytr = y.loc[train_mask]
    Xte = X.loc[test_mask]

    if len(Xtr) < int(min_train_rows) or Xte.empty:
        return [], {"status": "insufficient_data", "train_rows": int(len(Xtr)), "test_rows": int(len(Xte))}, None, None, None

    # Clean
    Xtr = Xtr.replace([np.inf, -np.inf], np.nan)
    ytr = ytr.replace([np.inf, -np.inf], np.nan)
    keep = Xtr.dropna().index.intersection(ytr.dropna().index)
    Xtr = Xtr.loc[keep]
    ytr = ytr.loc[keep]

    Xte = Xte.replace([np.inf, -np.inf], np.nan).dropna()
    if Xte.empty:
        return [], {"status": "no_features_latest", "train_rows": int(len(Xtr)), "test_rows": 0}, None, None, None

    yb = (ytr > 0.0).astype(int)
    if yb.nunique() < 2:
        return [], {"status": "single_class_train", "train_rows": int(len(Xtr)), "test_rows": int(len(Xte))}, None, None, None

    clf = _make_classifier()
    reg = _make_regressor()
    clf.fit(Xtr, yb)
    reg.fit(Xtr, ytr)

    p = clf.predict_proba(Xte)[:, 1]
    r = reg.predict(Xte)
    preds: list[TickerForecast] = []
    for i, idx in enumerate(Xte.index):
        _d, t = idx
        preds.append(TickerForecast(ticker=str(t), prob_up=float(p[i]), exp_return=float(r[i])))
    preds.sort(key=lambda z: (z.exp_return if z.exp_return is not None else -1e9), reverse=True)
    meta = {"status": "ok", "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)), "latest_date": str(latest_date.date())}
    return preds, meta, clf, reg, Xte


def _fit_latest_panel(
    *,
    X: pd.DataFrame,  # MultiIndex(date,ticker)
    y: pd.Series,
    latest_date: pd.Timestamp,
    min_train_rows: int = 1500,
) -> tuple[list[TickerForecast], dict[str, object]]:
    """
    Train on all history strictly before `latest_date`, then predict cross-section for `latest_date`.
    """
    if X.empty or y.empty:
        return [], {"status": "empty"}

    # Split by date to avoid leakage
    dates = X.index.get_level_values(0)
    train_mask = dates < latest_date
    test_mask = dates == latest_date

    Xtr = X.loc[train_mask]
    ytr = y.loc[train_mask]
    Xte = X.loc[test_mask]

    if len(Xtr) < int(min_train_rows) or Xte.empty:
        return [], {"status": "insufficient_data", "train_rows": int(len(Xtr)), "test_rows": int(len(Xte))}

    # Align & clean
    Xtr = Xtr.replace([np.inf, -np.inf], np.nan)
    ytr = ytr.replace([np.inf, -np.inf], np.nan)
    keep = Xtr.dropna().index.intersection(ytr.dropna().index)
    Xtr = Xtr.loc[keep]
    ytr = ytr.loc[keep]

    Xte = Xte.replace([np.inf, -np.inf], np.nan)
    Xte = Xte.dropna()
    if Xte.empty:
        return [], {"status": "no_features_latest"}

    # Targets
    yb = (ytr > 0.0).astype(int)
    if yb.nunique() < 2:
        return [], {"status": "single_class_train"}

    clf = _make_classifier()
    reg = _make_regressor()
    clf.fit(Xtr, yb)
    reg.fit(Xtr, ytr)

    p = clf.predict_proba(Xte)[:, 1]
    r = reg.predict(Xte)

    # Emit forecasts per ticker
    out: list[TickerForecast] = []
    for i, idx in enumerate(Xte.index):
        _d, t = idx
        out.append(TickerForecast(ticker=str(t), prob_up=float(p[i]), exp_return=float(r[i])))

    out.sort(key=lambda z: (z.exp_return if z.exp_return is not None else -1e9), reverse=True)
    return out, {"status": "ok", "train_rows": int(len(Xtr)), "test_rows": int(len(Xte))}


def fit_and_rank_latest(
    *,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[list[TickerForecast], dict[str, object]]:
    if X.empty:
        return [], {"status": "empty"}
    latest_date = pd.to_datetime(X.index.get_level_values(0).max())
    return _fit_latest_panel(X=X, y=y, latest_date=latest_date)



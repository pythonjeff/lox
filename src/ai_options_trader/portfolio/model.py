from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


Horizon = Literal["3m", "6m", "12m"]


@dataclass(frozen=True)
class HorizonForecast:
    horizon: Horizon
    prob_up: float | None
    exp_return: float | None
    auc_cv: float | None


def _make_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )


def _make_regressor() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=5.0)),
        ]
    )


def _cv_auc(model: Pipeline, X: pd.DataFrame, y: pd.Series, splits: int = 5) -> float | None:
    yb = (y > 0.0).astype(int)
    # Need both classes present
    if yb.nunique() < 2 or len(yb) < 200:
        return None
    tscv = TimeSeriesSplit(n_splits=splits)
    aucs: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = yb.iloc[train_idx], yb.iloc[test_idx]
        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue
        m = _make_classifier()
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        try:
            aucs.append(float(roc_auc_score(yte, p)))
        except Exception:
            continue
    if not aucs:
        return None
    return float(np.mean(aucs))


def fit_and_forecast(
    *,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[float | None, float | None, float | None]:
    """
    Fit on all available history and predict for the latest feature row.
    Returns (prob_up, exp_return, auc_cv).
    """
    if X.empty or y.empty:
        return None, None, None

    # Align and drop NaNs
    idx = X.index.intersection(y.index)
    X2 = X.loc[idx].copy()
    y2 = y.loc[idx].copy()
    X2 = X2.replace([np.inf, -np.inf], np.nan)
    y2 = y2.replace([np.inf, -np.inf], np.nan)
    keep = X2.dropna().index.intersection(y2.dropna().index)
    X2 = X2.loc[keep]
    y2 = y2.loc[keep]
    if len(X2) < 250:
        return None, None, None

    auc = _cv_auc(_make_classifier(), X2, y2)

    # Final fit
    yb = (y2 > 0.0).astype(int)
    clf = _make_classifier()
    reg = _make_regressor()
    clf.fit(X2, yb)
    reg.fit(X2, y2)

    x_last = X.iloc[[-1]]
    x_last = x_last.replace([np.inf, -np.inf], np.nan)
    if x_last.isna().any(axis=1).iloc[0]:
        return None, None, auc

    prob_up = float(clf.predict_proba(x_last)[:, 1][0])
    exp_ret = float(reg.predict(x_last)[0])
    return prob_up, exp_ret, auc


def build_forecasts(X: pd.DataFrame, y_df: pd.DataFrame) -> list[HorizonForecast]:
    out: list[HorizonForecast] = []
    for h, col in [("3m", "fwd_ret_3m"), ("6m", "fwd_ret_6m"), ("12m", "fwd_ret_12m")]:
        p, r, auc = fit_and_forecast(X=X, y=y_df[col])
        out.append(HorizonForecast(horizon=h, prob_up=p, exp_return=r, auc_cv=auc))
    return out



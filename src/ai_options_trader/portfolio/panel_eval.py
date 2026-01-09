from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_options_trader.portfolio.panel_model import _make_regressor, _make_classifier  # internal, but stable enough for MVP


@dataclass(frozen=True)
class PanelEvalResult:
    status: str
    n_dates: int
    n_preds: int
    spearman_mean: float | None
    hit_rate_mean: float | None
    top_bottom_spread_mean: float | None
    notes: str = ""


def walk_forward_panel_eval(
    *,
    X: pd.DataFrame,  # MultiIndex(date,ticker)
    y: pd.Series,  # forward return (%), aligned to X.index
    horizon_days: int = 63,
    min_train_days: int = 750,  # ~3y trading days of cross-sections
    step_days: int = 5,  # evaluate weekly to keep it light
    top_k: int = 3,
) -> PanelEvalResult:
    """
    Leak-resistant walk-forward evaluation:
    - Split by date
    - Purge last `horizon_days` trading days from training before each test date
    - Predict cross-section for test date
    Metrics per date:
    - Spearman(rank(pred), rank(realized))
    - Hit rate: share of tickers where sign(pred) == sign(realized)
    - Top-bottom spread: mean(realized(top_k)) - mean(realized(bottom_k))
    """
    if X.empty or y.empty:
        return PanelEvalResult(status="empty", n_dates=0, n_preds=0, spearman_mean=None, hit_rate_mean=None, top_bottom_spread_mean=None)

    # Align
    idx = X.index.intersection(y.index)
    X2 = X.loc[idx].replace([np.inf, -np.inf], np.nan)
    y2 = y.loc[idx].replace([np.inf, -np.inf], np.nan)
    keep = X2.dropna().index.intersection(y2.dropna().index)
    X2 = X2.loc[keep]
    y2 = y2.loc[keep]
    if X2.empty:
        return PanelEvalResult(status="empty_after_dropna", n_dates=0, n_preds=0, spearman_mean=None, hit_rate_mean=None, top_bottom_spread_mean=None)

    dates = pd.Index(sorted(set(X2.index.get_level_values(0))))
    if len(dates) < 200:
        return PanelEvalResult(status="insufficient_dates", n_dates=int(len(dates)), n_preds=0, spearman_mean=None, hit_rate_mean=None, top_bottom_spread_mean=None)

    # Use business-day stepping; evaluate every `step_days` dates in the available index.
    test_dates = dates[:: max(1, int(step_days))]
    spears: list[float] = []
    hits: list[float] = []
    spreads: list[float] = []
    n_preds = 0
    n_dates_used = 0

    for d in test_dates:
        # Purge window: exclude the last horizon_days dates before d
        # We do this by training only on dates <= (d - horizon_days business days).
        cutoff = pd.to_datetime(d) - pd.tseries.offsets.BDay(int(horizon_days))
        train_dates = dates[dates <= cutoff]
        if len(train_dates) < int(min_train_days):
            continue

        train_mask = X2.index.get_level_values(0).isin(train_dates)
        test_mask = X2.index.get_level_values(0) == d
        Xtr = X2.loc[train_mask]
        ytr = y2.loc[train_mask]
        Xte = X2.loc[test_mask]
        yte = y2.loc[test_mask]
        if Xte.empty or yte.empty or len(Xtr) < 1000:
            continue

        yb = (ytr > 0.0).astype(int)
        if yb.nunique() < 2:
            continue

        reg = _make_regressor()
        clf = _make_classifier()
        reg.fit(Xtr, ytr)
        clf.fit(Xtr, yb)

        pred_r = pd.Series(reg.predict(Xte), index=Xte.index)
        pred_p = pd.Series(clf.predict_proba(Xte)[:, 1], index=Xte.index)

        # Cross-sectional metrics
        rr = pd.Series(yte.values, index=yte.index)
        # Spearman on exp_return predictions
        if rr.shape[0] >= 5:
            # Avoid scipy dependency: compute Spearman via rank correlation.
            pr = pd.Series(pred_r.values).rank(method="average")
            tr = pd.Series(rr.values).rank(method="average")
            s = pr.corr(tr, method="pearson")
            if s is not None and np.isfinite(s):
                spears.append(float(s))

        # Hit rate: sign agreement (using exp_return sign)
        hit = float((np.sign(pred_r.values) == np.sign(rr.values)).mean()) if rr.shape[0] else np.nan
        if np.isfinite(hit):
            hits.append(hit)

        # Top/bottom spread
        order = pred_r.sort_values(ascending=False)
        k = int(min(top_k, max(1, rr.shape[0] // 2)))
        top_idx = order.index[:k]
        bot_idx = order.index[-k:]
        spread = float(rr.loc[top_idx].mean() - rr.loc[bot_idx].mean())
        if np.isfinite(spread):
            spreads.append(spread)

        n_preds += int(rr.shape[0])
        n_dates_used += 1

    if n_dates_used == 0:
        return PanelEvalResult(status="no_valid_folds", n_dates=0, n_preds=0, spearman_mean=None, hit_rate_mean=None, top_bottom_spread_mean=None)

    return PanelEvalResult(
        status="ok",
        n_dates=int(n_dates_used),
        n_preds=int(n_preds),
        spearman_mean=float(np.mean(spears)) if spears else None,
        hit_rate_mean=float(np.mean(hits)) if hits else None,
        top_bottom_spread_mean=float(np.mean(spreads)) if spreads else None,
        notes=f"purge={horizon_days}bd step={step_days} top_k={top_k}",
    )


def walk_forward_panel_portfolio(
    *,
    X: pd.DataFrame,  # MultiIndex(date,ticker)
    y: pd.Series,
    horizon_days: int = 63,
    min_train_days: int = 750,
    step_days: int = 5,
    top_k: int = 3,
    tc_bps: float = 5.0,
    stress_vol_z: float = 1.5,
    book: str = "longshort",  # "longshort"|"longonly"
) -> pd.DataFrame:
    """
    Trading-like walk-forward series for a simple book:
    - Long: top_k predicted exp_return tickers
    - Short (optional): bottom_k predicted exp_return tickers (book="longshort")
    - Return:
        - longonly: mean(realized top_k)
        - longshort: mean(realized top_k) - mean(realized bottom_k)
    - Turnover: fraction of positions changed vs previous rebalance
    - Costs:
        - longonly: tc_bps applied to long turnover
        - longshort: tc_bps applied to (long+short) turnover (single blended turnover rate)

    Also tags each rebalance date with:
    - decade bucket
    - stress flag (vol_z_vix >= stress_vol_z, when available)
    """
    if X.empty or y.empty:
        return pd.DataFrame()

    idx = X.index.intersection(y.index)
    X2 = X.loc[idx].replace([np.inf, -np.inf], np.nan)
    y2 = y.loc[idx].replace([np.inf, -np.inf], np.nan)
    keep = X2.dropna().index.intersection(y2.dropna().index)
    X2 = X2.loc[keep]
    y2 = y2.loc[keep]
    if X2.empty:
        return pd.DataFrame()

    dates = pd.Index(sorted(set(X2.index.get_level_values(0))))
    test_dates = dates[:: max(1, int(step_days))]

    prev_long: set[str] = set()
    prev_short: set[str] = set()
    rows = []

    book_s = (book or "longshort").strip().lower()
    if book_s not in {"longshort", "longonly"}:
        book_s = "longshort"

    for d in test_dates:
        cutoff = pd.to_datetime(d) - pd.tseries.offsets.BDay(int(horizon_days))
        train_dates = dates[dates <= cutoff]
        if len(train_dates) < int(min_train_days):
            continue

        train_mask = X2.index.get_level_values(0).isin(train_dates)
        test_mask = X2.index.get_level_values(0) == d
        Xtr = X2.loc[train_mask]
        ytr = y2.loc[train_mask]
        Xte = X2.loc[test_mask]
        yte = y2.loc[test_mask]
        if Xte.empty or yte.empty or len(Xtr) < 1000:
            continue
        yb = (ytr > 0.0).astype(int)
        if yb.nunique() < 2:
            continue

        reg = _make_regressor()
        reg.fit(Xtr, ytr)
        pred = pd.Series(reg.predict(Xte), index=Xte.index)
        realized = pd.Series(yte.values, index=yte.index)

        # Select top/bottom
        order = pred.sort_values(ascending=False)
        k = int(min(top_k, max(1, realized.shape[0] // 2)))
        top_idx = order.index[:k]
        bot_idx = order.index[-k:]
        long_syms = {str(t) for (_dd, t) in top_idx}
        short_syms = {str(t) for (_dd, t) in bot_idx}

        long_ret = float(realized.loc[top_idx].mean())
        short_ret = float(realized.loc[bot_idx].mean())
        ls_gross = long_ret - short_ret

        # Turnover vs previous
        long_changes = len(long_syms.symmetric_difference(prev_long))
        short_changes = len(short_syms.symmetric_difference(prev_short))
        denom = max(1, 2 * k)  # long+short book size
        turnover = float((long_changes + short_changes) / denom)

        # Long-only turnover/costs
        long_turnover = float(long_changes / max(1, k))
        long_cost = float(tc_bps) / 10000.0 * 100.0 * long_turnover
        long_net = long_ret - long_cost

        # Long/short blended turnover/costs
        cost = float(tc_bps) / 10000.0 * 100.0 * turnover  # convert bps to percent return drag
        ls_net = ls_gross - cost

        # Decade bucket
        year = int(pd.to_datetime(d).year)
        decade = int(year // 10 * 10)

        # Stress flag using vol_z_vix if present; fall back to vol_pressure_score.
        stress = False
        try:
            # Same across tickers; pick first row for date.
            any_row = Xte.iloc[0]
            vz = any_row.get("vol_z_vix")
            if isinstance(vz, (int, float)) and float(vz) >= float(stress_vol_z):
                stress = True
            else:
                vp = any_row.get("vol_pressure_score")
                if isinstance(vp, (int, float)) and float(vp) >= float(stress_vol_z):
                    stress = True
        except Exception:
            stress = False

        rows.append(
            {
                "date": pd.to_datetime(d),
                "k": k,
                "long_ret": long_ret,
                "short_ret": short_ret,
                "ls_ret_gross": ls_gross,
                "turnover": turnover,
                "long_turnover": long_turnover,
                "tc_bps": float(tc_bps),
                "ls_ret_net": ls_net,
                "long_ret_net": long_net,
                "book": book_s,
                "decade": str(decade),
                "stress": str(bool(stress)),
                "long": ",".join(sorted(long_syms)),
                "short": ",".join(sorted(short_syms)),
            }
        )

        prev_long, prev_short = long_syms, short_syms

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").sort_index()



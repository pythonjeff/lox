from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MoonshotStats:
    ticker: str
    direction: str  # bullish|bearish
    samples: int
    horizon_days: int
    k_analogs: int
    score: float

    # Distribution summary (analog-conditioned forward returns)
    q05: float | None
    q50: float | None
    q95: float | None
    best: float | None
    worst: float | None

    # "Extreme move happened before" evidence
    extreme_date: pd.Timestamp | None
    extreme_return: float | None


def _to_float_df(x: pd.DataFrame) -> pd.DataFrame:
    y = x.copy()
    for c in y.columns:
        y[c] = pd.to_numeric(y[c], errors="coerce")
    return y.astype(float)


def _analog_dates(
    *,
    regimes: pd.DataFrame,
    asof: pd.Timestamp,
    horizon_days: int,
    k_analogs: int,
) -> pd.DatetimeIndex:
    """
    Pick the k closest historical regime dates to `asof`, using mean absolute distance.
    Excludes the trailing horizon window to avoid lookahead.
    """
    if regimes.empty:
        return pd.DatetimeIndex([])

    r = _to_float_df(regimes).sort_index()
    if asof not in r.index:
        asof = pd.to_datetime(r.index.max())

    # Exclude the tail window (and the asof day itself) so we don't "peek" across label horizon.
    cutoff = pd.to_datetime(asof) - pd.Timedelta(days=int(horizon_days) + 1)
    hist = r.loc[:cutoff]
    if hist.empty:
        return pd.DatetimeIndex([])

    cur = r.loc[pd.to_datetime(asof)]
    cur = cur.fillna(0.0)
    hist = hist.fillna(0.0)

    # Mean absolute distance across all regime features (already roughly z-scored).
    d = (hist.sub(cur, axis=1)).abs().mean(axis=1)
    return pd.to_datetime(d.nsmallest(max(1, int(k_analogs))).index)


def rank_moonshots(
    *,
    px: pd.DataFrame,
    regimes: pd.DataFrame,
    asof: pd.Timestamp | None = None,
    horizon_days: int = 7,
    k_analogs: int = 250,
    min_abs_extreme: float = 0.15,
    min_samples: int = 40,
    direction: str = "both",  # bullish|bearish|both
) -> list[MoonshotStats]:
    """
    Rank "moonshot" candidates by looking for extreme historical forward moves that occurred under
    regime conditions similar to today.

    Returns a list of MoonshotStats sorted by score descending.
    """
    if px is None or px.empty or regimes is None or regimes.empty:
        return []

    px2 = px.sort_index().copy()
    px2 = px2.apply(pd.to_numeric, errors="coerce").astype(float)
    px2 = px2.dropna(how="all")

    reg = _to_float_df(regimes).sort_index()
    reg = reg.dropna(how="all")
    if reg.empty or px2.empty:
        return []

    if asof is None:
        asof = min(pd.to_datetime(reg.index.max()), pd.to_datetime(px2.index.max()))
    asof = pd.to_datetime(asof)

    dates = _analog_dates(regimes=reg, asof=asof, horizon_days=int(horizon_days), k_analogs=int(k_analogs))
    if len(dates) < max(10, int(min_samples) // 4):
        return []

    out: list[MoonshotStats] = []
    want = (direction or "both").strip().lower()
    if want not in {"bullish", "bearish", "both"}:
        want = "both"

    # Precompute forward returns for all tickers (vectorized).
    fwd = px2.shift(-int(horizon_days)).div(px2).sub(1.0)

    for tkr in px2.columns:
        s = fwd[tkr].dropna()
        if s.empty:
            continue
        ss = s.reindex(dates).dropna()
        if int(ss.shape[0]) < int(min_samples):
            continue

        q05 = float(ss.quantile(0.05))
        q50 = float(ss.quantile(0.50))
        q95 = float(ss.quantile(0.95))
        best = float(ss.max())
        worst = float(ss.min())

        # Bullish moonshot: large positive extremes happened before under similar regimes.
        if want in {"bullish", "both"} and best >= float(min_abs_extreme):
            extreme_date = pd.to_datetime(ss.idxmax())
            extreme_ret = float(ss.loc[extreme_date])
            # Score emphasizes tail (q95) and the single best analog ("it happened before").
            score = float(q95 + 0.35 * best)
            out.append(
                MoonshotStats(
                    ticker=str(tkr),
                    direction="bullish",
                    samples=int(ss.shape[0]),
                    horizon_days=int(horizon_days),
                    k_analogs=int(k_analogs),
                    score=score,
                    q05=q05,
                    q50=q50,
                    q95=q95,
                    best=best,
                    worst=worst,
                    extreme_date=extreme_date,
                    extreme_return=extreme_ret,
                )
            )

        # Bearish moonshot: large negative extremes happened before under similar regimes.
        if want in {"bearish", "both"} and abs(worst) >= float(min_abs_extreme):
            extreme_date = pd.to_datetime(ss.idxmin())
            extreme_ret = float(ss.loc[extreme_date])
            # Score emphasizes negative tail and worst analog.
            score = float((-q05) + 0.35 * abs(worst))
            out.append(
                MoonshotStats(
                    ticker=str(tkr),
                    direction="bearish",
                    samples=int(ss.shape[0]),
                    horizon_days=int(horizon_days),
                    k_analogs=int(k_analogs),
                    score=score,
                    q05=q05,
                    q50=q50,
                    q95=q95,
                    best=best,
                    worst=worst,
                    extreme_date=extreme_date,
                    extreme_return=extreme_ret,
                )
            )

    out.sort(key=lambda x: (float(x.score), float(x.samples)), reverse=True)
    return out


def rank_moonshots_unconditional(
    *,
    px: pd.DataFrame,
    asof: pd.Timestamp | None = None,
    horizon_days: int = 7,
    min_samples: int = 200,
    direction: str = "both",  # bullish|bearish|both
) -> list[MoonshotStats]:
    """
    Fallback "always rank something" moonshot list.

    This ignores regime similarity and ranks purely on unconditional forward-return tails.
    It is used when the analog-based filter finds no candidates.
    """
    if px is None or px.empty:
        return []

    px2 = px.sort_index().copy()
    px2 = px2.apply(pd.to_numeric, errors="coerce").astype(float)
    px2 = px2.dropna(how="all")
    if px2.empty:
        return []

    if asof is None:
        asof = pd.to_datetime(px2.index.max())
    asof = pd.to_datetime(asof)

    cutoff = pd.to_datetime(asof) - pd.Timedelta(days=int(horizon_days) + 1)
    px_hist = px2.loc[:cutoff]
    if px_hist.empty:
        return []

    want = (direction or "both").strip().lower()
    if want not in {"bullish", "bearish", "both"}:
        want = "both"

    fwd = px_hist.shift(-int(horizon_days)).div(px_hist).sub(1.0)
    out: list[MoonshotStats] = []
    for tkr in px_hist.columns:
        ss = fwd[tkr].dropna()
        if int(ss.shape[0]) < int(min_samples):
            continue
        q05 = float(ss.quantile(0.05))
        q50 = float(ss.quantile(0.50))
        q95 = float(ss.quantile(0.95))
        best = float(ss.max())
        worst = float(ss.min())

        if want in {"bullish", "both"}:
            # Score emphasizes right tail; "extreme evidence" is best historical point.
            extreme_date = pd.to_datetime(ss.idxmax()) if not ss.empty else None
            extreme_ret = float(ss.loc[extreme_date]) if extreme_date is not None else None
            score = float(q95 + 0.35 * best)
            out.append(
                MoonshotStats(
                    ticker=str(tkr),
                    direction="bullish",
                    samples=int(ss.shape[0]),
                    horizon_days=int(horizon_days),
                    k_analogs=int(ss.shape[0]),
                    score=score,
                    q05=q05,
                    q50=q50,
                    q95=q95,
                    best=best,
                    worst=worst,
                    extreme_date=extreme_date,
                    extreme_return=extreme_ret,
                )
            )
        if want in {"bearish", "both"}:
            extreme_date = pd.to_datetime(ss.idxmin()) if not ss.empty else None
            extreme_ret = float(ss.loc[extreme_date]) if extreme_date is not None else None
            score = float((-q05) + 0.35 * abs(worst))
            out.append(
                MoonshotStats(
                    ticker=str(tkr),
                    direction="bearish",
                    samples=int(ss.shape[0]),
                    horizon_days=int(horizon_days),
                    k_analogs=int(ss.shape[0]),
                    score=score,
                    q05=q05,
                    q50=q50,
                    q95=q95,
                    best=best,
                    worst=worst,
                    extreme_date=extreme_date,
                    extreme_return=extreme_ret,
                )
            )

    out.sort(key=lambda x: (float(x.score), float(x.samples)), reverse=True)
    return out


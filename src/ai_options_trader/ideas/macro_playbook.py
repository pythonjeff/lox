from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlaybookIdea:
    ticker: str
    direction: str  # bullish|bearish
    horizon_days: int
    n_matches: int
    exp_return: float  # mean fwd return (%)
    median_return: float
    hit_rate: float  # P(fwd_ret > 0)
    worst: float  # min fwd return (%)
    best: float  # max fwd return (%)
    score: float  # ranking scalar
    notes: dict


def _zscore_frame(X: pd.DataFrame) -> pd.DataFrame:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0).replace(0, np.nan)
    return (X - mu) / sd


def _forward_return(series: pd.Series, days: int) -> pd.Series:
    return (series.shift(-days) / series - 1.0) * 100.0


def build_forward_returns(prices: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    px = prices.sort_index().ffill()
    out = {}
    for c in px.columns:
        out[c] = _forward_return(px[c], int(horizon_days))
    return pd.DataFrame(out, index=px.index)


def _nearest_neighbors(
    X: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    k: int = 250,
    lookback_days: int = 252 * 7,
) -> pd.DatetimeIndex:
    """
    Pick k nearest historical dates to `asof` in feature space.

    Notes:
    - Uses z-scored features for distance.
    - Restricts to a trailing lookback window for stability.
    """
    X = X.sort_index()
    if asof not in X.index:
        # choose last available
        asof = X.index.max()

    start = asof - pd.Timedelta(days=int(lookback_days))
    hist = X.loc[(X.index < asof) & (X.index >= start)].dropna(how="any")
    if hist.empty:
        return pd.DatetimeIndex([])

    x0 = X.loc[[asof]].dropna(axis=1, how="any")
    cols = [c for c in x0.columns if c in hist.columns]
    hist2 = hist[cols].dropna(how="any")
    if hist2.empty or not cols:
        return pd.DatetimeIndex([])

    # z-score using historical window stats (numpy path for speed)
    mu = hist2.mean(axis=0).to_numpy(dtype=float)
    sd = hist2.std(axis=0, ddof=0).replace(0, np.nan).to_numpy(dtype=float)
    H = hist2.to_numpy(dtype=float)
    x = x0[cols].iloc[0].to_numpy(dtype=float)

    # handle NaNs / zero std
    sd = np.where(np.isfinite(sd), sd, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    H = np.where(np.isfinite(H), H, mu)
    x = np.where(np.isfinite(x), x, mu)

    Hz = (H - mu) / sd
    xz = (x - mu) / sd
    # Euclidean distances
    d = np.sqrt(((Hz - xz) ** 2).sum(axis=1))
    take = min(max(10, int(k)), d.shape[0])
    idx_sorted = np.argsort(d)[:take]
    return pd.DatetimeIndex(hist2.index[idx_sorted])


def rank_macro_playbook(
    *,
    features: pd.DataFrame,
    prices: pd.DataFrame,
    tickers: Iterable[str],
    horizon_days: int = 63,  # ~3 months
    k: int = 250,
    lookback_days: int = 252 * 7,
    min_matches: int = 60,
    asof: pd.Timestamp | None = None,
) -> list[PlaybookIdea]:
    """
    Regime-conditioned playbook:
    - Find historical dates with similar regime feature vectors (kNN)
    - For each ticker, look at forward returns over the horizon from those dates
    - Rank ideas by a simple quality score
    """
    X = features.sort_index()
    px = prices.sort_index().ffill()
    idx = X.index.intersection(px.index)
    X = X.loc[idx].dropna(how="any")
    px = px.loc[idx]
    if X.empty:
        return []

    asof = asof or X.index.max()
    nbrs = _nearest_neighbors(X, asof, k=k, lookback_days=lookback_days)
    if len(nbrs) < int(min_matches):
        return []

    fwd = build_forward_returns(px, horizon_days=int(horizon_days))
    ideas: list[PlaybookIdea] = []

    for t in tickers:
        if t not in fwd.columns:
            continue
        s = pd.to_numeric(fwd.loc[nbrs, t], errors="coerce").dropna()
        if s.shape[0] < int(min_matches):
            continue

        exp_ret = float(s.mean())
        med = float(s.median())
        hit = float((s > 0).mean())
        worst = float(s.min())
        best = float(s.max())
        direction = "bullish" if exp_ret >= 0 else "bearish"

        # Score: reward magnitude + consistency, penalize ugly tails a bit.
        score = (abs(exp_ret) * (0.5 + hit)) - 0.05 * abs(worst)

        ideas.append(
            PlaybookIdea(
                ticker=t,
                direction=direction,
                horizon_days=int(horizon_days),
                n_matches=int(s.shape[0]),
                exp_return=exp_ret,
                median_return=med,
                hit_rate=hit,
                worst=worst,
                best=best,
                score=float(score),
                notes={"asof": str(asof.date()), "neighbors_start": str(nbrs.min().date()), "neighbors_end": str(nbrs.max().date())},
            )
        )

    ideas.sort(key=lambda x: x.score, reverse=True)
    return ideas



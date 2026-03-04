"""
Macro Playbook — regime-conditioned forward-return scoring via k-NN analogs.

Answers: "In the k past days most similar to today's regime feature vector,
which assets delivered the best forward N-day returns?"

This is the statistical backbone for `lox suggest`. It provides expected
returns, hit rates, Sharpe estimates, and VaR directly from historical
analog distributions — no Monte Carlo required.

Usage:
    from lox.ideas.macro_playbook import rank_macro_playbook

    ideas = rank_macro_playbook(
        features=feature_matrix,   # pd.DataFrame (date-indexed, numeric columns)
        prices=price_panel,        # pd.DataFrame (date-indexed, ticker columns)
        tickers=["XLE", "GLD"],
        horizon_days=20,
        k=120,
        min_matches=50,
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PlaybookIdea:
    """Single playbook scoring result for one ticker."""

    ticker: str
    direction: str          # "bullish" or "bearish"
    exp_return: float       # mean forward return across analogs
    hit_rate: float         # fraction of analogs with positive forward return
    sharpe_est: float       # exp_return / std(forward returns), annualised
    var_5: float            # 5th percentile of forward returns (worst-case proxy)
    n_analogs: int          # how many analogs had valid price data for this ticker
    fwd_returns: np.ndarray # raw forward returns for downstream analysis


def rank_macro_playbook(
    *,
    features: pd.DataFrame,
    prices: pd.DataFrame,
    tickers: list[str],
    horizon_days: int = 20,
    k: int = 120,
    min_matches: int = 50,
) -> list[PlaybookIdea]:
    """
    Score tickers by regime-conditioned forward returns using k-NN analogs.

    Parameters
    ----------
    features : pd.DataFrame
        Regime feature matrix, date-indexed, numeric columns.
        The last row is treated as the query (current regime state).
    prices : pd.DataFrame
        Daily close prices, date-indexed, one column per ticker.
    tickers : list[str]
        Candidate tickers to score (must be columns in ``prices``).
    horizon_days : int
        Forward return horizon in trading days (default 20 ≈ 1 month).
    k : int
        Number of nearest neighbors to use (default 120).
    min_matches : int
        Minimum analogs with valid forward returns required to include a
        ticker in the output (default 50).

    Returns
    -------
    list[PlaybookIdea]
        Sorted by absolute expected return descending (strongest signal first).
    """
    features = features.copy()
    prices = prices.copy()

    feat_cols = [c for c in features.columns if features[c].dtype.kind in ("f", "i")]
    if not feat_cols:
        logger.warning("No numeric feature columns found")
        return []

    feat_num = features[feat_cols].astype(float)

    # Z-score features for distance calculation (per-column)
    means = feat_num.mean()
    stds = feat_num.std().replace(0, 1)
    feat_z = (feat_num - means) / stds
    feat_z = feat_z.fillna(0)

    if len(feat_z) < 2:
        logger.warning("Feature matrix too short (%d rows)", len(feat_z))
        return []

    query = feat_z.iloc[-1].values.astype(np.float64)

    # Historical rows (exclude last row — that's our query)
    hist = feat_z.iloc[:-1]
    hist_dates = hist.index
    hist_vals = hist.values.astype(np.float64)

    # Euclidean distance from query to every historical row
    diffs = hist_vals - query[np.newaxis, :]
    dists = np.sqrt(np.nansum(diffs ** 2, axis=1))

    # Select k nearest
    k_actual = min(k, len(dists))
    nearest_idx = np.argpartition(dists, k_actual)[:k_actual]
    analog_dates = hist_dates[nearest_idx]

    # Align prices to feature dates
    prices_aligned = prices.reindex(features.index).ffill()

    ideas: list[PlaybookIdea] = []
    for ticker in tickers:
        if ticker not in prices_aligned.columns:
            logger.debug("Ticker %s not in price panel, skipping", ticker)
            continue

        px = prices_aligned[ticker]
        fwd_returns = []

        for d in analog_dates:
            loc = prices_aligned.index.get_loc(d)
            end_loc = loc + horizon_days
            if end_loc >= len(px):
                continue
            p0 = px.iloc[loc]
            p1 = px.iloc[end_loc]
            if p0 is None or p1 is None or p0 == 0 or np.isnan(p0) or np.isnan(p1):
                continue
            fwd_returns.append((p1 - p0) / p0)

        if len(fwd_returns) < min_matches:
            logger.debug(
                "Ticker %s: only %d analogs (need %d), skipping",
                ticker, len(fwd_returns), min_matches,
            )
            continue

        fwd_arr = np.array(fwd_returns, dtype=np.float64)
        exp_ret = float(np.mean(fwd_arr))
        std_ret = float(np.std(fwd_arr, ddof=1)) if len(fwd_arr) > 1 else 1e-9
        hit_rate = float(np.mean(fwd_arr > 0))
        # Annualise Sharpe: scale by sqrt(252 / horizon_days)
        sharpe = (exp_ret / std_ret) * np.sqrt(252 / horizon_days) if std_ret > 1e-9 else 0.0
        var_5 = float(np.percentile(fwd_arr, 5))

        ideas.append(PlaybookIdea(
            ticker=ticker,
            direction="bullish" if exp_ret >= 0 else "bearish",
            exp_return=exp_ret,
            hit_rate=hit_rate,
            sharpe_est=sharpe,
            var_5=var_5,
            n_analogs=len(fwd_arr),
            fwd_returns=fwd_arr,
        ))

    ideas.sort(key=lambda x: abs(x.exp_return), reverse=True)
    return ideas

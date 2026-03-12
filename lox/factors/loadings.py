"""
Rolling OLS factor regression — compute factor betas per ticker.

Uses pure numpy (np.linalg.lstsq) to avoid statsmodels dependency.

Author: Lox Capital Research
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FACTOR_NAMES = ["Mkt", "SMB", "HML", "RMW", "CMA", "Mom"]
FACTOR_DF_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
MIN_OBS = 60


@dataclass(frozen=True)
class PositionLoadings:
    """Factor regression results for a single position."""

    ticker: str
    symbol: str
    position_type: str  # equity, call, put, short_equity
    weight_pct: float
    market_value: float

    # Factor betas
    mkt_beta: float
    smb: float
    hml: float
    rmw: float
    cma: float
    mom: float

    # Diagnostics
    r_squared: float
    alpha_ann: float       # annualized alpha (daily * 252)
    residual_vol: float    # annualized residual vol
    n_obs: int
    data_warning: str      # "" if OK

    def betas_dict(self) -> dict[str, float]:
        return {
            "Mkt": self.mkt_beta, "SMB": self.smb, "HML": self.hml,
            "RMW": self.rmw, "CMA": self.cma, "Mom": self.mom,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "symbol": self.symbol,
            "position_type": self.position_type,
            "weight_pct": round(self.weight_pct, 2),
            "market_value": round(self.market_value, 2),
            "betas": {k: round(v, 4) for k, v in self.betas_dict().items()},
            "r_squared": round(self.r_squared, 4),
            "alpha_ann_pct": round(self.alpha_ann * 100, 2),
            "residual_vol_pct": round(self.residual_vol * 100, 2),
            "n_obs": self.n_obs,
            "data_warning": self.data_warning,
        }


def _ols_regression(
    y: np.ndarray,
    X: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Pure numpy OLS: y = alpha + X @ beta + epsilon.

    Returns (betas, r_squared, alpha, residual_std).
    """
    T, K = X.shape
    X_aug = np.column_stack([np.ones(T), X])

    result = np.linalg.lstsq(X_aug, y, rcond=None)
    beta_full = result[0]

    alpha = float(beta_full[0])
    betas = beta_full[1:]

    fitted = X_aug @ beta_full
    residuals = y - fitted

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    r2 = max(0.0, min(1.0, r2))

    dof = T - K - 1
    residual_std = float(np.std(residuals, ddof=max(1, K + 1))) if dof > 0 else 0.0

    return betas, r2, alpha, residual_std


def compute_single_loadings(
    ticker: str,
    symbol: str,
    position_type: str,
    weight_pct: float,
    market_value: float,
    price_series: pd.Series,
    factor_df: pd.DataFrame,
    window: int = 252,
) -> PositionLoadings:
    """Compute factor loadings for one ticker via rolling OLS."""
    # Daily returns
    returns = price_series.pct_change().dropna()
    returns.index = pd.to_datetime(returns.index)

    # Align with factor data
    rf = factor_df["RF"]
    factors = factor_df[FACTOR_DF_COLS]

    # Inner join on dates
    combined = pd.DataFrame({"ret": returns}).join(factors).join(rf).dropna()

    # Take trailing window
    if len(combined) > window:
        combined = combined.iloc[-window:]

    n_obs = len(combined)
    warning = ""
    if n_obs < MIN_OBS:
        warning = f"Only {n_obs} obs (min {MIN_OBS})"
    if n_obs < 10:
        # Too few for any meaningful regression
        return PositionLoadings(
            ticker=ticker, symbol=symbol, position_type=position_type,
            weight_pct=weight_pct, market_value=market_value,
            mkt_beta=0.0, smb=0.0, hml=0.0, rmw=0.0, cma=0.0, mom=0.0,
            r_squared=0.0, alpha_ann=0.0, residual_vol=0.0,
            n_obs=n_obs, data_warning=f"Insufficient data ({n_obs} obs)",
        )

    # Excess returns
    y = (combined["ret"] - combined["RF"]).values
    X = combined[FACTOR_DF_COLS].values

    betas, r2, alpha, res_std = _ols_regression(y, X)

    return PositionLoadings(
        ticker=ticker,
        symbol=symbol,
        position_type=position_type,
        weight_pct=weight_pct,
        market_value=market_value,
        mkt_beta=float(betas[0]),
        smb=float(betas[1]),
        hml=float(betas[2]),
        rmw=float(betas[3]),
        cma=float(betas[4]),
        mom=float(betas[5]),
        r_squared=r2,
        alpha_ann=alpha * 252,
        residual_vol=res_std * np.sqrt(252),
        n_obs=n_obs,
        data_warning=warning,
    )


def compute_all_loadings(
    settings: Any,
    positions: list[dict],
    factor_df: pd.DataFrame,
    window: int = 252,
    refresh: bool = False,
) -> list[PositionLoadings]:
    """Compute factor loadings for all positions in parallel.

    Fetches daily closes for unique underlyings, runs OLS for each.
    """
    from lox.cli_commands.shared.book_impact import parse_position_type
    from lox.data.market import fetch_equity_daily_closes

    # Build position metadata
    pos_meta: list[dict] = []
    total_abs_mv = sum(abs(p["market_value"]) for p in positions)

    for p in positions:
        underlying, pos_type = parse_position_type(p["symbol"], p["qty"])
        weight = (p["market_value"] / total_abs_mv * 100) if total_abs_mv > 0 else 0
        # Puts and shorts have negative effective weight
        if pos_type in ("put", "short_equity"):
            weight = -abs(weight)
        pos_meta.append({
            "ticker": underlying,
            "symbol": p["symbol"],
            "position_type": pos_type,
            "weight_pct": weight,
            "market_value": p["market_value"],
        })

    # Unique tickers to fetch
    unique_tickers = list({pm["ticker"] for pm in pos_meta})

    # Fetch daily closes — need window + buffer for returns calc
    lookback_days = int((window + 60) * 1.5)  # calendar days
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    try:
        prices_df = fetch_equity_daily_closes(
            settings=settings,
            symbols=unique_tickers,
            start=start_date,
            refresh=refresh,
        )
    except Exception as e:
        logger.warning(f"Failed to fetch price data: {e}")
        return []

    # Compute loadings in parallel
    results: list[PositionLoadings] = []

    def _compute_one(pm: dict) -> PositionLoadings:
        ticker = pm["ticker"]
        if ticker not in prices_df.columns:
            return PositionLoadings(
                ticker=ticker, symbol=pm["symbol"],
                position_type=pm["position_type"],
                weight_pct=pm["weight_pct"], market_value=pm["market_value"],
                mkt_beta=0.0, smb=0.0, hml=0.0, rmw=0.0, cma=0.0, mom=0.0,
                r_squared=0.0, alpha_ann=0.0, residual_vol=0.0,
                n_obs=0, data_warning=f"No price data for {ticker}",
            )
        return compute_single_loadings(
            ticker=ticker,
            symbol=pm["symbol"],
            position_type=pm["position_type"],
            weight_pct=pm["weight_pct"],
            market_value=pm["market_value"],
            price_series=prices_df[ticker].dropna(),
            factor_df=factor_df,
            window=window,
        )

    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="factor") as pool:
        futures = {pool.submit(_compute_one, pm): pm for pm in pos_meta}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                pm = futures[future]
                logger.warning(f"Factor regression failed for {pm['ticker']}: {e}")

    # Sort by absolute weight descending
    results.sort(key=lambda r: -abs(r.weight_pct))
    return results

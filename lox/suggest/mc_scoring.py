"""
Monte Carlo forward scoring for suggest candidates (opt-in via --deep).

Runs regime-conditioned block-bootstrap simulation per candidate ticker
using the existing `compute_quant_scenarios` engine, then extracts
expected return and risk metrics for composite scoring.

This module is only called when the user passes `--deep` to `lox suggest`.
The default path uses playbook analog returns instead (faster, no MC).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lox.regimes.features import UnifiedRegimeState

logger = logging.getLogger(__name__)

N_SIMS = 5_000       # reduced from 10k for batch speed
HORIZON_DAYS = 63     # 3 months
LOOKBACK_DAYS = 400   # ~1.5 years of trading history


@dataclass
class MCScore:
    """MC simulation result for one candidate ticker."""

    ticker: str
    current_price: float
    median_return: float     # (p50 - current) / current
    p5_return: float         # VaR proxy: (p5 - current) / current
    p25_return: float        # downside scenario
    p75_return: float        # upside scenario
    upside_ratio: float      # p75_return / abs(p25_return) — reward/risk
    realized_vol: float      # annualised historical vol


def score_candidates_mc(
    *,
    tickers: list[str],
    regime_state: "UnifiedRegimeState | None" = None,
    settings=None,
    benchmark: str = "SPY",
) -> dict[str, MCScore]:
    """
    Run regime-conditioned MC simulation for each candidate ticker.

    Returns dict of ticker -> MCScore. Tickers that fail (missing data,
    too few prices) are silently skipped.
    """
    from lox.data.market import fetch_equity_daily_closes
    from lox.llm.scenarios.quant_scenarios import compute_quant_scenarios

    if settings is None:
        return {}

    all_syms = list(set([benchmark] + [t.upper() for t in tickers]))
    start = (datetime.now() - timedelta(days=LOOKBACK_DAYS + 60)).strftime("%Y-%m-%d")

    try:
        px = fetch_equity_daily_closes(
            settings=settings, symbols=all_syms, start=start, refresh=False,
        )
    except Exception as e:
        logger.warning("MC scoring: price fetch failed: %s", e)
        return {}

    if px is None or px.empty:
        return {}

    # Run MC for benchmark first to get baseline distribution
    bench_result = _run_mc_for_ticker(px, benchmark, regime_state)

    results: dict[str, MCScore] = {}
    for ticker in tickers:
        t = ticker.upper()
        if t not in px.columns:
            continue
        mc = _run_mc_for_ticker(px, t, regime_state)
        if mc is not None:
            results[t] = mc

    return results


def _run_mc_for_ticker(
    px_df,
    ticker: str,
    regime_state: "UnifiedRegimeState | None",
) -> MCScore | None:
    """Run MC for a single ticker, return MCScore or None."""
    from lox.llm.scenarios.quant_scenarios import compute_quant_scenarios

    if ticker not in px_df.columns:
        return None

    series = px_df[ticker].dropna()
    if len(series) < 60:
        return None

    # Convert to FMP-style list[dict] (oldest first)
    hist_prices = [
        {"date": str(d.date()), "close": float(v)}
        for d, v in zip(series.index, series.values)
        if np.isfinite(v) and v > 0
    ]
    if len(hist_prices) < 30:
        return None

    current_price = hist_prices[-1]["close"]

    try:
        result = compute_quant_scenarios(
            historical_prices=hist_prices,
            regime_state=regime_state,
            current_price=current_price,
            horizon_days=HORIZON_DAYS,
            n_sims=N_SIMS,
        )
    except Exception as e:
        logger.debug("MC failed for %s: %s", ticker, e)
        return None

    dist = result.get("full_distribution", {})
    p5 = dist.get("p5", current_price)
    p25 = dist.get("p25", current_price)
    p50 = dist.get("p50", current_price)
    p75 = dist.get("p75", current_price)

    median_ret = (p50 - current_price) / current_price
    p5_ret = (p5 - current_price) / current_price
    p25_ret = (p25 - current_price) / current_price
    p75_ret = (p75 - current_price) / current_price

    upside = abs(p75_ret) if p75_ret != 0 else 1e-9
    downside = abs(p25_ret) if p25_ret != 0 else 1e-9
    upside_ratio = upside / downside

    return MCScore(
        ticker=ticker,
        current_price=current_price,
        median_return=median_ret,
        p5_return=p5_ret,
        p25_return=p25_ret,
        p75_return=p75_ret,
        upside_ratio=upside_ratio,
        realized_vol=result.get("realized_vol_annual", 0.0),
    )

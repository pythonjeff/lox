"""
Regime-conditional quantitative scenario generation.

Uses block-bootstrap simulation on historical returns, conditioned on
the current macro regime state, to produce probability-weighted price
targets for BULL / BASE / BEAR / TAIL RISK scenarios.

The LLM then fills in narrative triggers only — it no longer invents
price targets.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from lox.regimes.features import UnifiedRegimeState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Block length for bootstrap resampling (preserves short-term autocorrelation)
BLOCK_LEN = 5  # trading days

# Base jump probability per day (scaled by regime adjustments)
BASE_JUMP_PROB = 0.02  # ~2% chance of a jump on any given day

# Jump size as multiple of daily vol
JUMP_SIZE_SIGMA = 3.0


def compute_multi_horizon(
    historical_prices: list[dict],
    regime_state: "UnifiedRegimeState | None" = None,
    current_price: float | None = None,
    horizons: list[int] | None = None,
    n_sims: int = 5_000,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """
    Run MC at multiple horizons and return a list of summary dicts.

    Default horizons: 21 (1M), 63 (3M), 126 (6M) trading days.
    Uses reduced sim count per horizon for speed.

    Returns list of:
        {"horizon_days": int, "horizon_label": str, "expected_return": float,
         "var_95": float, "cvar_95": float, "prob_loss": float,
         "max_drawdown_median": float, "sortino": float,
         "p25": float, "p50": float, "p75": float}
    """
    if horizons is None:
        horizons = [21, 63, 126]

    labels = {21: "1M", 42: "2M", 63: "3M", 126: "6M", 252: "1Y"}

    results = []
    for h in horizons:
        try:
            res = compute_quant_scenarios(
                historical_prices=historical_prices,
                regime_state=regime_state,
                current_price=current_price,
                horizon_days=h,
                n_sims=n_sims,
                seed=seed,
            )
            rm = res.get("risk_metrics", {})
            dist = res.get("full_distribution", {})
            cp = res.get("current_price", current_price or 0)

            results.append({
                "horizon_days": h,
                "horizon_label": labels.get(h, f"{h}d"),
                "expected_return": rm.get("expected_return", 0),
                "var_95": rm.get("var_95", 0),
                "cvar_95": rm.get("cvar_95", 0),
                "prob_loss": rm.get("prob_loss", 0),
                "max_drawdown_median": rm.get("max_drawdown_median", 0),
                "sortino": rm.get("sortino", 0),
                "p25": round((dist.get("p25", cp) / cp - 1) * 100, 1) if cp else 0,
                "p50": round((dist.get("p50", cp) / cp - 1) * 100, 1) if cp else 0,
                "p75": round((dist.get("p75", cp) / cp - 1) * 100, 1) if cp else 0,
            })
        except Exception as exc:
            logger.warning("Multi-horizon MC failed for %dd: %s", h, exc)
            continue

    return results


def compute_quant_scenarios(
    historical_prices: list[dict],
    regime_state: "UnifiedRegimeState | None" = None,
    current_price: float | None = None,
    horizon_days: int = 63,
    n_sims: int = 10_000,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Compute regime-conditional scenario targets via block-bootstrap simulation.

    Parameters
    ----------
    historical_prices : list[dict]
        FMP-style historical price dicts, each with at least a ``"close"`` key.
        Expected in reverse-chronological order (most recent first) as returned
        by the FMP API, but the function handles both orderings.
    regime_state : UnifiedRegimeState | None
        If provided, drift / vol / jump parameters are adjusted to reflect the
        current macro regime.
    current_price : float | None
        Spot price to anchor simulations.  Falls back to latest close.
    horizon_days : int
        Simulation horizon in trading days (default 63 ≈ 3 months).
    n_sims : int
        Number of Monte-Carlo paths (default 10 000).
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    dict
        {
            "method": str,
            "horizon": str,
            "n_simulations": int,
            "current_price": float,
            "realized_vol_annual": float,
            "regime_adjustments": dict | None,
            "scenarios": {
                "BULL":      {"target": float, "range": [lo, hi], "probability": int},
                "BASE":      {"target": float, "range": [lo, hi], "probability": int},
                "BEAR":      {"target": float, "range": [lo, hi], "probability": int},
                "TAIL_RISK": {"target": float, "range": [lo, hi], "probability": int},
            },
            "full_distribution": {p5, p10, p25, p50, p75, p90, p95},
        }
    """
    # ------------------------------------------------------------------
    # 1. Parse closes → sorted oldest-first
    # ------------------------------------------------------------------
    closes = _extract_closes(historical_prices)
    if len(closes) < 30:
        raise ValueError(f"Need >= 30 prices for bootstrap, got {len(closes)}")

    if current_price is None:
        current_price = float(closes[-1])

    # ------------------------------------------------------------------
    # 2. Log returns & realised statistics
    # ------------------------------------------------------------------
    log_rets = np.diff(np.log(closes))
    daily_mu = float(np.mean(log_rets))
    daily_sigma = float(np.std(log_rets, ddof=1))
    annual_vol = daily_sigma * np.sqrt(252)

    # ------------------------------------------------------------------
    # 3. Regime adjustments (drift shift, vol scale, jump process)
    # ------------------------------------------------------------------
    regime_adj: dict[str, Any] | None = None
    drift_adj_daily = 0.0
    vol_scale = 1.0
    jump_prob = BASE_JUMP_PROB
    jump_sigma = JUMP_SIZE_SIGMA

    if regime_state is not None:
        try:
            mc = regime_state.to_monte_carlo_params()
            # Annualised drift adj → daily
            drift_adj_daily = mc.get("equity_drift_adj", 0.0) / 252.0
            vol_scale = mc.get("equity_vol_adj", 1.0)
            jump_prob = BASE_JUMP_PROB * mc.get("jump_prob_adj", 1.0)
            jump_sigma = JUMP_SIZE_SIGMA * mc.get("jump_size_adj", 1.0)

            regime_adj = {
                "equity_drift_adj": mc.get("equity_drift_adj", 0.0),
                "equity_vol_adj": mc.get("equity_vol_adj", 1.0),
                "jump_prob_adj": mc.get("jump_prob_adj", 1.0),
                "jump_size_adj": mc.get("jump_size_adj", 1.0),
            }
        except Exception as exc:
            logger.warning("Could not extract regime MC params: %s", exc)

    adjusted_sigma = daily_sigma * vol_scale
    adjusted_mu = daily_mu + drift_adj_daily

    # ------------------------------------------------------------------
    # 4. Block-bootstrap simulation (store path-level stats)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)

    # Demeaned returns (we re-inject adjusted drift below)
    demeaned = log_rets - daily_mu

    n_returns = len(demeaned)
    n_blocks = int(np.ceil(horizon_days / BLOCK_LEN))

    terminal_prices = np.empty(n_sims)
    max_drawdowns = np.empty(n_sims)
    drawdown_days = np.empty(n_sims)        # day of max drawdown trough
    terminal_returns = np.empty(n_sims)     # log return over horizon

    for i in range(n_sims):
        # Resample blocks
        block_starts = rng.integers(0, n_returns - BLOCK_LEN + 1, size=n_blocks)
        path_rets = np.concatenate(
            [demeaned[s : s + BLOCK_LEN] for s in block_starts]
        )[:horizon_days]

        # Rescale to adjusted vol and re-inject adjusted drift
        path_rets = path_rets * (vol_scale) + adjusted_mu

        # Jump process overlay
        jump_mask = rng.random(horizon_days) < jump_prob
        if jump_mask.any():
            jump_signs = rng.choice([-1, 1], size=int(jump_mask.sum()))
            path_rets[jump_mask] += jump_signs * jump_sigma * adjusted_sigma

        # Cumulative log returns along the path
        cum_log_rets = np.cumsum(path_rets)

        # Terminal price & return
        terminal_returns[i] = cum_log_rets[-1]
        terminal_prices[i] = current_price * np.exp(cum_log_rets[-1])

        # Path prices for drawdown calculation
        path_prices = current_price * np.exp(cum_log_rets)
        running_max = np.maximum.accumulate(
            np.concatenate([[current_price], path_prices])
        )
        drawdown_series = (
            np.concatenate([[current_price], path_prices]) - running_max
        ) / running_max
        max_drawdowns[i] = float(np.min(drawdown_series))     # most negative
        drawdown_days[i] = float(np.argmin(drawdown_series))   # day of trough

    # ------------------------------------------------------------------
    # 5. Extract percentiles & risk metrics
    # ------------------------------------------------------------------
    pcts = np.percentile(terminal_prices, [5, 10, 25, 50, 75, 90, 95])
    p5, p10, p25, p50, p75, p90, p95 = (float(v) for v in pcts)

    # Risk metrics from path-level data
    risk_metrics = _compute_risk_metrics(
        terminal_prices=terminal_prices,
        terminal_returns=terminal_returns,
        max_drawdowns=max_drawdowns,
        drawdown_days=drawdown_days,
        current_price=current_price,
        horizon_days=horizon_days,
        daily_mu=adjusted_mu,
    )

    # ------------------------------------------------------------------
    # 6. Map to scenarios with regime-aware probabilities
    # ------------------------------------------------------------------
    scenario_probs = _regime_scenario_probabilities(regime_state)

    scenarios = {
        "BULL": {
            "target": _r2(p75),
            "range": [_r2(p50), _r2(p90)],
            "probability": scenario_probs["BULL"],
        },
        "BASE": {
            "target": _r2(p50),
            "range": [_r2(p25), _r2(p75)],
            "probability": scenario_probs["BASE"],
        },
        "BEAR": {
            "target": _r2(p25),
            "range": [_r2(p10), _r2(p50)],
            "probability": scenario_probs["BEAR"],
        },
        "TAIL_RISK": {
            "target": _r2(p5),
            "range": [_r2(p5), _r2(p10)],
            "probability": scenario_probs["TAIL_RISK"],
        },
    }

    # Histogram of terminal returns
    pct_returns = (terminal_prices - current_price) / current_price * 100
    histogram = _compute_return_histogram(pct_returns)

    return {
        "method": "regime_conditional_bootstrap",
        "horizon": f"3M ({horizon_days} trading days)",
        "n_simulations": n_sims,
        "current_price": _r2(current_price),
        "realized_vol_annual": round(annual_vol * 100, 1),
        "regime_adjustments": regime_adj,
        "scenarios": scenarios,
        "full_distribution": {
            "p5": _r2(p5),
            "p10": _r2(p10),
            "p25": _r2(p25),
            "p50": _r2(p50),
            "p75": _r2(p75),
            "p90": _r2(p90),
            "p95": _r2(p95),
        },
        "risk_metrics": risk_metrics,
        "histogram": histogram,
    }


# ======================================================================
# Helpers
# ======================================================================

def _compute_risk_metrics(
    *,
    terminal_prices: np.ndarray,
    terminal_returns: np.ndarray,
    max_drawdowns: np.ndarray,
    drawdown_days: np.ndarray,
    current_price: float,
    horizon_days: int,
    daily_mu: float,
) -> dict[str, Any]:
    """
    Compute path-level risk metrics from MC simulation results.

    Returns dict with VaR, CVaR, drawdown stats, probability-of-loss
    thresholds, and Sortino ratio estimate.
    """
    # Terminal percentage returns
    pct_returns = (terminal_prices - current_price) / current_price

    # VaR / CVaR (Expected Shortfall) at 95% confidence
    var_95 = float(np.percentile(pct_returns, 5))
    tail_mask = pct_returns <= var_95
    cvar_95 = float(np.mean(pct_returns[tail_mask])) if tail_mask.any() else var_95

    # Max drawdown distribution
    dd_p50 = float(np.percentile(max_drawdowns, 50))
    dd_p95 = float(np.percentile(max_drawdowns, 5))   # 5th pctl of neg values = worst
    dd_mean = float(np.mean(max_drawdowns))
    dd_day_median = float(np.median(drawdown_days))

    # Probability of loss thresholds
    prob_loss = float(np.mean(pct_returns < 0))
    prob_loss_5 = float(np.mean(pct_returns < -0.05))
    prob_loss_10 = float(np.mean(pct_returns < -0.10))
    prob_loss_20 = float(np.mean(pct_returns < -0.20))

    # Probability of gain thresholds
    prob_gain_5 = float(np.mean(pct_returns > 0.05))
    prob_gain_10 = float(np.mean(pct_returns > 0.10))

    # Sortino ratio estimate (annualised)
    # target return = 0 (MAR), downside deviation from negative returns only
    neg_rets = pct_returns[pct_returns < 0]
    if len(neg_rets) > 0:
        downside_dev = float(np.sqrt(np.mean(neg_rets ** 2)))
        # Annualise: scale horizon return to annual
        annual_factor = 252.0 / horizon_days
        mean_annual_ret = float(np.mean(pct_returns)) * annual_factor
        downside_dev_annual = downside_dev * np.sqrt(annual_factor)
        sortino = mean_annual_ret / downside_dev_annual if downside_dev_annual > 0 else 0.0
    else:
        sortino = 99.0  # all paths positive

    # Expected return & vol
    expected_return = float(np.mean(pct_returns))
    return_std = float(np.std(pct_returns))

    return {
        "var_95": round(var_95 * 100, 2),
        "cvar_95": round(cvar_95 * 100, 2),
        "max_drawdown_median": round(dd_p50 * 100, 2),
        "max_drawdown_p95": round(dd_p95 * 100, 2),
        "max_drawdown_mean": round(dd_mean * 100, 2),
        "drawdown_trough_day_median": round(dd_day_median, 0),
        "prob_loss": round(prob_loss * 100, 1),
        "prob_loss_5pct": round(prob_loss_5 * 100, 1),
        "prob_loss_10pct": round(prob_loss_10 * 100, 1),
        "prob_loss_20pct": round(prob_loss_20 * 100, 1),
        "prob_gain_5pct": round(prob_gain_5 * 100, 1),
        "prob_gain_10pct": round(prob_gain_10 * 100, 1),
        "expected_return": round(expected_return * 100, 2),
        "return_std": round(return_std * 100, 2),
        "sortino": round(sortino, 2),
    }


def _compute_return_histogram(
    pct_returns: np.ndarray,
    n_bins: int = 25,
) -> dict[str, Any]:
    """
    Compute histogram of percentage returns for display.

    Returns dict with bin edges, counts, and marked percentiles.
    """
    # Clip extremes for cleaner display (keep 1st-99th percentile range)
    lo = float(np.percentile(pct_returns, 1))
    hi = float(np.percentile(pct_returns, 99))
    # Ensure range is at least +-1%
    if hi - lo < 2:
        mid = (hi + lo) / 2
        lo, hi = mid - 1, mid + 1

    counts, edges = np.histogram(pct_returns, bins=n_bins, range=(lo, hi))
    bin_centers = [(float(edges[i]) + float(edges[i + 1])) / 2 for i in range(len(counts))]

    median_ret = float(np.median(pct_returns))

    return {
        "bin_centers": [round(c, 1) for c in bin_centers],
        "counts": [int(c) for c in counts],
        "bin_width": round(float(edges[1] - edges[0]), 2),
        "median_return": round(median_ret, 2),
        "range": [round(lo, 1), round(hi, 1)],
    }


def _r2(v: float) -> float:
    """Round to 2 decimal places."""
    return round(v, 2)


def _extract_closes(historical: list[dict]) -> np.ndarray:
    """
    Pull close prices from FMP historical data and return as oldest-first
    numpy array, filtering out any nulls/zeroes.
    """
    raw = [(d.get("date", ""), d.get("close") or d.get("adjClose")) for d in historical]
    # Drop entries with no close price
    raw = [(dt, c) for dt, c in raw if c is not None and c > 0]
    if not raw:
        raise ValueError("No valid close prices in historical data")

    # Sort oldest-first (FMP returns newest-first)
    raw.sort(key=lambda x: x[0])
    return np.array([c for _, c in raw], dtype=np.float64)


def _regime_scenario_probabilities(
    regime_state: "UnifiedRegimeState | None",
) -> dict[str, int]:
    """
    Derive scenario probability weights from the current regime category.

    Uses the overall_category from UnifiedRegimeState to tilt probabilities:
    - risk_on  → higher BULL weight
    - risk_off → higher BEAR / TAIL weight
    - cautious → balanced
    """
    # Defaults (cautious / no regime)
    default = {"BULL": 20, "BASE": 45, "BEAR": 25, "TAIL_RISK": 10}

    if regime_state is None:
        return default

    cat = getattr(regime_state, "overall_category", "cautious")

    if cat == "risk_on":
        return {"BULL": 30, "BASE": 45, "BEAR": 18, "TAIL_RISK": 7}
    elif cat == "risk_off":
        return {"BULL": 10, "BASE": 35, "BEAR": 35, "TAIL_RISK": 20}
    else:
        return default

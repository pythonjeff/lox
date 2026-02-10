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
    # 4. Block-bootstrap simulation
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)

    # Demeaned returns (we re-inject adjusted drift below)
    demeaned = log_rets - daily_mu

    n_returns = len(demeaned)
    n_blocks = int(np.ceil(horizon_days / BLOCK_LEN))

    terminal_prices = np.empty(n_sims)

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

        # Terminal price
        cum_return = float(np.sum(path_rets))
        terminal_prices[i] = current_price * np.exp(cum_return)

    # ------------------------------------------------------------------
    # 5. Extract percentiles
    # ------------------------------------------------------------------
    pcts = np.percentile(terminal_prices, [5, 10, 25, 50, 75, 90, 95])
    p5, p10, p25, p50, p75, p90, p95 = (float(v) for v in pcts)

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
    }


# ======================================================================
# Helpers
# ======================================================================

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

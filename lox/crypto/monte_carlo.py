"""Crypto perps Monte Carlo scenario simulator.

Simulates forward price paths for crypto assets under different macro regime
assumptions.  V1 uses empirical benchmarks for BTC/ETH/SOL; these can be
refined with real historical data in v2.

Key design decisions (v1 — basic benchmarks):
  - Daily returns are modeled as drift + diffusion + jump (Merton jump-diffusion)
  - Parameters vary by macro regime category (risk_on / neutral / risk_off)
  - Crypto-specific adjustments: higher vol, fatter tails, leverage-driven jumps
  - Correlation between coins is simplified (single common factor)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── V1 Benchmark Parameters ─────────────────────────────────────────────────
# These are rough empirical calibrations.  Ann. drift, ann. vol, jump params.
# "Risk-on" = crypto bull (Goldilocks, low rates, ample liquidity)
# "Neutral" = mixed macro (current-ish conditions)
# "Risk-off" = crypto bear (tightening, credit stress, risk unwind)

REGIME_PARAMS: dict[str, dict[str, dict[str, float]]] = {
    # jump_freq = expected number of jump events per YEAR (converted to daily in engine)
    # jump_mean / jump_std = log-return size of each jump event
    "risk_on": {
        "BTC": {"drift": 0.80, "vol": 0.55, "jump_freq": 3.0, "jump_mean": -0.06, "jump_std": 0.06},
        "ETH": {"drift": 1.00, "vol": 0.70, "jump_freq": 4.0, "jump_mean": -0.08, "jump_std": 0.08},
        "SOL": {"drift": 1.20, "vol": 0.90, "jump_freq": 5.0, "jump_mean": -0.10, "jump_std": 0.10},
        "_default": {"drift": 0.90, "vol": 0.75, "jump_freq": 4.0, "jump_mean": -0.08, "jump_std": 0.08},
    },
    "neutral": {
        "BTC": {"drift": 0.15, "vol": 0.65, "jump_freq": 5.0, "jump_mean": -0.08, "jump_std": 0.07},
        "ETH": {"drift": 0.10, "vol": 0.80, "jump_freq": 6.0, "jump_mean": -0.10, "jump_std": 0.09},
        "SOL": {"drift": 0.05, "vol": 1.00, "jump_freq": 8.0, "jump_mean": -0.12, "jump_std": 0.11},
        "_default": {"drift": 0.10, "vol": 0.80, "jump_freq": 6.0, "jump_mean": -0.10, "jump_std": 0.09},
    },
    "risk_off": {
        "BTC": {"drift": -0.30, "vol": 0.85, "jump_freq": 10.0, "jump_mean": -0.12, "jump_std": 0.10},
        "ETH": {"drift": -0.50, "vol": 1.00, "jump_freq": 12.0, "jump_mean": -0.15, "jump_std": 0.12},
        "SOL": {"drift": -0.70, "vol": 1.30, "jump_freq": 15.0, "jump_mean": -0.18, "jump_std": 0.14},
        "_default": {"drift": -0.50, "vol": 1.00, "jump_freq": 12.0, "jump_mean": -0.15, "jump_std": 0.12},
    },
}

# Cross-coin correlation (simplified common-factor model)
COMMON_FACTOR_LOADING = {
    "BTC": 0.80,   # BTC is the common factor itself
    "ETH": 0.85,   # ETH highly correlated to BTC
    "SOL": 0.75,   # SOL somewhat less
    "_default": 0.70,
}


@dataclass
class CryptoMCResult:
    """Results of a single-coin Monte Carlo simulation."""
    coin: str
    regime: str
    current_price: float
    horizon_days: int
    n_paths: int

    # Summary statistics (at terminal date)
    mean_return: float
    median_return: float
    vol_ann: float
    var_95: float               # 5th percentile return (loss)
    cvar_95: float              # conditional VaR (expected shortfall)
    prob_positive: float        # probability of positive return
    prob_2x: float              # probability of doubling
    prob_half: float            # probability of halving

    # Percentile returns
    pct_5: float
    pct_25: float
    pct_50: float
    pct_75: float
    pct_95: float

    # Price levels
    price_5: float
    price_25: float
    price_50: float
    price_75: float
    price_95: float

    # Max drawdown stats
    avg_max_drawdown: float
    worst_max_drawdown: float

    # Parameters used
    params: dict[str, float] = field(default_factory=dict)


def run_crypto_mc(
    coin: str,
    current_price: float,
    regime: str = "neutral",
    horizon_days: int = 90,
    n_paths: int = 5_000,
    leverage: float = 1.0,
    seed: int | None = None,
    param_overrides: dict[str, float] | None = None,
) -> CryptoMCResult:
    """Run Monte Carlo simulation for a single crypto asset.

    Parameters
    ----------
    coin : str
        Coin ticker (BTC, ETH, SOL, etc.)
    current_price : float
        Current spot/mark price.
    regime : str
        Macro regime category: "risk_on", "neutral", "risk_off".
    horizon_days : int
        Simulation horizon in calendar days.
    n_paths : int
        Number of simulation paths.
    leverage : float
        Leverage multiplier (1.0 = spot, 5.0 = 5x perp).
    seed : int | None
        Random seed for reproducibility.
    param_overrides : dict | None
        Override any parameter (drift, vol, jump_prob, jump_mean, jump_std).
    """
    rng = np.random.default_rng(seed)

    # Get parameters
    regime_key = regime if regime in REGIME_PARAMS else "neutral"
    coin_key = coin.upper()
    params = REGIME_PARAMS[regime_key].get(coin_key, REGIME_PARAMS[regime_key]["_default"]).copy()
    if param_overrides:
        params.update(param_overrides)

    # Convert annual params to daily
    dt = 1.0 / 365.0
    daily_drift = params["drift"] * dt
    daily_vol = params["vol"] * np.sqrt(dt)
    # jump_freq = expected jumps per year → daily probability
    daily_jump_prob = params["jump_freq"] / 365.0

    # Apply leverage to drift and vol
    lev_drift = daily_drift * leverage
    lev_vol = daily_vol * leverage

    # Simulate paths using log returns
    # r_t = drift + vol * Z + jump_indicator * jump_size
    trading_days = int(horizon_days * 5 / 7)  # crypto trades 24/7, but use calendar days
    trading_days = max(horizon_days, 1)  # actually crypto trades every day

    Z = rng.standard_normal((n_paths, trading_days))
    jump_indicator = rng.binomial(1, daily_jump_prob, (n_paths, trading_days))
    jump_sizes = rng.normal(params["jump_mean"], params["jump_std"], (n_paths, trading_days))

    # Log returns per day
    log_returns = (lev_drift - 0.5 * lev_vol**2) + lev_vol * Z + jump_indicator * jump_sizes * leverage

    # Cumulative log returns → price paths
    cum_log_returns = np.cumsum(log_returns, axis=1)
    price_paths = current_price * np.exp(cum_log_returns)

    # Terminal returns (percentage)
    terminal_prices = price_paths[:, -1]
    terminal_returns = (terminal_prices / current_price - 1) * 100  # percentage

    # Max drawdown per path
    running_max = np.maximum.accumulate(price_paths, axis=1)
    drawdowns = (price_paths - running_max) / running_max
    max_drawdowns = np.min(drawdowns, axis=1) * 100  # percentage

    # Statistics
    mean_ret = float(np.mean(terminal_returns))
    median_ret = float(np.median(terminal_returns))
    var_95 = float(np.percentile(terminal_returns, 5))
    tail = terminal_returns[terminal_returns <= np.percentile(terminal_returns, 5)]
    cvar_95 = float(np.mean(tail)) if len(tail) > 0 else var_95

    pcts = [5, 25, 50, 75, 95]
    ret_pcts = {p: float(np.percentile(terminal_returns, p)) for p in pcts}
    price_pcts = {p: float(np.percentile(terminal_prices, p)) for p in pcts}

    return CryptoMCResult(
        coin=coin_key,
        regime=regime_key,
        current_price=current_price,
        horizon_days=horizon_days,
        n_paths=n_paths,
        mean_return=mean_ret,
        median_return=median_ret,
        vol_ann=params["vol"] * leverage,
        var_95=var_95,
        cvar_95=cvar_95,
        prob_positive=float(np.mean(terminal_returns > 0)) * 100,
        prob_2x=float(np.mean(terminal_returns > 100)) * 100,
        prob_half=float(np.mean(terminal_returns < -50)) * 100,
        pct_5=ret_pcts[5],
        pct_25=ret_pcts[25],
        pct_50=ret_pcts[50],
        pct_75=ret_pcts[75],
        pct_95=ret_pcts[95],
        price_5=price_pcts[5],
        price_25=price_pcts[25],
        price_50=price_pcts[50],
        price_75=price_pcts[75],
        price_95=price_pcts[95],
        avg_max_drawdown=float(np.mean(max_drawdowns)),
        worst_max_drawdown=float(np.min(max_drawdowns)),
        params=params,
    )


def run_multi_regime_mc(
    coin: str,
    current_price: float,
    horizon_days: int = 90,
    n_paths: int = 5_000,
    leverage: float = 1.0,
    seed: int | None = None,
) -> dict[str, CryptoMCResult]:
    """Run MC across all three regime scenarios for comparison."""
    results = {}
    for regime in ["risk_on", "neutral", "risk_off"]:
        results[regime] = run_crypto_mc(
            coin=coin,
            current_price=current_price,
            regime=regime,
            horizon_days=horizon_days,
            n_paths=n_paths,
            leverage=leverage,
            seed=seed,
        )
    return results

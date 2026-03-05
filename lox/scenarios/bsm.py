"""
Black-Scholes option pricing for scenario P&L estimation.

Provides scalar and vectorized (numpy) European put/call pricing.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


# ── Scalar versions (single price) ──────────────────────────────────────


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European call price via Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European put price via Black-Scholes (put-call parity)."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def option_price(
    opt_type: str, S: float, K: float, T: float, r: float, sigma: float,
) -> float:
    """Price a put or call. opt_type should be 'put' or 'call'."""
    if opt_type == "put":
        return bs_put_price(S, K, T, r, sigma)
    return bs_call_price(S, K, T, r, sigma)


# ── Vectorized versions (numpy arrays) ──────────────────────────────────
# S and T can be arrays of any shape; K, r, sigma are scalars.
# Returns array of the same shape as S (or T, whichever is array).


def bs_call_price_vec(
    S: np.ndarray, K: float, T: np.ndarray, r: float, sigma: float,
) -> np.ndarray:
    """Vectorized European call price. S and T are numpy arrays."""
    safe_T = np.maximum(T, 1e-10)
    sqrt_T = np.sqrt(safe_T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * safe_T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    price = S * norm.cdf(d1) - K * np.exp(-r * safe_T) * norm.cdf(d2)
    # Where T <= 0, use intrinsic value
    expired = T <= 0
    if np.any(expired):
        price = np.where(expired, np.maximum(S - K, 0.0), price)
    return price


def bs_put_price_vec(
    S: np.ndarray, K: float, T: np.ndarray, r: float, sigma: float,
) -> np.ndarray:
    """Vectorized European put price. S and T are numpy arrays."""
    safe_T = np.maximum(T, 1e-10)
    sqrt_T = np.sqrt(safe_T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * safe_T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    price = K * np.exp(-r * safe_T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    expired = T <= 0
    if np.any(expired):
        price = np.where(expired, np.maximum(K - S, 0.0), price)
    return price


def option_price_vec(
    opt_type: str, S: np.ndarray, K: float, T: np.ndarray,
    r: float, sigma: float,
) -> np.ndarray:
    """Vectorized option pricing. opt_type is 'put' or 'call'."""
    if opt_type == "put":
        return bs_put_price_vec(S, K, T, r, sigma)
    return bs_call_price_vec(S, K, T, r, sigma)

"""
yfinance options chain adapter — free, no API key required.

Fetches OI, IV, volume, bid/ask from Yahoo Finance across all expiries and
computes gamma analytically via Black-Scholes (requires scipy).

Used as the Polygon fallback in lox.positioning.data._fetch_options_chain so
that GEX, PCR, and skew are populated even without a Polygon/Massive key.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, timezone

logger = logging.getLogger(__name__)

# Risk-free rate used for Black-Scholes gamma (3-month T-Bill proxy, updated rarely)
_RF_RATE = 0.043


def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma (same for calls and puts)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        from scipy.stats import norm
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return float(norm.pdf(d1) / (S * sigma * math.sqrt(T)))
    except Exception:
        return 0.0


def _bs_delta(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> float:
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        from scipy.stats import norm
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        if opt_type == "call":
            return float(norm.cdf(d1))
        else:
            return float(norm.cdf(d1) - 1)
    except Exception:
        return 0.0


def _safe_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _safe_int(v, default: int = 0) -> int:
    try:
        f = float(v)
        return default if math.isnan(f) else int(f)
    except (TypeError, ValueError):
        return default


def fetch_options_chain_yfinance(
    ticker: str,
    *,
    max_expiries: int = 6,
    rf_rate: float = _RF_RATE,
) -> list:
    """
    Fetch SPY (or any ticker) options chain from Yahoo Finance and return a
    list of OptionCandidate objects ready for compute_gex_from_chain / compute_pcr_from_chain.

    Greeks are computed analytically from the IV Yahoo provides, so gamma
    values are accurate enough for GEX and regime classification.

    Args:
        ticker: Underlying ticker (e.g. "SPY").
        max_expiries: Number of near-term expiries to include (caps API calls).
        rf_rate: Risk-free rate for Black-Scholes (default ~4.3%).

    Returns:
        List of OptionCandidate, empty on failure.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — run: pip install yfinance")
        return []

    from lox.data.alpaca import OptionCandidate

    try:
        ticker_obj = yf.Ticker(ticker)
        spot = float(ticker_obj.fast_info.last_price or 0)
        if spot <= 0:
            logger.warning(f"[yfinance] Could not get spot price for {ticker}")
            return []

        expiries = ticker_obj.options
        if not expiries:
            logger.warning(f"[yfinance] No expiries for {ticker}")
            return []

        today = date.today()
        candidates: list[OptionCandidate] = []

        for expiry_str in expiries[:max_expiries]:
            try:
                chain = ticker_obj.option_chain(expiry_str)
            except Exception as e:
                logger.debug(f"[yfinance] Failed to fetch {ticker} chain for {expiry_str}: {e}")
                continue

            try:
                expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            dte = max((expiry_dt - today).days, 0)
            T = dte / 365.0  # time to expiry in years

            for side_df, opt_type in ((chain.calls, "call"), (chain.puts, "put")):
                for _, row in side_df.iterrows():
                    strike = float(row.get("strike") or 0)
                    if strike <= 0:
                        continue

                    iv = _safe_float(row.get("impliedVolatility"))
                    oi = _safe_int(row.get("openInterest"))
                    volume = _safe_int(row.get("volume"))
                    bid = _safe_float(row.get("bid")) or None
                    ask = _safe_float(row.get("ask")) or None
                    last = _safe_float(row.get("lastPrice")) or None
                    symbol = str(row.get("contractSymbol") or "")

                    # Compute greeks analytically from IV
                    gamma = _bs_gamma(spot, strike, T, rf_rate, iv) if iv > 0 else None
                    delta = _bs_delta(spot, strike, T, rf_rate, iv, opt_type) if iv > 0 else None

                    candidates.append(OptionCandidate(
                        symbol=symbol,
                        opt_type=opt_type,
                        expiry=expiry_dt,
                        strike=strike,
                        dte_days=dte,
                        delta=delta,
                        gamma=gamma,
                        theta=None,
                        vega=None,
                        iv=iv if iv > 0 else None,
                        oi=oi if oi > 0 else None,
                        volume=volume if volume > 0 else None,
                        bid=bid,
                        ask=ask,
                        last=last,
                    ))

        logger.info(f"[yfinance] {ticker}: {len(candidates)} contracts across {min(max_expiries, len(expiries))} expiries")
        return candidates

    except Exception as e:
        logger.warning(f"[yfinance] Chain fetch failed for {ticker}: {e}")
        return []

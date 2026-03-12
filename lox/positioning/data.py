"""
Positioning & flow data layer.

Fetches and computes raw inputs for the positioning regime classifier.
No classification logic lives here — just data retrieval and normalization.

Data sources (all existing APIs, zero new keys):
  - FMP /api/v4/commitment_of_traders_report: CFTC COT net speculative
  - Alpaca options chain: GEX, put/call ratio, IV skew
  - FMP /api/v4/short-interest: short interest as % float
  - FRED VIXCLS + VIX3M: VIX term structure (via volatility signals)
  - Trading Economics / FMP: AAII sentiment

Author: Lox Capital Research
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from lox.config import Settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# COT asset mapping — CFTC symbol names used by FMP
# ─────────────────────────────────────────────────────────────────────────────

COT_SYMBOLS: dict[str, str] = {
    "ES": "E-MINI S&P 500",
    "NQ": "NASDAQ-100 STOCK INDEX",
    "ZN": "10-YEAR U.S. TREASURY NOTES",
    "GC": "GOLD",
    "CL": "CRUDE OIL, LIGHT SWEET",
}

# Alternate names FMP may use — fallback matching
COT_SYMBOL_ALTS: dict[str, list[str]] = {
    "ES": ["S&P 500", "E-MINI S&P", "SP 500"],
    "NQ": ["NASDAQ", "E-MINI NASDAQ"],
    "ZN": ["10-YEAR", "T-NOTE", "10 YEAR"],
    "GC": ["GOLD"],
    "CL": ["CRUDE OIL", "WTI"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PositioningInputs:
    """Raw inputs for the positioning regime classifier."""

    # VIX term structure
    vix_level: float | None = None
    vix3m_level: float | None = None
    vix_term_slope: float | None = None         # VIX3M / VIX (>1 = contango, <1 = backwardation)

    # Options flow
    put_call_ratio: float | None = None          # OI-weighted equity P/C
    skew_25d: float | None = None                # 25-delta risk reversal (put IV - call IV)

    # Sentiment
    aaii_bull_pct: float | None = None            # AAII % bullish

    # CFTC COT
    cot_net_spec: dict[str, float] | None = None   # net speculative by asset key
    cot_z_score: dict[str, float] | None = None    # z-score of net spec
    cot_dates: dict[str, str] | None = None        # report date per asset

    # Dealer Gamma Exposure
    gex_total: float | None = None               # aggregate dealer GEX ($bn notional)
    gex_flip_level: float | None = None          # price where GEX flips negative
    gex_spot: float | None = None                # current spot price (for context)

    # Short Interest
    short_interest_pct: dict[str, float] | None = None  # SI as % float by ticker

    # Metadata
    ticker: str = "SPY"
    asof: str = ""
    error: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# CFTC COT data via FMP
# ─────────────────────────────────────────────────────────────────────────────

def fetch_cot_data(
    settings: Settings,
    symbols: list[str] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    """Fetch CFTC Commitment of Traders net speculative positioning from FMP.

    Returns:
        (net_spec, z_scores, dates) — dicts keyed by asset code (ES, NQ, etc.)
    """
    if symbols is None:
        symbols = list(COT_SYMBOLS.keys())

    if not settings.fmp_api_key:
        return {}, {}, {}

    from lox.altdata.cache import cache_path, read_cache, write_cache
    import requests

    cache_key = "positioning_cot_data"
    p = cache_path(cache_key)
    cached = read_cache(p, max_age=timedelta(hours=24))
    if isinstance(cached, dict) and "net_spec" in cached:
        return (
            cached.get("net_spec", {}),
            cached.get("z_scores", {}),
            cached.get("dates", {}),
        )

    net_spec: dict[str, float] = {}
    z_scores: dict[str, float] = {}
    dates: dict[str, str] = {}

    try:
        # Fetch recent COT reports (last ~52 weeks for z-score calc)
        url = f"https://financialmodelingprep.com/api/v4/commitment_of_traders_report"
        resp = requests.get(
            url,
            params={"apikey": settings.fmp_api_key},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list) or not data:
            return {}, {}, {}

        # Group by market name
        for sym_code, fmp_name in COT_SYMBOLS.items():
            if sym_code not in symbols:
                continue

            alts = [fmp_name.lower()] + [a.lower() for a in COT_SYMBOL_ALTS.get(sym_code, [])]

            # Filter rows matching this asset
            rows = []
            for row in data:
                if not isinstance(row, dict):
                    continue
                market = str(row.get("short_name") or row.get("market_and_exchange_names") or "").lower()
                if any(alt in market for alt in alts):
                    rows.append(row)

            if not rows:
                continue

            # Sort by date descending
            rows.sort(key=lambda r: str(r.get("date") or r.get("report_date") or ""), reverse=True)

            # Latest report
            latest = rows[0]
            report_date = str(latest.get("date") or latest.get("report_date") or "")

            # Net speculative = noncommercial_long - noncommercial_short
            nc_long = _safe_float(latest.get("noncomm_positions_long_all") or latest.get("noncommercial_long"))
            nc_short = _safe_float(latest.get("noncomm_positions_short_all") or latest.get("noncommercial_short"))

            if nc_long is not None and nc_short is not None:
                net = nc_long - nc_short
                net_spec[sym_code] = net
                dates[sym_code] = report_date[:10]

                # Z-score from trailing ~52 weeks of data
                historical_nets = []
                for r in rows[:52]:
                    nl = _safe_float(r.get("noncomm_positions_long_all") or r.get("noncommercial_long"))
                    ns = _safe_float(r.get("noncomm_positions_short_all") or r.get("noncommercial_short"))
                    if nl is not None and ns is not None:
                        historical_nets.append(nl - ns)

                if len(historical_nets) >= 10:
                    import numpy as np
                    arr = np.array(historical_nets)
                    mean = float(np.mean(arr))
                    std = float(np.std(arr, ddof=1))
                    if std > 0:
                        z_scores[sym_code] = (net - mean) / std

        # Cache results
        if net_spec:
            write_cache(p, {
                "net_spec": net_spec,
                "z_scores": z_scores,
                "dates": dates,
            })

    except Exception as e:
        logger.warning(f"Failed to fetch COT data: {e}")

    return net_spec, z_scores, dates


# ─────────────────────────────────────────────────────────────────────────────
# GEX computation from options chain
# ─────────────────────────────────────────────────────────────────────────────

def compute_gex_from_chain(
    candidates: list[Any],
    spot: float,
) -> tuple[float | None, float | None]:
    """Compute dealer Gamma Exposure (GEX) from options chain.

    GEX per contract = OI × gamma × spot × 100 × direction
    Direction: +1 for calls (dealers long gamma), -1 for puts (dealers short gamma)

    Returns:
        (gex_total_bn, gex_flip_level)
        gex_total_bn: total GEX in $billions notional
        gex_flip_level: price where cumulative GEX crosses zero
    """
    if not candidates or spot <= 0:
        return None, None

    gex_by_strike: dict[float, float] = {}
    total_gex = 0.0

    for c in candidates:
        # Prefer OI; fall back to volume if OI unavailable
        oi = getattr(c, "oi", None) or 0
        if oi <= 0:
            oi = getattr(c, "volume", None) or 0
        gamma = getattr(c, "gamma", None)
        strike = getattr(c, "strike", None) or 0
        opt_type = getattr(c, "opt_type", "")
        delta = getattr(c, "delta", None)

        if oi <= 0 or gamma is None or gamma <= 0:
            continue

        # Infer type from delta if opt_type not set
        if not opt_type and delta is not None:
            opt_type = "put" if delta < 0 else "call"

        # Dealers are short options → they're long gamma on calls, short on puts
        # (Standard assumption: retail buys, dealers sell)
        direction = 1.0 if opt_type.lower() in ("call", "c") else -1.0

        contract_gex = oi * gamma * spot * 100 * direction
        total_gex += contract_gex

        if strike > 0:
            gex_by_strike[strike] = gex_by_strike.get(strike, 0.0) + contract_gex

    if total_gex == 0.0 and not gex_by_strike:
        return None, None

    # Convert to $billions
    gex_total_bn = total_gex / 1e9

    # Find GEX flip level: where cumulative GEX crosses zero
    flip_level = _find_gex_flip(gex_by_strike, spot)

    return gex_total_bn, flip_level


def _find_gex_flip(gex_by_strike: dict[float, float], spot: float) -> float | None:
    """Find the strike price where cumulative GEX flips from positive to negative."""
    if not gex_by_strike:
        return None

    sorted_strikes = sorted(gex_by_strike.keys())
    cum_gex = 0.0
    last_positive_strike = None

    for strike in sorted_strikes:
        cum_gex += gex_by_strike[strike]
        if cum_gex > 0:
            last_positive_strike = strike
        elif cum_gex <= 0 and last_positive_strike is not None:
            return strike

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Put/Call ratio from options chain
# ─────────────────────────────────────────────────────────────────────────────

def compute_pcr_from_chain(candidates: list[Any]) -> float | None:
    """Compute OI-weighted put/call ratio from options chain."""
    if not candidates:
        return None

    put_oi = 0
    call_oi = 0

    for c in candidates:
        # Prefer OI; fall back to volume if OI unavailable
        oi = getattr(c, "oi", None) or 0
        if oi <= 0:
            oi = getattr(c, "volume", None) or 0
        opt_type = getattr(c, "opt_type", "")
        delta = getattr(c, "delta", None)

        if oi <= 0:
            continue

        # Infer type from delta if needed
        if not opt_type and delta is not None:
            opt_type = "put" if delta < 0 else "call"

        if opt_type.lower() in ("put", "p"):
            put_oi += oi
        elif opt_type.lower() in ("call", "c"):
            call_oi += oi

    if call_oi == 0:
        return None

    return put_oi / call_oi


# ─────────────────────────────────────────────────────────────────────────────
# IV Skew from options chain
# ─────────────────────────────────────────────────────────────────────────────

def compute_skew_from_chain(
    candidates: list[Any],
    spot: float,
) -> float | None:
    """Compute 25-delta risk reversal: avg(put IV at ~25d) - avg(call IV at ~25d)."""
    if not candidates or spot <= 0:
        return None

    put_ivs: list[float] = []
    call_ivs: list[float] = []

    for c in candidates:
        delta = getattr(c, "delta", None)
        iv = getattr(c, "iv", None)
        if delta is None or iv is None or iv <= 0:
            continue

        abs_delta = abs(delta)
        # Accept deltas in ~15-35% range as "25-delta" neighborhood
        if 0.15 <= abs_delta <= 0.35:
            if delta < 0:
                put_ivs.append(iv)
            else:
                call_ivs.append(iv)

    if not put_ivs or not call_ivs:
        return None

    avg_put_iv = sum(put_ivs) / len(put_ivs)
    avg_call_iv = sum(call_ivs) / len(call_ivs)

    # Risk reversal: positive = puts are richer (downside fear)
    return (avg_put_iv - avg_call_iv) * 100  # in vol points


# ─────────────────────────────────────────────────────────────────────────────
# Short Interest via FMP
# ─────────────────────────────────────────────────────────────────────────────

def fetch_short_interest(
    settings: Settings,
    tickers: list[str],
) -> dict[str, float]:
    """Fetch short interest as % of float from FMP.

    Returns dict: ticker -> SI% (e.g., {"AAPL": 1.2, "TSLA": 3.5})
    """
    if not settings.fmp_api_key or not tickers:
        return {}

    from lox.altdata.cache import cache_path, read_cache, write_cache
    import requests

    clean = [t.strip().upper() for t in tickers if t.strip()]
    cache_key = f"positioning_si_{'_'.join(sorted(clean[:10]))}"
    p = cache_path(cache_key)
    cached = read_cache(p, max_age=timedelta(hours=24))
    if isinstance(cached, dict):
        return cached

    result: dict[str, float] = {}

    for ticker in clean[:10]:  # Cap to avoid rate limits
        try:
            url = f"https://financialmodelingprep.com/api/v4/short-interest"
            resp = requests.get(
                url,
                params={"symbol": ticker, "apikey": settings.fmp_api_key},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, list) and data:
                row = data[0]
                si_pct = _safe_float(row.get("shortInterestPercentOfFloat") or row.get("shortPercentOfFloat"))
                if si_pct is not None:
                    result[ticker] = si_pct
        except Exception as e:
            logger.debug(f"Short interest fetch failed for {ticker}: {e}")

    if result:
        write_cache(p, result)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# VIX Term Structure (from volatility infrastructure)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_vix_term(settings: Settings, refresh: bool = False) -> tuple[float | None, float | None, float | None]:
    """Get VIX level, VIX3M level, and VIX term slope from volatility signals.

    Returns (vix, vix3m, slope) where slope = VIX3M / VIX.
    """
    try:
        from lox.volatility.signals import build_volatility_state

        state = build_volatility_state(settings=settings, refresh=refresh)
        inputs = state.inputs if state else None
        if inputs is None:
            return None, None, None

        vix = inputs.vix
        vix3m = inputs.vix3m

        if vix is not None and vix > 0 and vix3m is not None and vix3m > 0:
            slope = vix3m / vix
            return vix, vix3m, slope

        return vix, vix3m, None
    except Exception as e:
        logger.warning(f"Failed to fetch VIX term structure: {e}")
        return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Options chain fetch (reuses existing Alpaca/Polygon infrastructure)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_options_chain(settings: Settings, ticker: str) -> list[Any]:
    """Fetch options chain for a ticker, returning list of OptionCandidate.

    Prefers Polygon (has OI + opt_type), falls back to Alpaca (has greeks but no OI).
    For Alpaca data, infers opt_type from delta sign.
    """
    # Try Polygon first — it has OI and opt_type which are critical for GEX/PCR
    try:
        from lox.data.polygon import fetch_options_chain_polygon
        candidates = fetch_options_chain_polygon(settings=settings, ticker=ticker, limit=250)
        if candidates:
            has_oi = sum(1 for c in candidates if c.oi is not None and c.oi > 0)
            if has_oi > 10:
                logger.debug(f"Polygon chain for {ticker}: {len(candidates)} contracts, {has_oi} with OI")
                return candidates
    except Exception as e:
        logger.debug(f"Polygon chain fetch failed for {ticker}: {e}")

    # Fallback to Alpaca — has greeks but no OI
    try:
        from lox.data.alpaca import make_clients, fetch_option_chain, to_candidates

        _, data_client = make_clients(settings)
        chain = fetch_option_chain(data_client, ticker)
        candidates = list(to_candidates(chain, ticker))

        # Alpaca doesn't provide opt_type or OI — infer type from delta
        enriched = []
        for c in candidates:
            opt_type = c.opt_type
            if not opt_type and c.delta is not None:
                opt_type = "put" if c.delta < 0 else "call"
            from lox.data.alpaca import OptionCandidate
            enriched.append(OptionCandidate(
                symbol=c.symbol, opt_type=opt_type, expiry=c.expiry,
                strike=c.strike, dte_days=c.dte_days,
                delta=c.delta, gamma=c.gamma, theta=c.theta,
                vega=c.vega, iv=c.iv, oi=c.oi, volume=c.volume,
                bid=c.bid, ask=c.ask, last=c.last,
            ))
        return enriched
    except Exception as e:
        logger.debug(f"Alpaca chain fetch also failed for {ticker}: {e}")

    return []


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def compute_positioning_inputs(
    *,
    settings: Settings,
    ticker: str = "SPY",
    refresh: bool = False,
) -> PositioningInputs:
    """Fetch all positioning regime inputs.

    Returns a PositioningInputs dataclass with all raw data for the classifier.
    Gracefully handles missing data — classifier will skip sub-scores with None.
    """
    asof = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ── VIX Term Structure ─────────────────────────────────────────────────
    vix_level, vix3m_level, vix_term_slope = _fetch_vix_term(settings, refresh=refresh)

    # ── AAII Sentiment ─────────────────────────────────────────────────────
    aaii_bull: float | None = None
    try:
        from lox.altdata.trading_economics import get_aaii_bullish_sentiment
        aaii_bull = get_aaii_bullish_sentiment()
    except Exception as e:
        logger.debug(f"AAII sentiment fetch failed: {e}")

    # ── CFTC COT ──────────────────────────────────────────────────────────
    cot_net, cot_z, cot_dates = fetch_cot_data(settings)

    # ── Options Chain (GEX, P/C, Skew) ────────────────────────────────────
    gex_total: float | None = None
    gex_flip: float | None = None
    gex_spot: float | None = None
    pcr: float | None = None
    skew: float | None = None

    try:
        # Get spot price
        from lox.altdata.fmp import fetch_realtime_quotes
        quotes = fetch_realtime_quotes(settings=settings, tickers=[ticker])
        spot = quotes.get(ticker.upper())

        if spot and spot > 0:
            gex_spot = spot

            # Fetch chain
            chain = _fetch_options_chain(settings, ticker)

            if chain:
                gex_total, gex_flip = compute_gex_from_chain(chain, spot)
                pcr = compute_pcr_from_chain(chain)
                skew = compute_skew_from_chain(chain, spot)
    except Exception as e:
        logger.warning(f"Options chain processing failed: {e}")

    # ── Short Interest ────────────────────────────────────────────────────
    si_pct: dict[str, float] | None = None
    try:
        si_tickers = [ticker]
        # Add portfolio tickers if we can get them
        si_result = fetch_short_interest(settings, si_tickers)
        si_pct = si_result if si_result else None
    except Exception as e:
        logger.debug(f"Short interest fetch failed: {e}")

    return PositioningInputs(
        vix_level=vix_level,
        vix3m_level=vix3m_level,
        vix_term_slope=vix_term_slope,
        put_call_ratio=pcr,
        skew_25d=skew,
        aaii_bull_pct=aaii_bull,
        cot_net_spec=cot_net if cot_net else None,
        cot_z_score=cot_z if cot_z else None,
        cot_dates=cot_dates if cot_dates else None,
        gex_total=gex_total,
        gex_flip_level=gex_flip,
        gex_spot=gex_spot,
        short_interest_pct=si_pct,
        ticker=ticker.upper(),
        asof=asof,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val: Any) -> float | None:
    """Safely convert a value to float, returning None on failure."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

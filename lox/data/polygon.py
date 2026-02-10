"""
Polygon.io / Massive API adapter for options data.

Provides OI, volume, greeks, and IV that Alpaca snapshots lack.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import requests

from lox.config import Settings
from lox.data.alpaca import OptionCandidate

logger = logging.getLogger(__name__)

POLYGON_BASE = "https://api.polygon.io"


def _parse_occ_symbol(occ: str) -> tuple[str, date, str, float]:
    """
    Parse OCC option symbol like 'O:SPY250117C00600000'
    Returns: (underlying, expiry, opt_type, strike)
    """
    # Remove 'O:' prefix if present
    if occ.startswith("O:"):
        occ = occ[2:]
    
    # Format: UNDERLYING + YYMMDD + C/P + STRIKE (8 digits, 3 decimal implied)
    # Find where the date starts (first digit after letters)
    i = 0
    while i < len(occ) and not occ[i].isdigit():
        i += 1
    
    underlying = occ[:i]
    rest = occ[i:]
    
    # Date is 6 digits YYMMDD
    date_str = rest[:6]
    opt_type = rest[6]  # C or P
    strike_str = rest[7:]
    
    # Parse date
    year = 2000 + int(date_str[:2])
    month = int(date_str[2:4])
    day = int(date_str[4:6])
    expiry = date(year, month, day)
    
    # Strike is in 1000ths (e.g., 00600000 = 600.000)
    strike = int(strike_str) / 1000.0
    
    return underlying, expiry, "call" if opt_type == "C" else "put", strike


def fetch_options_chain_polygon(
    settings: Settings,
    ticker: str,
    *,
    contract_type: str | None = None,  # "call", "put", or None for both
    expiration_date_gte: str | None = None,  # YYYY-MM-DD
    expiration_date_lte: str | None = None,
    limit: int = 250,
) -> list[OptionCandidate]:
    """
    Fetch options chain from Polygon with OI, volume, greeks.
    
    Returns list of OptionCandidate objects with full data.
    """
    api_key = settings.massive_api_key
    if not api_key:
        logger.warning("MASSIVE_API_KEY not set - cannot fetch Polygon options data")
        return []
    
    url = f"{POLYGON_BASE}/v3/snapshot/options/{ticker}"
    
    params: dict[str, Any] = {
        "apiKey": api_key,
        "limit": limit,
    }
    
    if contract_type:
        params["contract_type"] = contract_type
    if expiration_date_gte:
        params["expiration_date.gte"] = expiration_date_gte
    if expiration_date_lte:
        params["expiration_date.lte"] = expiration_date_lte
    
    candidates: list[OptionCandidate] = []
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        results = data.get("results", [])
        logger.info(f"Polygon returned {len(results)} options for {ticker}")
        
        for item in results:
            details = item.get("details", {})
            greeks = item.get("greeks", {})
            day = item.get("day", {})
            underlying = item.get("underlying_asset", {})
            
            # Parse symbol
            symbol = details.get("ticker", "")
            contract_type_val = details.get("contract_type", "").lower()
            expiry_str = details.get("expiration_date", "")
            strike = details.get("strike_price", 0.0)
            
            # Parse expiry
            try:
                expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date() if expiry_str else date(1970, 1, 1)
            except ValueError:
                expiry = date(1970, 1, 1)
            
            # Calculate DTE
            today = date.today()
            dte_days = (expiry - today).days if expiry > today else 0
            
            # Extract greeks
            delta = greeks.get("delta")
            gamma = greeks.get("gamma")
            theta = greeks.get("theta")
            vega = greeks.get("vega")
            
            # IV from item level
            iv = item.get("implied_volatility")
            
            # OI and volume from day data
            oi = item.get("open_interest")
            volume = day.get("volume")
            
            # Quotes
            last_quote = item.get("last_quote", {})
            bid = last_quote.get("bid")
            ask = last_quote.get("ask")
            
            last_trade = item.get("last_trade", {})
            last_price = last_trade.get("price")
            
            candidates.append(OptionCandidate(
                symbol=symbol,
                opt_type=contract_type_val or "call",
                expiry=expiry,
                strike=float(strike) if strike else 0.0,
                dte_days=dte_days,
                delta=float(delta) if delta is not None else None,
                gamma=float(gamma) if gamma is not None else None,
                theta=float(theta) if theta is not None else None,
                vega=float(vega) if vega is not None else None,
                iv=float(iv) if iv is not None else None,
                oi=int(oi) if oi is not None else None,
                volume=int(volume) if volume is not None else None,
                bid=float(bid) if bid is not None else None,
                ask=float(ask) if ask is not None else None,
                last=float(last_price) if last_price is not None else None,
            ))
        
        # Handle pagination if needed
        next_url = data.get("next_url")
        while next_url and len(candidates) < 1000:  # Cap at 1000 to avoid runaway
            next_resp = requests.get(f"{next_url}&apiKey={api_key}", timeout=30)
            next_resp.raise_for_status()
            next_data = next_resp.json()
            
            for item in next_data.get("results", []):
                details = item.get("details", {})
                greeks = item.get("greeks", {})
                day = item.get("day", {})
                
                symbol = details.get("ticker", "")
                contract_type_val = details.get("contract_type", "").lower()
                expiry_str = details.get("expiration_date", "")
                strike = details.get("strike_price", 0.0)
                
                try:
                    expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date() if expiry_str else date(1970, 1, 1)
                except ValueError:
                    expiry = date(1970, 1, 1)
                
                today = date.today()
                dte_days = (expiry - today).days if expiry > today else 0
                
                delta = greeks.get("delta")
                gamma = greeks.get("gamma")
                theta = greeks.get("theta")
                vega = greeks.get("vega")
                iv = item.get("implied_volatility")
                oi = item.get("open_interest")
                volume = day.get("volume")
                
                last_quote = item.get("last_quote", {})
                bid = last_quote.get("bid")
                ask = last_quote.get("ask")
                
                last_trade = item.get("last_trade", {})
                last_price = last_trade.get("price")
                
                candidates.append(OptionCandidate(
                    symbol=symbol,
                    opt_type=contract_type_val or "call",
                    expiry=expiry,
                    strike=float(strike) if strike else 0.0,
                    dte_days=dte_days,
                    delta=float(delta) if delta is not None else None,
                    gamma=float(gamma) if gamma is not None else None,
                    theta=float(theta) if theta is not None else None,
                    vega=float(vega) if vega is not None else None,
                    iv=float(iv) if iv is not None else None,
                    oi=int(oi) if oi is not None else None,
                    volume=int(volume) if volume is not None else None,
                    bid=float(bid) if bid is not None else None,
                    ask=float(ask) if ask is not None else None,
                    last=float(last_price) if last_price is not None else None,
                ))
            
            next_url = next_data.get("next_url")
        
    except requests.RequestException as e:
        logger.error(f"Polygon API error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error parsing Polygon response: {e}")
        return []
    
    return candidates


def fetch_oi_map(
    settings: Settings,
    ticker: str,
    *,
    min_dte: int = 7,
    max_dte: int = 90,
) -> dict[str, int]:
    """
    Fetch OI data from Polygon and return a map of symbol -> open_interest.
    
    This is meant to enrich Alpaca data which lacks OI.
    """
    from datetime import timedelta
    
    today = date.today()
    exp_gte = (today + timedelta(days=min_dte)).strftime("%Y-%m-%d")
    exp_lte = (today + timedelta(days=max_dte)).strftime("%Y-%m-%d")
    
    candidates = fetch_options_chain_polygon(
        settings,
        ticker,
        expiration_date_gte=exp_gte,
        expiration_date_lte=exp_lte,
        limit=250,
    )
    
    # Build symbol -> OI map
    # Polygon symbols are like "O:SPY250117C00600000"
    # Alpaca symbols are like "SPY250117C00600000"
    oi_map: dict[str, int] = {}
    for c in candidates:
        sym = c.symbol
        # Normalize: remove "O:" prefix if present
        if sym.startswith("O:"):
            sym = sym[2:]
        if c.oi is not None:
            oi_map[sym] = c.oi
    
    return oi_map


def enrich_candidates_with_oi(
    candidates: list[OptionCandidate],
    oi_map: dict[str, int],
) -> list[OptionCandidate]:
    """
    Enrich Alpaca OptionCandidates with OI data from Polygon.
    
    Returns new list with OI populated where available.
    """
    enriched = []
    for c in candidates:
        sym = c.symbol
        # Try with and without O: prefix
        oi = oi_map.get(sym) or oi_map.get(f"O:{sym}")
        
        if oi is not None or c.oi is not None:
            # Create new candidate with OI
            enriched.append(OptionCandidate(
                symbol=c.symbol,
                opt_type=c.opt_type,
                expiry=c.expiry,
                strike=c.strike,
                dte_days=c.dte_days,
                delta=c.delta,
                gamma=c.gamma,
                theta=c.theta,
                vega=c.vega,
                iv=c.iv,
                oi=oi if oi is not None else c.oi,
                volume=c.volume,
                bid=c.bid,
                ask=c.ask,
                last=c.last,
            ))
        else:
            enriched.append(c)
    
    return enriched


def fetch_high_oi_options(
    settings: Settings,
    tickers: list[str],
    *,
    max_premium: float = 2.0,  # Max premium per contract (e.g., 2.0 = $200)
    min_dte: int = 7,
    max_dte: int = 60,
    min_oi: int = 500,
    contract_type: str | None = None,  # "call", "put", or None
) -> list[OptionCandidate]:
    """
    Scan multiple tickers for high OI options under a budget.
    
    Uses Alpaca for quotes/greeks and Polygon for OI.
    
    Args:
        tickers: List of underlying symbols to scan
        max_premium: Maximum premium per contract (in dollars per share, so 2.0 = $200/contract)
        min_dte: Minimum days to expiration
        max_dte: Maximum days to expiration
        min_oi: Minimum open interest
        contract_type: Filter by call/put or None for both
    
    Returns:
        List of OptionCandidate objects sorted by OI descending
    """
    from lox.data.alpaca import fetch_option_chain, make_clients, to_candidates
    from lox.utils.occ import parse_occ_option_symbol
    
    _, data_client = make_clients(settings)
    
    all_candidates: list[OptionCandidate] = []
    
    for ticker in tickers:
        # Get quotes/greeks from Alpaca
        try:
            chain = fetch_option_chain(data_client, ticker, feed=settings.alpaca_options_feed)
            alpaca_candidates = list(to_candidates(chain, ticker))
        except Exception as e:
            logger.warning(f"Alpaca chain fetch failed for {ticker}: {e}")
            continue
        
        # Get OI from Polygon
        try:
            oi_map = fetch_oi_map(settings, ticker, min_dte=min_dte, max_dte=max_dte)
            logger.info(f"Polygon OI map for {ticker}: {len(oi_map)} symbols")
        except Exception as e:
            logger.warning(f"Polygon OI fetch failed for {ticker}: {e}")
            oi_map = {}
        
        # Enrich Alpaca data with Polygon OI
        enriched = enrich_candidates_with_oi(alpaca_candidates, oi_map)
        
        # Now filter
        today = date.today()
        for c in enriched:
            # Parse symbol to get type and expiry
            try:
                expiry, opt_type, strike = parse_occ_option_symbol(c.symbol, ticker)
                dte = (expiry - today).days
            except Exception:
                continue
            
            # Filter by type
            if contract_type and opt_type != contract_type:
                continue
            
            # Filter by DTE
            if dte < min_dte or dte > max_dte:
                continue
            
            # Filter by OI
            if c.oi is None or c.oi < min_oi:
                continue
            
            # Filter by premium (use mid or ask)
            mid = c.mid
            price = mid if mid else c.ask
            if price is None or price > max_premium:
                continue
            
            # Update with parsed data
            all_candidates.append(OptionCandidate(
                symbol=c.symbol,
                opt_type=opt_type,
                expiry=expiry,
                strike=strike,
                dte_days=dte,
                delta=c.delta,
                gamma=c.gamma,
                theta=c.theta,
                vega=c.vega,
                iv=c.iv,
                oi=c.oi,
                volume=c.volume,
                bid=c.bid,
                ask=c.ask,
                last=c.last,
            ))
    
    # Sort by OI descending
    all_candidates.sort(key=lambda x: x.oi or 0, reverse=True)
    
    return all_candidates


def get_liquid_etf_universe() -> list[str]:
    """
    Return a list of highly liquid broad ETFs for options scanning.
    """
    return [
        # S&P 500
        "SPY", "VOO", "IVV",
        # Nasdaq
        "QQQ", "TQQQ",
        # Dow Jones
        "DIA",
        # Russell 2000
        "IWM",
        # Total Market
        "VTI",
        # Sector ETFs (highly liquid)
        "XLF", "XLE", "XLK", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB",
        # International
        "EEM", "EFA", "FXI", "EWZ",
        # Bonds
        "TLT", "IEF", "HYG", "LQD",
        # Volatility
        "VXX", "UVXY", "SVXY",
        # Gold/Commodities
        "GLD", "SLV", "USO", "GDX",
        # Leveraged
        "SQQQ", "SPXU", "TNA", "TZA",
    ]

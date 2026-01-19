"""
Options chain scanning and filtering utilities.

Provides clean interfaces for:
- Fetching and parsing option chains
- Filtering by DTE, moneyness, liquidity
- Computing option metrics (spread, distance-to-strike)

Author: Lox Capital Research
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


class OptionType(str, Enum):
    """Option contract type."""
    CALL = "call"
    PUT = "put"
    BOTH = "both"


@dataclass
class OptionContract:
    """Parsed option contract with computed metrics."""
    symbol: str
    underlying: str
    strike: float
    expiry: date
    option_type: OptionType
    
    # Pricing
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    
    # Greeks
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    iv: Optional[float] = None
    
    # Liquidity
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    
    @property
    def mid(self) -> Optional[float]:
        """Mid-price if bid/ask available."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.last
    
    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread in dollars."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None
    
    @property
    def spread_pct(self) -> Optional[float]:
        """Spread as percentage of mid-price."""
        mid = self.mid
        spread = self.spread
        if mid and mid > 0 and spread is not None:
            return spread / mid
        return None
    
    @property
    def dte(self) -> int:
        """Days to expiration."""
        return (self.expiry - date.today()).days
    
    @property
    def premium_usd(self) -> float:
        """Premium per contract (price * 100)."""
        px = self.ask or self.mid or self.last or 0.0
        return px * 100.0
    
    def distance_to_strike(self, underlying_price: float) -> float:
        """
        Percentage distance from current price to strike.
        
        Positive = OTM for calls, ITM for puts
        Negative = ITM for calls, OTM for puts
        """
        if underlying_price <= 0:
            return 0.0
        return (self.strike - underlying_price) / underlying_price


@dataclass 
class ScanFilter:
    """Filter criteria for option scanning."""
    min_dte: int = 7
    max_dte: int = 90
    option_type: OptionType = OptionType.BOTH
    
    # Price filters
    min_price: float = 0.05
    max_premium: Optional[float] = None  # Max per-contract premium
    price_basis: Literal["ask", "mid", "last"] = "ask"
    
    # Spread/liquidity
    max_spread_pct: float = 0.30
    min_volume: Optional[int] = None
    min_open_interest: Optional[int] = None
    
    # Greeks
    require_delta: bool = True
    min_abs_delta: Optional[float] = None
    max_abs_delta: Optional[float] = None
    target_delta: Optional[float] = None
    
    # Moneyness
    max_otm_pct: Optional[float] = None  # e.g., 0.20 = 20% OTM max


def scan_options(
    contracts: List[OptionContract],
    filter: ScanFilter,
    underlying_price: Optional[float] = None,
) -> List[OptionContract]:
    """
    Filter option contracts based on criteria.
    
    Args:
        contracts: List of OptionContract to filter
        filter: ScanFilter with criteria
        underlying_price: Current underlying price (for moneyness)
    
    Returns:
        Filtered list of contracts matching all criteria
    """
    results: List[OptionContract] = []
    
    for c in contracts:
        # DTE filter
        if c.dte < filter.min_dte or c.dte > filter.max_dte:
            continue
        
        # Type filter
        if filter.option_type != OptionType.BOTH:
            if c.option_type != filter.option_type:
                continue
        
        # Price filter
        px = _get_price(c, filter.price_basis)
        if px is None or px < filter.min_price:
            continue
        
        # Premium filter
        if filter.max_premium is not None:
            if c.premium_usd > filter.max_premium:
                continue
        
        # Spread filter
        if c.spread_pct is not None and c.spread_pct > filter.max_spread_pct:
            continue
        
        # Liquidity filters
        if filter.min_volume is not None:
            if c.volume is None or c.volume < filter.min_volume:
                continue
        
        if filter.min_open_interest is not None:
            if c.open_interest is None or c.open_interest < filter.min_open_interest:
                continue
        
        # Delta filters
        if filter.require_delta and c.delta is None:
            continue
        
        if c.delta is not None:
            abs_delta = abs(c.delta)
            if filter.min_abs_delta is not None and abs_delta < filter.min_abs_delta:
                continue
            if filter.max_abs_delta is not None and abs_delta > filter.max_abs_delta:
                continue
        
        # Moneyness filter
        if filter.max_otm_pct is not None and underlying_price:
            dist = c.distance_to_strike(underlying_price)
            # For calls: positive dist = OTM, for puts: negative dist = OTM
            if c.option_type == OptionType.CALL and dist > filter.max_otm_pct:
                continue
            if c.option_type == OptionType.PUT and dist < -filter.max_otm_pct:
                continue
        
        results.append(c)
    
    return results


def rank_by_delta_theta(
    contracts: List[OptionContract],
    target_delta: float = 0.30,
    delta_weight: float = 1.0,
    theta_weight: float = 1.0,
) -> List[OptionContract]:
    """
    Rank contracts by proximity to target delta and low theta decay.
    
    Lower score = better (closer to target delta, lower decay).
    
    Args:
        contracts: Filtered contracts to rank
        target_delta: Target absolute delta value
        delta_weight: Weight for delta distance in score
        theta_weight: Weight for theta magnitude in score
    
    Returns:
        Contracts sorted by score (best first)
    """
    scored: List[tuple[float, OptionContract]] = []
    
    for c in contracts:
        if c.delta is None:
            continue
        
        # Delta distance (lower is better)
        delta_dist = abs(abs(c.delta) - target_delta)
        
        # Theta penalty (more negative theta = worse)
        theta_penalty = abs(c.theta) if c.theta else 0.0
        
        # Combined score
        score = (delta_weight * delta_dist) + (theta_weight * theta_penalty)
        scored.append((score, c))
    
    # Sort by score ascending
    scored.sort(key=lambda x: x[0])
    
    return [c for _, c in scored]


def parse_chain_to_contracts(
    chain: Dict[str, Any],
    underlying: str,
) -> List[OptionContract]:
    """
    Parse Alpaca option chain snapshot to OptionContract list.
    
    Args:
        chain: Dict of symbol -> OptionsSnapshot from Alpaca
        underlying: Underlying ticker symbol
    
    Returns:
        List of OptionContract objects
    """
    from ai_options_trader.utils.occ import parse_occ_option_symbol
    
    contracts: List[OptionContract] = []
    
    for symbol, snapshot in chain.items():
        try:
            expiry, opt_type, strike = parse_occ_option_symbol(symbol, underlying)
            
            contract = OptionContract(
                symbol=symbol,
                underlying=underlying,
                strike=strike,
                expiry=expiry,
                option_type=OptionType(opt_type),
                bid=_safe_float(getattr(snapshot, "bid_price", None)),
                ask=_safe_float(getattr(snapshot, "ask_price", None)),
                last=_safe_float(getattr(snapshot, "last_price", None)),
                delta=_safe_float(getattr(snapshot, "delta", None)),
                gamma=_safe_float(getattr(snapshot, "gamma", None)),
                theta=_safe_float(getattr(snapshot, "theta", None)),
                vega=_safe_float(getattr(snapshot, "vega", None)),
                iv=_safe_float(getattr(snapshot, "implied_volatility", None)),
                volume=_safe_int(getattr(snapshot, "volume", None)),
                open_interest=_safe_int(getattr(snapshot, "open_interest", None)),
            )
            contracts.append(contract)
        except (ValueError, AttributeError):
            continue
    
    return contracts


def _get_price(c: OptionContract, basis: str) -> Optional[float]:
    """Get price based on specified basis."""
    if basis == "bid":
        return c.bid
    elif basis == "ask":
        return c.ask
    elif basis == "mid":
        return c.mid
    elif basis == "last":
        return c.last
    return c.ask or c.mid or c.last


def _safe_float(x: Any) -> Optional[float]:
    """Safe float conversion."""
    try:
        return float(x) if x is not None else None
    except (ValueError, TypeError):
        return None


def _safe_int(x: Any) -> Optional[int]:
    """Safe int conversion."""
    try:
        return int(x) if x is not None else None
    except (ValueError, TypeError):
        return None

"""
Portfolio greeks calculation from Alpaca positions.

Uses unified utilities from utils/occ.py and alpaca_adapter.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime
import math

from ai_options_trader.config import Settings
from ai_options_trader.data.alpaca import make_clients
from ai_options_trader.utils.occ import parse_occ_option_full


@dataclass
class PortfolioGreeks:
    """Calculated greeks for entire portfolio."""
    net_delta: float  # As % of NAV
    net_vega: float   # As % of NAV per VIX point
    net_theta: float  # As % of NAV per day
    net_gamma: float  # As % of NAV per 1% SPX move
    positions: List['PositionGreeks']
    total_nav: float
    equity_value: float
    options_value: float
    cash: float
    has_tail_hedges: bool
    n_positions: int


@dataclass
class PositionGreeks:
    """Greeks for a single position."""
    symbol: str
    quantity: float
    market_value: float
    delta: float
    vega: float
    theta: float
    gamma: float
    position_delta: float
    position_vega: float
    position_theta: float
    position_gamma: float
    asset_type: str
    underlying: str | None
    strike: float | None
    expiry: datetime | None
    dte: int | None


def estimate_option_greeks(
    option_type: str,
    strike: float,
    expiry: datetime,
    underlying_price: float,
    implied_vol: float = 0.25,
) -> Tuple[float, float, float, float]:
    """
    Estimate option greeks using simplified Black-Scholes.
    Returns: (delta, vega, theta, gamma)
    """
    from scipy.stats import norm
    
    dte = (expiry - datetime.now()).days
    if dte <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    
    t = dte / 365.0
    r = 0.045  # Risk-free rate
    
    d1 = (math.log(underlying_price / strike) + (r + 0.5 * implied_vol**2) * t) / (implied_vol * math.sqrt(t))
    d2 = d1 - implied_vol * math.sqrt(t)
    
    if option_type.upper() in ("CALL", "C"):
        delta = norm.cdf(d1)
        theta = (
            -underlying_price * norm.pdf(d1) * implied_vol / (2 * math.sqrt(t))
            - r * strike * math.exp(-r * t) * norm.cdf(d2)
        ) / 365.0
    else:
        delta = norm.cdf(d1) - 1.0
        theta = (
            -underlying_price * norm.pdf(d1) * implied_vol / (2 * math.sqrt(t))
            + r * strike * math.exp(-r * t) * norm.cdf(-d2)
        ) / 365.0
    
    vega = underlying_price * norm.pdf(d1) * math.sqrt(t) / 100.0
    gamma = norm.pdf(d1) / (underlying_price * implied_vol * math.sqrt(t))
    
    return (delta, vega, theta, gamma)


def calculate_portfolio_greeks(settings: Settings) -> PortfolioGreeks:
    """
    Calculate greeks for entire portfolio from Alpaca.
    """
    trading, _ = make_clients(settings)
    
    acct = trading.get_account()
    nav = float(getattr(acct, "equity", 0.0) or 0.0)
    cash = float(getattr(acct, "cash", 0.0) or 0.0)
    
    positions = trading.get_all_positions()
    
    position_greeks: List[PositionGreeks] = []
    total_delta = 0.0
    total_vega = 0.0
    total_theta = 0.0
    total_gamma = 0.0
    equity_value = 0.0
    options_value = 0.0
    has_tail_hedges = False
    
    for pos in positions:
        symbol = str(getattr(pos, "symbol", ""))
        qty = float(getattr(pos, "qty", 0.0) or 0.0)
        market_value = float(getattr(pos, "market_value", 0.0) or 0.0)
        
        # Try parsing as option
        parsed = parse_occ_option_full(symbol)
        
        if parsed:
            underlying = parsed['underlying']
            expiry = parsed['expiry']
            strike = parsed['strike']
            option_type = parsed['type']
            dte = (expiry - datetime.now()).days
            
            # Estimate underlying price (use strike as proxy)
            underlying_price = strike
            delta, vega, theta, gamma = estimate_option_greeks(
                option_type=option_type,
                strike=strike,
                expiry=expiry,
                underlying_price=underlying_price,
            )
            
            pos_delta = delta * 100 * qty
            pos_vega = vega * 100 * qty
            pos_theta = theta * 100 * qty
            pos_gamma = gamma * 100 * qty
            
            options_value += abs(market_value)
            
            if option_type == "put" and strike < underlying_price * 0.9:
                has_tail_hedges = True
        else:
            underlying = symbol
            expiry = None
            strike = None
            dte = None
            option_type = "stock"
            
            delta = 1.0
            vega = 0.0
            theta = 0.0
            gamma = 0.0
            
            pos_delta = qty
            pos_vega = 0.0
            pos_theta = 0.0
            pos_gamma = 0.0
            
            equity_value += abs(market_value)
        
        total_delta += pos_delta
        total_vega += pos_vega
        total_theta += pos_theta
        total_gamma += pos_gamma
        
        position_greeks.append(PositionGreeks(
            symbol=symbol,
            quantity=qty,
            market_value=market_value,
            delta=delta,
            vega=vega,
            theta=theta,
            gamma=gamma,
            position_delta=pos_delta,
            position_vega=pos_vega,
            position_theta=pos_theta,
            position_gamma=pos_gamma,
            asset_type=option_type,
            underlying=underlying,
            strike=strike,
            expiry=expiry,
            dte=dte,
        ))
    
    net_delta = total_delta / nav if nav > 0 else 0.0
    net_vega = total_vega / nav if nav > 0 else 0.0
    net_theta = total_theta / nav if nav > 0 else 0.0
    net_gamma = total_gamma / nav if nav > 0 else 0.0
    
    return PortfolioGreeks(
        net_delta=net_delta,
        net_vega=net_vega,
        net_theta=net_theta,
        net_gamma=net_gamma,
        positions=position_greeks,
        total_nav=nav,
        equity_value=equity_value,
        options_value=options_value,
        cash=cash,
        has_tail_hedges=has_tail_hedges,
        n_positions=len(position_greeks),
    )

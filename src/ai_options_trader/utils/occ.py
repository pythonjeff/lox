"""
Unified OCC option symbol parser.

Handles both formats:
- Standard OCC: "GOOG251219C00355000"
- Alpaca format: "SPY/250321P00400000"
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional


def parse_occ_option_symbol(symbol: str, underlying: Optional[str] = None) -> tuple[date, str, float]:
    """
    Parse OCC-style option symbol.
    
    Args:
        symbol: OCC option symbol (e.g., "GOOG251219C00355000" or "SPY/250321P00400000")
        underlying: Optional underlying ticker for validation (ignored if symbol has /)
    
    Returns:
        expiry: datetime.date
        opt_type: 'call' or 'put'
        strike: float
    
    Raises:
        ValueError: If symbol cannot be parsed
    """
    # Handle Alpaca "/" format
    if '/' in symbol:
        parts = symbol.split('/')
        option_part = parts[1] if len(parts) > 1 else symbol
    else:
        # Standard OCC format - find where date starts
        if underlying:
            if not symbol.startswith(underlying):
                raise ValueError(f"Symbol {symbol} does not start with underlying {underlying}.")
            option_part = symbol[len(underlying):]
        else:
            # Find where the date starts (first digit sequence)
            idx = 0
            while idx < len(symbol) and not symbol[idx].isdigit():
                idx += 1
            option_part = symbol[idx:]
    
    # Validate minimum length for YYMMDD + C/P + 8-digit strike
    if len(option_part) < 15:
        raise ValueError(f"Symbol {symbol} too short to be OCC-style (YYMMDD+C/P+8 strike).")
    
    date_code = option_part[:6]      # YYMMDD
    cp_code = option_part[6]         # C or P
    strike_code = option_part[7:15]  # 8 digits
    
    year = 2000 + int(date_code[0:2])
    month = int(date_code[2:4])
    day = int(date_code[4:6])
    expiry = date(year, month, day)
    
    if cp_code.upper() == "C":
        opt_type = "call"
    elif cp_code.upper() == "P":
        opt_type = "put"
    else:
        raise ValueError(f"Unknown call/put code '{cp_code}' in symbol {symbol}.")
    
    strike = int(strike_code) / 1000.0
    return expiry, opt_type, strike


def parse_occ_option_full(symbol: str) -> dict | None:
    """
    Parse OCC option symbol and return full details including underlying.
    
    Args:
        symbol: OCC option symbol
    
    Returns:
        Dict with underlying, expiry, type, strike, or None if parsing fails
    """
    try:
        # Handle Alpaca "/" format
        if '/' in symbol:
            parts = symbol.split('/')
            underlying = parts[0]
            option_part = parts[1] if len(parts) > 1 else symbol
        else:
            # Find where the date starts
            idx = 0
            while idx < len(symbol) and not symbol[idx].isdigit():
                idx += 1
            underlying = symbol[:idx]
            option_part = symbol[idx:]
        
        if len(option_part) < 15:
            return None
        
        date_str = option_part[:6]
        year = 2000 + int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        expiry = datetime(year, month, day)
        
        option_type = option_part[6]
        if option_type.upper() not in ['C', 'P']:
            return None
        
        strike_str = option_part[7:15]
        strike = float(strike_str) / 1000.0
        
        return {
            'underlying': underlying,
            'expiry': expiry,
            'type': 'call' if option_type.upper() == 'C' else 'put',
            'strike': strike,
        }
    except Exception:
        return None

"""
Convert Alpaca positions to Portfolio object for Monte Carlo analysis.

Uses existing infrastructure from data/alpaca.py (make_clients).
"""
from typing import List, Dict

from ai_options_trader.config import Settings
from ai_options_trader.data.alpaca import make_clients
from ai_options_trader.portfolio.positions import Position, Portfolio
from ai_options_trader.altdata.fmp import fetch_realtime_quotes
from ai_options_trader.utils.occ import parse_occ_option_full


def alpaca_to_portfolio(settings: Settings) -> Portfolio:
    """
    Fetch positions from Alpaca and convert to Portfolio object.
    
    Uses existing make_clients() infrastructure.
    """
    print("Fetching positions from Alpaca...")
    
    trading, _ = make_clients(settings)
    
    # Get account
    try:
        acct = trading.get_account()
        cash = float(getattr(acct, "cash", 0.0) or 0.0)
    except Exception as e:
        print(f"Error fetching account: {e}")
        return Portfolio(positions=[], cash=0.0)
    
    # Get positions
    try:
        alpaca_positions = trading.get_all_positions()
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return Portfolio(positions=[], cash=cash)
    
    print(f"Found {len(alpaca_positions)} positions, cash: ${cash:,.0f}")
    
    # Extract unique underlying tickers for options
    underlying_tickers = set()
    for p in alpaca_positions:
        symbol = str(getattr(p, "symbol", ""))
        parsed = parse_occ_option_full(symbol)
        if parsed:
            underlying_tickers.add(parsed['underlying'])
    
    # Fetch real underlying prices from FMP
    print(f"Fetching real-time prices for {len(underlying_tickers)} underlyings...")
    underlying_prices: Dict[str, float] = {}
    if underlying_tickers:
        try:
            underlying_prices = fetch_realtime_quotes(
                settings=settings,
                tickers=list(underlying_tickers),
            )
            print(f"Fetched prices: {', '.join(f'{t}=${p:.2f}' for t, p in underlying_prices.items())}")
        except Exception as e:
            print(f"Could not fetch underlying prices: {e}")
            print("Falling back to strike prices as proxies")
    
    # Convert to Position objects
    positions: List[Position] = []
    
    for p in alpaca_positions:
        symbol = str(getattr(p, "symbol", ""))
        qty = float(getattr(p, "qty", 0.0) or 0.0)
        current_price = float(getattr(p, "current_price", 0.0) or 0.0)
        avg_entry = float(getattr(p, "avg_entry_price", 0.0) or 0.0)
        
        # Try to parse as option
        parsed = parse_occ_option_full(symbol)
        
        if parsed:
            # Option position
            underlying_symbol = parsed['underlying']
            
            # Try to get real price, fall back to strike if unavailable
            if underlying_symbol in underlying_prices:
                underlying_price = underlying_prices[underlying_symbol]
            else:
                underlying_price = parsed['strike']
                print(f"No price for {underlying_symbol}, using strike ${underlying_price:.2f}")
            
            position = Position(
                ticker=symbol,
                quantity=qty,
                position_type=parsed['type'],
                strike=parsed['strike'],
                expiry=parsed['expiry'],
                entry_price=avg_entry,
                entry_underlying_price=underlying_price,
                entry_iv=0.25,
            )
            
            # Calculate greeks
            position.calculate_greeks(underlying_price=underlying_price, iv=0.25)
        else:
            # Stock/ETF position
            position = Position(
                ticker=symbol,
                quantity=qty,
                position_type="stock",
                entry_price=avg_entry,
            )
            position.calculate_greeks(underlying_price=current_price, iv=0.0)
        
        positions.append(position)
        
        # Show what we added
        if position.is_option:
            print(f"  {symbol}: {qty:+.1f} @ ${current_price:.2f} "
                  f"(D{position.delta:+.2f}, V${position.position_vega_usd:.0f}, "
                  f"T${position.position_theta_usd:.0f}/day)")
        else:
            if abs(qty) < 1:
                print(f"  {symbol}: {qty:+.4f} shares @ ${current_price:.2f}")
            else:
                print(f"  {symbol}: {qty:+.0f} shares @ ${current_price:.2f}")
    
    portfolio = Portfolio(positions=positions, cash=cash)
    
    print(f"\nPortfolio loaded:")
    print(f"  NAV: ${portfolio.nav:,.0f}")
    print(f"  Net Delta: {portfolio.net_delta_pct*100:+.1f}%")
    print(f"  Net Vega: ${portfolio.net_vega:,.0f}")
    print(f"  Net Theta: ${portfolio.net_theta_per_day:,.0f} /day")
    
    return portfolio

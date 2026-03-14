"""Portfolio analysis, categorization, and construction for the LOX FUND Dashboard."""


def describe_portfolio(positions):
    """Build a concise description of portfolio positioning from actual positions."""
    if not positions:
        return "No positions", {}

    longs = []
    shorts = []

    for p in positions:
        symbol = p.get("symbol", "")
        qty = p.get("qty", 0)
        opt = p.get("opt_info")

        if opt:
            # Option position
            underlying = opt.get("underlying", symbol[:3])
            opt_type = opt.get("opt_type", opt.get("type", "")).upper()

            if qty > 0:  # Long options
                if opt_type in ["PUT", "P"]:
                    shorts.append(f"{underlying} (puts)")
                elif opt_type in ["CALL", "C"]:
                    longs.append(f"{underlying} (calls)")
            else:  # Short options
                if opt_type in ["PUT", "P"]:
                    longs.append(f"{underlying} (short puts)")
                elif opt_type in ["CALL", "C"]:
                    shorts.append(f"{underlying} (short calls)")
        else:
            # Stock/ETF position
            ticker = symbol
            if qty > 0:
                longs.append(ticker)
            elif qty < 0:
                shorts.append(ticker)

    desc_parts = []
    if longs:
        desc_parts.append(f"Long: {', '.join(longs[:4])}")
    if shorts:
        desc_parts.append(f"Short: {', '.join(shorts[:4])}")

    return "; ".join(desc_parts) if desc_parts else "No clear directional exposure"


def categorize_portfolio_positions(positions):
    """
    Categorize portfolio positions by their macro sensitivity for scenario analysis.

    Returns dict with structured position breakdown and scenario impacts.
    """
    if not positions:
        return {
            "summary": "No positions",
            "by_category": {},
            "scenario_matrix": {},
        }

    # Category buckets
    categories = {
        "long_equity_calls": [],      # Bullish equity (profit if market rallies)
        "long_equity_puts": [],       # Bearish equity (profit if market sells off)
        "long_vol": [],               # Long volatility (profit if VIX spikes)
        "long_rates_sensitive": [],   # Rates plays (TLT calls = profit if yields fall)
        "long_credit_puts": [],       # Credit stress plays (HYG puts = profit if spreads widen)
        "long_commodity": [],         # Commodity exposure (gold, oil, etc.)
        "short_equity_calls": [],     # Short calls (profit if market flat/down)
        "short_equity_puts": [],      # Short puts (profit if market flat/up)
        "etf_long": [],               # Long ETF shares
        "etf_short": [],              # Short ETF shares
    }

    # Known ETF classifications
    vol_etfs = ["VIXM", "VXX", "UVXY", "SVXY", "VIXY"]
    rates_etfs = ["TLT", "IEF", "SHY", "TBT", "TMF", "TMV"]
    credit_etfs = ["HYG", "JNK", "LQD", "BKLN", "HYGH"]
    commodity_etfs = ["GLDM", "GLD", "SLV", "USO", "UNG", "DBA", "DBB"]
    em_etfs = ["FXI", "EEM", "EWZ", "EWY", "EWT", "MCHI"]
    growth_etfs = ["QQQ", "ARKK", "TAN", "ICLN", "SOXX"]

    for p in positions:
        symbol = p.get("symbol", "")
        qty = p.get("qty", 0)
        opt = p.get("opt_info")
        pnl = p.get("pnl", 0)
        mv = p.get("market_value", 0)

        if not symbol or qty == 0:
            continue

        pos_entry = {
            "symbol": symbol,
            "qty": qty,
            "pnl": pnl,
            "market_value": mv,
        }

        if opt:
            underlying = opt.get("underlying", symbol[:3]).upper()
            opt_type = opt.get("opt_type", opt.get("type", "")).upper()
            strike = opt.get("strike", 0)
            expiry = opt.get("expiry", "")

            pos_entry["underlying"] = underlying
            pos_entry["opt_type"] = opt_type
            pos_entry["strike"] = strike
            pos_entry["expiry"] = expiry

            is_call = opt_type in ["CALL", "C"]
            is_put = opt_type in ["PUT", "P"]
            is_long = qty > 0

            # Categorize by underlying and option type
            if underlying in vol_etfs:
                if is_long and is_call:
                    categories["long_vol"].append(pos_entry)
                elif is_long and is_put:
                    categories["short_equity_puts"].append(pos_entry)  # Rare but possible
            elif underlying in rates_etfs:
                if is_long and is_call:
                    categories["long_rates_sensitive"].append(pos_entry)
                elif is_long and is_put:
                    categories["long_equity_puts"].append(pos_entry)  # Bearish bonds
            elif underlying in credit_etfs:
                if is_long and is_put:
                    categories["long_credit_puts"].append(pos_entry)
                elif is_long and is_call:
                    categories["long_equity_calls"].append(pos_entry)
            elif underlying in commodity_etfs:
                if is_long:
                    categories["long_commodity"].append(pos_entry)
            else:
                # General equity/sector ETF
                if is_long and is_call:
                    categories["long_equity_calls"].append(pos_entry)
                elif is_long and is_put:
                    categories["long_equity_puts"].append(pos_entry)
                elif not is_long and is_call:
                    categories["short_equity_calls"].append(pos_entry)
                elif not is_long and is_put:
                    categories["short_equity_puts"].append(pos_entry)
        else:
            # Stock/ETF shares
            ticker = symbol.upper()
            if qty > 0:
                if ticker in vol_etfs:
                    categories["long_vol"].append(pos_entry)
                elif ticker in commodity_etfs:
                    categories["long_commodity"].append(pos_entry)
                else:
                    categories["etf_long"].append(pos_entry)
            else:
                categories["etf_short"].append(pos_entry)

    # Build scenario impact matrix
    scenario_matrix = {
        "risk_off_spike": {  # VIX +10pts, HY spreads +100bp, equities -10%
            "winners": [],
            "losers": [],
        },
        "rates_surge": {  # 10Y +50bp, growth equities -5%
            "winners": [],
            "losers": [],
        },
        "goldilocks_rally": {  # VIX -5pts, equities +5%
            "winners": [],
            "losers": [],
        },
        "credit_stress": {  # HY spreads +150bp, HYG -5%
            "winners": [],
            "losers": [],
        },
    }

    # Map categories to scenario impacts
    for pos in categories["long_vol"]:
        scenario_matrix["risk_off_spike"]["winners"].append(pos["symbol"])
        scenario_matrix["goldilocks_rally"]["losers"].append(pos["symbol"])

    for pos in categories["long_equity_puts"]:
        scenario_matrix["risk_off_spike"]["winners"].append(pos["symbol"])
        scenario_matrix["goldilocks_rally"]["losers"].append(pos["symbol"])

    for pos in categories["long_equity_calls"]:
        scenario_matrix["goldilocks_rally"]["winners"].append(pos["symbol"])
        scenario_matrix["risk_off_spike"]["losers"].append(pos["symbol"])
        scenario_matrix["rates_surge"]["losers"].append(pos["symbol"])

    for pos in categories["long_rates_sensitive"]:
        scenario_matrix["rates_surge"]["losers"].append(pos["symbol"])
        # Rates falling = TLT calls win (flight to quality)
        scenario_matrix["risk_off_spike"]["winners"].append(pos["symbol"])

    for pos in categories["long_credit_puts"]:
        scenario_matrix["credit_stress"]["winners"].append(pos["symbol"])
        scenario_matrix["risk_off_spike"]["winners"].append(pos["symbol"])
        scenario_matrix["goldilocks_rally"]["losers"].append(pos["symbol"])

    for pos in categories["long_commodity"]:
        # Gold typically wins in risk-off
        if "GLD" in pos["symbol"] or "GLDM" in pos["symbol"]:
            scenario_matrix["risk_off_spike"]["winners"].append(pos["symbol"])

    # Build summary string
    active_categories = {k: v for k, v in categories.items() if v}
    summary_parts = []

    if categories["long_vol"]:
        tickers = [p["underlying"] if "underlying" in p else p["symbol"] for p in categories["long_vol"]]
        summary_parts.append(f"Long Vol: {', '.join(set(tickers))}")

    if categories["long_equity_puts"]:
        tickers = [p.get("underlying", p["symbol"]) for p in categories["long_equity_puts"]]
        summary_parts.append(f"Long Puts: {', '.join(set(tickers))}")

    if categories["long_equity_calls"]:
        tickers = [p.get("underlying", p["symbol"]) for p in categories["long_equity_calls"]]
        summary_parts.append(f"Long Calls: {', '.join(set(tickers))}")

    if categories["long_credit_puts"]:
        tickers = [p.get("underlying", p["symbol"]) for p in categories["long_credit_puts"]]
        summary_parts.append(f"Credit Puts: {', '.join(set(tickers))}")

    if categories["long_rates_sensitive"]:
        tickers = [p.get("underlying", p["symbol"]) for p in categories["long_rates_sensitive"]]
        summary_parts.append(f"Rates Plays: {', '.join(set(tickers))}")

    if categories["long_commodity"]:
        tickers = [p.get("underlying", p["symbol"]) for p in categories["long_commodity"]]
        summary_parts.append(f"Commodities: {', '.join(set(tickers))}")

    return {
        "summary": " | ".join(summary_parts) if summary_parts else "No directional exposure",
        "by_category": active_categories,
        "scenario_matrix": scenario_matrix,
    }


def build_portfolio_from_alpaca(positions_data, cash_available):
    """
    Build a Portfolio object from Alpaca positions for Monte Carlo simulation.

    Returns a Portfolio with Position objects including calculated greeks.
    """
    from lox.portfolio.positions import Portfolio, Position
    from datetime import datetime

    portfolio_positions = []

    for p in positions_data:
        symbol = p.get("symbol", "")
        qty = p.get("qty", 0)
        current_price = p.get("current_price", 0) or 0
        market_value = abs(p.get("market_value", 0) or 0)
        opt_info = p.get("opt_info")

        if not symbol or qty == 0:
            continue

        if opt_info:
            # Option position
            underlying = opt_info.get("underlying", symbol[:3])
            strike = opt_info.get("strike", 0)
            expiry_str = opt_info.get("expiry", "")
            opt_type = opt_info.get("opt_type", "P").upper()

            # Parse expiry
            try:
                expiry = datetime.strptime(expiry_str, "%Y-%m-%d") if expiry_str else None
            except:
                expiry = None

            # Estimate underlying price from strike proximity
            # (In production, fetch live underlying price)
            underlying_price = strike * 1.05 if opt_type in ["P", "PUT"] else strike * 0.95

            # Entry IV estimate based on position type
            entry_iv = 0.25  # Default
            if "VIX" in underlying.upper():
                entry_iv = 0.90
            elif "HYG" in underlying.upper():
                entry_iv = 0.18
            elif "TAN" in underlying.upper():
                entry_iv = 0.35

            pos = Position(
                ticker=symbol,
                quantity=qty,
                position_type="put" if opt_type in ["P", "PUT"] else "call",
                strike=strike,
                expiry=expiry,
                entry_price=current_price if current_price > 0 else (market_value / (abs(qty) * 100) if qty != 0 else 1),
                entry_underlying_price=underlying_price,
                entry_iv=entry_iv,
            )

            # Calculate greeks
            pos.calculate_greeks(underlying_price, entry_iv)
            portfolio_positions.append(pos)
        else:
            # Stock/ETF position
            pos = Position(
                ticker=symbol,
                quantity=qty,
                position_type="etf" if len(symbol) <= 5 else "stock",
                entry_price=current_price if current_price > 0 else (market_value / abs(qty) if qty != 0 else 1),
            )
            pos.calculate_greeks(pos.entry_price, 0.20)
            portfolio_positions.append(pos)

    return Portfolio(positions=portfolio_positions, cash=cash_available)

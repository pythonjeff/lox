"""
LOX FUND - Professional Investor Report Generator

Generates a branded PDF report for sharing with investors.
"""

from __future__ import annotations

import os
import tempfile
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich import print as rprint

from ai_options_trader.config import load_settings
from ai_options_trader.utils.settings import safe_load_settings


def _get_nav_summary() -> dict:
    """Get NAV and fund summary data."""
    from ai_options_trader.nav.store import read_nav_sheet
    from ai_options_trader.nav.investors import read_investor_flows
    
    rows = read_nav_sheet()
    flows = read_investor_flows()
    
    if not rows:
        return {}
    
    last = rows[-1]
    
    # Calculate totals from investor flows
    total_capital = sum(float(f.amount) for f in flows if float(f.amount) > 0)
    investor_codes = set(f.code for f in flows if float(f.amount) > 0)
    
    return {
        "equity": float(last.equity),
        "cash": float(last.cash),
        "capital": total_capital,
        "twr_cum": float(last.twr_cum) if last.twr_cum else 0,
        "investor_count": len(investor_codes),
        "snapshot_date": last.ts,
    }


def _get_investor_ledger() -> list[dict]:
    """Get investor ownership data."""
    from ai_options_trader.nav.investors import read_investor_flows
    from ai_options_trader.data.alpaca import make_clients
    
    settings = safe_load_settings()
    flows = read_investor_flows()
    
    if not flows:
        return []
    
    # Get live equity
    try:
        trading, _ = make_clients(settings)
        account = trading.get_account()
        live_equity = float(getattr(account, 'equity', 0) or 0)
    except Exception:
        live_equity = 0
    
    # Aggregate by investor
    investors = {}
    for f in flows:
        code = f.code
        if code not in investors:
            investors[code] = {"code": code, "deposits": 0, "units": 0}
        investors[code]["deposits"] += float(f.amount)
        investors[code]["units"] += float(f.units)
    
    # Calculate total units
    total_units = sum(inv["units"] for inv in investors.values())
    nav_per_unit = live_equity / total_units if total_units > 0 else 1.0
    
    # Calculate values and returns
    result = []
    for inv in sorted(investors.values(), key=lambda x: x["units"], reverse=True):
        value = inv["units"] * nav_per_unit
        pnl = value - inv["deposits"]
        ret = (pnl / inv["deposits"] * 100) if inv["deposits"] > 0 else 0
        ownership = (inv["units"] / total_units * 100) if total_units > 0 else 0
        
        result.append({
            "code": inv["code"],
            "ownership": ownership,
            "deposits": inv["deposits"],
            "value": value,
            "pnl": pnl,
            "return": ret,
        })
    
    return result


def _get_closed_trades() -> list[dict]:
    """Get closed trades data with dates."""
    from collections import defaultdict
    from ai_options_trader.data.alpaca import make_clients
    
    settings = safe_load_settings()
    
    try:
        trading, _ = make_clients(settings)
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500)
        orders = trading.get_orders(req) or []
    except Exception:
        return []
    
    # Group filled orders by symbol
    trades_by_symbol = defaultdict(lambda: {'buys': [], 'sells': []})
    
    for o in orders:
        status = str(getattr(o, 'status', '?')).split('.')[-1].lower()
        if 'filled' not in status:
            continue
        
        sym = getattr(o, 'symbol', '?')
        side = str(getattr(o, 'side', '?')).split('.')[-1].lower()
        filled_qty = float(getattr(o, 'filled_qty', 0) or 0)
        filled_price = getattr(o, 'filled_avg_price', 0)
        filled_at = getattr(o, 'filled_at', None)
        
        try:
            price = float(filled_price) if filled_price else 0
        except (ValueError, TypeError):
            price = 0
        
        if filled_qty <= 0 or price <= 0:
            continue
        
        is_option = len(sym) > 10 and any(c.isdigit() for c in sym[-8:]) and '/' not in sym
        mult = 100 if is_option else 1
        
        trade = {'qty': filled_qty, 'price': price, 'mult': mult, 'filled_at': filled_at}
        
        if side == 'buy':
            trades_by_symbol[sym]['buys'].append(trade)
        else:
            trades_by_symbol[sym]['sells'].append(trade)
    
    # FIFO matching
    closed_trades = []
    
    for sym, data in trades_by_symbol.items():
        buys = data['buys']
        sells = data['sells']
        
        if not buys or not sells:
            continue
        
        total_cost = sum(b['qty'] * b['price'] * b['mult'] for b in buys)
        total_proceeds = sum(s['qty'] * s['price'] * s['mult'] for s in sells)
        
        buy_qty = sum(b['qty'] for b in buys)
        sell_qty = sum(s['qty'] for s in sells)
        closed_qty = min(buy_qty, sell_qty)
        
        # Get the most recent sell date as the close date
        close_date = None
        for s in sells:
            if s.get('filled_at'):
                try:
                    dt = s['filled_at']
                    if hasattr(dt, 'date'):
                        close_date = dt.date()
                    elif isinstance(dt, str):
                        close_date = datetime.fromisoformat(dt.replace('Z', '+00:00')).date()
                except Exception:
                    pass
        
        if closed_qty > 0:
            # Simplified P&L
            avg_buy = total_cost / buy_qty if buy_qty > 0 else 0
            avg_sell = total_proceeds / sell_qty if sell_qty > 0 else 0
            pnl = (avg_sell - avg_buy) * closed_qty
            pnl_pct = (pnl / (avg_buy * closed_qty) * 100) if avg_buy > 0 else 0
            
            display_sym = _parse_option_display(sym)
            
            closed_trades.append({
                "symbol": display_sym,
                "cost": avg_buy * closed_qty,
                "proceeds": avg_sell * closed_qty,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "close_date": close_date,
            })
    
    # Sort by P&L
    closed_trades.sort(key=lambda x: x['pnl'], reverse=True)
    return closed_trades


def _get_week_bounds() -> tuple:
    """Get Monday and Friday of current week."""
    from datetime import date
    today = date.today()
    # Monday is 0, Sunday is 6
    days_since_monday = today.weekday()
    monday = today - timedelta(days=days_since_monday)
    friday = monday + timedelta(days=4)
    return monday, friday


def _filter_this_week_wins(trades: list[dict]) -> list[dict]:
    """Filter trades to only this week's winners."""
    monday, friday = _get_week_bounds()
    
    week_wins = []
    for t in trades:
        close_date = t.get("close_date")
        if close_date and monday <= close_date <= friday and t["pnl"] > 0:
            week_wins.append(t)
    
    # Sort by P&L descending
    week_wins.sort(key=lambda x: x['pnl'], reverse=True)
    return week_wins


def _parse_option_display(sym: str) -> str:
    """Parse option symbol for display."""
    if '/' in sym or len(sym) <= 6:
        return sym
    
    try:
        i = 0
        while i < len(sym) and not sym[i].isdigit():
            i += 1
        
        if i == 0 or i >= len(sym):
            return sym
        
        ticker = sym[:i]
        rest = sym[i:]
        
        if len(rest) >= 15:
            exp = f"{rest[2:4]}/{rest[4:6]}"
            opt_type = "C" if rest[6] == 'C' else "P"
            strike = int(rest[7:]) / 1000
            return f"{ticker} ${strike:.0f}{opt_type} {exp}"
    except Exception:
        pass
    
    return sym


def _get_market_outlook() -> dict:
    """Get market data for outlook section using FMP and FRED APIs."""
    import requests
    
    settings = safe_load_settings()
    
    outlook = {
        "vix": None,
        "vix_level": "N/A",
        "vix_color": "#64748b",
        "curve_2_10": None,
        "curve_status": "N/A",
        "mortgage_30y": None,
        "sp500_weekly": None,
        "sp500_color": "#64748b",
        "earnings": [],
    }
    
    fmp_key = getattr(settings, 'FMP_API_KEY', None) if settings else None
    fred_key = getattr(settings, 'FRED_API_KEY', None) if settings else None
    
    # VIX via FMP
    if fmp_key:
        try:
            url = "https://financialmodelingprep.com/api/v3/quote/%5EVIX"
            resp = requests.get(url, params={"apikey": fmp_key}, timeout=15)
            if resp.ok:
                data = resp.json()
                if data and isinstance(data, list) and len(data) > 0:
                    vix_price = data[0].get("price")
                    if vix_price:
                        outlook["vix"] = float(vix_price)
                        vix = outlook["vix"]
                        if vix < 13:
                            outlook["vix_level"] = "Very Low"
                            outlook["vix_color"] = "#22c55e"  # Green - complacent
                        elif vix < 16:
                            outlook["vix_level"] = "Low"
                            outlook["vix_color"] = "#84cc16"  # Lime
                        elif vix < 20:
                            outlook["vix_level"] = "Normal"
                            outlook["vix_color"] = "#64748b"  # Gray
                        elif vix < 25:
                            outlook["vix_level"] = "Elevated"
                            outlook["vix_color"] = "#f59e0b"  # Amber
                        elif vix < 30:
                            outlook["vix_level"] = "High"
                            outlook["vix_color"] = "#f97316"  # Orange
                        elif vix < 40:
                            outlook["vix_level"] = "Very High"
                            outlook["vix_color"] = "#ef4444"  # Red
                        else:
                            outlook["vix_level"] = "Extreme"
                            outlook["vix_color"] = "#dc2626"  # Dark red
        except Exception:
            pass
        
        # SPY weekly change via FMP
        try:
            url = "https://financialmodelingprep.com/api/v3/historical-price-full/SPY"
            resp = requests.get(url, params={"apikey": fmp_key, "serietype": "line"}, timeout=15)
            if resp.ok:
                data = resp.json()
                historical = data.get("historical", [])
                if len(historical) >= 5:
                    current = float(historical[0].get("close", 0))
                    week_ago = float(historical[4].get("close", 0))
                    if current and week_ago:
                        pct = ((current - week_ago) / week_ago) * 100
                        outlook["sp500_weekly"] = pct
                        outlook["sp500_color"] = "#22c55e" if pct >= 0 else "#ef4444"
        except Exception:
            pass
    
    # FRED data: 2/10 curve and 30Y mortgage
    if fred_key:
        try:
            from ai_options_trader.data.fred import FredClient
            fred = FredClient(api_key=fred_key)
            
            # 2/10 Treasury Spread (T10Y2Y)
            try:
                df = fred.fetch_series(series_id="T10Y2Y", start_date="2024-01-01", refresh=False)
                if df is not None and not df.empty:
                    df = df.sort_values("date")
                    df = df[df["value"].notna()]
                    if len(df) > 0:
                        spread = float(df.iloc[-1]["value"])
                        outlook["curve_2_10"] = spread
                        if spread < 0:
                            outlook["curve_status"] = "Inverted"
                        elif spread < 0.25:
                            outlook["curve_status"] = "Flat"
                        else:
                            outlook["curve_status"] = "Normal"
            except Exception:
                pass
            
            # 30Y Mortgage Rate (MORTGAGE30US)
            try:
                df = fred.fetch_series(series_id="MORTGAGE30US", start_date="2024-01-01", refresh=False)
                if df is not None and not df.empty:
                    df = df.sort_values("date")
                    df = df[df["value"].notna()]
                    if len(df) > 0:
                        outlook["mortgage_30y"] = float(df.iloc[-1]["value"])
            except Exception:
                pass
        except Exception:
            pass
    
    outlook["earnings"] = _get_upcoming_earnings()
    return outlook


def _get_upcoming_earnings() -> list[str]:
    """Get notable upcoming earnings."""
    # Hardcoded notable names - in production would use API
    return [
        "Check earnings calendars for this week's reports",
    ]


def _generate_html_report(nav: dict, investors: list, trades: list, week_wins: list, outlook: dict) -> str:
    """Generate HTML for the investor report with two pages."""
    
    now = datetime.now(timezone.utc)
    report_date = now.strftime("%B %d, %Y")
    monday, friday = _get_week_bounds()
    week_range = f"{monday.strftime('%b %d')} - {friday.strftime('%b %d, %Y')}"
    
    # Calculate summary stats
    total_pnl = nav.get("equity", 0) - nav.get("capital", 0)
    twr_pct = nav.get("twr_cum", 0) * 100
    
    # Trade summary
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t["pnl"] >= 0)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_realized = sum(t["pnl"] for t in trades)
    
    # Build investor rows
    investor_rows = ""
    for inv in investors:
        pnl_class = "positive" if inv["pnl"] >= 0 else "negative"
        investor_rows += f"""
        <tr>
            <td class="investor-code">{inv['code']}</td>
            <td>{inv['ownership']:.1f}%</td>
            <td>${inv['deposits']:,.0f}</td>
            <td>${inv['value']:,.0f}</td>
            <td class="{pnl_class}">${inv['pnl']:+,.0f}</td>
            <td class="{pnl_class}">{inv['return']:+.1f}%</td>
        </tr>
        """
    
    # Build this week's wins rows (for cover page)
    week_wins_rows = ""
    week_wins_total = sum(t["pnl"] for t in week_wins)
    for t in week_wins[:10]:  # Top 10 wins this week
        week_wins_rows += f"""
        <tr>
            <td>{t['symbol']}</td>
            <td>${t['cost']:,.0f}</td>
            <td>${t['proceeds']:,.0f}</td>
            <td class="positive">${t['pnl']:+,.0f}</td>
            <td class="positive">{t['pnl_pct']:+.1f}%</td>
        </tr>
        """
    
    # Build ALL trade rows (for Exhibit A)
    all_trade_rows = ""
    for t in trades:
        pnl_class = "positive" if t["pnl"] >= 0 else "negative"
        close_date_str = t.get("close_date").strftime("%m/%d") if t.get("close_date") else "-"
        all_trade_rows += f"""
        <tr>
            <td>{t['symbol']}</td>
            <td>{close_date_str}</td>
            <td>${t['cost']:,.0f}</td>
            <td>${t['proceeds']:,.0f}</td>
            <td class="{pnl_class}">${t['pnl']:+,.0f}</td>
            <td class="{pnl_class}">{t['pnl_pct']:+.1f}%</td>
        </tr>
        """
    
    # Market outlook
    vix_text = f"{outlook['vix']:.1f}" if outlook['vix'] else "N/A"
    vix_color = outlook.get('vix_color', '#64748b')
    vix_level = outlook.get('vix_level', 'N/A')
    
    curve_text = f"{outlook['curve_2_10']:+.2f}%" if outlook['curve_2_10'] is not None else "N/A"
    curve_status = outlook.get('curve_status', 'N/A')
    curve_color = "#ef4444" if outlook.get('curve_2_10', 0) and outlook['curve_2_10'] < 0 else "#22c55e" if outlook.get('curve_2_10', 0) and outlook['curve_2_10'] > 0.25 else "#f59e0b"
    
    mortgage_text = f"{outlook['mortgage_30y']:.2f}%" if outlook['mortgage_30y'] else "N/A"
    
    sp500_text = f"{outlook['sp500_weekly']:+.1f}%" if outlook['sp500_weekly'] else "N/A"
    sp500_color = outlook.get('sp500_color', '#64748b')
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LOX FUND - Investor Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #ffffff;
            color: #1a1a1a;
            line-height: 1.5;
            font-size: 11px;
        }}
        
        .container {{
            max-width: 800px;
            margin: 0 auto;
            padding: 40px;
        }}
        
        /* Header */
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 0;
            padding-bottom: 16px;
        }}
        
        .logo {{
            display: flex;
            align-items: baseline;
            gap: 8px;
        }}
        
        .logo-text {{
            font-size: 28px;
            font-weight: 700;
            color: #1a1a1a;
            letter-spacing: -1px;
        }}
        
        .logo-fund {{
            font-size: 14px;
            font-weight: 600;
            color: #666666;
            letter-spacing: 0.5px;
        }}
        
        .blue-accent {{
            height: 2px;
            background: linear-gradient(90deg, #0066ff, transparent);
            margin-bottom: 32px;
        }}
        
        .report-meta {{
            text-align: right;
            color: #64748b;
        }}
        
        .report-title {{
            font-size: 14px;
            font-weight: 600;
            color: #0f172a;
            margin-bottom: 4px;
        }}
        
        .report-date {{
            font-size: 12px;
        }}
        
        /* Hero Section */
        .hero {{
            background: linear-gradient(145deg, #0f172a 0%, #1e293b 100%);
            border-radius: 12px;
            padding: 32px;
            margin-bottom: 32px;
            color: white;
        }}
        
        .hero-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr;
            gap: 24px;
        }}
        
        .hero-stat {{
            text-align: center;
        }}
        
        .hero-label {{
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 1px;
            color: rgba(255,255,255,0.6);
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        
        .hero-value {{
            font-size: 28px;
            font-weight: 700;
        }}
        
        .hero-value.positive {{ color: #4ade80; }}
        .hero-value.negative {{ color: #f87171; }}
        
        .hero-context {{
            font-size: 11px;
            color: rgba(255,255,255,0.5);
            margin-top: 4px;
        }}
        
        /* Sections */
        .section {{
            margin-bottom: 32px;
        }}
        
        .section-title {{
            font-size: 14px;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid #e2e8f0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
        }}
        
        th {{
            text-align: left;
            padding: 10px 12px;
            background: #f8fafc;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 10px;
            border-bottom: 2px solid #e2e8f0;
        }}
        
        th:not(:first-child) {{
            text-align: right;
        }}
        
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #f1f5f9;
        }}
        
        td:not(:first-child) {{
            text-align: right;
        }}
        
        .investor-code {{
            font-weight: 600;
            color: #0f172a;
        }}
        
        .positive {{ color: #16a34a; }}
        .negative {{ color: #dc2626; }}
        
        /* Market Outlook */
        .outlook-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }}
        
        .outlook-card {{
            background: #f8fafc;
            border-radius: 8px;
            padding: 14px 12px;
            text-align: center;
            border: 1px solid #e2e8f0;
        }}
        
        .outlook-label {{
            font-size: 9px;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }}
        
        .outlook-value {{
            font-size: 18px;
            font-weight: 700;
            color: #0f172a;
        }}
        
        .outlook-context {{
            font-size: 9px;
            margin-top: 4px;
            font-weight: 500;
        }}
        
        .outlook-positive {{ color: #22c55e; }}
        .outlook-negative {{ color: #ef4444; }}
        .outlook-neutral {{ color: #64748b; }}
        
        /* Footer */
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            text-align: center;
            color: #94a3b8;
            font-size: 10px;
        }}
        
        .disclaimer {{
            font-size: 9px;
            color: #94a3b8;
            margin-top: 20px;
            padding: 16px;
            background: #f8fafc;
            border-radius: 8px;
        }}
        
        .page-break {{
            page-break-before: always;
            margin-top: 40px;
            padding-top: 40px;
            border-top: 1px solid #e2e8f0;
        }}
        
        .exhibit-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
        }}
        
        .exhibit-title {{
            font-size: 18px;
            font-weight: 700;
            color: #1a1a1a;
        }}
        
        .exhibit-subtitle {{
            font-size: 12px;
            color: #666666;
        }}
        
        @media print {{
            body {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
            .container {{ padding: 20px; }}
            .page-break {{ page-break-before: always; border-top: none; margin-top: 0; padding-top: 20px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="logo">
                <span class="logo-text">LOX</span>
                <span class="logo-fund">FUND</span>
            </div>
            <div class="report-meta">
                <div class="report-title">Investor Report</div>
                <div class="report-date">{report_date}</div>
            </div>
        </div>
        
        <!-- Blue accent line (matches dashboard) -->
        <div class="blue-accent"></div>
        
        <!-- Hero Stats -->
        <div class="hero">
            <div class="hero-grid">
                <div class="hero-stat">
                    <div class="hero-label">Fund Return (TWR)</div>
                    <div class="hero-value {'positive' if twr_pct >= 0 else 'negative'}">{twr_pct:+.1f}%</div>
                    <div class="hero-context">Since inception</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-label">Net Asset Value</div>
                    <div class="hero-value">${nav.get('equity', 0):,.0f}</div>
                    <div class="hero-context">Liquidation value</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-label">Total P&L</div>
                    <div class="hero-value {'positive' if total_pnl >= 0 else 'negative'}">${total_pnl:+,.0f}</div>
                    <div class="hero-context">Unrealized + Realized</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-label">Investors</div>
                    <div class="hero-value">{nav.get('investor_count', 0)}</div>
                    <div class="hero-context">${nav.get('capital', 0):,.0f} AUM</div>
                </div>
            </div>
        </div>
        
        <!-- Investor Ledger -->
        <div class="section">
            <div class="section-title">Investor Ownership</div>
            <table>
                <thead>
                    <tr>
                        <th>Investor</th>
                        <th>Ownership</th>
                        <th>Contributed</th>
                        <th>Current Value</th>
                        <th>P&L</th>
                        <th>Return</th>
                    </tr>
                </thead>
                <tbody>
                    {investor_rows}
                </tbody>
            </table>
        </div>
        
        <!-- This Week's Wins -->
        <div class="section">
            <div class="section-title">This Week's Wins ({week_range})</div>
            {'<div style="color: #666; font-size: 12px; margin-bottom: 16px;">No winning trades closed this week.</div>' if not week_wins_rows else f"""
            <div style="display: flex; gap: 24px; margin-bottom: 16px;">
                <div><strong>{len(week_wins)}</strong> winning trades</div>
                <div><strong class="positive">${week_wins_total:+,.0f}</strong> realized gains</div>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Trade</th>
                        <th>Cost</th>
                        <th>Proceeds</th>
                        <th>P&L</th>
                        <th>Return</th>
                    </tr>
                </thead>
                <tbody>
                    {week_wins_rows}
                </tbody>
            </table>
            """}
            <div style="font-size: 11px; color: #666; margin-top: 12px;">See Exhibit A for complete trade history.</div>
        </div>
        
        <!-- Market Outlook -->
        <div class="section">
            <div class="section-title">Market Conditions</div>
            <div class="outlook-grid">
                <div class="outlook-card">
                    <div class="outlook-label">VIX</div>
                    <div class="outlook-value" style="color: {vix_color}">{vix_text}</div>
                    <div class="outlook-context" style="color: {vix_color}">{vix_level}</div>
                </div>
                <div class="outlook-card">
                    <div class="outlook-label">2/10 Curve</div>
                    <div class="outlook-value" style="color: {curve_color}">{curve_text}</div>
                    <div class="outlook-context" style="color: {curve_color}">{curve_status}</div>
                </div>
                <div class="outlook-card">
                    <div class="outlook-label">30Y Mortgage</div>
                    <div class="outlook-value">{mortgage_text}</div>
                    <div class="outlook-context outlook-neutral">Weekly avg</div>
                </div>
                <div class="outlook-card">
                    <div class="outlook-label">S&P 500 (Weekly)</div>
                    <div class="outlook-value" style="color: {sp500_color}">{sp500_text}</div>
                    <div class="outlook-context" style="color: {sp500_color}">SPY change</div>
                </div>
            </div>
        </div>
        
        <!-- Disclaimer -->
        <div class="disclaimer">
            <strong>Disclaimer:</strong> This report is provided for informational purposes only and does not constitute investment advice. 
            Past performance is not indicative of future results. The fund involves substantial risk of loss and is not suitable for all investors.
            Returns shown are Time-Weighted Returns (TWR) which remove the impact of cash flow timing.
        </div>
        
        <!-- ==================== EXHIBIT A: FULL TRADE HISTORY ==================== -->
        <div class="page-break">
            <div class="exhibit-header">
                <div>
                    <div class="exhibit-title">Exhibit A: Complete Trade History</div>
                    <div class="exhibit-subtitle">All closed positions since inception</div>
                </div>
                <div style="text-align: right;">
                    <div class="logo" style="justify-content: flex-end;">
                        <span class="logo-text" style="font-size: 20px;">LOX</span>
                        <span class="logo-fund" style="font-size: 11px;">FUND</span>
                    </div>
                </div>
            </div>
            <div class="blue-accent"></div>
            
            <div style="display: flex; gap: 24px; margin: 20px 0; font-size: 12px;">
                <div><strong>{total_trades}</strong> total trades</div>
                <div><strong>{win_rate:.0f}%</strong> win rate</div>
                <div><strong>{winning_trades}</strong> wins / <strong>{total_trades - winning_trades}</strong> losses</div>
                <div><strong class="{'positive' if total_realized >= 0 else 'negative'}">${total_realized:+,.0f}</strong> total realized</div>
            </div>
            
            <table style="font-size: 10px;">
                <thead>
                    <tr>
                        <th>Trade</th>
                        <th>Closed</th>
                        <th>Cost</th>
                        <th>Proceeds</th>
                        <th>P&L</th>
                        <th>Return</th>
                    </tr>
                </thead>
                <tbody>
                    {all_trade_rows}
                </tbody>
            </table>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <div style="font-weight: 600; margin-bottom: 8px;">LOX FUND</div>
            <div>Generated {now.strftime("%Y-%m-%d %H:%M UTC")}</div>
        </div>
    </div>
</body>
</html>
    """
    
    return html


def _save_html_report(html: str, output_path: str) -> bool:
    """Save HTML report to file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return True
    except Exception as e:
        rprint(f"[red]Error saving report: {e}[/red]")
        return False


def register(app: typer.Typer) -> None:
    """Register the investor report command."""
    
    @app.command("report")
    def investor_report(
        share: bool = typer.Option(False, "--share", "-s", help="Generate shareable HTML report"),
        output: str = typer.Option(None, "--output", "-o", help="Output path for report"),
        open_file: bool = typer.Option(True, "--open/--no-open", help="Open report after generation"),
    ):
        """
        Generate a professional investor report.
        
        Examples:
            lox weekly report              # Display in terminal
            lox weekly report --share      # Generate HTML and open in browser
            lox weekly report -s -o report.html  # Save to specific path
        
        The HTML report can be printed to PDF from the browser (Cmd+P / Ctrl+P).
        """
        console = Console()
        
        console.print("\n[bold cyan]LOX FUND[/bold cyan] [dim]Investor Report[/dim]\n")
        
        # Gather data
        with console.status("[bold green]Gathering fund data..."):
            nav = _get_nav_summary()
            investors = _get_investor_ledger()
            trades = _get_closed_trades()
            week_wins = _filter_this_week_wins(trades)
            outlook = _get_market_outlook()
        
        if not share:
            # Terminal display
            _display_terminal_report(console, nav, investors, trades, outlook)
        else:
            # Generate HTML report
            console.print("[bold]Generating investor report...[/bold]")
            
            html = _generate_html_report(nav, investors, trades, week_wins, outlook)
            
            # Determine output path
            if output:
                report_path = output
            else:
                timestamp = datetime.now().strftime("%Y%m%d")
                output_dir = Path.home() / "Desktop"
                if not output_dir.exists():
                    output_dir = Path(tempfile.gettempdir())
                report_path = str(output_dir / f"lox_fund_report_{timestamp}.html")
            
            if _save_html_report(html, report_path):
                console.print(f"\n[green]✓ Report saved to:[/green] {report_path}")
                console.print("[dim]Tip: Print to PDF from browser with Cmd+P (Mac) or Ctrl+P (Windows)[/dim]")
                
                if open_file:
                    try:
                        if sys.platform == "darwin":
                            subprocess.run(["open", report_path], check=False)
                        elif sys.platform == "win32":
                            subprocess.run(["start", report_path], shell=True, check=False)
                        else:
                            subprocess.run(["xdg-open", report_path], check=False)
                    except Exception:
                        pass
            else:
                console.print("[red]Failed to generate report[/red]")


def _display_terminal_report(console: Console, nav: dict, investors: list, trades: list, outlook: dict):
    """Display report in terminal."""
    from rich.table import Table
    from rich.panel import Panel
    
    # Summary
    twr_pct = nav.get("twr_cum", 0) * 100
    total_pnl = nav.get("equity", 0) - nav.get("capital", 0)
    
    summary = f"""
[bold]Fund Return (TWR):[/bold] [{'green' if twr_pct >= 0 else 'red'}]{twr_pct:+.1f}%[/]
[bold]Net Asset Value:[/bold] ${nav.get('equity', 0):,.0f}
[bold]Total P&L:[/bold] [{'green' if total_pnl >= 0 else 'red'}]${total_pnl:+,.0f}[/]
[bold]Investors:[/bold] {nav.get('investor_count', 0)} | AUM: ${nav.get('capital', 0):,.0f}
    """
    
    console.print(Panel(summary.strip(), title="[bold]Fund Summary[/bold]", border_style="blue"))
    
    # Investor table
    table = Table(title="Investor Ownership", box=None)
    table.add_column("Investor", style="bold")
    table.add_column("Ownership", justify="right")
    table.add_column("Contributed", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Return", justify="right")
    
    for inv in investors:
        pnl_style = "green" if inv["pnl"] >= 0 else "red"
        table.add_row(
            inv["code"],
            f"{inv['ownership']:.1f}%",
            f"${inv['deposits']:,.0f}",
            f"${inv['value']:,.0f}",
            f"[{pnl_style}]${inv['pnl']:+,.0f}[/]",
            f"[{pnl_style}]{inv['return']:+.1f}%[/]",
        )
    
    console.print(table)
    console.print()
    
    # Trade stats
    total_trades = len(trades)
    winning = sum(1 for t in trades if t["pnl"] >= 0)
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
    realized = sum(t["pnl"] for t in trades)
    
    console.print(f"[bold]Trades:[/bold] {total_trades} | Win Rate: {win_rate:.0f}% | Realized: ${realized:+,.0f}")
    console.print()
    
    # Market outlook
    vix = outlook.get("vix")
    curve = outlook.get("curve_2_10")
    mortgage = outlook.get("mortgage_30y")
    spy = outlook.get("sp500_weekly")
    
    vix_str = f"{vix:.1f}" if vix else "N/A"
    vix_level = outlook.get("vix_level", "")
    vix_color = "green" if vix and vix < 16 else "yellow" if vix and vix < 25 else "red" if vix else "white"
    
    curve_str = f"{curve:+.2f}%" if curve is not None else "N/A"
    curve_status = outlook.get("curve_status", "")
    curve_color = "red" if curve and curve < 0 else "green" if curve and curve > 0.25 else "yellow"
    
    mortgage_str = f"{mortgage:.2f}%" if mortgage else "N/A"
    
    spy_str = f"{spy:+.1f}%" if spy else "N/A"
    spy_color = "green" if spy and spy >= 0 else "red" if spy else "white"
    
    outlook_text = f"""
[bold]VIX:[/bold] [{vix_color}]{vix_str}[/] ({vix_level})
[bold]2/10 Curve:[/bold] [{curve_color}]{curve_str}[/] ({curve_status})
[bold]30Y Mortgage:[/bold] {mortgage_str}
[bold]S&P 500 (weekly):[/bold] [{spy_color}]{spy_str}[/]
    """
    
    console.print(Panel(outlook_text.strip(), title="[bold]Market Conditions[/bold]", border_style="cyan"))

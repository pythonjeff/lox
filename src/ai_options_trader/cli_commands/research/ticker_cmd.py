"""
LOX Research: Ticker Command

Hedge fund level single-stock analysis with visualizations.

Usage:
    lox research ticker AAPL          # Full analysis
    lox research ticker NVDA --chart  # Generate price chart
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ai_options_trader.config import load_settings


def register(app: typer.Typer) -> None:
    """Register the ticker command."""
    
    @app.command("ticker")
    def ticker_cmd(
        symbol: str = typer.Argument(..., help="Ticker symbol (e.g., AAPL, NVDA, SPY)"),
        chart: bool = typer.Option(True, "--chart/--no-chart", help="Generate price chart"),
        llm: bool = typer.Option(True, "--llm/--no-llm", help="Include LLM analysis"),
    ):
        """
        Hedge fund level ticker analysis.
        
        Includes:
        - Price chart with key levels (matplotlib)
        - Fundamentals snapshot (P/E, revenue, margins)
        - Technical levels (support/resistance, RSI, MAs)
        - LLM synthesis with bull/bear case
        
        Examples:
            lox research ticker AAPL
            lox research ticker NVDA --no-chart
            lox research ticker SPY --no-llm
        """
        console = Console()
        settings = load_settings()
        
        symbol = symbol.upper()
        console.print()
        console.print(f"[bold cyan]LOX RESEARCH[/bold cyan]  [bold]{symbol}[/bold]")
        console.print()
        
        # Fetch all data
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Fetching data...[/bold cyan]"),
            transient=True,
        ) as progress:
            progress.add_task("fetch", total=None)
            
            price_data = _fetch_price_data(settings, symbol)
            fundamentals = _fetch_fundamentals(settings, symbol)
            technicals = _compute_technicals(price_data)
        
        # Display price info
        if price_data:
            _show_price_panel(console, symbol, price_data, technicals)
        
        # Display fundamentals
        if fundamentals:
            _show_fundamentals(console, fundamentals)
        
        # ETF Flow analysis (only for ETFs)
        is_etf = fundamentals.get("profile", {}).get("isEtf", False) or fundamentals.get("etf_info")
        if is_etf and price_data.get("historical"):
            _show_etf_flows(console, price_data, fundamentals)
        
        # Bond ETF: Refinancing wall (maturity distribution)
        if is_etf:
            asset_class = (fundamentals.get("etf_info", {}).get("assetClass") or "").lower()
            description = (fundamentals.get("profile", {}).get("description") or "").lower()
            is_bond_etf = "fixed income" in asset_class or "bond" in description or "credit" in description
            if is_bond_etf:
                _show_refinancing_wall(console, settings, symbol)
        
        # Display technicals
        if technicals:
            _show_technicals(console, technicals)
        
        # Generate chart
        if chart and price_data:
            chart_path = _generate_chart(symbol, price_data, technicals)
            if chart_path:
                console.print(f"\n[green]Chart saved:[/green] {chart_path}")
                _open_chart(chart_path)
        
        # LLM Analysis
        if llm:
            _show_llm_analysis(console, settings, symbol, price_data, fundamentals, technicals)
        
        console.print()


def _fetch_price_data(settings, symbol: str) -> dict:
    """Fetch historical price data."""
    try:
        import requests
        from datetime import datetime, timedelta
        
        if not settings.fmp_api_key:
            return {}
        
        # Get quote
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
        resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=15)
        quote = {}
        if resp.ok:
            data = resp.json()
            if data and isinstance(data, list):
                quote = data[0]
        
        # Get historical
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
        resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=15)
        historical = []
        if resp.ok:
            data = resp.json()
            historical = data.get("historical", [])[:252]  # 1 year
        
        return {
            "symbol": symbol,
            "quote": quote,
            "historical": historical,
        }
    except Exception:
        return {}


def _fetch_fundamentals(settings, symbol: str) -> dict:
    """Fetch fundamental data (auto-detects ETFs vs stocks)."""
    try:
        import requests
        
        if not settings.fmp_api_key:
            return {}
        
        result = {}
        
        # Company profile (works for both stocks and ETFs)
        url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
        resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=15)
        if resp.ok:
            data = resp.json()
            if data and isinstance(data, list):
                result["profile"] = data[0]
        
        # Check if ETF — fetch ETF-specific data instead of stock ratios
        is_etf = result.get("profile", {}).get("isEtf", False)
        
        if is_etf:
            # ETF info (AUM, expense ratio, holdings count, etc.)
            url = f"https://financialmodelingprep.com/api/v4/etf-info"
            resp = requests.get(url, params={"symbol": symbol, "apikey": settings.fmp_api_key}, timeout=15)
            if resp.ok:
                data = resp.json()
                if data and isinstance(data, list):
                    result["etf_info"] = data[0]
        else:
            # Stock-specific: key metrics and ratios
            url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}"
            resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=15)
            if resp.ok:
                data = resp.json()
                if data and isinstance(data, list):
                    result["metrics"] = data[0]
            
            url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}"
            resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=15)
            if resp.ok:
                data = resp.json()
                if data and isinstance(data, list):
                    result["ratios"] = data[0]
        
        return result
    except Exception:
        return {}


def _compute_technicals(price_data: dict) -> dict:
    """Compute technical indicators from price data."""
    if not price_data or not price_data.get("historical"):
        return {}
    
    try:
        import pandas as pd
        import numpy as np
        
        historical = price_data["historical"]
        df = pd.DataFrame(historical)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        
        # Current price
        current = closes[-1] if len(closes) > 0 else 0
        
        # Moving averages
        ma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else None
        ma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else None
        ma_200 = np.mean(closes[-200:]) if len(closes) >= 200 else None
        
        # RSI (14-day)
        rsi = None
        if len(closes) >= 15:
            deltas = np.diff(closes[-15:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
        
        # 52-week high/low
        high_52w = max(highs) if len(highs) > 0 else None
        low_52w = min(lows) if len(lows) > 0 else None
        
        # Support/Resistance (simple: recent lows/highs)
        support = min(lows[-20:]) if len(lows) >= 20 else None
        resistance = max(highs[-20:]) if len(highs) >= 20 else None
        
        # Volatility (20-day)
        returns = np.diff(closes[-21:]) / closes[-21:-1] if len(closes) >= 21 else []
        volatility = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 0 else None
        
        # Trend (above/below 50 MA)
        trend = None
        if ma_50:
            trend = "bullish" if current > ma_50 else "bearish"
        
        return {
            "current": current,
            "ma_20": ma_20,
            "ma_50": ma_50,
            "ma_200": ma_200,
            "rsi": rsi,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "support": support,
            "resistance": resistance,
            "volatility": volatility,
            "trend": trend,
            "df": df,
        }
    except Exception:
        return {}


def _show_price_panel(console: Console, symbol: str, price_data: dict, technicals: dict):
    """Display price information panel."""
    quote = price_data.get("quote", {})
    
    price = quote.get("price", technicals.get("current", 0))
    change = quote.get("change", 0)
    change_pct = quote.get("changesPercentage", 0)
    
    change_color = "green" if change >= 0 else "red"
    
    content = f"""[bold]{symbol}[/bold]  ${price:,.2f}  [{change_color}]{change:+.2f} ({change_pct:+.2f}%)[/{change_color}]

52W Range: ${technicals.get('low_52w', 0):,.2f} - ${technicals.get('high_52w', 0):,.2f}
Support: ${technicals.get('support', 0):,.2f}  |  Resistance: ${technicals.get('resistance', 0):,.2f}
Trend: {technicals.get('trend', 'N/A').upper()}  |  Volatility: {technicals.get('volatility', 0):.1f}% (ann.)"""
    
    console.print(Panel(content, title="[bold]Price[/bold]", border_style="blue"))


def _show_fundamentals(console: Console, fundamentals: dict):
    """Display fundamentals table (detects ETFs vs stocks)."""
    profile = fundamentals.get("profile", {})
    etf_info = fundamentals.get("etf_info", {})
    
    is_etf = profile.get("isEtf", False) or bool(etf_info)
    
    if is_etf:
        _show_etf_fundamentals(console, profile, etf_info)
    else:
        _show_stock_fundamentals(console, fundamentals)


def _show_etf_fundamentals(console: Console, profile: dict, etf_info: dict):
    """Display ETF-specific fundamentals."""
    table = Table(title="[bold]ETF Profile[/bold]", box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    
    def fmt(val, pct=False, billions=False, dollar=False):
        if val is None:
            return "N/A"
        try:
            v = float(val)
            if billions:
                return f"${v/1e9:.1f}B"
            if pct:
                return f"{v:.2f}%"
            if dollar:
                return f"${v:.2f}"
            return f"{v:,.0f}"
        except Exception:
            return str(val)[:20]
    
    aum = etf_info.get("aum") or profile.get("mktCap")
    expense = etf_info.get("expenseRatio")
    nav = etf_info.get("nav")
    holdings = etf_info.get("holdingsCount")
    div_yield = profile.get("lastDiv")
    price = profile.get("price")
    beta = profile.get("beta")
    inception = etf_info.get("inceptionDate") or profile.get("ipoDate")
    asset_class = etf_info.get("assetClass", "N/A")
    company = etf_info.get("etfCompany", "N/A")
    
    # Compute yield % if we have div and price
    yield_pct = None
    if div_yield and price:
        try:
            yield_pct = (float(div_yield) / float(price)) * 100
        except Exception:
            pass
    
    table.add_row("AUM", fmt(aum, billions=True), "Expense Ratio", fmt(expense, pct=True))
    table.add_row("NAV", fmt(nav, dollar=True), "Holdings", fmt(holdings))
    table.add_row("Yield", fmt(yield_pct, pct=True) if yield_pct else fmt(div_yield, dollar=True), "Beta", fmt(beta))
    table.add_row("Asset Class", str(asset_class)[:15], "Issuer", str(company)[:15])
    table.add_row("Inception", str(inception)[:10] if inception else "N/A", "Avg Volume", fmt(etf_info.get("avgVolume")))
    
    console.print()
    console.print(table)
    
    # Show description if available
    desc = etf_info.get("description") or profile.get("description")
    if desc:
        from rich.text import Text
        console.print()
        console.print(f"[dim]{desc[:200]}[/dim]")


def _show_etf_flows(console: Console, price_data: dict, fundamentals: dict):
    """Compute and display ETF flow signals from volume data."""
    import numpy as np
    
    historical = price_data.get("historical", [])
    if len(historical) < 21:
        return
    
    # Sort oldest first
    hist = list(reversed(historical[:60]))
    
    closes = [h["close"] for h in hist]
    volumes = [h["volume"] for h in hist]
    highs = [h["high"] for h in hist]
    lows = [h["low"] for h in hist]
    
    # Money Flow Index (14-day)
    mfi = None
    if len(hist) >= 15:
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        pos_mf = sum(
            typical_prices[i] * volumes[i]
            for i in range(-14, 0)
            if typical_prices[i] > typical_prices[i - 1]
        )
        neg_mf = sum(
            typical_prices[i] * volumes[i]
            for i in range(-14, 0)
            if typical_prices[i] <= typical_prices[i - 1]
        )
        if neg_mf > 0:
            mfi = 100 - (100 / (1 + pos_mf / neg_mf))
        elif pos_mf > 0:
            mfi = 100.0
    
    # Dollar volume averages
    dv = [c * v for c, v in zip(closes, volumes)]
    dv_5d = np.mean(dv[-5:]) if len(dv) >= 5 else None
    dv_20d = np.mean(dv[-20:]) if len(dv) >= 20 else None
    
    # Volume vs 20d average
    vol_20d = np.mean(volumes[-20:])
    vol_today = volumes[-1]
    vol_ratio = vol_today / vol_20d if vol_20d > 0 else 1.0
    
    # OBV trend
    obv = [0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    
    obv_5d = "Inflows" if len(obv) >= 6 and obv[-1] > obv[-6] else "Outflows"
    obv_20d = "Inflows" if len(obv) >= 21 and obv[-1] > obv[-21] else "Outflows"
    
    # Up/Down volume ratio (20d)
    if len(closes) >= 21:
        up_vol = sum(v for c1, c0, v in zip(closes[-20:], closes[-21:-1], volumes[-20:]) if c1 > c0)
        dn_vol = sum(v for c1, c0, v in zip(closes[-20:], closes[-21:-1], volumes[-20:]) if c1 <= c0)
        flow_ratio = up_vol / dn_vol if dn_vol > 0 else 999.0
    else:
        flow_ratio = 1.0
    
    # Net flow signal
    if flow_ratio > 1.3:
        net_signal = "[green]STRONG INFLOWS[/green]"
    elif flow_ratio > 1.1:
        net_signal = "[green]INFLOWS[/green]"
    elif flow_ratio < 0.7:
        net_signal = "[red]STRONG OUTFLOWS[/red]"
    elif flow_ratio < 0.9:
        net_signal = "[red]OUTFLOWS[/red]"
    else:
        net_signal = "[yellow]BALANCED[/yellow]"
    
    # Premium/Discount to NAV
    etf_info = fundamentals.get("etf_info", {})
    nav = etf_info.get("nav")
    current_price = closes[-1] if closes else None
    prem_disc = None
    if nav and current_price:
        try:
            prem_disc = ((current_price - float(nav)) / float(nav)) * 100
        except Exception:
            pass
    
    # MFI signal
    mfi_signal = ""
    if mfi is not None:
        if mfi > 80:
            mfi_signal = "[red]Overbought[/red]"
        elif mfi < 20:
            mfi_signal = "[green]Oversold[/green]"
        elif mfi < 40:
            mfi_signal = "[yellow]Weak[/yellow]"
        elif mfi > 60:
            mfi_signal = "[green]Strong[/green]"
        else:
            mfi_signal = "Neutral"
    
    # Build table
    table = Table(title="[bold]Fund Flows (Volume-Based)[/bold]", box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    
    table.add_row(
        "20d Net Flow",
        net_signal,
        "MFI (14d)",
        f"{mfi:.0f} {mfi_signal}" if mfi is not None else "N/A",
    )
    
    obv_5d_color = "green" if obv_5d == "Inflows" else "red"
    obv_20d_color = "green" if obv_20d == "Inflows" else "red"
    table.add_row(
        "OBV 5d",
        f"[{obv_5d_color}]{obv_5d}[/{obv_5d_color}]",
        "OBV 20d",
        f"[{obv_20d_color}]{obv_20d}[/{obv_20d_color}]",
    )
    
    table.add_row(
        "$ Vol 5d Avg",
        f"${dv_5d / 1e6:.0f}M" if dv_5d else "N/A",
        "$ Vol 20d Avg",
        f"${dv_20d / 1e6:.0f}M" if dv_20d else "N/A",
    )
    
    vol_color = "green" if vol_ratio > 1.2 else "red" if vol_ratio < 0.8 else "yellow"
    table.add_row(
        "Vol vs 20d Avg",
        f"[{vol_color}]{vol_ratio:.2f}x[/{vol_color}]",
        "Up/Down Ratio",
        f"{'[green]' if flow_ratio > 1.1 else '[red]' if flow_ratio < 0.9 else ''}{flow_ratio:.2f}x{'[/green]' if flow_ratio > 1.1 else '[/red]' if flow_ratio < 0.9 else ''}",
    )
    
    if prem_disc is not None:
        pd_color = "green" if prem_disc > 0.1 else "red" if prem_disc < -0.1 else "yellow"
        table.add_row(
            "NAV Premium",
            f"[{pd_color}]{prem_disc:+.2f}%[/{pd_color}]",
            "",
            "",
        )
    
    console.print()
    console.print(table)


def _show_refinancing_wall(console: Console, settings, symbol: str):
    """Fetch bond ETF holdings and display maturity (refinancing) wall."""
    import re
    import requests
    from collections import defaultdict
    from datetime import datetime

    try:
        url = f"https://financialmodelingprep.com/api/v3/etf-holder/{symbol}"
        resp = requests.get(
            url, params={"apikey": settings.fmp_api_key}, timeout=20
        )
        if not resp.ok:
            return
        holdings = resp.json()
        if not holdings or not isinstance(holdings, list):
            return
    except Exception:
        return

    # Parse maturity year from bond names (format: "COMPANY 144A MM/DD/YYYY")
    date_pattern = re.compile(r"(\d{2}/\d{2}/(\d{4}))\s*$")
    by_year: dict[int, dict] = defaultdict(lambda: {"count": 0, "mv": 0, "weight": 0.0})
    total_mv = 0
    parsed = 0
    current_year = datetime.now().year

    for h in holdings:
        name = h.get("name", "")
        mv = h.get("marketValue", 0) or 0
        wt = h.get("weightPercentage", 0) or 0
        match = date_pattern.search(name)
        if match:
            year = int(match.group(2))
            by_year[year]["count"] += 1
            by_year[year]["mv"] += mv
            by_year[year]["weight"] += wt
            total_mv += mv
            parsed += 1

    if parsed < 10 or total_mv <= 0:
        return  # Not enough bond data

    # Build display — only show meaningful years
    years_to_show = sorted(y for y in by_year if current_year <= y <= current_year + 12)
    if not years_to_show:
        return

    max_mv = max(by_year[y]["mv"] for y in years_to_show) if years_to_show else 1

    # Near-term / mid-term buckets
    near_mv = sum(by_year[y]["mv"] for y in years_to_show if y <= current_year + 2)
    mid_mv = sum(by_year[y]["mv"] for y in years_to_show if current_year + 3 <= y <= current_year + 5)
    near_pct = near_mv / total_mv * 100 if total_mv else 0
    mid_pct = mid_mv / total_mv * 100 if total_mv else 0

    # Build table
    table = Table(
        title=f"[bold]Refinancing Wall ({parsed:,} bonds, ${total_mv / 1e9:.1f}B)[/bold]",
        box=None, padding=(0, 1),
    )
    table.add_column("Year", style="bold", min_width=6, no_wrap=True)
    table.add_column("Bonds", justify="right", min_width=6)
    table.add_column("Mkt Value", justify="right", min_width=10)
    table.add_column("% Fund", justify="right", min_width=7)
    table.add_column("Distribution", min_width=30)

    for year in years_to_show:
        d = by_year[year]
        pct = d["mv"] / total_mv * 100
        bar_len = int(d["mv"] / max_mv * 25)

        # Color: red for near-term (pressure), yellow for mid-term, green for far
        if year <= current_year + 1:
            color = "red"
            bar_char = "█"
        elif year <= current_year + 3:
            color = "yellow"
            bar_char = "█"
        else:
            color = "green"
            bar_char = "█"

        bar = f"[{color}]{bar_char * bar_len}[/{color}]"
        year_str = f"[{color}]{year}[/{color}]"
        pct_str = f"[{color}]{pct:.1f}%[/{color}]"

        table.add_row(
            year_str,
            str(d["count"]),
            f"${d['mv'] / 1e9:.2f}B",
            pct_str,
            bar,
        )

    console.print()
    console.print(table)

    # Summary line
    near_color = "red" if near_pct > 15 else "yellow" if near_pct > 8 else "green"
    console.print(
        f"\n  [{near_color}]Near-term (≤{current_year + 2}): ${near_mv / 1e9:.1f}B ({near_pct:.0f}%)[/{near_color}]"
        f"  |  Mid-term ({current_year + 3}-{current_year + 5}): ${mid_mv / 1e9:.1f}B ({mid_pct:.0f}%)"
    )


def _show_stock_fundamentals(console: Console, fundamentals: dict):
    """Display stock fundamentals table."""
    profile = fundamentals.get("profile", {})
    metrics = fundamentals.get("metrics", {})
    ratios = fundamentals.get("ratios", {})
    
    table = Table(title="[bold]Fundamentals[/bold]", box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    
    def fmt(val, pct=False, billions=False):
        if val is None:
            return "N/A"
        try:
            v = float(val)
            if billions:
                return f"${v/1e9:.1f}B"
            if pct:
                return f"{v:.1f}%"
            return f"{v:.2f}"
        except Exception:
            return str(val)[:15]
    
    mkt_cap = profile.get("mktCap")
    pe = ratios.get("peRatioTTM") or profile.get("pe")
    ps = ratios.get("priceToSalesRatioTTM")
    pb = ratios.get("priceToBookRatioTTM")
    
    profit_margin = ratios.get("netProfitMarginTTM")
    roe = ratios.get("returnOnEquityTTM")
    roa = ratios.get("returnOnAssetsTTM")
    
    table.add_row("Market Cap", fmt(mkt_cap, billions=True), "P/E Ratio", fmt(pe))
    table.add_row("P/S Ratio", fmt(ps), "P/B Ratio", fmt(pb))
    table.add_row("Profit Margin", fmt(profit_margin, pct=True), "ROE", fmt(roe, pct=True))
    table.add_row("ROA", fmt(roa, pct=True), "Sector", profile.get("sector", "N/A")[:15])
    
    console.print()
    console.print(table)


def _show_technicals(console: Console, technicals: dict):
    """Display technicals table."""
    table = Table(title="[bold]Technical Levels[/bold]", box=None, padding=(0, 2))
    table.add_column("Indicator", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Signal")
    
    current = technicals.get("current", 0)
    
    # Moving averages
    ma_20 = technicals.get("ma_20")
    ma_50 = technicals.get("ma_50")
    ma_200 = technicals.get("ma_200")
    
    def ma_signal(ma):
        if not ma or not current:
            return "[dim]N/A[/dim]"
        if current > ma:
            return "[green]Above[/green]"
        return "[red]Below[/red]"
    
    table.add_row("20-Day MA", f"${ma_20:,.2f}" if ma_20 else "N/A", ma_signal(ma_20))
    table.add_row("50-Day MA", f"${ma_50:,.2f}" if ma_50 else "N/A", ma_signal(ma_50))
    table.add_row("200-Day MA", f"${ma_200:,.2f}" if ma_200 else "N/A", ma_signal(ma_200))
    
    # RSI
    rsi = technicals.get("rsi")
    rsi_signal = "[dim]N/A[/dim]"
    if rsi:
        if rsi > 70:
            rsi_signal = "[red]Overbought[/red]"
        elif rsi < 30:
            rsi_signal = "[green]Oversold[/green]"
        else:
            rsi_signal = "[yellow]Neutral[/yellow]"
    
    table.add_row("RSI (14)", f"{rsi:.1f}" if rsi else "N/A", rsi_signal)
    
    console.print()
    console.print(table)


def _generate_chart(symbol: str, price_data: dict, technicals: dict) -> str | None:
    """Generate institutional-grade price chart (candlestick + volume + RSI)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib.ticker as mticker
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
        import numpy as np
        import pandas as pd
        from datetime import datetime
        import tempfile
        from pathlib import Path

        df = technicals.get("df")
        if df is None or df.empty:
            return None

        # Use last 6 months of trading days
        df = df.tail(126).copy().reset_index(drop=True)

        if len(df) < 20:
            return None

        # ── Compute indicators ─────────────────────────────────────────────
        closes = df["close"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        opens = df["open"].values.astype(float)
        volumes = df["volume"].values.astype(float)

        # Moving averages
        df["sma20"] = df["close"].rolling(20).mean()
        df["sma50"] = df["close"].rolling(50).mean()

        # Bollinger Bands (20, 2) — subtle envelope only
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["sma20"] + 2 * bb_std
        df["bb_lower"] = df["sma20"] - 2 * bb_std

        # RSI (14-day, Wilder smoothing)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Volume MA for reference
        df["vol_ma20"] = df["volume"].rolling(20).mean()

        # ── Color palette (institutional dark) ─────────────────────────────
        BG = "#0a0e17"
        PANEL_BG = "#0f1318"
        GRID = "#1a2030"
        TEXT = "#c9d1d9"
        TEXT_DIM = "#6e7681"
        GREEN = "#00d26a"
        RED = "#ff4757"
        BLUE = "#3b82f6"
        CYAN = "#22d3ee"
        ORANGE = "#f59e0b"
        MAGENTA = "#a855f7"
        WHITE = "#e6edf3"
        BB_FILL = "#3b82f620"

        # ── Figure layout: 3 panels (price, volume, RSI) ──────────────────
        fig = plt.figure(figsize=(14, 9), facecolor=BG)
        gs = fig.add_gridspec(
            3, 1,
            height_ratios=[5, 1.2, 1.5],
            hspace=0.03,
            left=0.07, right=0.93, top=0.92, bottom=0.06,
        )
        ax_price = fig.add_subplot(gs[0])
        ax_vol = fig.add_subplot(gs[1], sharex=ax_price)
        ax_rsi = fig.add_subplot(gs[2], sharex=ax_price)

        for ax in (ax_price, ax_vol, ax_rsi):
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_DIM, labelsize=9)
            ax.grid(True, alpha=0.15, color=GRID, linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_color(GRID)
                spine.set_linewidth(0.5)

        dates = mdates.date2num(df["date"])

        # ── Panel 1: Candlestick + Overlays ────────────────────────────────
        up = closes >= opens
        dn = ~up
        candle_width = 0.6

        # Candle bodies
        ax_price.bar(
            dates[up], (closes - opens)[up], candle_width,
            bottom=opens[up], color=GREEN, alpha=0.9, linewidth=0, zorder=3,
        )
        ax_price.bar(
            dates[dn], (opens - closes)[dn], candle_width,
            bottom=closes[dn], color=RED, alpha=0.9, linewidth=0, zorder=3,
        )

        # Candle wicks
        ax_price.vlines(
            dates[up], lows[up], highs[up],
            color=GREEN, linewidth=0.7, zorder=2,
        )
        ax_price.vlines(
            dates[dn], lows[dn], highs[dn],
            color=RED, linewidth=0.7, zorder=2,
        )

        # Moving averages
        ax_price.plot(dates, df["sma20"], color=BLUE, linewidth=1.3, alpha=0.85, label="SMA 20")
        if len(df) >= 50:
            ax_price.plot(dates, df["sma50"], color=ORANGE, linewidth=1.3, alpha=0.85, label="SMA 50")

        # Bollinger Bands — subtle envelope (no edge lines, just fill)
        bb_valid = df["bb_upper"].notna()
        if bb_valid.any():
            ax_price.fill_between(
                dates[bb_valid],
                df["bb_upper"][bb_valid],
                df["bb_lower"][bb_valid],
                color=BLUE, alpha=0.06, label="BBand (2σ)",
            )

        # Support / Resistance levels
        support = technicals.get("support")
        resistance = technicals.get("resistance")
        if support:
            ax_price.axhline(y=support, color=GREEN, linestyle=":", linewidth=0.8, alpha=0.6, zorder=1)
            ax_price.text(
                dates[-1] + 1, support, f" S ${support:.2f}",
                color=GREEN, fontsize=8, va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=BG, edgecolor=GREEN, alpha=0.8, linewidth=0.5),
            )
        if resistance:
            ax_price.axhline(y=resistance, color=RED, linestyle=":", linewidth=0.8, alpha=0.6, zorder=1)
            ax_price.text(
                dates[-1] + 1, resistance, f" R ${resistance:.2f}",
                color=RED, fontsize=8, va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=BG, edgecolor=RED, alpha=0.8, linewidth=0.5),
            )

        # Last price annotation
        last_price = closes[-1]
        last_color = GREEN if closes[-1] >= opens[-1] else RED
        ax_price.axhline(y=last_price, color=last_color, linewidth=0.5, alpha=0.4, linestyle="-", zorder=1)
        ax_price.text(
            dates[-1] + 1.5, last_price, f" ${last_price:.2f}",
            color=last_color, fontsize=9, va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=last_color, edgecolor="none", alpha=0.15),
        )

        # Y-axis: tight to price range with 2% padding
        price_min = float(df["low"].min())
        price_max = float(df["high"].max())
        price_range = price_max - price_min
        pad = price_range * 0.05
        ax_price.set_ylim(price_min - pad, price_max + pad)
        ax_price.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))

        # Legend — compact, out of the way
        legend_elements = [
            Line2D([0], [0], color=BLUE, lw=1.3, label="SMA 20"),
            Line2D([0], [0], color=ORANGE, lw=1.3, label="SMA 50"),
            mpatches.Patch(color=BLUE, alpha=0.15, label="BBand 2σ"),
        ]
        ax_price.legend(
            handles=legend_elements, loc="upper left", fontsize=7.5,
            framealpha=0.7, facecolor=BG, edgecolor=GRID, labelcolor=TEXT_DIM,
            ncol=3, columnspacing=1.2, handlelength=1.5,
        )
        ax_price.tick_params(labelbottom=False)
        ax_price.set_ylabel("Price", fontsize=9, color=TEXT_DIM, labelpad=8)

        # ── Panel 2: Volume ────────────────────────────────────────────────
        vol_colors = np.where(up, GREEN, RED)
        ax_vol.bar(dates, volumes, candle_width, color=vol_colors, alpha=0.6, zorder=2)
        vol_ma = df["vol_ma20"]
        vol_ma_valid = vol_ma.notna()
        if vol_ma_valid.any():
            ax_vol.plot(dates[vol_ma_valid], vol_ma[vol_ma_valid], color=BLUE, linewidth=1, alpha=0.7)

        ax_vol.set_ylabel("Vol", fontsize=8, color=TEXT_DIM, labelpad=8)
        ax_vol.tick_params(labelbottom=False)

        # Format volume axis (M/B)
        def _vol_fmt(x, _):
            if x >= 1e9:
                return f"{x / 1e9:.1f}B"
            if x >= 1e6:
                return f"{x / 1e6:.0f}M"
            if x >= 1e3:
                return f"{x / 1e3:.0f}K"
            return str(int(x))
        ax_vol.yaxis.set_major_formatter(mticker.FuncFormatter(_vol_fmt))

        # ── Panel 3: RSI ───────────────────────────────────────────────────
        rsi_valid = df["rsi"].notna()
        if rsi_valid.any():
            ax_rsi.plot(dates[rsi_valid], df["rsi"][rsi_valid], color=CYAN, linewidth=1.2)
            ax_rsi.axhline(70, color=RED, linewidth=0.6, alpha=0.5, linestyle="--")
            ax_rsi.axhline(30, color=GREEN, linewidth=0.6, alpha=0.5, linestyle="--")
            ax_rsi.axhline(50, color=TEXT_DIM, linewidth=0.4, alpha=0.3, linestyle="-")

            # Shade overbought/oversold zones
            ax_rsi.fill_between(
                dates[rsi_valid], 70, df["rsi"][rsi_valid],
                where=df["rsi"][rsi_valid] >= 70,
                color=RED, alpha=0.1, interpolate=True,
            )
            ax_rsi.fill_between(
                dates[rsi_valid], 30, df["rsi"][rsi_valid],
                where=df["rsi"][rsi_valid] <= 30,
                color=GREEN, alpha=0.1, interpolate=True,
            )

            # Current RSI value label
            rsi_last = float(df["rsi"].iloc[-1]) if pd.notna(df["rsi"].iloc[-1]) else None
            if rsi_last is not None:
                rsi_color = RED if rsi_last >= 70 else GREEN if rsi_last <= 30 else CYAN
                ax_rsi.text(
                    dates[-1] + 1, rsi_last, f" {rsi_last:.1f}",
                    color=rsi_color, fontsize=8, va="center", fontweight="bold",
                )

        ax_rsi.set_ylim(10, 90)
        ax_rsi.set_ylabel("RSI", fontsize=8, color=TEXT_DIM, labelpad=8)
        ax_rsi.set_yticks([30, 50, 70])

        # X-axis formatting
        ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax_rsi.xaxis.set_major_locator(mdates.MonthLocator())
        ax_rsi.tick_params(axis="x", labelsize=8, rotation=0)

        # ── Title bar ─────────────────────────────────────────────────────
        current = technicals.get("current", 0)
        change_pct = price_data.get("quote", {}).get("changesPercentage", 0)
        change_val = price_data.get("quote", {}).get("change", 0)
        chg_color = GREEN if change_pct >= 0 else RED
        chg_sign = "+" if change_pct >= 0 else ""

        # Main title
        fig.text(
            0.07, 0.96, symbol,
            fontsize=20, fontweight="bold", color=WHITE,
            fontfamily="monospace",
        )
        fig.text(
            0.07 + len(symbol) * 0.018, 0.96,
            f"   ${current:,.2f}   {chg_sign}{change_val:,.2f} ({chg_sign}{change_pct:.2f}%)",
            fontsize=13, color=chg_color,
            fontfamily="monospace",
        )

        # Info bar (right-aligned)
        rsi_val = technicals.get("rsi")
        vol_val = technicals.get("volatility")
        info_parts = []
        if rsi_val is not None:
            info_parts.append(f"RSI {rsi_val:.1f}")
        if vol_val is not None:
            info_parts.append(f"Vol {vol_val:.1f}%")
        info_parts.append(datetime.now().strftime("%b %d, %Y"))
        fig.text(
            0.93, 0.96, "  |  ".join(info_parts),
            fontsize=9, color=TEXT_DIM, ha="right",
            fontfamily="monospace",
        )

        # Branding
        fig.text(
            0.93, 0.015, "LOX FUND",
            fontsize=8, color=TEXT_DIM, ha="right", fontweight="bold",
            fontfamily="monospace", alpha=0.5,
        )

        # ── Save ───────────────────────────────────────────────────────────
        output_dir = Path(tempfile.gettempdir()) / "lox_charts"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{symbol}_{timestamp}.png"

        fig.savefig(
            output_path, dpi=180, facecolor=BG, edgecolor="none",
            bbox_inches="tight", pad_inches=0.15,
        )
        plt.close(fig)

        return str(output_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def _open_chart(path: str):
    """Open chart in system viewer."""
    import subprocess
    import sys
    
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        elif sys.platform == "win32":
            subprocess.run(["start", path], shell=True, check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception:
        pass


def _compute_refinancing_wall(settings, symbol: str) -> dict | None:
    """Compute refinancing wall summary for LLM context."""
    import re
    import requests
    from collections import defaultdict
    from datetime import datetime

    try:
        url = f"https://financialmodelingprep.com/api/v3/etf-holder/{symbol}"
        resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=20)
        if not resp.ok:
            return None
        holdings = resp.json()
    except Exception:
        return None

    date_pattern = re.compile(r"(\d{2}/\d{2}/(\d{4}))\s*$")
    by_year: dict[int, float] = defaultdict(float)
    total_mv = 0
    current_year = datetime.now().year

    for h in holdings:
        name = h.get("name", "")
        mv = h.get("marketValue", 0) or 0
        match = date_pattern.search(name)
        if match:
            year = int(match.group(2))
            by_year[year] += mv
            total_mv += mv

    if total_mv <= 0:
        return None

    near_term = sum(v for y, v in by_year.items() if y <= current_year + 2)
    mid_term = sum(v for y, v in by_year.items() if current_year + 3 <= y <= current_year + 5)

    wall = {}
    for y in sorted(by_year):
        if current_year <= y <= current_year + 10:
            wall[str(y)] = f"${by_year[y] / 1e9:.2f}B ({by_year[y] / total_mv * 100:.1f}%)"

    return {
        "maturity_by_year": wall,
        "near_term_pct": round(near_term / total_mv * 100, 1),
        "mid_term_pct": round(mid_term / total_mv * 100, 1),
        "total_bonds_parsed": sum(1 for _ in by_year),
        "total_market_value": f"${total_mv / 1e9:.1f}B",
        "peak_year": str(max(by_year, key=by_year.get)),
        "peak_year_pct": round(by_year[max(by_year, key=by_year.get)] / total_mv * 100, 1),
    }


def _compute_flow_context(price_data: dict) -> dict | None:
    """Compute ETF flow metrics for LLM context."""
    import numpy as np
    
    historical = price_data.get("historical", [])
    if len(historical) < 21:
        return None
    
    hist = list(reversed(historical[:60]))
    closes = [h["close"] for h in hist]
    volumes = [h["volume"] for h in hist]
    highs = [h["high"] for h in hist]
    lows = [h["low"] for h in hist]
    
    # MFI
    mfi = None
    if len(hist) >= 15:
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        pos_mf = sum(typical_prices[i] * volumes[i] for i in range(-14, 0) if typical_prices[i] > typical_prices[i - 1])
        neg_mf = sum(typical_prices[i] * volumes[i] for i in range(-14, 0) if typical_prices[i] <= typical_prices[i - 1])
        mfi = 100 - (100 / (1 + pos_mf / neg_mf)) if neg_mf > 0 else (100.0 if pos_mf > 0 else 50.0)
    
    # Dollar volume
    dv = [c * v for c, v in zip(closes, volumes)]
    dv_5d = float(np.mean(dv[-5:])) if len(dv) >= 5 else None
    dv_20d = float(np.mean(dv[-20:])) if len(dv) >= 20 else None
    
    # Up/Down ratio
    if len(closes) >= 21:
        up_vol = sum(v for c1, c0, v in zip(closes[-20:], closes[-21:-1], volumes[-20:]) if c1 > c0)
        dn_vol = sum(v for c1, c0, v in zip(closes[-20:], closes[-21:-1], volumes[-20:]) if c1 <= c0)
        flow_ratio = up_vol / dn_vol if dn_vol > 0 else 10.0
    else:
        flow_ratio = 1.0
    
    if flow_ratio > 1.3:
        signal = "STRONG INFLOWS"
    elif flow_ratio > 1.1:
        signal = "INFLOWS"
    elif flow_ratio < 0.7:
        signal = "STRONG OUTFLOWS"
    elif flow_ratio < 0.9:
        signal = "OUTFLOWS"
    else:
        signal = "BALANCED"
    
    return {
        "net_flow_signal_20d": signal,
        "up_down_volume_ratio": round(flow_ratio, 2),
        "mfi_14d": round(mfi, 1) if mfi else None,
        "dollar_vol_5d_avg": f"${dv_5d / 1e6:.0f}M" if dv_5d else None,
        "dollar_vol_20d_avg": f"${dv_20d / 1e6:.0f}M" if dv_20d else None,
    }


def _show_llm_analysis(console: Console, settings, symbol: str, price_data: dict, fundamentals: dict, technicals: dict):
    """Show LLM analysis."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Analyzing...[/bold cyan]"),
        transient=True,
    ) as progress:
        progress.add_task("llm", total=None)
        
        try:
            from ai_options_trader.llm.core.analyst import llm_analyze_regime
            
            # Build snapshot
            snapshot = {
                "symbol": symbol,
                "price": technicals.get("current"),
                "change_pct": price_data.get("quote", {}).get("changesPercentage"),
                "52w_high": technicals.get("high_52w"),
                "52w_low": technicals.get("low_52w"),
                "rsi": technicals.get("rsi"),
                "trend": technicals.get("trend"),
                "volatility": technicals.get("volatility"),
                "support": technicals.get("support"),
                "resistance": technicals.get("resistance"),
            }
            
            profile = fundamentals.get("profile", {})
            if profile:
                snapshot["sector"] = profile.get("sector")
                snapshot["industry"] = profile.get("industry")
                snapshot["mkt_cap"] = profile.get("mktCap")
            
            ratios = fundamentals.get("ratios", {})
            if ratios:
                snapshot["pe_ratio"] = ratios.get("peRatioTTM")
                snapshot["profit_margin"] = ratios.get("netProfitMarginTTM")
            
            # Add ETF-specific context (flows, AUM, yield)
            etf_info = fundamentals.get("etf_info", {})
            is_etf = profile.get("isEtf", False) or bool(etf_info)
            if is_etf:
                snapshot["asset_type"] = "ETF"
                if etf_info:
                    snapshot["aum"] = etf_info.get("totalAssets")
                    snapshot["expense_ratio"] = etf_info.get("expenseRatio")
                    snapshot["etf_yield"] = etf_info.get("yield")
                    snapshot["holdings_count"] = etf_info.get("holdingsCount")
                    snapshot["asset_class"] = etf_info.get("assetClass")
                # Compute flow signals for LLM context
                flow_ctx = _compute_flow_context(price_data)
                if flow_ctx:
                    snapshot["fund_flows"] = flow_ctx
                # Refinancing wall for bond ETFs
                refi_wall = _compute_refinancing_wall(settings, symbol)
                if refi_wall:
                    snapshot["refinancing_wall"] = refi_wall
            
            analysis = llm_analyze_regime(
                settings=settings,
                domain="growth",  # Use growth domain for equities
                snapshot=snapshot,
                regime_label=f"{symbol} Analysis",
                include_news=True,
                include_prices=False,  # Already have price data
                include_calendar=True,
                ticker=symbol,  # Ticker-specific scenarios & trade ideas
            )
            
            console.print()
            from rich.markdown import Markdown
            console.print(Panel(
                Markdown(analysis),
                title="[bold]AI Analysis[/bold]",
                border_style="blue",
            ))
            
        except Exception as e:
            console.print(f"\n[dim]Analysis unavailable: {e}[/dim]")

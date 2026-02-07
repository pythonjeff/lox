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
    """Fetch fundamental data."""
    try:
        import requests
        
        if not settings.fmp_api_key:
            return {}
        
        result = {}
        
        # Company profile
        url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
        resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=15)
        if resp.ok:
            data = resp.json()
            if data and isinstance(data, list):
                result["profile"] = data[0]
        
        # Key metrics
        url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}"
        resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=15)
        if resp.ok:
            data = resp.json()
            if data and isinstance(data, list):
                result["metrics"] = data[0]
        
        # Ratios
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
    """Display fundamentals table."""
    profile = fundamentals.get("profile", {})
    metrics = fundamentals.get("metrics", {})
    ratios = fundamentals.get("ratios", {})
    
    table = Table(title="[bold]Fundamentals[/bold]", box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    
    # Format values
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
    
    rev_growth = metrics.get("revenuePerShareTTM")
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
    """Generate matplotlib price chart."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        import tempfile
        from pathlib import Path
        
        df = technicals.get("df")
        if df is None or df.empty:
            return None
        
        # Use last 6 months
        df = df.tail(126).copy()
        
        # LOX Fund styling
        plt.style.use('dark_background')
        bg_color = '#0d1117'
        grid_color = '#21262d'
        price_color = '#0066ff'
        ma_colors = {'20': '#22c55e', '50': '#f59e0b', '200': '#ef4444'}
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        fig.patch.set_facecolor(bg_color)
        ax1.set_facecolor(bg_color)
        ax2.set_facecolor(bg_color)
        
        # Price chart
        ax1.plot(df["date"], df["close"], color=price_color, linewidth=2, label="Price")
        ax1.fill_between(df["date"], df["close"], alpha=0.1, color=price_color)
        
        # Moving averages
        if len(df) >= 20:
            df["ma20"] = df["close"].rolling(20).mean()
            ax1.plot(df["date"], df["ma20"], color=ma_colors['20'], linewidth=1, linestyle='--', label="20 MA", alpha=0.7)
        if len(df) >= 50:
            df["ma50"] = df["close"].rolling(50).mean()
            ax1.plot(df["date"], df["ma50"], color=ma_colors['50'], linewidth=1, linestyle='--', label="50 MA", alpha=0.7)
        
        # Support/Resistance
        support = technicals.get("support")
        resistance = technicals.get("resistance")
        if support:
            ax1.axhline(y=support, color='#22c55e', linestyle=':', linewidth=1, alpha=0.5)
            ax1.text(df["date"].iloc[-1], support, f" S: ${support:.0f}", color='#22c55e', fontsize=9, va='center')
        if resistance:
            ax1.axhline(y=resistance, color='#ef4444', linestyle=':', linewidth=1, alpha=0.5)
            ax1.text(df["date"].iloc[-1], resistance, f" R: ${resistance:.0f}", color='#ef4444', fontsize=9, va='center')
        
        ax1.set_ylabel("Price ($)", fontsize=11, color='white')
        ax1.legend(loc='upper left', framealpha=0.9, facecolor=bg_color)
        ax1.grid(True, alpha=0.3, color=grid_color)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        
        # Volume chart
        if "volume" in df.columns:
            colors = ['#22c55e' if df["close"].iloc[i] >= df["open"].iloc[i] else '#ef4444' 
                     for i in range(len(df))]
            ax2.bar(df["date"], df["volume"], color=colors, alpha=0.7, width=0.8)
            ax2.set_ylabel("Volume", fontsize=11, color='white')
            ax2.grid(True, alpha=0.3, color=grid_color)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        
        # Title
        current = technicals.get("current", 0)
        change_pct = price_data.get("quote", {}).get("changesPercentage", 0)
        change_color = '#22c55e' if change_pct >= 0 else '#ef4444'
        
        fig.suptitle(f"{symbol}  ${current:,.2f}  ({change_pct:+.2f}%)", 
                    fontsize=16, fontweight='bold', color='white', y=0.98)
        
        # LOX branding
        fig.text(0.99, 0.01, 'LOX FUND', fontsize=9, color='#3d444d',
                ha='right', va='bottom', alpha=0.7, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save
        output_dir = Path(tempfile.gettempdir()) / "lox_charts"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{symbol}_{timestamp}.png"
        
        fig.savefig(output_path, dpi=150, facecolor=bg_color, edgecolor='none',
                   bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        
        return str(output_path)
    except Exception as e:
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
            
            analysis = llm_analyze_regime(
                settings=settings,
                domain="growth",  # Use growth domain for equities
                snapshot=snapshot,
                regime_label=f"{symbol} Analysis",
                include_news=True,
                include_prices=False,  # Already have price data
                include_calendar=True,
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

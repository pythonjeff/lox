from __future__ import annotations

from datetime import date
import typer

app = typer.Typer(
    add_completion=False, 
    help="""Lox Capital CLI — Hedge fund research & portfolio management.

\b
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. FUND INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  lox status              Portfolio health (NAV, P&L, cash)
  lox nav snapshot        NAV and investor ledger
  lox account             Account summary

\b
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. MACRO DASHBOARD & REGIMES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  lox dashboard           All regime pillars at a glance
  lox labs volatility     Volatility regime
  lox labs funding        Funding/liquidity regime
  lox labs fiscal         Fiscal regime

\b
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. PORTFOLIO ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  lox labs mc-v01         Monte Carlo simulation
  lox model predict       ML regime predictions
  lox labs stress         Portfolio stress tests

\b
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. TRADE IDEAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  lox scan-extremes       Overbought/oversold opportunities
  lox suggest             Quick trade suggestions
  lox ideas catalyst      Event-driven ideas
  lox ideas screen        ML-ranked ticker screen

\b
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. RESEARCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  lox research -t NVDA    Full ticker research report
  lox chat                Interactive research chat
  lox scan -t NVDA        Options chain scanner

\b
Run 'lox <command> --help' for details.
"""
)

# ---------------------------------------------------------------------------
# TOP-LEVEL COMMANDS (flat, most commonly used)
# ---------------------------------------------------------------------------

@app.command("scan")
def scan_cmd(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
    want: str = typer.Option("put", "--want", "-w", help="call or put"),
    min_days: int = typer.Option(30, "--min-days", help="Min DTE"),
    max_days: int = typer.Option(365, "--max-days", help="Max DTE"),
    filter_delta: float = typer.Option(None, "--delta", "-d", help="Filter by delta (e.g. 0.3 for ~30 delta)"),
    max_iv: float = typer.Option(None, "--max-iv", help="Max IV (e.g. 0.5 for 50%)"),
    min_iv: float = typer.Option(None, "--min-iv", help="Min IV (e.g. 0.3 for 30%)"),
    max_theta: float = typer.Option(None, "--max-theta", help="Max theta (daily decay, e.g. -0.05)"),
    show: int = typer.Option(30, "--show", "-n", help="Number of results"),
):
    """
    Options chain scanner.
    
    Examples:
        lox scan -t NVDA
        lox scan -t CRWV --want put --min-days 100
        lox scan -t MSTR --delta 0.2 --max-iv 0.8
    """
    from ai_options_trader.config import load_settings
    from ai_options_trader.data.alpaca import fetch_option_chain, make_clients
    from ai_options_trader.utils.occ import parse_occ_option_symbol
    from rich.console import Console
    from rich.table import Table
    
    settings = load_settings()
    _, data = make_clients(settings)
    
    w = want.strip().lower()
    if w not in {"call", "put"}:
        w = "put"
    
    console = Console()
    t = ticker.upper()
    today = date.today()
    
    # Build filter description
    filters = []
    if filter_delta is not None:
        filters.append(f"Δ~{abs(filter_delta):.0%}")
    if min_iv is not None or max_iv is not None:
        iv_range = f"IV:{min_iv or 0:.0%}-{max_iv or 999:.0%}"
        filters.append(iv_range)
    if max_theta is not None:
        filters.append(f"θ≥{max_theta}")
    filter_str = f" | {', '.join(filters)}" if filters else ""
    console.print(f"\n[bold cyan]{t} {w.upper()}s[/bold cyan] | DTE: {min_days}-{max_days}{filter_str}\n")
    
    chain = fetch_option_chain(data, t, feed=settings.alpaca_options_feed)
    if not chain:
        console.print("[yellow]No options data[/yellow]")
        return
    
    opts = []
    for opt in chain.values():
        symbol = str(getattr(opt, "symbol", ""))
        if not symbol:
            continue
        try:
            expiry, opt_type, strike = parse_occ_option_symbol(symbol, t)
            if opt_type != w:
                continue
            dte = (expiry - today).days
            if dte < min_days or dte > max_days:
                continue
            
            greeks = getattr(opt, "greeks", None)
            opt_delta = getattr(greeks, "delta", None) if greeks else None
            opt_theta = getattr(greeks, "theta", None) if greeks else None
            opt_gamma = getattr(greeks, "gamma", None) if greeks else None
            opt_vega = getattr(greeks, "vega", None) if greeks else None
            opt_iv = getattr(opt, "implied_volatility", None)
            
            quote = getattr(opt, "latest_quote", None)
            bid = getattr(quote, "bid_price", None) if quote else None
            ask = getattr(quote, "ask_price", None) if quote else None
            trade = getattr(opt, "latest_trade", None)
            last = getattr(trade, "price", None) if trade else None
            
            opts.append({
                "symbol": symbol, "strike": strike, "dte": dte,
                "delta": float(opt_delta) if opt_delta else None,
                "theta": float(opt_theta) if opt_theta else None,
                "gamma": float(opt_gamma) if opt_gamma else None,
                "vega": float(opt_vega) if opt_vega else None,
                "iv": float(opt_iv) if opt_iv else None,
                "bid": float(bid) if bid else None,
                "ask": float(ask) if ask else None,
                "last": float(last) if last else None,
            })
        except Exception:
            continue
    
    if not opts:
        console.print(f"[yellow]No {w}s in {min_days}-{max_days} DTE[/yellow]")
        return
    
    # Apply filters
    if filter_delta is not None:
        target_delta = abs(filter_delta)
        tolerance = 0.05  # +/- 5 delta points
        opts = [o for o in opts if o["delta"] is not None and abs(abs(o["delta"]) - target_delta) <= tolerance]
    
    if min_iv is not None:
        opts = [o for o in opts if o["iv"] is not None and o["iv"] >= min_iv]
    
    if max_iv is not None:
        opts = [o for o in opts if o["iv"] is not None and o["iv"] <= max_iv]
    
    if max_theta is not None:
        # Theta is negative, so max_theta=-0.05 means "theta >= -0.05" (less decay)
        opts = [o for o in opts if o["theta"] is not None and o["theta"] >= max_theta]
    
    if not opts:
        console.print(f"[yellow]No {w}s matching filters[/yellow]")
        return
    
    # Get current stock price for ATM sorting
    from ai_options_trader.data.quotes import fetch_stock_last_prices
    stock_price = None
    try:
        prices, _, _ = fetch_stock_last_prices(settings=settings, symbols=[t])
        stock_price = prices.get(t)
    except Exception:
        pass
    
    if stock_price:
        # Sort by distance from ATM, then by DTE
        opts.sort(key=lambda x: (abs(x["strike"] - stock_price), x["dte"]))
        console.print(f"[dim]Stock: ${stock_price:.2f} | Found {len(opts)} contracts (sorted ATM first)[/dim]\n")
    else:
        # Fallback to strike sort
        opts.sort(key=lambda x: (x["strike"], x["dte"]))
        console.print(f"[dim]Found {len(opts)} contracts[/dim]\n")
    
    opts = opts[:show]
    
    table = Table(show_header=True, expand=False)
    table.add_column("Symbol", style="cyan")
    table.add_column("Strike", justify="right")
    table.add_column("DTE", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("IV", justify="right")
    table.add_column("Theta", justify="right")
    table.add_column("Bid", justify="right")
    table.add_column("Ask", justify="right", style="yellow")
    
    for o in opts:
        table.add_row(
            o["symbol"],
            f"${o['strike']:.2f}",
            str(o["dte"]),
            f"{o['delta']:+.2f}" if o["delta"] else "—",
            f"{o['iv']:.0%}" if o["iv"] else "—",
            f"{o['theta']:.3f}" if o["theta"] else "—",
            f"${o['bid']:.2f}" if o["bid"] else "—",
            f"${o['ask']:.2f}" if o["ask"] else "—",
        )
    console.print(table)


@app.command("research")
def research_cmd(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
    llm: bool = typer.Option(False, "--llm", help="Include LLM analysis"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick mode (skip SEC/news)"),
):
    """
    Comprehensive research report on a ticker.
    
    Provides uniform output for both stocks and ETFs:
    - Momentum metrics (RSI, returns, trend signals)
    - Hedge fund metrics (Sharpe, volatility, drawdown, beta)
    - Fundamentals (valuation, profitability, growth)
    - SEC filings summary
    - Analyst ratings and targets
    - News sentiment analysis
    
    Examples:
        lox research -t NVDA           # Full report
        lox research -t SPY            # ETF report
        lox research -t AAPL --llm     # With LLM summary
        lox research -t MSFT --quick   # Quick mode
    """
    _run_comprehensive_research(ticker, llm=llm, json_out=json_out, quick=quick)


def _run_comprehensive_research(ticker: str, llm: bool = False, json_out: bool = False, quick: bool = False):
    """Run comprehensive research report."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.markdown import Markdown
    
    from ai_options_trader.config import load_settings
    from ai_options_trader.research.ticker_report import build_ticker_research_report
    
    console = Console()
    settings = load_settings()
    t = ticker.strip().upper()
    
    console.print(f"\n[bold cyan]RESEARCH REPORT: {t}[/bold cyan]")
    console.print("[dim]Building comprehensive analysis...[/dim]\n")
    
    report = build_ticker_research_report(settings, t, include_llm=llm)
    
    if json_out:
        import json
        from dataclasses import asdict
        output = {
            "ticker": report.ticker,
            "asset_type": report.asset_type,
            "generated_at": report.generated_at,
            "current_price": report.current_price,
            "momentum": asdict(report.momentum),
            "hf_metrics": asdict(report.hf_metrics),
            "overall_score": report.overall_score,
            "key_strengths": report.key_strengths,
            "key_risks": report.key_risks,
        }
        console.print(json.dumps(output, indent=2))
        return
    
    # Header with overall score
    score = report.overall_score
    score_color = "green" if score >= 60 else "yellow" if score >= 40 else "red"
    filled = int(score / 5)
    gauge = "█" * filled + "░" * (20 - filled)
    
    header_lines = [
        f"[bold]{t}[/bold] - {report.asset_type.upper()}",
        f"Price: [bold]${report.current_price:.2f}[/bold]",
        f"Generated: {report.generated_at}",
        "",
        f"[bold]Overall Score:[/bold] [{score_color}]{score}/100[/{score_color}]",
        f"[{score_color}]{gauge}[/{score_color}]",
    ]
    console.print(Panel("\n".join(header_lines), title="Overview", border_style="cyan", expand=False))
    
    # Momentum Section
    m = report.momentum
    mom_color = "green" if m.momentum_score > 20 else "red" if m.momentum_score < -20 else "yellow"
    
    mom_table = Table(title="Momentum Metrics", expand=False, show_header=True)
    mom_table.add_column("Metric", style="bold")
    mom_table.add_column("Value", justify="right")
    mom_table.add_column("Metric", style="bold")
    mom_table.add_column("Value", justify="right")
    
    def ret_color(v):
        return "green" if v > 0 else "red" if v < 0 else "white"
    
    mom_table.add_row(
        "1D Return", f"[{ret_color(m.return_1d)}]{m.return_1d:+.2f}%[/{ret_color(m.return_1d)}]",
        "RSI(14)", f"[bold]{m.rsi_14}[/bold] ({m.rsi_interpretation})",
    )
    mom_table.add_row(
        "1W Return", f"[{ret_color(m.return_5d)}]{m.return_5d:+.2f}%[/{ret_color(m.return_5d)}]",
        "vs SMA20", "✓ Above" if m.above_sma_20 else "✗ Below",
    )
    mom_table.add_row(
        "1M Return", f"[{ret_color(m.return_1m)}]{m.return_1m:+.2f}%[/{ret_color(m.return_1m)}]",
        "vs SMA50", "✓ Above" if m.above_sma_50 else "✗ Below",
    )
    mom_table.add_row(
        "3M Return", f"[{ret_color(m.return_3m)}]{m.return_3m:+.2f}%[/{ret_color(m.return_3m)}]",
        "vs SMA200", "✓ Above" if m.above_sma_200 else "✗ Below",
    )
    mom_table.add_row(
        "YTD Return", f"[{ret_color(m.return_ytd)}]{m.return_ytd:+.2f}%[/{ret_color(m.return_ytd)}]",
        "Signal", "[green]Golden Cross[/green]" if m.golden_cross else "[red]Death Cross[/red]" if m.death_cross else "Neutral",
    )
    mom_table.add_row(
        "vs 52W High", f"{m.pct_from_52w_high:.1f}%",
        "Trend", f"{m.trend_direction.title()} ({m.trend_strength})",
    )
    mom_table.add_row(
        "vs 52W Low", f"+{m.pct_from_52w_low:.1f}%",
        "[bold]Score[/bold]", f"[{mom_color}][bold]{m.momentum_score}[/bold] ({m.momentum_label})[/{mom_color}]",
    )
    
    console.print()
    console.print(mom_table)
    
    # Hedge Fund Metrics Section
    hf = report.hf_metrics
    risk_color = "green" if hf.risk_label == "Low" else "yellow" if hf.risk_label == "Moderate" else "red"
    
    hf_table = Table(title="Hedge Fund Grade Metrics", expand=False, show_header=True)
    hf_table.add_column("Risk Metrics", style="bold")
    hf_table.add_column("Value", justify="right")
    hf_table.add_column("Return Metrics", style="bold")
    hf_table.add_column("Value", justify="right")
    
    hf_table.add_row(
        "Volatility (30D)", f"{hf.volatility_30d:.1f}%",
        "Sharpe (YTD)", f"{hf.sharpe_ratio_ytd:.2f}",
    )
    hf_table.add_row(
        "Volatility (1Y)", f"{hf.volatility_1y:.1f}%",
        "Sharpe (1Y)", f"{hf.sharpe_ratio_1y:.2f}",
    )
    hf_table.add_row(
        "Beta (vs SPY)", f"{hf.beta_1y:.2f}",
        "Sortino (1Y)", f"{hf.sortino_ratio_1y:.2f}",
    )
    hf_table.add_row(
        "Max Drawdown (1Y)", f"[red]-{hf.max_drawdown_1y:.1f}%[/red]",
        "Calmar Ratio", f"{hf.calmar_ratio:.2f}",
    )
    hf_table.add_row(
        "Current DD", f"-{hf.current_drawdown:.1f}%",
        "Best Day (1Y)", f"[green]+{hf.best_day_1y:.1f}%[/green]",
    )
    hf_table.add_row(
        "VaR 95% (1D)", f"-{hf.var_95_1d:.2f}%",
        "Worst Day (1Y)", f"[red]{hf.worst_day_1y:.1f}%[/red]",
    )
    hf_table.add_row(
        "Corr (SPY)", f"{hf.correlation_spy:.2f}",
        "Corr (QQQ)", f"{hf.correlation_qqq:.2f}",
    )
    hf_table.add_row(
        "[bold]Risk Score[/bold]", f"[{risk_color}][bold]{hf.risk_score}[/bold][/{risk_color}]",
        "[bold]Risk Level[/bold]", f"[{risk_color}][bold]{hf.risk_label}[/bold][/{risk_color}]",
    )
    
    console.print()
    console.print(hf_table)
    
    # Fundamentals Section (varies by type)
    if report.fundamentals:
        console.print()
        if report.asset_type == "etf":
            _print_etf_fundamentals(console, report.fundamentals)
        else:
            _print_stock_fundamentals(console, report.fundamentals)
    
    # Analyst Data (stocks only)
    if report.analyst_data:
        a = report.analyst_data
        console.print()
        
        upside_color = "green" if a.upside_to_target > 10 else "red" if a.upside_to_target < -10 else "yellow"
        
        analyst_lines = [
            f"[bold]Consensus:[/bold] {a.consensus_rating} ({a.num_analysts} analysts)",
            "",
            f"Target Price Mean: [bold]${a.target_price_mean:.2f}[/bold]",
            f"Target Range: ${a.target_price_low:.2f} - ${a.target_price_high:.2f}",
            f"Upside to Target: [{upside_color}]{a.upside_to_target:+.1f}%[/{upside_color}]",
        ]
        console.print(Panel("\n".join(analyst_lines), title="Analyst Ratings", expand=False))
    
    # SEC Filings (if available)
    if report.sec_filings and not quick:
        sec = report.sec_filings
        console.print()
        
        insider_status = ""
        if sec.has_insider_buying and not sec.has_insider_selling:
            insider_status = "[green]Net Buying[/green]"
        elif sec.has_insider_selling and not sec.has_insider_buying:
            insider_status = "[red]Net Selling[/red]"
        elif sec.has_insider_buying and sec.has_insider_selling:
            insider_status = "[yellow]Mixed[/yellow]"
        else:
            insider_status = "No Activity"
        
        sec_lines = [
            f"[bold]Filings (30D):[/bold] {sec.filings_30d}",
            f"[bold]Filings (90D):[/bold] {sec.filings_90d}",
            f"[bold]Most Recent:[/bold] {sec.most_recent_form} ({sec.most_recent_date})",
            f"[bold]Insider Activity:[/bold] {insider_status}",
        ]
        
        if sec.recent_8k_items:
            sec_lines.append(f"[bold]Recent 8-K Items:[/bold] {', '.join(sec.recent_8k_items[:3])}")
        
        if sec.notable_filings:
            sec_lines.append("")
            sec_lines.append("[bold]Recent Filings:[/bold]")
            for f in sec.notable_filings[:3]:
                sec_lines.append(f"  • {f['date']}: {f['form']} - {f['description'][:50]}")
        
        console.print(Panel("\n".join(sec_lines), title="SEC Filings", expand=False))
    
    # News Analysis (if available)
    if report.news_analysis and not quick:
        news = report.news_analysis
        console.print()
        
        sent_color = "green" if news.sentiment_label == "Positive" else "red" if news.sentiment_label == "Negative" else "yellow"
        
        news_lines = [
            f"[bold]Articles (30D):[/bold] {news.total_articles_30d}",
            f"[bold]Sentiment:[/bold] [{sent_color}]{news.sentiment_label}[/{sent_color}] ({news.sentiment_score:+.2f})",
            f"[bold]Key Themes:[/bold] {', '.join(news.key_themes) if news.key_themes else 'None'}",
        ]
        
        if news.recent_headlines:
            news_lines.append("")
            news_lines.append("[bold]Recent Headlines:[/bold]")
            for h in news.recent_headlines[:3]:
                sent_icon = "📈" if h["sentiment"] == "positive" else "📉" if h["sentiment"] == "negative" else "➖"
                news_lines.append(f"  {sent_icon} {h['date']}: {h['title'][:60]}")
        
        console.print(Panel("\n".join(news_lines), title="News Analysis", expand=False))
    
    # Key Strengths and Risks
    console.print()
    
    cols = []
    if report.key_strengths:
        strength_lines = ["[bold green]KEY STRENGTHS[/bold green]", ""]
        for s in report.key_strengths:
            strength_lines.append(f"  ✓ {s}")
        cols.append(Panel("\n".join(strength_lines), border_style="green", expand=True))
    
    if report.key_risks:
        risk_lines = ["[bold red]KEY RISKS[/bold red]", ""]
        for r in report.key_risks:
            risk_lines.append(f"  ⚠ {r}")
        cols.append(Panel("\n".join(risk_lines), border_style="red", expand=True))
    
    if cols:
        console.print(Columns(cols))
    
    # LLM Summary (if available)
    if report.llm_summary:
        console.print()
        console.print(Panel(Markdown(report.llm_summary), title="AI Research Summary", border_style="magenta", expand=False))
    
    console.print()


# Formatting functions moved to cli_commands/utils/formatting.py
from ai_options_trader.cli_commands.utils.formatting import (
    print_stock_fundamentals as _print_stock_fundamentals,
    print_etf_fundamentals as _print_etf_fundamentals,
)


@app.command("scan-extremes")
def scan_extremes_cmd(
    universe: str = typer.Option("default", "--universe", "-u", help="Universe: default, tech, finance, etf, or comma-separated tickers"),
    min_conviction: int = typer.Option(50, "--min-conviction", help="Minimum conviction score (0-100)"),
    max_ideas: int = typer.Option(5, "--max-ideas", "-n", help="Max ideas per category"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    Scan for overbought/oversold opportunities.
    
    Identifies stocks that may have moved irrationally and could be 
    candidates for mean reversion trades.
    
    Idea Types:
    - oversold_bounce: Deeply oversold stocks that may bounce
    - overbought_fade: Extended rallies that may pull back
    - momentum_continuation: Healthy trends with room to run
    
    Universes:
    - default: Mix of mega caps, sectors, and ETFs
    - tech: Technology stocks
    - finance: Financial stocks
    - etf: Major ETFs
    - Or comma-separated: AAPL,MSFT,NVDA
    
    Examples:
        lox scan-extremes                    # Default scan
        lox scan-extremes -u tech            # Tech stocks only
        lox scan-extremes -u NVDA,AMD,INTC   # Custom tickers
        lox scan-extremes --min-conviction 70  # Higher conviction only
    """
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    
    from ai_options_trader.config import load_settings
    from ai_options_trader.research.ideas import generate_trade_ideas, DEFAULT_SCAN_UNIVERSE
    
    console = Console()
    settings = load_settings()
    
    # Determine universe
    universe_lower = universe.lower()
    if universe_lower == "default":
        scan_universe = DEFAULT_SCAN_UNIVERSE
    elif universe_lower == "tech":
        scan_universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", 
                        "AMD", "AVGO", "CRM", "ADBE", "ORCL", "CSCO", "INTC", "QCOM",
                        "IBM", "TXN", "MU", "AMAT", "LRCX", "KLAC", "ADI", "MRVL",
                        "CRWV", "SMCI", "VRT", "ARM", "PLTR", "SNOW", "NET", "DDOG"]
    elif universe_lower == "finance":
        scan_universe = ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", 
                        "SCHW", "CME", "ICE", "SPGI", "MCO", "V", "MA", "PYPL",
                        "COF", "USB", "PNC", "TFC", "BK", "STT", "MTB", "HBAN"]
    elif universe_lower == "etf":
        scan_universe = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV",
                        "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE", "GLD", "SLV",
                        "TLT", "HYG", "LQD", "EEM", "VWO", "FXI", "KWEB", "EWJ",
                        "SMH", "SOXX", "XBI", "XHB", "ITB", "ARKK", "ARKG"]
    else:
        # Custom comma-separated
        scan_universe = [t.strip().upper() for t in universe.split(",")]
    
    console.print(f"\n[bold cyan]TRADE IDEAS GENERATOR[/bold cyan]")
    console.print(f"[dim]Scanning {len(scan_universe)} tickers for opportunities...[/dim]\n")
    
    report = generate_trade_ideas(
        settings=settings,
        universe=scan_universe,
        min_conviction=min_conviction,
        max_ideas=max_ideas,
    )
    
    if json_out:
        import json
        from dataclasses import asdict
        output = {
            "generated_at": report.generated_at,
            "universe_size": report.universe_size,
            "market_breadth": report.market_breadth,
            "avg_rsi": report.avg_rsi,
            "pct_oversold": report.pct_oversold,
            "pct_overbought": report.pct_overbought,
            "oversold_ideas": [asdict(i) for i in report.oversold_ideas],
            "overbought_ideas": [asdict(i) for i in report.overbought_ideas],
            "momentum_ideas": [asdict(i) for i in report.momentum_ideas],
        }
        console.print(json.dumps(output, indent=2))
        return
    
    # Market Breadth Summary
    breadth_color = "green" if report.market_breadth == "bullish" else "red" if report.market_breadth == "bearish" else "yellow"
    
    breadth_lines = [
        f"[bold]Market Breadth:[/bold] [{breadth_color}]{report.market_breadth.upper()}[/{breadth_color}]",
        "",
        f"Average RSI: {report.avg_rsi}",
        f"Oversold (<30): {report.pct_oversold:.1f}%",
        f"Overbought (>70): {report.pct_overbought:.1f}%",
        "",
        f"[dim]Universe: {report.universe_size} tickers scanned[/dim]",
        f"[dim]Generated: {report.generated_at}[/dim]",
    ]
    console.print(Panel("\n".join(breadth_lines), title="Market Overview", border_style="cyan", expand=False))
    
    # Oversold Ideas (Long Candidates)
    if report.oversold_ideas:
        console.print()
        console.print("[bold green]OVERSOLD - POTENTIAL BOUNCE CANDIDATES (Long)[/bold green]")
        
        oversold_table = Table(expand=False, show_header=True)
        oversold_table.add_column("Ticker", style="bold cyan")
        oversold_table.add_column("Price", justify="right")
        oversold_table.add_column("RSI", justify="right")
        oversold_table.add_column("1M Ret", justify="right")
        oversold_table.add_column("vs 52W High", justify="right")
        oversold_table.add_column("Conviction", justify="right")
        oversold_table.add_column("Signal", style="dim")
        
        for idea in report.oversold_ideas:
            conv_color = "green" if idea.conviction_score >= 70 else "yellow" if idea.conviction_score >= 50 else "white"
            
            oversold_table.add_row(
                idea.ticker,
                f"${idea.current_price:.2f}",
                f"[red]{idea.rsi:.0f}[/red]",
                f"[red]{idea.return_1m:+.1f}%[/red]",
                f"{idea.pct_from_52w_high:.0f}%",
                f"[{conv_color}]{idea.conviction_score}[/{conv_color}]",
                idea.reasoning[0][:40] if idea.reasoning else "",
            )
        
        console.print(oversold_table)
        
        # Show details for top idea
        if report.oversold_ideas:
            top = report.oversold_ideas[0]
            detail_lines = [
                f"[bold green]TOP OVERSOLD: {top.ticker}[/bold green]",
                "",
                "[bold]Why:[/bold]",
            ]
            for r in top.reasoning[:4]:
                detail_lines.append(f"  • {r}")
            detail_lines.extend([
                "",
                f"[bold]Suggested Trade:[/bold]",
                f"  Entry: {top.suggested_entry}",
                f"  Target: +{top.suggested_target_pct}%",
                f"  Stop: -{top.suggested_stop_pct}%",
                f"  R:R: {top.risk_reward}:1",
            ])
            console.print()
            console.print(Panel("\n".join(detail_lines), border_style="green", expand=False))
    
    # Overbought Ideas (Short/Fade Candidates)
    if report.overbought_ideas:
        console.print()
        console.print("[bold red]OVERBOUGHT - POTENTIAL PULLBACK CANDIDATES (Fade)[/bold red]")
        
        overbought_table = Table(expand=False, show_header=True)
        overbought_table.add_column("Ticker", style="bold cyan")
        overbought_table.add_column("Price", justify="right")
        overbought_table.add_column("RSI", justify="right")
        overbought_table.add_column("1M Ret", justify="right")
        overbought_table.add_column("vs 52W High", justify="right")
        overbought_table.add_column("Conviction", justify="right")
        overbought_table.add_column("Signal", style="dim")
        
        for idea in report.overbought_ideas:
            conv_color = "green" if idea.conviction_score >= 70 else "yellow" if idea.conviction_score >= 50 else "white"
            
            overbought_table.add_row(
                idea.ticker,
                f"${idea.current_price:.2f}",
                f"[green]{idea.rsi:.0f}[/green]",
                f"[green]{idea.return_1m:+.1f}%[/green]",
                f"{idea.pct_from_52w_high:.0f}%",
                f"[{conv_color}]{idea.conviction_score}[/{conv_color}]",
                idea.reasoning[0][:40] if idea.reasoning else "",
            )
        
        console.print(overbought_table)
    
    # Momentum Ideas
    if report.momentum_ideas:
        console.print()
        console.print("[bold yellow]MOMENTUM - CONTINUATION CANDIDATES[/bold yellow]")
        
        momentum_table = Table(expand=False, show_header=True)
        momentum_table.add_column("Ticker", style="bold cyan")
        momentum_table.add_column("Price", justify="right")
        momentum_table.add_column("RSI", justify="right")
        momentum_table.add_column("Score", justify="right")
        momentum_table.add_column("Conviction", justify="right")
        momentum_table.add_column("Signal", style="dim")
        
        for idea in report.momentum_ideas:
            conv_color = "green" if idea.conviction_score >= 70 else "yellow" if idea.conviction_score >= 50 else "white"
            
            momentum_table.add_row(
                idea.ticker,
                f"${idea.current_price:.2f}",
                f"{idea.rsi:.0f}",
                f"{idea.momentum_score}",
                f"[{conv_color}]{idea.conviction_score}[/{conv_color}]",
                idea.reasoning[0][:40] if idea.reasoning else "",
            )
        
        console.print(momentum_table)
    
    # No ideas found
    if not report.oversold_ideas and not report.overbought_ideas and not report.momentum_ideas:
        console.print()
        console.print("[yellow]No high-conviction ideas found in current scan.[/yellow]")
        console.print("[dim]Try lowering --min-conviction or expanding the universe.[/dim]")
    
    console.print()


# GPU commands moved to cli_commands/gpu/gpu_cmd.py


# GPU commands moved to cli_commands/gpu/gpu_cmd.py
# All GPU code (gpu, gpu-debt, and helper functions) extracted to:
# - cli_commands/gpu/gpu_cmd.py




# ---------------------------------------------------------------------------
# REGIME SUBGROUP (consolidated)
# ---------------------------------------------------------------------------
regime_app = typer.Typer(add_completion=False, help="Economic regime analysis")
app.add_typer(regime_app, name="regime")


@regime_app.command("vol")
def regime_vol(llm: bool = typer.Option(False, "--llm", help="Include LLM analysis")):
    """Volatility regime (VIX level, term structure)."""
    from ai_options_trader.cli_commands.regimes.volatility_cmd import volatility_snapshot
    volatility_snapshot(llm=llm)


@regime_app.command("fiscal")
def regime_fiscal():
    """US fiscal regime (deficits, TGA, issuance)."""
    from ai_options_trader.cli_commands.regimes.fiscal_cmd import fiscal_snapshot
    fiscal_snapshot()


@regime_app.command("funding")
def regime_funding():
    """Funding markets (SOFR, repo spreads)."""
    from ai_options_trader.cli_commands.regimes.funding_cmd import funding_snapshot
    funding_snapshot()


@regime_app.command("rates")
def regime_rates():
    """Rates / yield curve analysis."""
    from ai_options_trader.cli_commands.regimes.rates_cmd import rates_snapshot
    rates_snapshot()


@regime_app.command("macro")
def regime_macro():
    """Macro regime overview."""
    from ai_options_trader.cli_commands.regimes.macro_cmd import macro_snapshot
    macro_snapshot()


# ---------------------------------------------------------------------------
# SCENARIO SUBGROUP (consolidated)
# ---------------------------------------------------------------------------
scenario_app = typer.Typer(add_completion=False, help="Portfolio scenario analysis")
app.add_typer(scenario_app, name="scenario")


@scenario_app.command("monte-carlo")
def scenario_monte_carlo(
    real: bool = typer.Option(False, "--real", help="Use real positions"),
):
    """Monte Carlo simulation with 10,000+ scenarios."""
    from ai_options_trader.cli_commands.analysis.scenarios_cmd import scenarios_monte_carlo
    scenarios_monte_carlo(real=real)


@scenario_app.command("stress")
def scenario_stress():
    """Stress test portfolio under extreme conditions."""
    from ai_options_trader.cli_commands.analysis.stress_cmd import stress_test
    stress_test()


# ---------------------------------------------------------------------------
# SUBGROUPS (organized by function)
# ---------------------------------------------------------------------------

# Chat - interactive research
from ai_options_trader.cli_commands.chat_cmd import app as chat_app
app.add_typer(chat_app, name="chat")

# Trade execution (renamed from autopilot)
trade_app = typer.Typer(add_completion=False, help="Trade execution and automation")
app.add_typer(trade_app, name="trade")

# Options (full commands)
options_app = typer.Typer(add_completion=False, help="Full options toolset")
app.add_typer(options_app, name="options")

# Ideas
ideas_app = typer.Typer(add_completion=False, help="Trade ideas from screens and catalysts")
app.add_typer(ideas_app, name="ideas")

# Account
nav_app = typer.Typer(add_completion=False, help="NAV tracking and fund accounting")
app.add_typer(nav_app, name="nav")

# Labs - advanced/power user
labs_app = typer.Typer(add_completion=False, help="Advanced research tools (power users)")
app.add_typer(labs_app, name="labs")

# Hidden/legacy subgroups for labs
macro_app = typer.Typer(add_completion=False, help="Macro signals")
labs_app.add_typer(macro_app, name="macro")
tariff_app = typer.Typer(add_completion=False, help="Tariff regime")
labs_app.add_typer(tariff_app, name="tariff")
funding_app = typer.Typer(add_completion=False, help="Funding markets")
labs_app.add_typer(funding_app, name="funding")
usd_app = typer.Typer(add_completion=False, help="USD regime")
labs_app.add_typer(usd_app, name="usd")
monetary_app = typer.Typer(add_completion=False, help="Fed liquidity")
labs_app.add_typer(monetary_app, name="monetary")
rates_app = typer.Typer(add_completion=False, help="Rates regime")
labs_app.add_typer(rates_app, name="rates")
fiscal_app = typer.Typer(add_completion=False, help="Fiscal regime")
labs_app.add_typer(fiscal_app, name="fiscal")
vol_app = typer.Typer(add_completion=False, help="Volatility regime")
labs_app.add_typer(vol_app, name="volatility")
commod_app = typer.Typer(add_completion=False, help="Commodities")
labs_app.add_typer(commod_app, name="commodities")
crypto_app = typer.Typer(add_completion=False, help="Crypto")
labs_app.add_typer(crypto_app, name="crypto")
ticker_app = typer.Typer(add_completion=False, help="Ticker analysis")
labs_app.add_typer(ticker_app, name="ticker")
housing_app = typer.Typer(add_completion=False, help="Housing regime")
labs_app.add_typer(housing_app, name="housing")
household_app = typer.Typer(add_completion=False, help="Household wealth")
labs_app.add_typer(household_app, name="household")
news_app = typer.Typer(add_completion=False, help="News sentiment")
labs_app.add_typer(news_app, name="news")
solar_app = typer.Typer(add_completion=False, help="Solar regime")
labs_app.add_typer(solar_app, name="solar")
silver_app = typer.Typer(add_completion=False, help="Silver regime")
labs_app.add_typer(silver_app, name="silver")
track_app = typer.Typer(add_completion=False, help="Tracking")
labs_app.add_typer(track_app, name="track")

# Additional apps needed for registration
model_app = typer.Typer(add_completion=False, help="ML models")
app.add_typer(model_app, name="model")
live_app = typer.Typer(add_completion=False, help="Live monitoring")
app.add_typer(live_app, name="live")
weekly_app = typer.Typer(add_completion=False, help="Weekly reports")
app.add_typer(weekly_app, name="weekly")
autopilot_app = trade_app  # Alias for backward compat

_COMMANDS_REGISTERED = False

def _register_commands() -> None:
    global _COMMANDS_REGISTERED
    if _COMMANDS_REGISTERED:
        return
    # Import here to keep `ai_options_trader.cli` lightweight at import time.
    
    # Core commands
    from ai_options_trader.cli_commands.core.core_cmd import register_core
    from ai_options_trader.cli_commands.core.dashboard_cmd import register as register_dashboard
    from ai_options_trader.cli_commands.core.dashboard_cmd import register_pillar_commands
    from ai_options_trader.cli_commands.core.nav_cmd import register as register_nav
    from ai_options_trader.cli_commands.core.account_cmd import register as register_account
    from ai_options_trader.cli_commands.core.weekly_report_cmd import register as register_weekly_report
    from ai_options_trader.cli_commands.core.closed_trades_cmd import register as register_closed_trades
    from ai_options_trader.cli_commands.core.live_cmd import register as register_live
    from ai_options_trader.cli_commands.core.portfolio_cmd import register as register_portfolio
    
    # Regime commands
    from ai_options_trader.cli_commands.regimes.macro_cmd import register as register_macro
    from ai_options_trader.cli_commands.regimes.tariff_cmd import register as register_tariff
    from ai_options_trader.cli_commands.regimes.funding_cmd import register as register_funding
    from ai_options_trader.cli_commands.regimes.usd_cmd import register as register_usd
    from ai_options_trader.cli_commands.regimes.monetary_cmd import register as register_monetary
    from ai_options_trader.cli_commands.regimes.rates_cmd import register as register_rates
    from ai_options_trader.cli_commands.regimes.fiscal_cmd import register as register_fiscal
    from ai_options_trader.cli_commands.regimes.volatility_cmd import register as register_volatility
    from ai_options_trader.cli_commands.regimes.commodities_cmd import register as register_commodities
    from ai_options_trader.cli_commands.regimes.crypto_cmd import register as register_crypto
    from ai_options_trader.cli_commands.regimes.housing_cmd import register as register_housing
    from ai_options_trader.cli_commands.regimes.household_cmd import register as register_household
    from ai_options_trader.cli_commands.regimes.news_cmd import register as register_news
    from ai_options_trader.cli_commands.regimes.solar_cmd import register as register_solar
    from ai_options_trader.cli_commands.regimes.silver_cmd import register as register_silver
    from ai_options_trader.cli_commands.regimes.regimes_cmd import register as register_regimes
    from ai_options_trader.cli_commands.regimes.fedfunds_cmd import register as register_fedfunds
    
    # Options commands (consolidated)
    from ai_options_trader.cli_commands.options.options_cmd import register as register_options
    
    # Analysis commands
    from ai_options_trader.cli_commands.analysis.scenarios_cmd import register as register_scenarios
    from ai_options_trader.cli_commands.analysis.model_cmd import register_model
    from ai_options_trader.cli_commands.analysis.ticker_cmd import register as register_ticker
    from ai_options_trader.cli_commands.analysis.deep_cmd import register as register_deep
    from ai_options_trader.cli_commands.analysis.stress_cmd import register_stress
    from ai_options_trader.cli_commands.analysis.fundamentals_cmd import register as register_fundamentals
    
    # Ideas commands
    from ai_options_trader.cli_commands.ideas.ideas_clean import register_ideas as register_ideas_clean
    from ai_options_trader.cli_commands.ideas.hedges_cmd import register as register_hedges
    
    # Scanner commands
    from ai_options_trader.cli_commands.scanner.bubble_finder_cmd import register as register_bubble_finder
    
    # GPU commands
    from ai_options_trader.cli_commands.gpu.gpu_cmd import register_gpu_commands
    
    # Other commands (remain at root)
    from ai_options_trader.cli_commands.track_cmd import register as register_track
    from ai_options_trader.cli_commands.autopilot_cmd import register as register_autopilot

    # Core commands (top-level for quick access)
    register_core(app)
    register_dashboard(app)  # Main dashboard command
    register_closed_trades(app)  # Closed trades P&L
    
    # Options: all commands (consolidated)
    register_options(options_app)
    
    # Ideas: clean commands
    register_ideas_clean(ideas_app)  # catalyst, screen
    
    # Model: clean unified commands
    register_model(model_app)  # predict, eval, inspect
    
    register_track(track_app)
    register_nav(nav_app)
    register_autopilot(autopilot_app)
    register_account(app)
    register_live(live_app)
    register_weekly_report(weekly_app)
    
    # GPU commands (top-level)
    register_gpu_commands(app)

    # Labs: keep everything else accessible under `lox labs ...`
    register_portfolio(labs_app)
    register_regimes(labs_app)
    register_scenarios(labs_app)  # Includes basic + ML-enhanced + Monte Carlo
    register_hedges(labs_app)  # Simplified hedge recommendations
    register_deep(labs_app)  # Deep dive ticker analysis
    register_stress(labs_app)  # Stress testing
    register_macro(macro_app)
    register_tariff(tariff_app)
    register_funding(funding_app)
    register_usd(usd_app)
    register_monetary(monetary_app)
    register_fedfunds(monetary_app)
    register_rates(rates_app)
    register_fiscal(fiscal_app)
    register_volatility(vol_app)
    register_commodities(commod_app)
    register_crypto(crypto_app)
    register_ticker(ticker_app)
    register_housing(housing_app)
    register_household(household_app)
    register_news(news_app)
    register_solar(solar_app)
    register_silver(silver_app)
    register_bubble_finder(labs_app)  # Bubble finder scanner
    register_fundamentals(labs_app)  # CFA-level financial modeling
    
    # Quick pillar access under labs
    register_pillar_commands(labs_app)
    _COMMANDS_REGISTERED = True


def main():
    _register_commands()
    app()

# Register commands when imported as a console-script entrypoint (`pyproject.toml` uses `ai_options_trader.cli:app`).
_register_commands()


if __name__ == "__main__":
    main()

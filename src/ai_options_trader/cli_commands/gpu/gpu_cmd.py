"""
GPU-related CLI commands - tracker and debt analysis.

Commands:
- gpu: GPU-backed securities tracker
- gpu-debt: GPU-backed debt market analysis
"""
from __future__ import annotations

from datetime import date

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def register_gpu_commands(app: typer.Typer) -> None:
    """Register GPU commands with the main app."""
    
    @app.command("gpu-debt")
    def gpu_debt_cmd(
        ticker: str = typer.Option("", "-t", "--ticker", help="Specific ticker for detailed analysis"),
        crwv: bool = typer.Option(False, "--crwv", help="Show detailed CoreWeave analysis"),
    ):
        """
        GPU-backed debt market analysis.
        
        Shows the emerging GPU-backed debt market across multiple companies,
        including depreciation analysis and refinancing risks.
        
        Examples:
            lox gpu-debt              # Market overview
            lox gpu-debt --crwv       # Detailed CoreWeave analysis
            lox gpu-debt -t ORCL      # Oracle's GPU-related debt
        """
        from ai_options_trader.gpu.debt_analysis import (
            CRWV_DEBT_STRUCTURE,
            GPU_DEBT_MARKET_OVERVIEW,
            GPU_DEBT_COMPANIES,
            assess_crwv_debt_risk,
            get_gpu_depreciation_analysis,
        )
        
        console = Console()
        
        # If specific ticker requested
        if ticker and ticker.upper() != "CRWV":
            _show_ticker_debt(console, ticker.upper())
            return
        
        # If --crwv flag
        if crwv or (ticker and ticker.upper() == "CRWV"):
            _show_crwv_detailed(console)
            return
        
        # Market overview
        console.print("\n[bold cyan]GPU-BACKED DEBT MARKET OVERVIEW[/bold cyan]")
        console.print("[dim]An emerging $50B+ asset class with unique risks[/dim]\n")
        
        market = GPU_DEBT_MARKET_OVERVIEW
        market_lines = [
            f"[bold]Estimated Market Size:[/bold] ${market['total_market_size_b']}B+",
            f"[bold]2024 Growth:[/bold] {market['growth_2024']}",
            "",
            "[bold]Key Players:[/bold]",
        ]
        for player in market["key_players"]:
            market_lines.append(f"  • {player['type']}: {', '.join(player['examples'])}")
        
        console.print(Panel("\n".join(market_lines), title="Market Snapshot", border_style="cyan", expand=False))
        
        # Companies table
        console.print("\n[bold yellow]COMPANIES WITH GPU DEBT/CAPEX EXPOSURE[/bold yellow]\n")
        
        comp_table = Table(title="GPU Debt Exposure by Company", expand=False)
        comp_table.add_column("Ticker", style="bold cyan")
        comp_table.add_column("Company")
        comp_table.add_column("Total Debt", justify="right")
        comp_table.add_column("GPU-Backed", justify="right", style="red")
        comp_table.add_column("GPU CapEx", justify="right", style="yellow")
        comp_table.add_column("LTV Ratio", justify="right")
        comp_table.add_column("Key Risk", width=35)
        
        for ticker_sym, data in GPU_DEBT_COMPANIES.items():
            ltv = data.get("ltv_ratio", 0)
            ltv_str = f"{ltv:.1f}x" if ltv > 0 else "N/A"
            ltv_color = "red" if ltv > 2 else "yellow" if ltv > 1 else ""
            
            gpu_debt = data.get("gpu_backed_debt_b", 0)
            gpu_capex = data.get("gpu_capex_b", 0)
            
            comp_table.add_row(
                ticker_sym,
                data["name"],
                f"${data['total_debt_b']:.1f}B",
                f"${gpu_debt:.1f}B" if gpu_debt > 0 else "-",
                f"${gpu_capex:.0f}B" if gpu_capex > 0 else "-",
                f"[{ltv_color}]{ltv_str}[/{ltv_color}]" if ltv_color else ltv_str,
                data["key_risk"][:35],
            )
        
        console.print(comp_table)
        
        # Depreciation summary
        console.print("\n[bold cyan]GPU DEPRECIATION: THE COLLATERAL PROBLEM[/bold cyan]\n")
        
        dep = get_gpu_depreciation_analysis()
        dep_table = Table(title="GPU Value Decline", expand=False)
        dep_table.add_column("GPU", style="bold")
        dep_table.add_column("Peak Price", justify="right")
        dep_table.add_column("Current", justify="right")
        dep_table.add_column("Decline", justify="right", style="red")
        dep_table.add_column("Accounting", justify="right")
        dep_table.add_column("Reality vs Books")
        
        for gpu_type in ["h100", "a100"]:
            data = dep[gpu_type]
            vs_books = data["vs_accounting"]
            status_color = "green" if vs_books > 0 else "red"
            dep_table.add_row(
                gpu_type.upper(),
                f"${data['peak_price']:,}",
                f"${data['current_price']:,}",
                f"-{data['decline_pct']:.0f}%",
                f"-{data['accounting_depreciation']}%",
                f"[{status_color}]{vs_books:+.0f}%[/{status_color}]",
            )
        
        console.print(dep_table)
        
        # CRWV spotlight
        console.print("\n[bold red]SPOTLIGHT: COREWEAVE (CRWV)[/bold red]\n")
        
        crwv_risk = assess_crwv_debt_risk()
        crwv_lines = [
            f"[bold]Total Debt:[/bold] ${crwv_risk.total_debt_b:.1f}B",
            f"[bold]GPU Collateral:[/bold] ${crwv_risk.collateral_value_b:.1f}B",
            f"[bold]LTV Ratio:[/bold] [red]{crwv_risk.ltv_ratio:.1f}x[/red]",
            f"[bold]Refinancing Risk:[/bold] [red]{crwv_risk.refinancing_risk}[/red]",
        ]
        console.print(Panel("\n".join(crwv_lines), title="CRWV Risk Summary", border_style="red", expand=False))
        
        console.print()
        console.print("[dim]Run 'lox gpu-debt --crwv' for detailed CoreWeave analysis[/dim]\n")
    
    @app.command("gpu")
    def gpu_tracker_cmd(
        basket: str = typer.Option("short_stack", "--basket", "-b", help="all|short_stack|pure_gpu|openai|supply_chain"),
        puts: bool = typer.Option(False, "--puts", help="Show put options for short stack"),
    ):
        """
        GPU-backed securities tracker. Monitor the GPU infrastructure stack.
        
        Baskets:
            short_stack  - High-conviction shorts (CRWV, SMCI, NVDA, VRT, MRVL)
            pure_gpu     - Companies with 50%+ GPU revenue
            openai       - OpenAI ecosystem (MSFT, ORCL, NVDA, CRWV)
            supply_chain - NVDA suppliers (TSM, MU, ASML)
            all          - Complete GPU stack
        
        Examples:
            lox gpu                     # Short stack overview
            lox gpu --puts              # With put options
            lox gpu -b openai           # OpenAI ecosystem
        """
        from ai_options_trader.config import load_settings
        from ai_options_trader.gpu.tracker import (
            build_gpu_tracker_report,
            GPU_BASKETS,
        )
        
        console = Console()
        settings = load_settings()
        
        basket_info = GPU_BASKETS.get(basket, GPU_BASKETS["short_stack"])
        
        console.print(f"\n[bold cyan]GPU-BACKED SECURITIES TRACKER[/bold cyan]")
        console.print(f"[dim]{basket_info['name']}: {basket_info['description']}[/dim]\n")
        
        console.print("[dim]Fetching market data...[/dim]")
        report = build_gpu_tracker_report(settings, basket)
        
        # NVDA Reference
        nvda_color = "green" if report.nvda_change_1d > 0 else "red"
        console.print(Panel(
            f"[bold]NVDA[/bold] (Center of GPU Universe)\n\n"
            f"Price: ${report.nvda_price:.2f}\n"
            f"Today: [{nvda_color}]{report.nvda_change_1d:+.2f}%[/{nvda_color}]",
            title="Reference",
            expand=False,
        ))
        
        # Main table
        table = Table(title=f"GPU Securities ({len(report.securities)} tickers)", expand=False)
        table.add_column("Ticker", style="bold")
        table.add_column("Name")
        table.add_column("Category", style="cyan")
        table.add_column("GPU Rev%", justify="right")
        table.add_column("Sensitivity", justify="center")
        table.add_column("Price", justify="right")
        table.add_column("Today", justify="right")
        table.add_column("Mkt Cap", justify="right")
        
        for sec in report.securities:
            sens_colors = {"extreme": "red", "high": "orange1", "medium": "yellow", "low": "green"}
            sens_color = sens_colors.get(sec.bear_sensitivity, "white")
            chg_color = "green" if sec.change_1d > 0 else "red" if sec.change_1d < 0 else "white"
            
            table.add_row(
                sec.ticker,
                sec.name[:12],
                sec.category.replace("_", " ")[:15],
                f"{sec.gpu_revenue_pct}%",
                f"[{sens_color}]{sec.bear_sensitivity.upper()}[/{sens_color}]",
                f"${sec.price:.2f}" if sec.price else "—",
                f"[{chg_color}]{sec.change_1d:+.2f}%[/{chg_color}]" if sec.price else "—",
                f"${sec.market_cap_b:.0f}B" if sec.market_cap_b else "—",
            )
        
        console.print(table)
        
        # Bear signals
        if report.bear_signals:
            bear_lines = ["[bold red]BEAR SIGNALS[/bold red]", ""]
            for s in report.bear_signals:
                bear_lines.append(f"  • {s}")
            console.print()
            console.print(Panel("\n".join(bear_lines), title="Signals", border_style="red", expand=False))
        
        # Summary
        stack_color = "green" if report.short_stack_change_1d > 0 else "red"
        summary_lines = [
            f"[bold]Short Stack Today:[/bold] [{stack_color}]{report.short_stack_change_1d:+.2f}%[/{stack_color}]",
            f"[bold]Total GPU Market Cap:[/bold] ${report.total_gpu_market_cap_b:,.0f}B",
        ]
        console.print()
        console.print(Panel("\n".join(summary_lines), title="Summary", expand=False))
        
        # Show puts if requested
        if puts:
            _show_gpu_puts(console, settings)
        
        # Available baskets
        basket_lines = ["[bold]Available Baskets[/bold] (--basket <name>)", ""]
        for bname, binfo in GPU_BASKETS.items():
            marker = " ← current" if bname == basket else ""
            basket_lines.append(f"  [cyan]{bname}[/cyan]: {binfo['description'][:50]}{marker}")
        
        console.print()
        console.print(Panel("\n".join(basket_lines), title="Baskets", expand=False))


def _show_ticker_debt(console: Console, ticker: str) -> None:
    """Show debt analysis for a specific ticker."""
    from ai_options_trader.config import load_settings
    from ai_options_trader.gpu.debt_analysis import fetch_ticker_debt_analysis
    
    settings = load_settings()
    
    console.print(f"\n[bold cyan]DEBT ANALYSIS: {ticker}[/bold cyan]\n")
    
    analysis = fetch_ticker_debt_analysis(ticker, settings)
    
    if not analysis:
        console.print(f"[red]Could not fetch debt data for {ticker}[/red]")
        return
    
    metrics_lines = [
        f"[bold]{analysis.name}[/bold]",
        "",
        f"[bold]Total Debt:[/bold] ${analysis.total_debt_b:.1f}B",
        f"[bold]Total Equity:[/bold] ${analysis.total_equity_b:.1f}B",
        f"[bold]Debt/Equity:[/bold] {analysis.debt_to_equity:.2f}x",
        f"[bold]Interest Coverage:[/bold] {analysis.interest_coverage:.1f}x",
        f"[bold]Current Ratio:[/bold] {analysis.current_ratio:.2f}",
    ]
    
    risk_color = "red" if analysis.risk_assessment == "HIGH RISK" else "yellow" if analysis.risk_assessment == "ELEVATED" else "green"
    metrics_lines.append(f"\n[bold]Risk Assessment:[/bold] [{risk_color}]{analysis.risk_assessment}[/{risk_color}]")
    
    console.print(Panel("\n".join(metrics_lines), title="Debt Metrics", border_style="cyan", expand=False))
    
    if analysis.key_risks:
        console.print()
        risk_lines = ["[bold]Key Debt Risks[/bold]", ""]
        for risk in analysis.key_risks:
            risk_lines.append(f"  • {risk}")
        console.print(Panel("\n".join(risk_lines), title="Risk Factors", border_style="red", expand=False))


def _show_crwv_detailed(console: Console) -> None:
    """Show detailed CoreWeave debt analysis."""
    from ai_options_trader.gpu.debt_analysis import (
        CRWV_DEBT_STRUCTURE,
        CRWV_OPTIONS_AT_MATURITY,
        GPU_PRICING,
        assess_crwv_debt_risk,
        get_gpu_depreciation_analysis,
        build_debt_maturity_timeline,
    )
    
    console.print("\n[bold cyan]COREWEAVE DEBT STRUCTURE & GPU PRICING ANALYSIS[/bold cyan]")
    console.print("[dim]Detailed analysis of the largest GPU-backed debtor[/dim]\n")
    
    risk = assess_crwv_debt_risk()
    
    risk_color = "red" if risk.refinancing_risk == "EXTREME" else "yellow"
    risk_lines = [
        f"[bold]CRWV Debt Risk Assessment[/bold]",
        "",
        f"Total Debt: [bold]${risk.total_debt_b:.1f}B[/bold]",
        f"GPU Collateral Value: ${risk.collateral_value_b:.1f}B",
        f"Loan-to-Value Ratio: [{risk_color}]{risk.ltv_ratio:.1f}x[/{risk_color}]",
        f"Interest Coverage: [{risk_color}]{risk.debt_coverage_ratio:.2f}x[/{risk_color}]",
        f"Refinancing Risk: [{risk_color}]{risk.refinancing_risk}[/{risk_color}]",
    ]
    console.print(Panel("\n".join(risk_lines), title="Risk Summary", border_style=risk_color, expand=False))
    
    # Debt breakdown
    debt = CRWV_DEBT_STRUCTURE
    debt_table = Table(title="CRWV Debt Breakdown ($12.9B Total)", expand=False)
    debt_table.add_column("Facility", style="bold")
    debt_table.add_column("Amount", justify="right")
    debt_table.add_column("Rate", justify="right")
    debt_table.add_column("Maturity", justify="center")
    debt_table.add_column("Collateral", style="cyan")
    
    for d in debt["debt_breakdown"]:
        debt_table.add_row(
            d["name"][:25],
            f"${d['amount_b']:.1f}B",
            d["interest_rate"],
            d["maturity"],
            d["collateral"][:15],
        )
    
    console.print()
    console.print(debt_table)
    
    # Timeline
    timeline = build_debt_maturity_timeline()
    
    timeline_table = Table(title="When Debt Comes Due: Collateral Shortfall", expand=False)
    timeline_table.add_column("Year", style="bold", justify="center")
    timeline_table.add_column("Facility", style="cyan")
    timeline_table.add_column("Debt Due", justify="right")
    timeline_table.add_column("Value @ Maturity", justify="right", style="red")
    timeline_table.add_column("Shortfall", justify="right", style="bold red")
    
    total_shortfall = 0
    for item in timeline:
        total_shortfall += item.shortfall_b
        timeline_table.add_row(
            str(item.year),
            item.facility[:20],
            f"${item.debt_due_b:.1f}B",
            f"${item.collateral_at_maturity_b:.1f}B",
            f"${item.shortfall_b:.1f}B",
        )
    
    console.print()
    console.print(timeline_table)
    
    # Key risks
    risk_lines = ["[bold red]KEY RISKS FOR CRWV DEBT HOLDERS[/bold red]", ""]
    for r in risk.key_risks:
        risk_lines.append(f"  • {r}")
    
    console.print()
    console.print(Panel("\n".join(risk_lines), title="Summary", border_style="red", expand=False))


def _show_gpu_puts(console: Console, settings) -> None:
    """Show put options for short stack tickers."""
    from ai_options_trader.data.alpaca import fetch_option_chain, make_clients
    from ai_options_trader.utils.occ import parse_occ_option_symbol
    
    console.print("\n[bold cyan]PUT OPTIONS FOR SHORT STACK[/bold cyan]\n")
    
    _, data = make_clients(settings)
    today = date.today()
    
    for ticker in ["CRWV", "SMCI", "NVDA"]:
        console.print(f"[bold]{ticker}[/bold]")
        
        chain = fetch_option_chain(data, ticker, feed=settings.alpaca_options_feed)
        if not chain:
            console.print("  [dim]No options data[/dim]")
            continue
        
        puts_list = []
        for opt in chain.values():
            symbol = str(getattr(opt, "symbol", ""))
            if not symbol:
                continue
            try:
                expiry, opt_type, strike = parse_occ_option_symbol(symbol, ticker)
                if opt_type != "put":
                    continue
                dte = (expiry - today).days
                if dte < 60 or dte > 180:
                    continue
                
                greeks = getattr(opt, "greeks", None)
                delta = getattr(greeks, "delta", None) if greeks else None
                quote = getattr(opt, "latest_quote", None)
                ask = getattr(quote, "ask_price", None) if quote else None
                
                if delta and -0.40 <= delta <= -0.20:
                    puts_list.append({
                        "strike": strike,
                        "dte": dte,
                        "delta": delta,
                        "ask": float(ask) if ask else 0,
                    })
            except Exception:
                continue
        
        if puts_list:
            puts_list.sort(key=lambda x: x["dte"])
            for p in puts_list[:3]:
                cost = p["ask"] * 100
                console.print(f"  ${p['strike']:.0f} put, {p['dte']} DTE, Δ={p['delta']:.2f}, ~${cost:.0f}")
        else:
            console.print("  [dim]No puts in target delta range[/dim]")
        
        console.print()

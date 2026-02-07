"""
LOX Capital CLI - Simplified

Primary commands:
- lox research regimes/ticker/portfolio
- lox nav/account/status  
- lox dashboard
- lox weekly report
"""
from __future__ import annotations

from datetime import date
import typer

app = typer.Typer(
    add_completion=False, 
    help="""LOX Capital CLI — Research & Portfolio Management

\b
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESEARCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  lox research regimes       Unified regime view + LLM
  lox research ticker NVDA   Hedge fund level analysis
  lox research portfolio     Outlook on open positions

\b
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACCOUNTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  lox status                 Portfolio health (NAV, P&L)
  lox nav snapshot           NAV and investor ledger
  lox account                Account summary

\b
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REPORTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  lox dashboard              Regime dashboard (Heroku)
  lox weekly report --share  Investor report

\b
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRADING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  lox scan -t NVDA           Options chain scanner

\b
Run 'lox <command> --help' for details.
"""
)


# ---------------------------------------------------------------------------
# TOP-LEVEL COMMANDS
# ---------------------------------------------------------------------------

@app.command("scan")
def scan_cmd(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
    want: str = typer.Option("put", "--want", "-w", help="call or put"),
    min_days: int = typer.Option(30, "--min-days", help="Min DTE"),
    max_days: int = typer.Option(365, "--max-days", help="Max DTE"),
    filter_delta: float = typer.Option(None, "--delta", "-d", help="Filter by delta"),
    max_iv: float = typer.Option(None, "--max-iv", help="Max IV"),
    min_iv: float = typer.Option(None, "--min-iv", help="Min IV"),
    show: int = typer.Option(30, "--show", "-n", help="Number of results"),
):
    """Options chain scanner."""
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
    
    console.print(f"\n[bold cyan]{t} {w.upper()}s[/bold cyan] | DTE: {min_days}-{max_days}\n")
    
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
            opt_iv = getattr(opt, "implied_volatility", None)
            
            quote = getattr(opt, "latest_quote", None)
            bid = getattr(quote, "bid_price", None) if quote else None
            ask = getattr(quote, "ask_price", None) if quote else None
            
            opts.append({
                "symbol": symbol, "strike": strike, "dte": dte,
                "delta": float(opt_delta) if opt_delta else None,
                "theta": float(opt_theta) if opt_theta else None,
                "iv": float(opt_iv) if opt_iv else None,
                "bid": float(bid) if bid else None,
                "ask": float(ask) if ask else None,
            })
        except Exception:
            continue
    
    if not opts:
        console.print(f"[yellow]No {w}s in {min_days}-{max_days} DTE[/yellow]")
        return
    
    # Apply filters
    if filter_delta is not None:
        target_delta = abs(filter_delta)
        opts = [o for o in opts if o["delta"] is not None and abs(abs(o["delta"]) - target_delta) <= 0.05]
    
    if min_iv is not None:
        opts = [o for o in opts if o["iv"] is not None and o["iv"] >= min_iv]
    
    if max_iv is not None:
        opts = [o for o in opts if o["iv"] is not None and o["iv"] <= max_iv]
    
    # Sort by strike
    opts.sort(key=lambda x: (x["strike"], x["dte"]))
    opts = opts[:show]
    
    table = Table(show_header=True, expand=False)
    table.add_column("Symbol", style="cyan")
    table.add_column("Strike", justify="right")
    table.add_column("DTE", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("IV", justify="right")
    table.add_column("Bid", justify="right")
    table.add_column("Ask", justify="right", style="yellow")
    
    for o in opts:
        table.add_row(
            o["symbol"],
            f"${o['strike']:.2f}",
            str(o["dte"]),
            f"{o['delta']:+.2f}" if o["delta"] else "—",
            f"{o['iv']:.0%}" if o["iv"] else "—",
            f"${o['bid']:.2f}" if o["bid"] else "—",
            f"${o['ask']:.2f}" if o["ask"] else "—",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# SUBGROUPS
# ---------------------------------------------------------------------------

# NAV / Accounting
nav_app = typer.Typer(add_completion=False, help="NAV and fund accounting")
app.add_typer(nav_app, name="nav")

# Weekly reports
weekly_app = typer.Typer(add_completion=False, help="Weekly reports")
app.add_typer(weekly_app, name="weekly")

# Regimes (for drill-down)
regime_app = typer.Typer(add_completion=False, help="Regime analysis")
app.add_typer(regime_app, name="regime")


@regime_app.command("vol")
def regime_vol(llm: bool = typer.Option(False, "--llm", help="Include LLM")):
    """Volatility regime."""
    from ai_options_trader.cli_commands.regimes.volatility_cmd import volatility_snapshot
    volatility_snapshot(llm=llm)


@regime_app.command("fiscal")
def regime_fiscal():
    """Fiscal regime."""
    from ai_options_trader.cli_commands.regimes.fiscal_cmd import fiscal_snapshot
    fiscal_snapshot()


@regime_app.command("funding")
def regime_funding():
    """Funding regime."""
    from ai_options_trader.cli_commands.regimes.funding_cmd import funding_snapshot
    funding_snapshot()


@regime_app.command("rates")
def regime_rates():
    """Rates regime."""
    from ai_options_trader.cli_commands.regimes.rates_cmd import rates_snapshot
    rates_snapshot()


@regime_app.command("macro")
def regime_macro():
    """Macro regime."""
    from ai_options_trader.cli_commands.regimes.macro_cmd import macro_snapshot
    macro_snapshot()


# ---------------------------------------------------------------------------
# COMMAND REGISTRATION
# ---------------------------------------------------------------------------

_COMMANDS_REGISTERED = False


def _register_commands() -> None:
    global _COMMANDS_REGISTERED
    if _COMMANDS_REGISTERED:
        return
    
    # Research module (primary interface)
    from ai_options_trader.cli_commands.research import register_research_commands
    register_research_commands(app)
    
    # Core commands
    from ai_options_trader.cli_commands.core.core_cmd import register_core
    from ai_options_trader.cli_commands.core.dashboard_cmd import register as register_dashboard
    from ai_options_trader.cli_commands.core.nav_cmd import register as register_nav
    from ai_options_trader.cli_commands.core.account_cmd import register as register_account
    from ai_options_trader.cli_commands.core.weekly_report_cmd import register as register_weekly_report
    from ai_options_trader.cli_commands.core.investor_report_cmd import register as register_investor_report
    from ai_options_trader.cli_commands.core.closed_trades_cmd import register as register_closed_trades
    
    register_core(app)
    register_dashboard(app)
    register_closed_trades(app)
    register_nav(nav_app)
    register_account(app)
    register_weekly_report(weekly_app)
    register_investor_report(weekly_app)
    
    _COMMANDS_REGISTERED = True


def main():
    _register_commands()
    app()


# Register on import
_register_commands()


if __name__ == "__main__":
    main()

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
    from lox.config import load_settings
    from lox.data.alpaca import fetch_option_chain, make_clients
    from lox.utils.occ import parse_occ_option_symbol
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


def _make_db_app():
    """Build a minimal Flask app for CLI database access (local SQLite or Heroku Postgres)."""
    import os as _os

    from flask import Flask
    from dashboard.models import db, bcrypt

    _app = Flask(__name__)
    _app.config["SECRET_KEY"] = _os.environ.get("FLASK_SECRET_KEY", "cli-temp")

    db_url = _os.environ.get(
        "DATABASE_URL",
        "sqlite:///" + _os.path.join(_os.path.dirname(__file__), "..", "..", "data", "lox_users.db"),
    )
    # Heroku uses "postgres://" but SQLAlchemy 2.x requires "postgresql://"
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    _app.config["SQLALCHEMY_DATABASE_URI"] = db_url
    _app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(_app)
    bcrypt.init_app(_app)
    return _app


@app.command("invite-investor")
def invite_investor_cmd(
    code: str = typer.Option(..., "--code", "-c", help="Investor code from nav_investor_flows.csv (e.g. JL)"),
    email: str = typer.Option(..., "--email", "-e", prompt="Investor email", help="Email address to invite"),
    base_url: str = typer.Option("http://localhost:5001", "--base-url", help="Dashboard base URL"),
):
    """Send an invite to an existing investor, linking their account to their investor code."""
    import sys as _sys
    import os as _os
    import csv

    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", ".."))

    from dotenv import load_dotenv
    load_dotenv()

    # Validate investor code exists in flows CSV
    flows_path = _os.path.join(_os.path.dirname(__file__), "..", "..", "data", "nav_investor_flows.csv")
    known_codes: set[str] = set()
    if _os.path.exists(flows_path):
        with open(flows_path, newline="") as f:
            for row in csv.DictReader(f):
                c = (row.get("code") or "").strip().upper()
                if c:
                    known_codes.add(c)

    code_upper = code.strip().upper()
    if code_upper not in known_codes:
        typer.echo(f"Warning: Code '{code_upper}' not found in nav_investor_flows.csv")
        typer.echo(f"  Known codes: {', '.join(sorted(known_codes))}")
        if not typer.confirm("Create invite anyway?"):
            raise typer.Exit()

    from dashboard.models import db, Invite, User

    _app = _make_db_app()

    with _app.app_context():
        db.create_all()

        # Check if user already exists with this code
        existing = User.query.filter_by(investor_code=code_upper).first()
        if existing:
            typer.echo(f"Error: Investor code '{code_upper}' is already linked to user '{existing.username}' ({existing.email})")
            raise typer.Exit(code=1)

        # Check for pending invite with same code
        pending = Invite.query.filter_by(investor_code=code_upper, accepted_at=None).first()
        if pending and pending.is_valid:
            typer.echo(f"Note: A pending invite for code '{code_upper}' already exists (sent to {pending.email})")
            if not typer.confirm("Create a new invite anyway?"):
                raise typer.Exit()

        invite = Invite.create(investor_code=code_upper, email=email.strip().lower())
        url = f"{base_url.rstrip('/')}/auth/register?invite={invite.token}"

        typer.echo("")
        typer.echo(f"  Invite created for investor {code_upper}")
        typer.echo(f"  Email:   {invite.email}")
        typer.echo(f"  Expires: {invite.expires_at.strftime('%Y-%m-%d')}")
        typer.echo(f"")
        typer.echo(f"  Registration link:")
        typer.echo(f"  {url}")
        typer.echo("")


@app.command("create-admin")
def create_admin_cmd(
    email: str = typer.Option(..., "--email", "-e", prompt="Admin email", help="Admin email address"),
    password: str = typer.Option(..., "--password", "-p", prompt="Password", hide_input=True, help="Admin password"),
    first_name: str = typer.Option("", "--first-name", "-f", help="First name"),
    last_name: str = typer.Option("", "--last-name", "-l", help="Last name"),
    investor_code: str = typer.Option(None, "--investor-code", "-c", help="Optional investor code to link (e.g. JL)"),
):
    """Create an admin user in the dashboard database."""
    import sys as _sys
    import os as _os

    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", ".."))

    from dotenv import load_dotenv
    load_dotenv()

    from dashboard.models import db, User

    _app = _make_db_app()

    with _app.app_context():
        db.create_all()

        if User.query.filter_by(email=email.lower()).first():
            typer.echo(f"Error: A user with email '{email}' already exists.", err=True)
            raise typer.Exit(code=1)
        if len(password) < 8:
            typer.echo("Error: Password must be at least 8 characters.", err=True)
            raise typer.Exit(code=1)

        # Auto-generate username from email prefix
        username = email.lower().split("@")[0]
        if User.query.filter_by(username=username).first():
            import uuid as _uuid
            username = f"{username}_{_uuid.uuid4().hex[:6]}"

        user = User(
            email=email.lower(),
            username=username,
            first_name=first_name.strip(),
            last_name=last_name.strip(),
            is_admin=True,
            investor_code=investor_code.strip().upper() if investor_code else None,
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        code_str = f", investor_code={user.investor_code}" if user.investor_code else ""
        typer.echo(f"Admin user created: {email}{code_str} [id={user.id}]")


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


# ── Keep regimes ──────────────────────────────────────────────────────────
@regime_app.command("vol")
def regime_vol(llm: bool = typer.Option(False, "--llm", help="Include LLM")):
    """Volatility regime."""
    from lox.cli_commands.regimes.volatility_cmd import volatility_snapshot
    volatility_snapshot(llm=llm)


@regime_app.command("fiscal")
def regime_fiscal():
    """Fiscal regime."""
    from lox.cli_commands.regimes.fiscal_cmd import fiscal_snapshot
    fiscal_snapshot()


@regime_app.command("funding")
def regime_funding():
    """Funding regime."""
    from lox.cli_commands.regimes.funding_cmd import funding_snapshot
    funding_snapshot()


@regime_app.command("rates")
def regime_rates():
    """Rates regime."""
    from lox.cli_commands.regimes.rates_cmd import rates_snapshot
    rates_snapshot()


@regime_app.command("commodities")
def regime_commodities(llm: bool = typer.Option(False, "--llm", help="Include LLM")):
    """Commodities regime."""
    from lox.cli_commands.regimes.commodities_cmd import _run_commodities_snapshot
    _run_commodities_snapshot(llm=llm)


@regime_app.command("monetary")
def regime_monetary(llm: bool = typer.Option(False, "--llm", help="Include LLM")):
    """Monetary regime."""
    from lox.cli_commands.regimes.monetary_cmd import _run_monetary_snapshot
    _run_monetary_snapshot(llm=llm)


@regime_app.command("usd")
def regime_usd(llm: bool = typer.Option(False, "--llm", help="Include LLM")):
    """USD regime."""
    from lox.cli_commands.regimes.usd_cmd import run_usd_snapshot
    run_usd_snapshot(llm=llm)


# ── NEW regimes (Feb 2026 restructure) ───────────────────────────────────
@regime_app.command("growth")
def regime_growth(llm: bool = typer.Option(False, "--llm", help="Include LLM")):
    """Growth regime (split from macro)."""
    from lox.cli_commands.regimes.growth_cmd import growth_snapshot
    growth_snapshot(llm=llm)


@regime_app.command("inflation")
def regime_inflation(llm: bool = typer.Option(False, "--llm", help="Include LLM")):
    """Inflation regime (split from macro)."""
    from lox.cli_commands.regimes.inflation_cmd import inflation_snapshot
    inflation_snapshot(llm=llm)


@regime_app.command("credit")
def regime_credit(llm: bool = typer.Option(False, "--llm", help="Include LLM")):
    """Credit / spreads regime."""
    from lox.cli_commands.regimes.credit_cmd import credit_snapshot
    credit_snapshot(llm=llm)


@regime_app.command("consumer")
def regime_consumer(llm: bool = typer.Option(False, "--llm", help="Include LLM")):
    """Consumer health regime (replaces housing)."""
    from lox.cli_commands.regimes.consumer_cmd import consumer_snapshot
    consumer_snapshot(llm=llm)


@regime_app.command("positioning")
def regime_positioning(llm: bool = typer.Option(False, "--llm", help="Include LLM")):
    """Market positioning regime."""
    from lox.cli_commands.regimes.positioning_cmd import positioning_snapshot
    positioning_snapshot(llm=llm)


# ── MACRO alias (shows Growth + Inflation + quadrant) ────────────────────
@regime_app.command("macro")
def regime_macro():
    """Macro regime (alias: shows Growth + Inflation + macro quadrant)."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print()
    console.print("[bold cyan]Macro regime has been split into Growth + Inflation.[/bold cyan]")
    console.print("[dim]Showing both sub-regimes...[/dim]\n")

    from lox.cli_commands.regimes.growth_cmd import growth_snapshot
    growth_snapshot()

    console.print()

    from lox.cli_commands.regimes.inflation_cmd import inflation_snapshot
    inflation_snapshot()

    # Show macro quadrant
    try:
        from lox.regimes.features import build_unified_regime_state, _compute_macro_quadrant
        from lox.config import load_settings as _ls
        state = build_unified_regime_state(settings=_ls())
        quadrant = _compute_macro_quadrant(state.growth, state.inflation)
        g_label = state.growth.label if state.growth else "?"
        g_score = f"{state.growth.score:.0f}" if state.growth else "?"
        i_label = state.inflation.label if state.inflation else "?"
        i_score = f"{state.inflation.score:.0f}" if state.inflation else "?"
        console.print()
        console.print(Panel(
            f"[bold]Macro Quadrant: {quadrant}[/bold]\n"
            f"Growth: {g_label} ({g_score}) + Inflation: {i_label} ({i_score})",
            border_style="cyan",
        ))
    except Exception:
        pass


# Register unified / transitions directly on regime_app
from lox.cli_commands.regimes.regimes_cmd import register as _register_regimes
_register_regimes(regime_app)


# ---------------------------------------------------------------------------
# COMMAND REGISTRATION
# ---------------------------------------------------------------------------

_COMMANDS_REGISTERED = False


def _register_commands() -> None:
    global _COMMANDS_REGISTERED
    if _COMMANDS_REGISTERED:
        return
    
    # Research module (primary interface)
    from lox.cli_commands.research import register_research_commands
    register_research_commands(app)
    
    # Core commands
    from lox.cli_commands.core.core_cmd import register_core
    from lox.cli_commands.core.dashboard_cmd import register as register_dashboard
    from lox.cli_commands.core.nav_cmd import register as register_nav
    from lox.cli_commands.core.account_cmd import register as register_account
    from lox.cli_commands.core.weekly_report_cmd import register as register_weekly_report
    from lox.cli_commands.core.investor_report_cmd import register as register_investor_report
    from lox.cli_commands.core.closed_trades_cmd import register as register_closed_trades
    
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

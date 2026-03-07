from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from datetime import datetime, timedelta, timezone

from lox.config import load_settings
from lox.data.alpaca import make_clients
from lox.overlay.context import build_trackers, extract_underlyings, fetch_calendar_events, fetch_news_payload
from lox.data.quotes import fetch_stock_last_prices


def _to_float(x) -> float | None:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def _make_db_app():
    """Build a minimal Flask app for CLI database access (local SQLite or Heroku Postgres)."""
    import os as _os

    from flask import Flask
    from dashboard.models import db, bcrypt

    _app = Flask(__name__)
    _app.config["SECRET_KEY"] = _os.environ.get("FLASK_SECRET_KEY", "cli-temp")

    db_url = _os.environ.get(
        "DATABASE_URL",
        "sqlite:///" + _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "data", "lox_users.db"),
    )
    # Heroku uses "postgres://" but SQLAlchemy 2.x requires "postgresql://"
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    _app.config["SQLALCHEMY_DATABASE_URI"] = db_url
    _app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(_app)
    bcrypt.init_app(_app)
    return _app


def register(app: typer.Typer) -> None:
    account_app = typer.Typer(add_completion=False, help="Account tools")
    app.add_typer(account_app, name="account")

    def _account_panel(console: Console) -> None:
        settings = load_settings()
        trading, _data = make_clients(settings)
        mode = "PAPER" if bool(settings.alpaca_paper) else "LIVE"
        try:
            acct = trading.get_account()
        except Exception as e:
            raise RuntimeError(
                f"Alpaca authorization failed (mode={mode}). "
                "Most common cause: ALPACA_PAPER=false but you're using PAPER keys. "
                "Fix by setting ALPACA_PAPER=true, or by creating LIVE keys and setting ALPACA_API_KEY/ALPACA_API_SECRET."
            ) from e

        cash = _to_float(getattr(acct, "cash", None)) or 0.0
        equity = _to_float(getattr(acct, "equity", None)) or 0.0
        bp = _to_float(getattr(acct, "buying_power", None)) or 0.0
        opt_bp = _to_float(getattr(acct, "options_buying_power", None))
        # Some Alpaca accounts expose these fields; best-effort only.
        cash_withdrawable = _to_float(getattr(acct, "cash_withdrawable", None))
        cash_transferable = _to_float(getattr(acct, "cash_transferable", None))

        console.print(
            Panel(
                f"[b]Mode:[/b] {mode}\n"
                f"[b]Cash:[/b] ${cash:,.2f}\n"
                f"[b]Equity:[/b] ${equity:,.2f}\n"
                f"[b]Buying power:[/b] ${bp:,.2f}\n"
                f"[b]Options BP:[/b] {('n/a' if opt_bp is None else f'${opt_bp:,.2f}')}\n"
                f"[b]Cash withdrawable:[/b] {('n/a' if cash_withdrawable is None else f'${cash_withdrawable:,.2f}')}\n"
                f"[b]Cash transferable:[/b] {('n/a' if cash_transferable is None else f'${cash_transferable:,.2f}')}",
                title="Alpaca account",
                expand=False,
            )
        )

    @account_app.callback(invoke_without_command=True)
    def account_root(ctx: typer.Context):
        """
        Verify Alpaca connectivity and print current capital.

        Uses `ALPACA_PAPER` to choose paper vs live.
        """
        if ctx.invoked_subcommand is not None:
            return
        console = Console()
        _account_panel(console)

    @account_app.command("summary")
    def account_summary(
        orders: int = typer.Option(10, "--orders", help="Best-effort: include this many recent closed orders (0 disables)"),
        news: bool = typer.Option(True, "--news/--no-news", help="Include recent headlines + reading links (FMP) and sentiment"),
        news_days: int = typer.Option(7, "--news-days", help="Lookback window (days) for news"),
        news_max_items: int = typer.Option(18, "--news-max-items", help="Max news items to include in the LLM payload"),
        calendar_days: int = typer.Option(10, "--calendar-days", help="How many days ahead to include economic calendar events"),
        calendar_max_items: int = typer.Option(18, "--calendar-max-items", help="Max calendar events to include"),
        model: str = typer.Option("", "--model", help="Optional OpenAI model override"),
        temperature: float = typer.Option(0.2, "--temperature", help="LLM temperature"),
    ):
        """
        LLM summary of your account + open positions: trades, themes, and risks to watch.
        """
        settings = load_settings()
        console = Console()
        trading, _data = make_clients(settings)

        acct = trading.get_account()
        cash = _to_float(getattr(acct, "cash", None)) or 0.0
        equity = _to_float(getattr(acct, "equity", None)) or 0.0
        bp = _to_float(getattr(acct, "buying_power", None)) or 0.0
        opt_bp = _to_float(getattr(acct, "options_buying_power", None))
        mode = "PAPER" if bool(settings.alpaca_paper) else "LIVE"

        # Positions
        raw_positions = trading.get_all_positions()
        positions: list[dict] = []
        for p in raw_positions:
            positions.append(
                {
                    "symbol": getattr(p, "symbol", ""),
                    "qty": _to_float(getattr(p, "qty", None)),
                    "avg_entry_price": _to_float(getattr(p, "avg_entry_price", None)),
                    "current_price": _to_float(getattr(p, "current_price", None)),
                    "unrealized_pl": _to_float(getattr(p, "unrealized_pl", None)),
                    "unrealized_plpc": _to_float(getattr(p, "unrealized_plpc", None)),
                }
            )

        held = extract_underlyings([str(p.get("symbol") or "") for p in positions])

        # Best-effort recent orders (Alpaca SDK versions differ; tolerate absence).
        recent_orders: list[dict] = []
        if int(orders) > 0:
            try:
                from alpaca.trading.requests import GetOrdersRequest  # type: ignore
                from alpaca.trading.enums import QueryOrderStatus  # type: ignore

                req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=int(orders), nested=True)
                ords = trading.get_orders(req)
                for o in ords or []:
                    recent_orders.append(
                        {
                            "id": getattr(o, "id", None),
                            "symbol": getattr(o, "symbol", None),
                            "side": str(getattr(o, "side", None) or ""),
                            "type": str(getattr(o, "type", None) or ""),
                            "qty": str(getattr(o, "qty", None) or ""),
                            "filled_qty": str(getattr(o, "filled_qty", None) or ""),
                            "filled_avg_price": str(getattr(o, "filled_avg_price", None) or ""),
                            "status": str(getattr(o, "status", None) or ""),
                            "submitted_at": str(getattr(o, "submitted_at", None) or ""),
                            "filled_at": str(getattr(o, "filled_at", None) or ""),
                        }
                    )
            except Exception:
                recent_orders = []

        from lox.llm.account_summary import llm_account_summary
        # Risk watch: current trackers + upcoming econ events (best-effort; uses caches where available).
        _feat_row, risk_watch = build_trackers(settings=settings, start_date="2012-01-01", refresh_fred=False)
        try:
            risk_watch["events"] = fetch_calendar_events(settings=settings, days_ahead=int(calendar_days), max_items=int(calendar_max_items))
        except Exception:
            risk_watch["events"] = []

        # News + sentiment (URLs for "reading list").
        news_payload: dict = {}
        if bool(news) and held:
            try:
                news_payload = fetch_news_payload(
                    settings=settings,
                    tickers=sorted(held),
                    lookback_days=int(news_days),
                    max_items=int(news_max_items),
                )
            except Exception:
                news_payload = {}

        asof = datetime.now(timezone.utc).date().isoformat()
        text = llm_account_summary(
            settings=settings,
            asof=asof,
            account={"mode": mode, "cash": cash, "equity": equity, "buying_power": bp, "options_buying_power": opt_bp},
            positions=positions,
            recent_orders=recent_orders,
            risk_watch=risk_watch,
            news=news_payload,
            model=(model.strip() or None),
            temperature=float(temperature),
        )
        console.print(Panel(text, title="Account summary (LLM)", expand=False))

    @account_app.command("invite-investor")
    def invite_investor_cmd(
        code: str = typer.Option(..., "--code", "-c", help="Investor code from nav_investor_flows.csv (e.g. JL)"),
        email: str = typer.Option(..., "--email", "-e", prompt="Investor email", help="Email address to invite"),
        base_url: str = typer.Option("http://localhost:5001", "--base-url", help="Dashboard base URL"),
    ):
        """Send an invite to an existing investor, linking their account to their investor code."""
        import sys as _sys
        import os as _os
        import csv

        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "..", ".."))

        from dotenv import load_dotenv
        load_dotenv()

        # Validate investor code exists in flows CSV
        flows_path = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "data", "nav_investor_flows.csv")
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

    @account_app.command("create-admin")
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

        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "..", ".."))

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
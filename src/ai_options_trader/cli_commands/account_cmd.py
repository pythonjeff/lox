from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from datetime import datetime, timedelta, timezone

from ai_options_trader.config import load_settings
from ai_options_trader.data.alpaca import make_clients
from ai_options_trader.overlay.context import build_trackers, extract_underlyings, fetch_calendar_events, fetch_news_payload


def _to_float(x) -> float | None:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


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

        console.print(
            Panel(
                f"[b]Mode:[/b] {mode}\n"
                f"[b]Cash:[/b] ${cash:,.2f}\n"
                f"[b]Equity:[/b] ${equity:,.2f}\n"
                f"[b]Buying power:[/b] ${bp:,.2f}",
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

        from ai_options_trader.llm.account_summary import llm_account_summary
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
            account={"mode": mode, "cash": cash, "equity": equity, "buying_power": bp},
            positions=positions,
            recent_orders=recent_orders,
            risk_watch=risk_watch,
            news=news_payload,
            model=(model.strip() or None),
            temperature=float(temperature),
        )
        console.print(Panel(text, title="Account summary (LLM)", expand=False))



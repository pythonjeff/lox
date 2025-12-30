from __future__ import annotations

import typer
from rich import print


def _json_short(s: str) -> str:
    import json

    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return ",".join(str(x) for x in obj)
        return str(obj)
    except Exception:
        return s


def register(track_app: typer.Typer) -> None:
    @track_app.command("recent")
    def track_recent(limit: int = typer.Option(20, "--limit")):
        """Show recent recommendations and executions from the local tracker DB."""
        from rich.table import Table
        from rich.console import Console
        from ai_options_trader.tracking.store import TrackerStore, default_tracker_db_path

        store = TrackerStore(default_tracker_db_path())
        recos = store.list_recent_recommendations(limit=limit)
        execs = store.list_recent_executions(limit=limit)

        t1 = Table(title="Recent recommendations")
        t1.add_column("created_at")
        t1.add_column("ticker")
        t1.add_column("dir")
        t1.add_column("score")
        t1.add_column("tags")
        for r in recos:
            t1.add_row(r.created_at[:19], r.ticker, r.direction, f"{r.score:.2f}", _json_short(r.tags_json))

        t2 = Table(title="Recent executions")
        t2.add_column("created_at")
        t2.add_column("symbol")
        t2.add_column("qty")
        t2.add_column("status")
        t2.add_column("order_id")
        for e in execs:
            t2.add_row(e.created_at[:19], e.symbol, str(e.qty), e.status, e.alpaca_order_id)

        c = Console()
        c.print(t1)
        c.print(t2)

    @track_app.command("sync")
    def track_sync(limit: int = typer.Option(50, "--limit", help="Max executions to sync (most recent first)")):
        """
        Sync recent executions with Alpaca to update status/fills in the local tracker DB.
        """
        from rich.console import Console
        from ai_options_trader.tracking.store import TrackerStore, default_tracker_db_path
        from ai_options_trader.config import load_settings
        from ai_options_trader.data.alpaca import make_clients

        settings = load_settings()
        trading, _data = make_clients(settings)

        store = TrackerStore(default_tracker_db_path())
        execs = store.list_recent_executions(limit=limit)

        c = Console()
        updated = 0
        for e in execs:
            try:
                o = trading.get_order_by_id(e.alpaca_order_id)
            except Exception as ex:
                c.print(f"[yellow]WARN[/yellow] could not fetch order {e.alpaca_order_id}: {ex}")
                continue

            store.update_execution_from_alpaca(
                alpaca_order_id=e.alpaca_order_id,
                status=str(getattr(o, "status", "")),
                filled_qty=int(getattr(o, "filled_qty", 0) or 0),
                filled_avg_price=float(getattr(o, "filled_avg_price", 0) or 0) or None,
                filled_at=str(getattr(o, "filled_at", None) or "") or None,
                raw=o.model_dump() if hasattr(o, "model_dump") else str(o),
            )
            updated += 1

        c.print(f"[green]Synced[/green] {updated} execution(s) into {default_tracker_db_path()}")

    @track_app.command("report")
    def track_report():
        """
        Show a quick performance snapshot for tracked option symbols using Alpaca positions.
        """
        from rich.table import Table
        from rich.console import Console
        from ai_options_trader.tracking.store import TrackerStore, default_tracker_db_path
        from ai_options_trader.config import load_settings
        from ai_options_trader.data.alpaca import make_clients

        settings = load_settings()
        trading, _data = make_clients(settings)
        store = TrackerStore(default_tracker_db_path())

        # Pull current positions (includes unrealized P/L on Alpaca)
        try:
            positions = trading.get_all_positions()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch Alpaca positions: {e}")

        pos_by_symbol = {p.symbol: p for p in positions}

        execs = store.list_recent_executions(limit=200)
        symbols = sorted({e.symbol for e in execs})

        tbl = Table(title="Tracked performance (current positions)")
        tbl.add_column("symbol", style="bold")
        tbl.add_column("qty", justify="right")
        tbl.add_column("avg_entry", justify="right")
        tbl.add_column("current", justify="right")
        tbl.add_column("uPL", justify="right")
        tbl.add_column("uPL%", justify="right")

        for sym in symbols:
            p = pos_by_symbol.get(sym)
            if not p:
                continue
            qty = getattr(p, "qty", "")
            avg = getattr(p, "avg_entry_price", None)
            cur = getattr(p, "current_price", None)
            upl = getattr(p, "unrealized_pl", None)
            uplpc = getattr(p, "unrealized_plpc", None)
            tbl.add_row(
                sym,
                str(qty),
                f"{float(avg):.2f}" if avg is not None else "—",
                f"{float(cur):.2f}" if cur is not None else "—",
                f"{float(upl):.2f}" if upl is not None else "—",
                f"{float(uplpc)*100:.1f}%" if uplpc is not None else "—",
            )

        Console().print(tbl)



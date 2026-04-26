"""Automatic daily NAV snapshots.

Piggybacks on dashboard API traffic — when an endpoint is hit after market
close on a weekday, a snapshot is appended to nav_sheet.csv if one hasn't
been written for today yet.  No background threads, no cron.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from lox.nav.store import NavSnapshot, default_nav_sheet_path, read_nav_sheet

_ET = ZoneInfo("America/New_York")

# Snapshot after 4:05 PM ET (5 min after close for settlement).
_CLOSE_HOUR = 16
_CLOSE_MINUTE = 5


def has_snapshot_today(*, path: str | None = None) -> bool:
    """True if the last row of nav_sheet.csv is dated today (ET)."""
    rows = read_nav_sheet(path=path)
    if not rows:
        return False
    last_ts = rows[-1].ts
    try:
        last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        last_date = last_dt.astimezone(_ET).date()
    except Exception:
        return False
    today = datetime.now(_ET).date()
    return last_date == today


def should_auto_snapshot(*, path: str | None = None) -> bool:
    """True when it's after market close on a weekday and no snapshot exists for today."""
    now = datetime.now(_ET)
    # Skip weekends (5 = Saturday, 6 = Sunday).
    if now.weekday() >= 5:
        return False
    # Wait until after close.
    if (now.hour, now.minute) < (_CLOSE_HOUR, _CLOSE_MINUTE):
        return False
    return not has_snapshot_today(path=path)


def take_auto_snapshot(*, path: str | None = None) -> NavSnapshot | None:
    """Take a snapshot if conditions are met.  Never raises."""
    try:
        if not should_auto_snapshot(path=path):
            return None

        from lox.config import load_settings
        from lox.data.alpaca import make_clients
        from lox.nav.store import append_nav_snapshot

        settings = load_settings()
        trading, _ = make_clients(settings)
        account = trading.get_account()

        equity = float(getattr(account, "equity", 0) or 0)
        cash = float(getattr(account, "cash", 0) or 0)
        buying_power = float(getattr(account, "buying_power", 0) or 0)
        positions = trading.get_all_positions()
        positions_count = len(list(positions))

        _, snap = append_nav_snapshot(
            equity=equity,
            cash=cash,
            buying_power=buying_power,
            positions_count=positions_count,
            note="auto",
            sheet_path=path,
        )
        print(f"[AutoSnapshot] Recorded equity=${equity:,.2f}  TWR={snap.twr_cum:.4%}")
        return snap
    except Exception as e:
        print(f"[AutoSnapshot] Error: {e}")
        return None

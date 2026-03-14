"""Investor ledger and unitized NAV computation."""

from datetime import datetime, timezone
import os

from lox.config import load_settings
from lox.nav.store import default_nav_sheet_path, read_nav_sheet
from lox.nav.investors import default_investor_flows_path, investor_report, read_investor_flows
from dashboard.cache import INVESTORS_CACHE, INVESTORS_CACHE_LOCK, INVESTORS_CACHE_TTL


def get_investor_data():
    """
    Unitized NAV investor ledger (hedge fund style) using LIVE Alpaca equity.
    Delegates to nav.investors.investor_report() for single source of truth.
    """
    try:
        if not os.path.exists(default_investor_flows_path()):
            return {"error": "Investor flows file not found", "investors": [], "fund_return": 0, "total_capital": 0}

        # Get live equity (Alpaca or nav_sheet fallback)
        live_equity = None
        try:
            # Lazy import to avoid circular imports
            from dashboard.positions import get_positions_data
            positions_data = get_positions_data()
            live_equity = positions_data.get("nav_equity")
            if not live_equity or live_equity <= 0:
                settings = load_settings()
                from lox.data.alpaca import make_clients
                trading, _ = make_clients(settings)
                account = trading.get_account()
                if account:
                    live_equity = float(getattr(account, "equity", 0) or 0)
        except Exception as e:
            print(f"[Investors] Live equity fetch error: {e}")

        rep = investor_report(
            nav_sheet_path=default_nav_sheet_path(),
            investor_flows_path=default_investor_flows_path(),
            live_equity=live_equity,
        )
        rows = rep.get("rows") or []
        total_capital = float(rep.get("total_capital") or 0)
        equity = float(rep.get("equity") or 0)

        # Prefer TWR from NAV sheet (strips out cash-flow timing bias)
        from lox.nav.store import read_nav_sheet
        nav_rows = read_nav_sheet()
        fund_return_decimal = float(nav_rows[-1].twr_cum) if nav_rows else float(rep.get("fund_return") or 0)

        # Map to dashboard shape: return_pct in percent, ownership in percent (0-100)
        investors = []
        for r in rows:
            ret = float(r.get("return") or 0)
            ownership = float(r.get("ownership") or 0)
            investors.append({
                "code": r.get("code", ""),
                "ownership": round(ownership * 100, 1),
                "units": round(float(r.get("units", 0)), 2),
                "basis": round(float(r.get("basis", 0)), 2),
                "value": round(float(r.get("value", 0)), 2),
                "pnl": round(float(r.get("pnl", 0)), 2),
                "return_pct": round(ret * 100, 1),
            })

        return {
            "investors": investors,
            "nav_per_unit": round(float(rep.get("nav_per_unit") or 1.0), 4),
            "total_units": round(float(rep.get("total_units") or 0), 2),
            "fund_return": round(fund_return_decimal * 100, 2),
            "total_capital": round(total_capital, 2),
            "equity": round(equity, 2),
            "fund_pnl": round(equity - total_capital, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "live": True,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "investors": [], "fund_return": 0, "total_capital": 0}

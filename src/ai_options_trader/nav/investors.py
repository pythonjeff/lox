from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ai_options_trader.nav.store import _parse_ts, read_nav_sheet


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_investor_flows_path() -> str:
    # Internal "fund ledger" of investor contributions/withdrawals.
    return os.environ.get("AOT_NAV_INVESTOR_FLOWS", "data/nav_investor_flows.csv")


@dataclass(frozen=True)
class InvestorFlow:
    ts: str
    code: str
    amount: float
    note: str


_FIELDS = ["ts", "code", "amount", "note"]


def append_investor_flow(
    *,
    code: str,
    amount: float,
    note: str = "",
    ts: str | None = None,
    path: str | None = None,
) -> str:
    path = path or default_investor_flows_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ts = ts or _utc_now_iso()
    row = {"ts": ts, "code": str(code).strip().upper(), "amount": f"{float(amount):.6f}", "note": note}
    file_exists = Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists:
            w.writeheader()
        w.writerow(row)
    return path


def read_investor_flows(*, path: str | None = None) -> list[InvestorFlow]:
    path = path or default_investor_flows_path()
    if not Path(path).exists():
        return []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        out: list[InvestorFlow] = []
        for row in r:
            out.append(
                InvestorFlow(
                    ts=str(row.get("ts") or ""),
                    code=str(row.get("code") or "").strip().upper(),
                    amount=float(row.get("amount") or 0.0),
                    note=str(row.get("note") or ""),
                )
            )
    out.sort(key=lambda x: _parse_ts(x.ts))
    return out


def compute_unitization(
    *,
    nav_sheet_path: str | None = None,
    investor_flows_path: str | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Compute a unitized NAV ledger:
    - Start nav_per_unit at 1.0
    - When investor flow happens at time t: units = amount / nav_per_unit
    - When a NAV snapshot happens at time t: nav_per_unit = equity / total_units (if total_units>0)

    Returns: (latest_nav_per_unit, units_by_investor)
    """
    nav_rows = read_nav_sheet(path=nav_sheet_path)  # sorted
    flows = read_investor_flows(path=investor_flows_path)  # sorted

    # Merge events chronologically.
    events: list[tuple[datetime, str, object]] = []
    for f in flows:
        events.append((_parse_ts(f.ts), "flow", f))
    for r in nav_rows:
        events.append((_parse_ts(r.ts), "nav", r))
    events.sort(key=lambda x: x[0])

    nav_per_unit = 1.0
    units_by: dict[str, float] = {}
    total_units = 0.0

    for _ts, kind, obj in events:
        if kind == "flow":
            f: InvestorFlow = obj  # type: ignore[assignment]
            if nav_per_unit <= 0:
                nav_per_unit = 1.0
            du = float(f.amount) / float(nav_per_unit)
            units_by[f.code] = float(units_by.get(f.code, 0.0)) + du
            total_units += du
        else:
            r = obj  # nav snapshot row
            if total_units > 0:
                nav_per_unit = float(getattr(r, "equity", 0.0)) / float(total_units)
            else:
                nav_per_unit = 1.0

    return float(nav_per_unit), units_by


def investor_report(
    *,
    nav_sheet_path: str | None = None,
    investor_flows_path: str | None = None,
) -> dict:
    nav_rows = read_nav_sheet(path=nav_sheet_path)
    last_equity = float(nav_rows[-1].equity) if nav_rows else 0.0
    nav_per_unit, units_by = compute_unitization(
        nav_sheet_path=nav_sheet_path,
        investor_flows_path=investor_flows_path,
    )
    total_units = sum(units_by.values()) if units_by else 0.0

    # Cost basis per investor is simply sum of their flows (signed).
    flows = read_investor_flows(path=investor_flows_path)
    basis_by: dict[str, float] = {}
    for f in flows:
        basis_by[f.code] = float(basis_by.get(f.code, 0.0)) + float(f.amount)

    out_rows = []
    for code in sorted(units_by.keys()):
        units = float(units_by.get(code, 0.0))
        value = units * float(nav_per_unit)
        basis = float(basis_by.get(code, 0.0))
        pnl = value - basis
        ret = (pnl / basis) if basis != 0 else None
        ownership = (units / total_units) if total_units > 0 else None
        out_rows.append(
            {
                "code": code,
                "units": units,
                "basis": basis,
                "value": value,
                "pnl": pnl,
                "return": ret,
                "ownership": ownership,
            }
        )

    return {
        "asof": nav_rows[-1].ts if nav_rows else None,
        "equity": last_equity,
        "nav_per_unit": nav_per_unit,
        "total_units": total_units,
        "rows": out_rows,
    }


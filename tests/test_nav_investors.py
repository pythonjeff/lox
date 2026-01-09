from __future__ import annotations

from pathlib import Path

from ai_options_trader.nav.investors import append_investor_flow, investor_report
from ai_options_trader.nav.store import append_nav_snapshot


def test_investor_unitization_basic(tmp_path: Path):
    sheet = str(tmp_path / "nav_sheet.csv")
    inv_flows = str(tmp_path / "nav_investor_flows.csv")

    # Two investors deposit before first snapshot.
    append_investor_flow(ts="2026-01-01T00:00:00+00:00", code="JL", amount=75.0, note="seed", path=inv_flows)
    append_investor_flow(ts="2026-01-01T00:00:00+00:00", code="TG", amount=100.0, note="seed", path=inv_flows)

    # First snapshot: equity is sum of deposits.
    append_nav_snapshot(
        ts="2026-01-01T12:00:00+00:00",
        equity=175.0,
        cash=175.0,
        buying_power=175.0,
        positions_count=0,
        note="start",
        sheet_path=sheet,
        flows_path=str(tmp_path / "unused.csv"),
    )

    # Account grows to 210 with no new investor flows => nav_per_unit increases.
    append_nav_snapshot(
        ts="2026-02-01T00:00:00+00:00",
        equity=210.0,
        cash=210.0,
        buying_power=210.0,
        positions_count=0,
        note="up",
        sheet_path=sheet,
        flows_path=str(tmp_path / "unused.csv"),
    )

    rep = investor_report(nav_sheet_path=sheet, investor_flows_path=inv_flows)
    assert rep["equity"] == 210.0
    rows = {r["code"]: r for r in rep["rows"]}

    # Ownership should be proportional to deposits (since both deposited at same nav_per_unit).
    assert abs(rows["JL"]["ownership"] - (75.0 / 175.0)) < 1e-9
    assert abs(rows["TG"]["ownership"] - (100.0 / 175.0)) < 1e-9

    # Values should sum (approximately) to equity.
    total_value = sum(r["value"] for r in rep["rows"])
    assert abs(total_value - 210.0) < 1e-6


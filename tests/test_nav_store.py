from __future__ import annotations

from pathlib import Path

from ai_options_trader.nav.store import append_cashflow, append_nav_snapshot, read_nav_sheet


def test_nav_snapshot_twr_with_flows(tmp_path: Path):
    sheet = str(tmp_path / "nav_sheet.csv")
    flows = str(tmp_path / "nav_flows.csv")

    # Initial deposit (should not create a return by itself; return begins from first snapshot onward)
    append_cashflow(ts="2026-01-01T00:00:00+00:00", amount=100.0, note="deposit", path=flows)

    # First snapshot: equity=100, no return computed (no prev equity baseline)
    _, s1 = append_nav_snapshot(
        ts="2026-01-01T12:00:00+00:00",
        equity=100.0,
        cash=100.0,
        buying_power=100.0,
        positions_count=0,
        note="start",
        sheet_path=sheet,
        flows_path=flows,
    )
    assert s1.twr_since_prev is None
    assert s1.twr_cum == 0.0

    # Another deposit between snapshots
    append_cashflow(ts="2026-01-10T00:00:00+00:00", amount=50.0, note="deposit2", path=flows)

    # Second snapshot: equity increases to 160; net flow is +50; pnl is +10; return is +10% on prev equity.
    _, s2 = append_nav_snapshot(
        ts="2026-01-20T00:00:00+00:00",
        equity=160.0,
        cash=160.0,
        buying_power=160.0,
        positions_count=0,
        note="mid",
        sheet_path=sheet,
        flows_path=flows,
    )
    assert round(float(s2.net_flow_since_prev or 0.0), 6) == 50.0
    assert round(float(s2.pnl_since_prev or 0.0), 6) == 10.0
    assert abs(float(s2.twr_since_prev or 0.0) - 0.10) < 1e-9
    assert abs(float(s2.twr_cum) - 0.10) < 1e-9

    # Third snapshot: no flows, equity to 152 => pnl -8, return -5% on 160, cum compounding.
    _, s3 = append_nav_snapshot(
        ts="2026-02-01T00:00:00+00:00",
        equity=152.0,
        cash=152.0,
        buying_power=152.0,
        positions_count=0,
        note="end",
        sheet_path=sheet,
        flows_path=flows,
    )
    assert round(float(s3.net_flow_since_prev or 0.0), 6) == 0.0
    assert round(float(s3.pnl_since_prev or 0.0), 6) == -8.0
    assert abs(float(s3.twr_since_prev or 0.0) - (-0.05)) < 1e-9
    expected_cum = (1.0 + 0.10) * (1.0 - 0.05) - 1.0
    assert abs(float(s3.twr_cum) - expected_cum) < 1e-9

    rows = read_nav_sheet(path=sheet)
    assert len(rows) == 3


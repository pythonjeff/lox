from datetime import date

from lox.cli_commands.volpicks_cmd import _pick_call_contract, _call_metrics, _append_snapshot
from lox.data.alpaca import OptionCandidate


def _c(dte, delta, bid=2.0, ask=2.1, oi=500, opt_type="call"):
    return OptionCandidate(
        symbol=f"X{dte}{delta}", opt_type=opt_type, expiry=date(2027, 1, 15), strike=100.0,
        dte_days=dte, delta=delta, gamma=None, theta=None, vega=None, iv=0.4,
        oi=oi, volume=10, bid=bid, ask=ask, last=None,
    )


def test_window_excludes_short_dated():
    picked = _pick_call_contract([_c(30, 0.40), _c(95, 0.45)], target_dte=90)
    assert picked.dte_days == 95


def test_illiquid_skipped():
    assert _pick_call_contract([_c(90, 0.40, oi=5), _c(90, 0.40, bid=1.0, ask=2.0)], target_dte=90) is None


def test_missing_oi_not_rejected():
    assert _pick_call_contract([_c(90, 0.40, oi=None)], target_dte=90) is not None


def test_delta_then_dte_tiebreak():
    picked = _pick_call_contract([_c(80, 0.40), _c(90, 0.40), _c(90, 0.30)], target_dte=90)
    assert picked.dte_days == 90 and picked.delta == 0.40


def test_puts_ignored():
    assert _pick_call_contract([_c(90, 0.40, opt_type="put")], target_dte=90) is None


def test_delta_cap_and_oi_lookup():
    assert _pick_call_contract([_c(90, 0.60)], target_dte=90) is None
    thin = _c(90, 0.40)
    picked = _pick_call_contract([thin], target_dte=90, oi_lookup=lambda s: {thin.symbol: 5})
    assert picked is None
    picked = _pick_call_contract([thin], target_dte=90, oi_lookup=lambda s: {thin.symbol: 900})
    assert picked.oi == 900


def test_call_metrics_breakeven_and_ratio():
    c = _c(87, 0.40, bid=11.8, ask=11.9)  # strike 100, mid 11.85, iv 0.4
    m = _call_metrics(c, spot=90.0, rv=0.20)
    assert abs(m["be_pct"] - (111.85 / 90.0 - 1)) < 1e-9
    assert abs(m["iv_rv"] - 2.0) < 1e-9
    assert abs(m["prem_pct"] - 11.85 / 90.0) < 1e-9
    assert m["be_sigma"] > 0


def test_snapshot_append_is_idempotent(tmp_path):
    p = tmp_path / "snap.csv"
    rows = [{"date": "2026-10-20", "symbol": "ACN", "atm_iv": 0.41}]
    assert _append_snapshot(rows, p) == 1
    assert _append_snapshot(rows, p) == 0
    assert _append_snapshot([{"date": "2026-10-21", "symbol": "ACN"}], p) == 1
    assert len(p.read_text().strip().splitlines()) == 3

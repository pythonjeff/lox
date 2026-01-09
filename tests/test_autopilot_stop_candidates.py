from __future__ import annotations

from ai_options_trader.cli_commands.autopilot_cmd import _stop_candidates


def test_stop_candidates_threshold():
    positions = [
        {"symbol": "A", "unrealized_plpc": -0.10},
        {"symbol": "B", "unrealized_plpc": -0.31},
        {"symbol": "C", "unrealized_plpc": None},
        {"symbol": "D", "unrealized_plpc": -0.30},
    ]
    out = _stop_candidates(positions, stop_loss_pct=0.30)
    syms = {p["symbol"] for p in out}
    assert syms == {"B", "D"}



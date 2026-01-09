from __future__ import annotations

from ai_options_trader.universe.sp500 import _parse_symbols_from_fmp


def test_parse_symbols_from_fmp():
    js = [
        {"symbol": "AAPL", "name": "Apple Inc."},
        {"symbol": "msft", "name": "Microsoft"},
        {"symbol": "BRK.B", "name": "Berkshire Hathaway"},
        {"Symbol": "GOOGL"},
        {"symbol": ""},
        {},
        "bad",
    ]
    out = _parse_symbols_from_fmp(js)
    assert out[:4] == ["AAPL", "MSFT", "BRK.B", "GOOGL"]



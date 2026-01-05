from __future__ import annotations

from typing import Any


def fmt_usd_from_millions(x: Any) -> str:
    """
    Format a numeric value expressed in USD *millions* into $M/$B/$T.
    Returns "n/a" when the value is not numeric.
    """
    if not isinstance(x, (int, float)):
        return "n/a"
    dollars = float(x) * 1_000_000.0
    if abs(dollars) >= 1_000_000_000_000:
        return f"${dollars/1_000_000_000_000:,.2f}T"
    if abs(dollars) >= 1_000_000_000:
        return f"${dollars/1_000_000_000:,.0f}B"
    return f"${dollars/1_000_000:,.0f}M"



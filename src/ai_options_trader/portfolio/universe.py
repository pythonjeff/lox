from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioUniverse:
    """
    Default "macro pod" universe:
    - Core equity beta proxies
    - Rates / duration
    - Credit
    - USD
    - Gold
    - Sectors

    v1 executes equities/ETFs only (no options, no crypto execution).
    """

    basket_equity: tuple[str, ...]
    tradable: tuple[str, ...]


DEFAULT_UNIVERSE = PortfolioUniverse(
    basket_equity=("SPY", "QQQ", "IWM", "XLF", "XLK", "XLE"),
    tradable=(
        "SPY",
        "QQQ",
        "IWM",
        "TLT",
        "HYG",
        "LQD",
        "UUP",
        "GLD",
        "XLF",
        "XLK",
        "XLE",
    ),
)



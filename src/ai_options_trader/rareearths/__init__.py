"""
Rare Earths & Critical Minerals Tracker

Tracks companies in the rare earth and critical minerals supply chain:
- Miners and processors
- Magnet manufacturers  
- Defense/EV supply chain
- China exposure and alternatives
"""

from .tracker import (
    RE_SECURITIES,
    RE_BASKETS,
    RareEarthSecurity,
    RareEarthReport,
    build_rareearth_report,
    get_market_signals,
)

__all__ = [
    "RE_SECURITIES",
    "RE_BASKETS",
    "RareEarthSecurity",
    "RareEarthReport",
    "build_rareearth_report",
    "get_market_signals",
]

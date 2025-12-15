from __future__ import annotations

"""Source/provider interfaces for the tariff module.

This package exists to keep imports stable as the project evolves.
The canonical protocol definition lives in `ai_options_trader.tariff.base`.
"""

from ai_options_trader.tariff.base import TariffDataProvider

__all__ = ["TariffDataProvider"]



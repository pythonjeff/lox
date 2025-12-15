from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Dict, List
import pandas as pd


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    frequency: str  # "daily" or "monthly"


class TariffDataProvider(Protocol):
    def get_cost_proxies(self, start_date: str) -> pd.DataFrame:
        """
        Return DataFrame indexed by date with one or more cost proxy columns.
        Example columns: import_price, apparel_ppi, freight_proxy
        """
        ...

    def get_equity_prices(self, symbols: List[str], start_date: str) -> pd.DataFrame:
        """
        Return DataFrame indexed by date with close prices columns per symbol.
        """
        ...

    def get_earnings_fragility(self, symbols: List[str], start_date: str) -> pd.DataFrame:
        """
        Return DataFrame indexed by date with fragility proxies per symbol.
        For now this can be empty; we will wire actual data later.
        """
        ...

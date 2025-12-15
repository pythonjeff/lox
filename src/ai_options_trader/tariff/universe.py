from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class Basket:
    name: str
    tickers: List[str]
    description: str


BASKETS: Dict[str, Basket] = {
    "import_retail_apparel": Basket(
        name="import_retail_apparel",
        tickers=["NKE", "LULU", "DECK", "VFC", "GPS", "UAA"],
        description="Import-heavy apparel/footwear brands (illustrative list; refine later).",
    ),
    "big_box_retail": Basket(
        name="big_box_retail",
        tickers=["WMT", "TGT", "COST", "AMZN"],
        description="Large retailers with significant imported goods exposure.",
    ),
}

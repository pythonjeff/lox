"""
ETF Tariff Impact Screener

Identifies ETFs most exposed to increased import duties based on:
1. Sector exposure to tariff-sensitive industries
2. Geographic exposure (China, EM supply chains)
3. Historical sensitivity to cost proxy changes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TariffETF:
    """ETF with tariff exposure characteristics."""
    ticker: str
    name: str
    category: str
    tariff_exposure: str  # "high", "medium", "low"
    exposure_rationale: str
    direction: str  # "hurt" or "benefit" from tariffs


# ETFs categorized by tariff exposure
TARIFF_EXPOSED_ETFS: Dict[str, List[TariffETF]] = {
    "import_dependent": [
        TariffETF("XRT", "SPDR S&P Retail", "Consumer", "high", 
                  "Retailers rely heavily on imported goods (apparel, electronics, home goods)", "hurt"),
        TariffETF("XLY", "Consumer Discretionary Select", "Consumer", "high",
                  "Discretionary spending on imported goods (autos, apparel, electronics)", "hurt"),
        TariffETF("XHB", "SPDR Homebuilders", "Housing", "medium",
                  "Imported lumber, appliances, fixtures drive costs", "hurt"),
        TariffETF("ITB", "iShares Home Construction", "Housing", "medium",
                  "Construction materials import exposure", "hurt"),
    ],
    "tech_hardware": [
        TariffETF("SMH", "VanEck Semiconductor", "Tech", "high",
                  "Chip manufacturing relies on global supply chains, China exposure", "hurt"),
        TariffETF("SOXX", "iShares Semiconductor", "Tech", "high",
                  "Semiconductor equipment and chips heavily traded globally", "hurt"),
        TariffETF("XLK", "Technology Select", "Tech", "medium",
                  "Hardware components, consumer electronics import exposure", "hurt"),
    ],
    "materials_industrials": [
        TariffETF("SLX", "VanEck Steel", "Materials", "high",
                  "Direct tariff target (Section 232); domestic steel may benefit", "benefit"),
        TariffETF("XLB", "Materials Select", "Materials", "medium",
                  "Mixed - some domestic producers benefit, importers hurt", "mixed"),
        TariffETF("XLI", "Industrial Select", "Industrials", "medium",
                  "Manufacturing equipment, machinery import/export exposure", "hurt"),
        TariffETF("IYT", "iShares Transportation", "Industrials", "medium",
                  "Lower trade volumes hurt shipping; fuel costs matter", "hurt"),
    ],
    "china_em_exposed": [
        TariffETF("FXI", "iShares China Large-Cap", "International", "high",
                  "Direct exposure to US-China trade tensions", "hurt"),
        TariffETF("MCHI", "iShares MSCI China", "International", "high",
                  "Broad China equity exposure - trade war sensitivity", "hurt"),
        TariffETF("KWEB", "KraneShares China Internet", "International", "high",
                  "Chinese tech companies facing regulatory and trade headwinds", "hurt"),
        TariffETF("EEM", "iShares MSCI Emerging Markets", "International", "medium",
                  "Supply chain disruption, EM export economies impacted", "hurt"),
    ],
    "agriculture_retaliation": [
        TariffETF("MOO", "VanEck Agribusiness", "Agriculture", "high",
                  "US ag exports face retaliatory tariffs (soybeans, pork, corn)", "hurt"),
        TariffETF("DBA", "Invesco DB Agriculture", "Commodities", "medium",
                  "Agricultural commodity prices sensitive to trade flows", "hurt"),
        TariffETF("CORN", "Teucrium Corn Fund", "Commodities", "medium",
                  "Corn exports face retaliatory tariff risk", "hurt"),
        TariffETF("SOYB", "Teucrium Soybean Fund", "Commodities", "high",
                  "Soybeans primary target of Chinese retaliation", "hurt"),
    ],
    "potential_beneficiaries": [
        TariffETF("SLX", "VanEck Steel", "Materials", "high",
                  "Domestic steel producers may benefit from import protection", "benefit"),
        TariffETF("XAR", "SPDR Aerospace & Defense", "Industrials", "low",
                  "Defense spending less trade-sensitive, domestic focus", "benefit"),
        TariffETF("PAVE", "Global X US Infrastructure", "Industrials", "medium",
                  "Domestic infrastructure focus, may benefit from reshoring", "benefit"),
    ],
}


def get_all_tariff_etfs() -> List[TariffETF]:
    """Get all tariff-exposed ETFs in a flat list."""
    all_etfs = []
    for etfs in TARIFF_EXPOSED_ETFS.values():
        all_etfs.extend(etfs)
    # Remove duplicates (SLX appears twice)
    seen = set()
    unique = []
    for etf in all_etfs:
        if etf.ticker not in seen:
            seen.add(etf.ticker)
            unique.append(etf)
    return unique


def get_high_exposure_etfs() -> List[TariffETF]:
    """Get only high-exposure ETFs."""
    return [e for e in get_all_tariff_etfs() if e.tariff_exposure == "high"]


def get_etfs_by_direction(direction: str) -> List[TariffETF]:
    """Get ETFs by their expected tariff impact direction."""
    return [e for e in get_all_tariff_etfs() if e.direction == direction]

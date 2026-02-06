"""
Rare Earths & Critical Minerals Tracker

Tracks the rare earth and critical minerals supply chain for investment analysis.
Key themes: China dominance, EV/defense demand, supply chain de-risking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import requests

from ai_options_trader.config import Settings


# =============================================================================
# RARE EARTH SECURITIES DATABASE
# =============================================================================

RE_SECURITIES = {
    # =========================================================================
    # PURE PLAY RARE EARTH MINERS
    # =========================================================================
    "MP": {
        "name": "MP Materials",
        "category": "miner",
        "subcategory": "pure_play",
        "description": "Only scaled rare earth mine in North America (Mountain Pass, CA)",
        "re_revenue_pct": 100,
        "key_products": ["NdPr oxide", "Lanthanum", "Cerium"],
        "production_mt": 43000,  # metric tons REO/year
        "reserves_years": 30,
        "china_exposure": "low",  # Processing independence
        "ev_exposure": "high",  # EV magnets
        "defense_exposure": "high",  # F-35, missiles
        "key_customers": ["GM (offtake)", "DOD contracts"],
        "vertical_integration": "Building magnetics facility in TX",
        "bear_sensitivity": "high",  # Commodity price sensitive
        "key_risk": "RE price volatility, China dumping, execution risk on magnetics",
        "thesis": "US rare earth independence play, but execution risk on downstream",
    },
    "UUUU": {
        "name": "Energy Fuels",
        "category": "miner",
        "subcategory": "diversified",
        "description": "Uranium miner pivoting to rare earth processing",
        "re_revenue_pct": 25,  # Growing
        "key_products": ["Uranium", "Rare earth carbonate", "Vanadium"],
        "china_exposure": "low",
        "ev_exposure": "medium",
        "defense_exposure": "high",  # Uranium for navy, RE for defense
        "key_customers": ["Utilities", "Neo Performance"],
        "vertical_integration": "Processing monazite sands",
        "bear_sensitivity": "medium",
        "key_risk": "RE is still small part of business, uranium price driven",
        "thesis": "Uranium + RE optionality, processing capability is valuable",
    },
    "LAC": {
        "name": "Lithium Americas",
        "category": "miner",
        "subcategory": "lithium",
        "description": "Developing Thacker Pass - largest lithium deposit in US",
        "re_revenue_pct": 0,  # Lithium, not RE, but critical mineral
        "key_products": ["Lithium carbonate"],
        "china_exposure": "low",
        "ev_exposure": "extreme",
        "defense_exposure": "medium",
        "key_customers": ["GM (offtake)"],
        "vertical_integration": "Mine to chemical",
        "bear_sensitivity": "extreme",  # Pre-revenue, lithium price
        "key_risk": "Permitting, lithium price crash, execution",
        "thesis": "US lithium independence, but years from production",
    },
    
    # =========================================================================
    # RARE EARTH PROCESSORS & MAGNET MAKERS
    # =========================================================================
    "AMTX": {
        "name": "Aemetis",
        "category": "processor",
        "subcategory": "emerging",
        "description": "Renewable fuels with rare earth processing ambitions",
        "re_revenue_pct": 5,
        "china_exposure": "low",
        "ev_exposure": "low",
        "defense_exposure": "low",
        "bear_sensitivity": "high",
        "key_risk": "RE is speculative, main business is biofuels",
        "thesis": "Speculative RE optionality",
    },
    
    # =========================================================================
    # INTERNATIONAL MINERS (NON-CHINA)
    # =========================================================================
    "LYSCF": {
        "name": "Lynas Rare Earths",
        "category": "miner",
        "subcategory": "pure_play",
        "description": "Largest RE producer outside China (Australia/Malaysia)",
        "re_revenue_pct": 100,
        "key_products": ["NdPr", "Heavy rare earths"],
        "production_mt": 22000,
        "china_exposure": "none",
        "ev_exposure": "high",
        "defense_exposure": "high",
        "key_customers": ["Siemens Gamesa", "DOD (building Texas facility)"],
        "vertical_integration": "Mine to separated oxides",
        "bear_sensitivity": "high",
        "key_risk": "Malaysia processing license, RE prices",
        "thesis": "Best-in-class non-China RE producer, DOD backing",
    },
    "ILMAF": {
        "name": "Iluka Resources",
        "category": "miner",
        "subcategory": "diversified",
        "description": "Mineral sands producer with RE project (Eneabba, Australia)",
        "re_revenue_pct": 10,
        "key_products": ["Zircon", "Titanium", "Rare earths"],
        "china_exposure": "low",
        "ev_exposure": "medium",
        "defense_exposure": "medium",
        "key_customers": ["Government grants for RE"],
        "bear_sensitivity": "medium",
        "key_risk": "RE still emerging, mineral sands cyclical",
        "thesis": "Diversified critical minerals with RE upside",
    },
    
    # =========================================================================
    # DOWNSTREAM / END USERS
    # =========================================================================
    "ALB": {
        "name": "Albemarle",
        "category": "processor",
        "subcategory": "lithium_leader",
        "description": "World's largest lithium producer",
        "re_revenue_pct": 0,  # Lithium, not RE
        "key_products": ["Lithium", "Bromine"],
        "china_exposure": "medium",  # Some China processing
        "ev_exposure": "extreme",
        "defense_exposure": "low",
        "key_customers": ["Tesla", "All EV OEMs"],
        "bear_sensitivity": "high",
        "key_risk": "Lithium price volatility, China competition",
        "thesis": "Lithium bellwether, EV demand proxy",
    },
    "SQM": {
        "name": "Sociedad Química y Minera",
        "category": "miner",
        "subcategory": "lithium",
        "description": "Chilean lithium producer (Atacama)",
        "re_revenue_pct": 0,
        "key_products": ["Lithium", "Potassium", "Iodine"],
        "china_exposure": "low",
        "ev_exposure": "extreme",
        "defense_exposure": "low",
        "key_customers": ["Global EV supply chain"],
        "bear_sensitivity": "high",
        "key_risk": "Chile nationalization risk, lithium prices",
        "thesis": "Low-cost lithium producer, but political risk",
    },
    "LTHM": {
        "name": "Livent",
        "category": "processor",
        "subcategory": "lithium",
        "description": "Pure-play lithium hydroxide producer (merged into Arcadium)",
        "re_revenue_pct": 0,
        "key_products": ["Lithium hydroxide"],
        "china_exposure": "medium",
        "ev_exposure": "extreme",
        "defense_exposure": "low",
        "bear_sensitivity": "high",
        "key_risk": "Lithium price, merged into Arcadium (ALTM)",
        "thesis": "Pure lithium play, now part of Arcadium",
    },
    
    # =========================================================================
    # ETFs & DIVERSIFIED EXPOSURE
    # =========================================================================
    "REMX": {
        "name": "VanEck Rare Earth ETF",
        "category": "etf",
        "subcategory": "pure_play",
        "description": "ETF tracking rare earth miners globally",
        "re_revenue_pct": 100,
        "key_holdings": ["Lynas", "MP Materials", "Pilbara", "China Northern RE"],
        "china_exposure": "high",  # ~40% China stocks
        "ev_exposure": "high",
        "defense_exposure": "high",
        "bear_sensitivity": "high",
        "key_risk": "Heavy China exposure, RE price volatility",
        "thesis": "Diversified RE exposure but China risk",
    },
    "LIT": {
        "name": "Global X Lithium & Battery ETF",
        "category": "etf",
        "subcategory": "lithium",
        "description": "ETF tracking lithium miners and battery makers",
        "re_revenue_pct": 0,  # Lithium focus
        "key_holdings": ["Albemarle", "SQM", "Ganfeng", "BYD"],
        "china_exposure": "high",  # ~30% China
        "ev_exposure": "extreme",
        "defense_exposure": "low",
        "bear_sensitivity": "high",
        "key_risk": "Lithium oversupply, China exposure",
        "thesis": "Broad EV supply chain exposure",
    },
    
    # =========================================================================
    # DEFENSE PRIMES (RE END USERS)
    # =========================================================================
    "LMT": {
        "name": "Lockheed Martin",
        "category": "defense",
        "subcategory": "prime",
        "description": "F-35 and missiles require rare earth magnets",
        "re_revenue_pct": 0,  # End user
        "re_dependency": "high",  # F-35 uses 920 lbs of RE
        "china_exposure": "concern",  # Supply chain risk
        "ev_exposure": "low",
        "defense_exposure": "extreme",
        "bear_sensitivity": "low",  # Diversified, defense budget
        "key_risk": "RE supply disruption could halt production",
        "thesis": "RE supply chain security is national priority",
    },
    "RTX": {
        "name": "RTX (Raytheon)",
        "category": "defense",
        "subcategory": "prime",
        "description": "Missiles and defense systems use RE magnets",
        "re_revenue_pct": 0,
        "re_dependency": "high",
        "china_exposure": "concern",
        "ev_exposure": "low",
        "defense_exposure": "extreme",
        "bear_sensitivity": "low",
        "key_risk": "RE supply chain vulnerability",
        "thesis": "Defense prime with RE supply chain concerns",
    },
}


# =============================================================================
# BASKETS FOR DIFFERENT VIEWS
# =============================================================================

RE_BASKETS = {
    "all": {
        "name": "All Rare Earth Securities",
        "description": "Complete rare earth and critical minerals universe",
        "tickers": list(RE_SECURITIES.keys()),
    },
    "pure_play": {
        "name": "Pure Play RE Miners",
        "description": "Companies with majority RE revenue",
        "tickers": ["MP", "LYSCF", "REMX"],
    },
    "us_focused": {
        "name": "US Supply Chain",
        "description": "US-based or US-focused critical mineral companies",
        "tickers": ["MP", "UUUU", "LAC"],
    },
    "lithium": {
        "name": "Lithium Plays",
        "description": "Lithium producers and processors",
        "tickers": ["ALB", "SQM", "LAC", "LIT"],
    },
    "etfs": {
        "name": "Critical Minerals ETFs",
        "description": "Diversified exposure via ETFs",
        "tickers": ["REMX", "LIT"],
    },
    "defense_chain": {
        "name": "Defense Supply Chain",
        "description": "RE suppliers and defense end users",
        "tickers": ["MP", "LYSCF", "LMT", "RTX"],
    },
    "ev_chain": {
        "name": "EV Supply Chain",
        "description": "EV battery and magnet materials",
        "tickers": ["ALB", "SQM", "LAC", "MP", "LIT"],
    },
}


# =============================================================================
# KEY MARKET DATA
# =============================================================================

RE_MARKET_CONTEXT = {
    "china_dominance": {
        "mining_pct": 60,
        "processing_pct": 90,
        "magnets_pct": 92,
        "trend": "Slight decline as West builds capacity",
    },
    "demand_drivers": [
        "EV motors (NdFeB magnets) - 30% of demand",
        "Wind turbines (permanent magnets) - 25% of demand",
        "Consumer electronics - 20% of demand",
        "Defense/aerospace - 10% of demand",
        "Industrial - 15% of demand",
    ],
    "supply_risks": [
        "China export restrictions (2010 precedent)",
        "Myanmar conflict (heavy RE source)",
        "Processing bottleneck (separation is hard)",
        "Environmental permitting in West",
    ],
    "catalysts_bull": [
        "US/EU supply chain legislation",
        "DOD contracts for domestic supply",
        "China export restrictions",
        "EV adoption acceleration",
    ],
    "catalysts_bear": [
        "China price dumping",
        "EV demand slowdown",
        "Substitution technology breakthroughs",
        "Project delays/cost overruns",
    ],
    "key_prices": {
        "NdPr_oxide_usd_kg": 60,  # Neodymium-Praseodymium (magnet grade)
        "dysprosium_usd_kg": 300,  # Heavy RE for high-temp magnets
        "lithium_carbonate_usd_t": 15000,  # Battery grade
    },
}


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class RareEarthSecurity:
    """A single rare earth related security with current data."""
    ticker: str
    name: str
    category: str
    price: float = 0
    change_1d: float = 0
    change_5d: float = 0
    change_1m: float = 0
    change_3m: float = 0
    change_6m: float = 0
    change_ytd: float = 0
    change_1y: float = 0
    week_52_high: float = 0
    week_52_low: float = 0
    pct_from_52w_high: float = 0
    pct_from_52w_low: float = 0
    re_revenue_pct: int = 0
    china_exposure: str = ""
    ev_exposure: str = ""
    defense_exposure: str = ""
    market_cap_b: float = 0
    pe_ratio: float = 0
    thesis: str = ""


@dataclass
class RareEarthReport:
    """Full rare earth sector report."""
    as_of: str
    securities: list[RareEarthSecurity] = field(default_factory=list)
    basket_performance: dict = field(default_factory=dict)
    bull_signals: list[str] = field(default_factory=list)
    bear_signals: list[str] = field(default_factory=list)
    total_market_cap_b: float = 0
    basket_change_1d: float = 0
    market_context: dict = field(default_factory=dict)


# =============================================================================
# REPORT BUILDING
# =============================================================================

def build_rareearth_report(settings: Settings, basket: str = "all") -> RareEarthReport:
    """Build the rare earth sector report with live data."""
    
    basket_info = RE_BASKETS.get(basket, RE_BASKETS["all"])
    tickers = basket_info["tickers"]
    
    base_url = "https://financialmodelingprep.com/api/v3"
    
    # Fetch quotes for all tickers
    securities = []
    total_market_cap = 0
    changes_1d = []
    
    for ticker in tickers:
        info = RE_SECURITIES.get(ticker, {})
        sec = RareEarthSecurity(
            ticker=ticker,
            name=info.get("name", ticker),
            category=info.get("category", "unknown"),
            re_revenue_pct=info.get("re_revenue_pct", 0),
            china_exposure=info.get("china_exposure", "unknown"),
            ev_exposure=info.get("ev_exposure", "unknown"),
            defense_exposure=info.get("defense_exposure", "unknown"),
            thesis=info.get("thesis", ""),
        )
        
        try:
            # Get quote with basic data
            resp = requests.get(
                f"{base_url}/quote/{ticker}",
                params={"apikey": settings.fmp_api_key},
                timeout=10,
            )
            if resp.ok and resp.json():
                q = resp.json()[0]
                sec.price = q.get("price", 0)
                sec.change_1d = q.get("changesPercentage", 0)
                sec.market_cap_b = q.get("marketCap", 0) / 1e9
                sec.pe_ratio = q.get("pe", 0) or 0
                sec.change_ytd = q.get("ytd", 0) or 0
                sec.week_52_high = q.get("yearHigh", 0) or 0
                sec.week_52_low = q.get("yearLow", 0) or 0
                
                # Calculate distance from 52-week levels
                if sec.price and sec.week_52_high:
                    sec.pct_from_52w_high = ((sec.price - sec.week_52_high) / sec.week_52_high) * 100
                if sec.price and sec.week_52_low and sec.week_52_low > 0:
                    sec.pct_from_52w_low = ((sec.price - sec.week_52_low) / sec.week_52_low) * 100
                
                total_market_cap += sec.market_cap_b
                if sec.change_1d != 0:
                    changes_1d.append(sec.change_1d)
        except Exception:
            pass
        
        # Get historical performance data
        try:
            perf_resp = requests.get(
                f"{base_url}/stock-price-change/{ticker}",
                params={"apikey": settings.fmp_api_key},
                timeout=10,
            )
            if perf_resp.ok and perf_resp.json():
                perf = perf_resp.json()[0]
                sec.change_5d = perf.get("5D", 0) or 0
                sec.change_1m = perf.get("1M", 0) or 0
                sec.change_3m = perf.get("3M", 0) or 0
                sec.change_6m = perf.get("6M", 0) or 0
                sec.change_1y = perf.get("1Y", 0) or 0
                # Override YTD if we got it from performance endpoint
                if perf.get("ytd"):
                    sec.change_ytd = perf.get("ytd", 0)
        except Exception:
            pass
        
        securities.append(sec)
    
    # Calculate basket performance
    basket_change_1d = sum(changes_1d) / len(changes_1d) if changes_1d else 0
    
    # Get market signals
    bull_signals, bear_signals = get_market_signals(securities)
    
    return RareEarthReport(
        as_of=datetime.now().strftime("%Y-%m-%d %H:%M"),
        securities=sorted(securities, key=lambda x: x.re_revenue_pct, reverse=True),
        bull_signals=bull_signals,
        bear_signals=bear_signals,
        total_market_cap_b=total_market_cap,
        basket_change_1d=basket_change_1d,
        market_context=RE_MARKET_CONTEXT,
    )


def get_market_signals(securities: list[RareEarthSecurity]) -> tuple[list[str], list[str]]:
    """Analyze current state for bull/bear signals."""
    bull = []
    bear = []
    
    # Check MP Materials (US bellwether)
    mp = next((s for s in securities if s.ticker == "MP"), None)
    if mp:
        if mp.change_1d > 3:
            bull.append(f"MP Materials up {mp.change_1d:.1f}% - US RE momentum")
        elif mp.change_1d < -3:
            bear.append(f"MP Materials down {mp.change_1d:.1f}% - sector weakness")
    
    # Check REMX ETF
    remx = next((s for s in securities if s.ticker == "REMX"), None)
    if remx:
        if remx.change_1d > 2:
            bull.append(f"REMX ETF up {remx.change_1d:.1f}% - broad RE strength")
        elif remx.change_1d < -2:
            bear.append(f"REMX ETF down {remx.change_1d:.1f}% - broad RE weakness")
    
    # Check lithium names
    lithium = [s for s in securities if s.ticker in ["ALB", "SQM", "LAC", "LIT"]]
    avg_lithium = sum(s.change_1d for s in lithium) / len(lithium) if lithium else 0
    if avg_lithium > 2:
        bull.append(f"Lithium stocks avg +{avg_lithium:.1f}% - EV demand signal")
    elif avg_lithium < -2:
        bear.append(f"Lithium stocks avg {avg_lithium:.1f}% - EV demand concerns")
    
    # Check defense names
    defense = [s for s in securities if s.ticker in ["LMT", "RTX"]]
    avg_defense = sum(s.change_1d for s in defense) / len(defense) if defense else 0
    if avg_defense > 1.5:
        bull.append(f"Defense primes up {avg_defense:.1f}% - defense spending tailwind")
    
    # Default signals
    if not bull:
        bull.append("No immediate bull signals detected")
    if not bear:
        bear.append("No immediate bear signals detected")
    
    return bull, bear


def get_thesis_summary() -> dict:
    """Get investment thesis summary for rare earths."""
    return {
        "bull_case": [
            "China supplies 60% of mined RE, 90% of processing - concentration risk",
            "EV adoption requires NdFeB magnets (1kg per EV motor)",
            "Defense systems (F-35: 920 lbs of RE) need secure supply",
            "US/EU legislation mandating domestic supply chain",
            "DOD funding domestic processing capacity",
        ],
        "bear_case": [
            "China can flood market to crush Western projects",
            "Long lead times for new mines (7-10 years)",
            "Processing is technically difficult (environmental)",
            "Substitution R&D could reduce RE demand",
            "EV slowdown would crush demand growth",
        ],
        "key_catalysts": [
            "China export restrictions (2010 precedent caused 10x price spike)",
            "IRA/CHIPS Act funding announcements",
            "DOD contract awards for domestic supply",
            "EV adoption rates and policy",
            "RE price movements (NdPr, Dysprosium)",
        ],
        "positioning": {
            "long": "MP Materials (US pure play), Lynas (best non-China producer)",
            "avoid": "China-heavy ETFs if concerned about geopolitical risk",
            "speculative": "UUUU (uranium + RE optionality), LAC (lithium)",
        },
    }

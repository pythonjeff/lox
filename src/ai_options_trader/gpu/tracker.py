"""
GPU-Backed Securities Tracker

Tracks the GPU infrastructure stack for bear thesis monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
import requests

from ai_options_trader.config import Settings


# =============================================================================
# GPU SECURITIES DATABASE
# =============================================================================

GPU_SECURITIES = {
    # GPU CLOUD PROVIDERS - Pure plays renting GPU capacity
    "CRWV": {
        "name": "CoreWeave",
        "category": "gpu_cloud",
        "subcategory": "pure_play",
        "gpu_revenue_pct": 100,
        "business_model": "Rents NVDA GPUs to AI companies",
        "key_customers": ["Microsoft (60%)", "OpenAI (15%)", "Meta (10%)"],
        "gpu_assets": "$8.7B in NVDA GPUs",
        "debt_backed_by_gpus": True,
        "debt_amount_b": 7.5,
        "nvda_investor": True,
        "nvda_customer": True,
        "bear_sensitivity": "extreme",  # Would collapse without AI demand
        "depreciation_risk": "extreme",  # GPUs depreciate fast
        "key_risk": "Microsoft builds own capacity, GPU obsolescence",
    },
    
    # AI INFRASTRUCTURE - Servers, cooling, data centers
    "SMCI": {
        "name": "Super Micro Computer",
        "category": "ai_infrastructure",
        "subcategory": "servers",
        "gpu_revenue_pct": 70,
        "business_model": "Builds AI servers with NVDA GPUs",
        "key_customers": ["Hyperscalers", "AI startups"],
        "gpu_assets": "N/A - assembler",
        "debt_backed_by_gpus": False,
        "nvda_investor": False,
        "nvda_customer": False,
        "bear_sensitivity": "extreme",
        "depreciation_risk": "low",
        "key_risk": "Low margins, accounting concerns, competition",
    },
    "VRT": {
        "name": "Vertiv",
        "category": "ai_infrastructure",
        "subcategory": "cooling",
        "gpu_revenue_pct": 50,
        "business_model": "Power and cooling for AI data centers",
        "key_customers": ["All hyperscalers"],
        "gpu_assets": "N/A",
        "debt_backed_by_gpus": False,
        "nvda_investor": False,
        "nvda_customer": False,
        "bear_sensitivity": "high",
        "depreciation_risk": "low",
        "key_risk": "AI CapEx slowdown hits orders",
    },
    "EQIX": {
        "name": "Equinix",
        "category": "ai_infrastructure",
        "subcategory": "data_centers",
        "gpu_revenue_pct": 25,
        "business_model": "Data center REIT with AI capacity",
        "key_customers": ["Cloud providers", "Enterprises"],
        "gpu_assets": "N/A",
        "debt_backed_by_gpus": False,
        "nvda_investor": False,
        "nvda_customer": False,
        "bear_sensitivity": "medium",
        "depreciation_risk": "low",
        "key_risk": "Hyperscalers build own capacity",
    },
    
    # OPENAI ECOSYSTEM - Direct OpenAI exposure
    "MSFT": {
        "name": "Microsoft",
        "category": "openai_ecosystem",
        "subcategory": "primary_investor",
        "gpu_revenue_pct": 15,
        "business_model": "Azure AI, OpenAI partnership, Copilot",
        "openai_investment_b": 13,
        "openai_profit_share_pct": 49,
        "key_customers": ["Enterprises", "Consumers"],
        "gpu_assets": "$50B+ in NVDA GPUs",
        "debt_backed_by_gpus": False,
        "nvda_investor": False,
        "nvda_customer": True,
        "bear_sensitivity": "low",  # Diversified
        "depreciation_risk": "medium",
        "key_risk": "OpenAI never profitable, AI hype fades",
    },
    "ORCL": {
        "name": "Oracle",
        "category": "openai_ecosystem",
        "subcategory": "cloud_partner",
        "gpu_revenue_pct": 20,
        "business_model": "OCI for OpenAI training, GPU cloud",
        "key_customers": ["OpenAI", "Enterprises"],
        "gpu_assets": "$10B+ in NVDA GPUs",
        "debt_backed_by_gpus": False,
        "nvda_investor": False,
        "nvda_customer": True,
        "bear_sensitivity": "medium",
        "depreciation_risk": "medium",
        "key_risk": "OpenAI shifts workloads, OCI growth stalls",
    },
    
    # NVDA SUPPLY CHAIN - NVDA's key suppliers
    "TSM": {
        "name": "Taiwan Semiconductor",
        "category": "nvda_supply_chain",
        "subcategory": "foundry",
        "gpu_revenue_pct": 25,
        "business_model": "Fabricates NVDA chips",
        "key_customers": ["NVDA", "AMD", "Apple", "Qualcomm"],
        "gpu_assets": "N/A",
        "debt_backed_by_gpus": False,
        "nvda_investor": False,
        "nvda_customer": False,
        "bear_sensitivity": "medium",  # Diversified
        "depreciation_risk": "low",
        "key_risk": "China/Taiwan risk, NVDA demand slowdown",
    },
    "MU": {
        "name": "Micron",
        "category": "nvda_supply_chain",
        "subcategory": "memory",
        "gpu_revenue_pct": 30,
        "business_model": "HBM memory for NVDA GPUs",
        "key_customers": ["NVDA", "AMD", "Intel"],
        "gpu_assets": "N/A",
        "debt_backed_by_gpus": False,
        "nvda_investor": False,
        "nvda_customer": False,
        "bear_sensitivity": "high",
        "depreciation_risk": "low",
        "key_risk": "HBM oversupply, pricing collapse",
    },
    "ASML": {
        "name": "ASML",
        "category": "nvda_supply_chain",
        "subcategory": "equipment",
        "gpu_revenue_pct": 15,
        "business_model": "EUV lithography for advanced chips",
        "key_customers": ["TSM", "Samsung", "Intel"],
        "gpu_assets": "N/A",
        "debt_backed_by_gpus": False,
        "nvda_investor": False,
        "nvda_customer": False,
        "bear_sensitivity": "low",  # Near monopoly
        "depreciation_risk": "low",
        "key_risk": "China restrictions, capex slowdown",
    },
    
    # GPU NETWORKING - AI interconnects
    "MRVL": {
        "name": "Marvell",
        "category": "ai_infrastructure",
        "subcategory": "networking",
        "gpu_revenue_pct": 40,
        "business_model": "Custom AI chips, networking",
        "key_customers": ["Hyperscalers", "NVDA competitors"],
        "gpu_assets": "N/A",
        "debt_backed_by_gpus": False,
        "nvda_investor": False,
        "nvda_customer": False,
        "bear_sensitivity": "high",
        "depreciation_risk": "low",
        "key_risk": "Custom silicon delays, NVDA dominance",
    },
    "ANET": {
        "name": "Arista Networks",
        "category": "ai_infrastructure",
        "subcategory": "networking",
        "gpu_revenue_pct": 35,
        "business_model": "AI data center networking",
        "key_customers": ["Meta", "Microsoft", "Google"],
        "gpu_assets": "N/A",
        "debt_backed_by_gpus": False,
        "nvda_investor": False,
        "nvda_customer": False,
        "bear_sensitivity": "medium",
        "depreciation_risk": "low",
        "key_risk": "AI capex slowdown, competition",
    },
    
    # THE CENTER - NVDA itself
    "NVDA": {
        "name": "NVIDIA",
        "category": "gpu_maker",
        "subcategory": "center",
        "gpu_revenue_pct": 85,
        "business_model": "Designs and sells GPUs for AI",
        "key_customers": ["All hyperscalers", "AI companies"],
        "gpu_assets": "N/A - makes them",
        "debt_backed_by_gpus": False,
        "nvda_investor": True,  # Invests in customers
        "nvda_customer": False,
        "bear_sensitivity": "extreme",  # All roads lead to NVDA
        "depreciation_risk": "creates it",
        "key_risk": "AI demand peak, custom silicon, China",
    },
    
    # AI COMPANIES - Dependent on GPUs
    "PLTR": {
        "name": "Palantir",
        "category": "ai_company",
        "subcategory": "enterprise_ai",
        "gpu_revenue_pct": 40,
        "business_model": "AI platform for enterprises/gov",
        "key_customers": ["Government", "Enterprises"],
        "gpu_assets": "Minimal - uses cloud",
        "debt_backed_by_gpus": False,
        "nvda_investor": False,
        "nvda_customer": True,
        "bear_sensitivity": "medium",
        "depreciation_risk": "low",
        "key_risk": "AI hype fades, valuation compression",
    },
    "AI": {
        "name": "C3.ai",
        "category": "ai_company",
        "subcategory": "enterprise_ai",
        "gpu_revenue_pct": 60,
        "business_model": "Enterprise AI platform",
        "key_customers": ["Enterprises"],
        "gpu_assets": "Minimal",
        "debt_backed_by_gpus": False,
        "nvda_investor": False,
        "nvda_customer": False,
        "bear_sensitivity": "high",
        "depreciation_risk": "low",
        "key_risk": "Unprofitable, competition from MSFT/GOOGL",
    },
}

# Baskets for different views
GPU_BASKETS = {
    "all": {
        "name": "All GPU Securities",
        "description": "Complete GPU infrastructure stack",
        "tickers": list(GPU_SECURITIES.keys()),
    },
    "short_stack": {
        "name": "GPU Short Stack",
        "description": "High-conviction shorts: extreme bear sensitivity",
        "tickers": ["CRWV", "SMCI", "NVDA", "VRT", "MRVL"],
    },
    "pure_gpu": {
        "name": "Pure GPU Plays",
        "description": "Companies with 50%+ GPU revenue exposure",
        "tickers": [t for t, v in GPU_SECURITIES.items() if v.get("gpu_revenue_pct", 0) >= 50],
    },
    "openai": {
        "name": "OpenAI Ecosystem",
        "description": "Companies tied to OpenAI success",
        "tickers": ["MSFT", "ORCL", "NVDA", "CRWV"],
    },
    "debt_risk": {
        "name": "GPU Debt Risk",
        "description": "Companies with GPU-backed debt",
        "tickers": [t for t, v in GPU_SECURITIES.items() if v.get("debt_backed_by_gpus")],
    },
    "supply_chain": {
        "name": "NVDA Supply Chain",
        "description": "NVDA's key suppliers",
        "tickers": ["TSM", "MU", "ASML"],
    },
}


@dataclass
class GPUSecurity:
    """A single GPU-related security with current data."""
    ticker: str
    name: str
    category: str
    price: float = 0
    change_1d: float = 0
    change_5d: float = 0
    change_1m: float = 0
    gpu_revenue_pct: int = 0
    bear_sensitivity: str = ""
    market_cap_b: float = 0
    pe_ratio: float = 0
    short_interest_pct: float = 0
    put_call_ratio: float = 0


@dataclass
class GPUTrackerReport:
    """Full GPU tracker report."""
    as_of: str
    nvda_price: float
    nvda_change_1d: float
    nvda_change_1m: float
    securities: list[GPUSecurity] = field(default_factory=list)
    basket_performance: dict = field(default_factory=dict)
    bear_signals: list[str] = field(default_factory=list)
    bull_signals: list[str] = field(default_factory=list)
    total_gpu_market_cap_b: float = 0
    short_stack_change_1d: float = 0


def build_gpu_tracker_report(settings: Settings, basket: str = "all") -> GPUTrackerReport:
    """Build the GPU tracker report with live data."""
    
    basket_info = GPU_BASKETS.get(basket, GPU_BASKETS["all"])
    tickers = basket_info["tickers"]
    
    base_url = "https://financialmodelingprep.com/api/v3"
    
    # Fetch quotes for all tickers
    securities = []
    nvda_price = 0
    nvda_change_1d = 0
    nvda_change_1m = 0
    total_market_cap = 0
    
    for ticker in tickers:
        info = GPU_SECURITIES.get(ticker, {})
        sec = GPUSecurity(
            ticker=ticker,
            name=info.get("name", ticker),
            category=info.get("category", "unknown"),
            gpu_revenue_pct=info.get("gpu_revenue_pct", 0),
            bear_sensitivity=info.get("bear_sensitivity", "unknown"),
        )
        
        try:
            # Get quote
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
                total_market_cap += sec.market_cap_b
                
                if ticker == "NVDA":
                    nvda_price = sec.price
                    nvda_change_1d = sec.change_1d
        except Exception:
            pass
        
        securities.append(sec)
    
    # Calculate basket performance
    short_stack_tickers = GPU_BASKETS["short_stack"]["tickers"]
    short_stack_changes = [s.change_1d for s in securities if s.ticker in short_stack_tickers and s.change_1d != 0]
    short_stack_change_1d = sum(short_stack_changes) / len(short_stack_changes) if short_stack_changes else 0
    
    # Get bear/bull signals
    bear_signals, bull_signals = get_bear_signals(securities)
    
    return GPUTrackerReport(
        as_of=datetime.now().strftime("%Y-%m-%d %H:%M"),
        nvda_price=nvda_price,
        nvda_change_1d=nvda_change_1d,
        nvda_change_1m=nvda_change_1m,
        securities=sorted(securities, key=lambda x: x.gpu_revenue_pct, reverse=True),
        bear_signals=bear_signals,
        bull_signals=bull_signals,
        total_gpu_market_cap_b=total_market_cap,
        short_stack_change_1d=short_stack_change_1d,
    )


def get_bear_signals(securities: list[GPUSecurity]) -> tuple[list[str], list[str]]:
    """Analyze current state for bear/bull signals."""
    bear = []
    bull = []
    
    # Check CRWV performance
    crwv = next((s for s in securities if s.ticker == "CRWV"), None)
    if crwv:
        if crwv.change_1d < -3:
            bear.append(f"CRWV down {crwv.change_1d:.1f}% today - GPU cloud weakness")
        if crwv.price < 70:
            bear.append(f"CRWV below $70 - critical support breakdown")
    
    # Check SMCI
    smci = next((s for s in securities if s.ticker == "SMCI"), None)
    if smci:
        if smci.change_1d < -5:
            bear.append(f"SMCI down {smci.change_1d:.1f}% - AI server demand concerns")
    
    # Check NVDA
    nvda = next((s for s in securities if s.ticker == "NVDA"), None)
    if nvda:
        if nvda.change_1d < -3:
            bear.append(f"NVDA down {nvda.change_1d:.1f}% - GPU demand concerns")
        elif nvda.change_1d > 3:
            bull.append(f"NVDA up {nvda.change_1d:.1f}% - GPU demand strong")
    
    # Check if high-sensitivity names underperforming
    high_sens = [s for s in securities if s.bear_sensitivity in ["extreme", "high"]]
    avg_change = sum(s.change_1d for s in high_sens) / len(high_sens) if high_sens else 0
    if avg_change < -2:
        bear.append(f"High-sensitivity names avg {avg_change:.1f}% - sector weakness")
    
    # Default signals if nothing notable
    if not bear:
        bear.append("No immediate bear signals detected")
    if not bull:
        bull.append("No immediate bull signals detected")
    
    return bear, bull

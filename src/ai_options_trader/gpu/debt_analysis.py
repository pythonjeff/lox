"""
GPU-Backed Debt Market Analysis

Covers the emerging GPU-backed debt market across multiple companies,
including CoreWeave, hyperscalers, and AI infrastructure providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# =============================================================================
# GPU-BACKED DEBT MARKET OVERVIEW
# =============================================================================

GPU_DEBT_MARKET_OVERVIEW = {
    "total_market_size_b": 50,  # Estimated total GPU-backed financing
    "growth_2024": "+200%",
    "key_players": [
        {"type": "GPU Cloud", "examples": ["CoreWeave", "Lambda Labs", "Crusoe Energy"]},
        {"type": "Hyperscalers", "examples": ["Microsoft", "Oracle", "Google"]},
        {"type": "Specialty Lenders", "examples": ["Magnetar", "BlackRock", "Blackstone"]},
    ],
    "market_dynamics": [
        "GPUs as collateral is a new and untested asset class",
        "Depreciation faster than traditional equipment (2-3 years vs 5-7 years)",
        "Collateral value highly correlated with AI demand",
        "Lenders pricing in tech obsolescence risk",
        "Interest rates typically 2-3% higher than traditional equipment financing",
    ],
    "risk_factors": [
        "Technology cycle risk - new GPU generations every 12-18 months",
        "Demand concentration - few large buyers (hyperscalers)",
        "Pricing power asymmetry - NVIDIA has leverage over buyers",
        "Energy costs - older GPUs become uneconomical to operate",
        "Regulatory risk - AI regulation could impact demand",
    ],
}

# Companies with significant GPU-related debt exposure
GPU_DEBT_COMPANIES = {
    "CRWV": {
        "name": "CoreWeave",
        "total_debt_b": 12.9,
        "gpu_backed_debt_b": 11.8,  # Excludes unsecured convertibles
        "gpu_collateral_value_b": 4.5,
        "ltv_ratio": 2.9,
        "key_risk": "Refinancing $7.5B in 2029 with depreciated H100s",
        "debt_details": "Detailed structure available",
    },
    "ORCL": {
        "name": "Oracle",
        "total_debt_b": 88.5,
        "gpu_backed_debt_b": 0,  # No GPU-backed debt
        "gpu_capex_b": 10,  # But massive GPU CapEx
        "gpu_collateral_value_b": 0,
        "ltv_ratio": 0,
        "key_risk": "Heavy CapEx for AI infrastructure funded by operating cash",
        "debt_details": "Standard corporate debt",
    },
    "MSFT": {
        "name": "Microsoft",
        "total_debt_b": 79.4,
        "gpu_backed_debt_b": 0,  # Funds GPU via CapEx, not debt
        "gpu_capex_b": 50,  # Massive GPU spending
        "gpu_collateral_value_b": 0,
        "ltv_ratio": 0,
        "key_risk": "AI CapEx may not generate ROI",
        "debt_details": "Investment-grade corporate bonds",
    },
    "SMCI": {
        "name": "Super Micro Computer",
        "total_debt_b": 2.0,
        "gpu_backed_debt_b": 0,  # Assembler, doesn't own GPUs
        "gpu_collateral_value_b": 0,
        "ltv_ratio": 0,
        "key_risk": "Inventory risk if AI server demand drops",
        "debt_details": "Working capital facilities",
    },
}

# =============================================================================
# COREWEAVE DEBT STRUCTURE (DETAILED)
# =============================================================================

CRWV_DEBT_STRUCTURE = {
    "total_debt_b": 12.9,  # As of Q3 2024
    "debt_breakdown": [
        {
            "name": "Equipment Financing",
            "amount_b": 2.0,
            "date": "Various",
            "type": "Equipment Loans",
            "collateral": "Specific GPU clusters",
            "collateral_gpus": ["A100", "Older GPUs"],
            "collateral_value_today_b": 0.8,
            "collateral_value_at_maturity_b": 0.2,
            "interest_rate": "8-10%",
            "maturity": "2026",
            "maturity_year": 2026,
            "lenders": ["Various equipment financiers"],
            "notes": "Direct GPU purchase financing",
        },
        {
            "name": "Magnetar Credit Facility",
            "amount_b": 2.3,
            "date": "Aug 2023",
            "type": "Senior Secured",
            "collateral": "GPU assets",
            "collateral_gpus": ["H100", "A100"],
            "collateral_value_today_b": 2.5,
            "collateral_value_at_maturity_b": 0.8,
            "interest_rate": "SOFR + 7-8%",
            "maturity": "2028",
            "maturity_year": 2028,
            "lenders": ["Magnetar Capital", "Blackstone"],
            "notes": "First major GPU-backed loan",
        },
        {
            "name": "BlackRock/Magnetar Facility",
            "amount_b": 7.5,
            "date": "May 2024",
            "type": "Senior Secured",
            "collateral": "GPU assets + contracts",
            "collateral_gpus": ["H100"],
            "collateral_value_today_b": 4.0,
            "collateral_value_at_maturity_b": 1.0,
            "interest_rate": "SOFR + 6.5-7.5%",
            "maturity": "2029",
            "maturity_year": 2029,
            "lenders": ["BlackRock", "Magnetar Capital"],
            "notes": "Largest GPU-backed debt facility ever",
        },
        {
            "name": "Convertible Notes",
            "amount_b": 1.1,
            "date": "Mar 2025 (IPO)",
            "type": "Convertible",
            "collateral": "Unsecured",
            "collateral_gpus": [],
            "collateral_value_today_b": 0,
            "collateral_value_at_maturity_b": 0,
            "interest_rate": "~5%",
            "maturity": "2030",
            "maturity_year": 2030,
            "lenders": ["Public market"],
            "notes": "IPO-related financing",
        },
    ],
    "key_covenants": [
        "Minimum liquidity: $500M",
        "Debt/EBITDA ratio: Must improve by 2026",
        "GPU utilization: Must maintain >70%",
        "Customer concentration: Microsoft <70% revenue",
        "CapEx limits: Cannot exceed cash flow by >3x",
    ],
    "collateral_details": {
        "total_gpu_assets_b": 8.7,
        "gpu_types": [
            {"type": "H100", "quantity": "~50,000", "value_new_b": 5.0, "current_value_b": 4.0},
            {"type": "A100", "quantity": "~100,000", "value_new_b": 3.0, "current_value_b": 1.5},
            {"type": "Older GPUs", "quantity": "~50,000", "value_new_b": 0.7, "current_value_b": 0.2},
        ],
        "depreciation_schedule": "5-year straight line",
        "actual_useful_life": "2-3 years (technology obsolescence)",
        "current_book_value_b": 5.7,
        "estimated_market_value_b": 4.5,
    },
    "risk_factors": [
        "GPU depreciation faster than debt amortization",
        "Technology obsolescence (H100 → H200 → Blackwell)",
        "Customer concentration (Microsoft 60%)",
        "Interest rate exposure (floating rate debt)",
        "Refinancing risk in 2028-2029",
        "Collateral value declining while debt stays fixed",
    ],
}


# =============================================================================
# CRWV OPTIONS WHEN DEBT COMES DUE
# =============================================================================

CRWV_OPTIONS_AT_MATURITY = {
    "options": [
        {
            "name": "Refinance w/ Old Chips",
            "description": "Take new loan backed by existing GPUs",
            "likelihood": "LOW",
            "reason": "By 2029, H100s worth ~20% of today. No one lends $7.5B against $1B of old chips.",
            "equity_impact": "Neutral if successful, but unlikely",
        },
        {
            "name": "Buy New Chips to Refinance",
            "description": "Buy Blackwell/next-gen, use as new collateral",
            "likelihood": "MEDIUM",
            "reason": "Requires cash to buy chips first. NVDA wants payment upfront. Need ~$5B+ for meaningful collateral.",
            "equity_impact": "Neutral IF they have cash/credit, but increases total debt",
        },
        {
            "name": "Equity Raise",
            "description": "Issue new shares to raise cash",
            "likelihood": "MEDIUM",
            "reason": "Possible but massively dilutive. Need to raise ~$10B+ to cover debt and operations.",
            "equity_impact": "SEVERE DILUTION - existing shares worth 50-80% less",
        },
        {
            "name": "Asset Sale",
            "description": "Sell GPUs and data centers",
            "likelihood": "MEDIUM",
            "reason": "Fire sale of depreciated assets. Buyers know you're desperate = low prices.",
            "equity_impact": "NEGATIVE - selling assets below book value",
        },
        {
            "name": "Strategic Acquisition",
            "description": "Get bought by larger company",
            "likelihood": "MEDIUM",
            "reason": "Microsoft, Google, or NVDA could acquire. But why pay premium for distressed assets?",
            "equity_impact": "UNCERTAIN - depends on acquisition price",
        },
        {
            "name": "Debt Restructuring",
            "description": "Negotiate with lenders for better terms",
            "likelihood": "HIGH",
            "reason": "Lenders may accept haircut (partial loss) rather than force bankruptcy.",
            "equity_impact": "NEGATIVE - usually involves equity dilution or conversion",
        },
        {
            "name": "Bankruptcy",
            "description": "Chapter 11 or liquidation",
            "likelihood": "POSSIBLE",
            "reason": "If none of the above work, debt holders take control. Equity = zero.",
            "equity_impact": "WIPEOUT - equity worth $0",
        },
    ],
    "most_likely_path": "Equity raise + debt restructuring = existing shareholders massively diluted",
    
    "bull_case_new_chips": {
        "theory": "Buy new Blackwell/next-gen chips, use as collateral, refinance old debt",
        "challenges": [
            "WHERE'S THE CASH? NVDA demands payment upfront. $5B of new chips = need $5B cash first.",
            "CHICKEN & EGG: Need to borrow to buy chips to borrow. But who lends when old debt is due?",
            "DEBT KEEPS GROWING: Even if successful, total debt increases. $12.9B becomes $15B+.",
            "CUSTOMER DEMAND: Need actual customers willing to pay for new chip capacity.",
            "CAPEX TREADMILL: Must keep buying newer chips every 2-3 years forever to stay solvent.",
        ],
        "what_it_requires": [
            "Strong revenue growth to service higher debt load",
            "Customers locked into long-term contracts (Microsoft currently 60%)",
            "GPU demand to remain strong through 2029+",
            "NVDA to extend favorable payment terms (unlikely - they have leverage)",
            "Interest rates to stay manageable",
        ],
        "bear_response": "Even the 'buy new chips' path requires everything to go right. One stumble (demand drop, Microsoft cancels, rates spike) and the house of cards falls.",
    },
}


# =============================================================================
# GPU PRICING DATA
# =============================================================================

GPU_PRICING = {
    "current_prices": {
        "H100_80GB": {
            "msrp": 40000,
            "street_price": 35000,
            "cloud_hourly": 3.50,  # Per hour rental
            "yoy_change_pct": -15,
            "availability": "improving",
        },
        "H200": {
            "msrp": 45000,
            "street_price": 42000,
            "cloud_hourly": 4.50,
            "yoy_change_pct": 0,  # New product
            "availability": "constrained",
        },
        "A100_80GB": {
            "msrp": 15000,
            "street_price": 8000,
            "cloud_hourly": 1.50,
            "yoy_change_pct": -45,
            "availability": "abundant",
        },
        "A100_40GB": {
            "msrp": 10000,
            "street_price": 4000,
            "cloud_hourly": 0.80,
            "yoy_change_pct": -60,
            "availability": "surplus",
        },
        "L40S": {
            "msrp": 12000,
            "street_price": 10000,
            "cloud_hourly": 1.20,
            "yoy_change_pct": -10,
            "availability": "good",
        },
    },
    "historical_trends": [
        {"period": "2023 Q1", "h100_street": 40000, "a100_street": 15000, "notes": "Peak shortage"},
        {"period": "2023 Q2", "h100_street": 42000, "a100_street": 14000, "notes": "Extreme demand"},
        {"period": "2023 Q3", "h100_street": 38000, "a100_street": 12000, "notes": "Supply improving"},
        {"period": "2023 Q4", "h100_street": 36000, "a100_street": 10000, "notes": "More supply"},
        {"period": "2024 Q1", "h100_street": 35000, "a100_street": 9000, "notes": "Prices stabilizing"},
        {"period": "2024 Q2", "h100_street": 34000, "a100_street": 8000, "notes": "A100 declining"},
        {"period": "2024 Q3", "h100_street": 33000, "a100_street": 7000, "notes": "H200 announced"},
        {"period": "2024 Q4", "h100_street": 32000, "a100_street": 6000, "notes": "Blackwell delays"},
        {"period": "2025 Q1", "h100_street": 35000, "a100_street": 8000, "notes": "Blackwell ramp"},
    ],
    "depreciation_reality": {
        "accounting_depreciation": "5 years (20%/year)",
        "actual_value_decline": {
            "year_1": -30,  # -30% in year 1
            "year_2": -50,  # -50% cumulative by year 2
            "year_3": -70,  # -70% cumulative by year 3
            "year_4": -85,  # Nearly worthless
            "year_5": -95,
        },
        "why_faster_than_accounting": [
            "New GPU generations every 12-18 months",
            "Performance/watt improvements make old GPUs uneconomical",
            "Cloud providers price based on newest tech",
            "AI models optimized for latest architecture",
            "Energy costs favor newer, efficient hardware",
        ],
    },
    "cloud_pricing_trends": {
        "h100_hourly": [
            {"date": "2023-06", "price": 4.50, "provider": "avg"},
            {"date": "2023-12", "price": 4.00, "provider": "avg"},
            {"date": "2024-06", "price": 3.75, "provider": "avg"},
            {"date": "2024-12", "price": 3.50, "provider": "avg"},
            {"date": "2025-01", "price": 3.50, "provider": "avg"},
        ],
        "trend": "declining",
        "annual_decline_pct": 15,
        "implications": [
            "CoreWeave revenue per GPU declining",
            "Must add more GPUs to maintain revenue",
            "Margin compression inevitable",
            "Older GPUs become unprofitable faster",
        ],
    },
}


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

@dataclass
class DebtMaturityItem:
    """Single debt maturity with collateral."""
    year: int
    facility: str
    debt_due_b: float
    collateral_gpus: list[str]
    collateral_today_b: float
    collateral_at_maturity_b: float
    coverage_today: float  # collateral/debt ratio today
    coverage_at_maturity: float  # collateral/debt ratio at maturity
    shortfall_b: float  # how much collateral is short of debt at maturity


def build_debt_maturity_timeline() -> list[DebtMaturityItem]:
    """Build timeline of debt maturities with collateral values."""
    items = []
    for d in CRWV_DEBT_STRUCTURE["debt_breakdown"]:
        if d["collateral_value_today_b"] == 0:  # Skip unsecured
            continue
        
        coverage_today = d["collateral_value_today_b"] / d["amount_b"] if d["amount_b"] > 0 else 0
        coverage_maturity = d["collateral_value_at_maturity_b"] / d["amount_b"] if d["amount_b"] > 0 else 0
        shortfall = d["amount_b"] - d["collateral_value_at_maturity_b"]
        
        items.append(DebtMaturityItem(
            year=d["maturity_year"],
            facility=d["name"],
            debt_due_b=d["amount_b"],
            collateral_gpus=d["collateral_gpus"],
            collateral_today_b=d["collateral_value_today_b"],
            collateral_at_maturity_b=d["collateral_value_at_maturity_b"],
            coverage_today=coverage_today,
            coverage_at_maturity=coverage_maturity,
            shortfall_b=shortfall,
        ))
    
    return sorted(items, key=lambda x: x.year)


@dataclass
class DebtRiskAssessment:
    """Assessment of CRWV debt risk."""
    total_debt_b: float
    collateral_value_b: float
    ltv_ratio: float  # Loan-to-value
    debt_coverage_ratio: float
    refinancing_risk: str
    key_risks: list[str] = field(default_factory=list)
    timeline_risks: list[str] = field(default_factory=list)


def assess_crwv_debt_risk() -> DebtRiskAssessment:
    """Assess CoreWeave's debt risk."""
    
    debt = CRWV_DEBT_STRUCTURE
    total_debt = debt["total_debt_b"]
    collateral_value = debt["collateral_details"]["estimated_market_value_b"]
    
    ltv = total_debt / collateral_value if collateral_value > 0 else 999
    
    # Estimate debt coverage (rough)
    # Revenue ~$2B, EBITDA margin ~20% = $400M EBITDA
    # Interest expense ~$1B/year at 7% on $12.9B
    ebitda_est = 0.4  # $400M
    interest_est = total_debt * 0.07  # ~$900M
    coverage = ebitda_est / interest_est if interest_est > 0 else 0
    
    # Determine refinancing risk
    if ltv > 2.5:
        refi_risk = "EXTREME"
    elif ltv > 2.0:
        refi_risk = "HIGH"
    elif ltv > 1.5:
        refi_risk = "ELEVATED"
    else:
        refi_risk = "MODERATE"
    
    key_risks = [
        f"LTV ratio {ltv:.1f}x - debt exceeds collateral value by {(ltv-1)*100:.0f}%",
        f"Interest coverage ~{coverage:.2f}x - below 1x means EBITDA doesn't cover interest",
        "GPU depreciation accelerating - collateral value declining 20-30%/year",
        "Floating rate debt exposed to rate increases",
        "Microsoft concentration - 60% of revenue from one customer",
    ]
    
    timeline_risks = [
        "2026: Equipment financing matures (~$2B)",
        "2028: Magnetar facility matures (~$2.3B)",
        "2029: BlackRock facility matures (~$7.5B) - THE BIG ONE",
        "By 2029: H100s will be worth <20% of current value",
    ]
    
    return DebtRiskAssessment(
        total_debt_b=total_debt,
        collateral_value_b=collateral_value,
        ltv_ratio=ltv,
        debt_coverage_ratio=coverage,
        refinancing_risk=refi_risk,
        key_risks=key_risks,
        timeline_risks=timeline_risks,
    )


def get_gpu_depreciation_analysis() -> dict:
    """Analyze GPU depreciation vs accounting."""
    
    pricing = GPU_PRICING
    
    # Calculate implied depreciation from price trends
    h100_peak = 42000  # 2023 Q2
    h100_current = pricing["current_prices"]["H100_80GB"]["street_price"]
    h100_decline = (h100_peak - h100_current) / h100_peak * 100
    
    a100_peak = 15000  # 2023 Q1
    a100_current = pricing["current_prices"]["A100_80GB"]["street_price"]
    a100_decline = (a100_peak - a100_current) / a100_peak * 100
    
    return {
        "h100": {
            "peak_price": h100_peak,
            "current_price": h100_current,
            "decline_pct": h100_decline,
            "time_period": "~2 years",
            "accounting_depreciation": 40,  # 2 years at 20%/year
            "status": "HELD VALUE" if h100_decline < 40 else "DEPRECIATED FASTER",
            "vs_accounting": 40 - h100_decline,  # positive = held value better
        },
        "a100": {
            "peak_price": a100_peak,
            "current_price": a100_current,
            "decline_pct": a100_decline,
            "time_period": "~2 years",
            "accounting_depreciation": 40,
            "status": "HELD VALUE" if a100_decline < 40 else "DEPRECIATED FASTER",
            "vs_accounting": 40 - a100_decline,  # negative = depreciated more
        },
        "key_insight": "Current-gen GPUs hold value while hot, but PRIOR-gen crashes when replaced",
        "pattern": "H100 holding value NOW, but will crash like A100 when Blackwell scales",
        "crwv_implication": "H100 collateral looks OK today, but faces cliff when next-gen arrives",
    }


# =============================================================================
# MARKET-WIDE GPU DEBT ANALYSIS
# =============================================================================

@dataclass
class GPUDebtMarketSummary:
    """Summary of the GPU-backed debt market."""
    total_market_size_b: float
    companies: list[dict]
    key_risks: list[str]
    market_dynamics: list[str]


def get_gpu_debt_market_summary() -> GPUDebtMarketSummary:
    """Get a summary of the GPU-backed debt market across all tracked companies."""
    
    companies = []
    for ticker, data in GPU_DEBT_COMPANIES.items():
        companies.append({
            "ticker": ticker,
            "name": data["name"],
            "total_debt_b": data["total_debt_b"],
            "gpu_backed_debt_b": data.get("gpu_backed_debt_b", 0),
            "gpu_capex_b": data.get("gpu_capex_b", 0),
            "ltv_ratio": data.get("ltv_ratio", 0),
            "key_risk": data["key_risk"],
        })
    
    # Sort by GPU-backed debt exposure
    companies.sort(key=lambda x: x["gpu_backed_debt_b"], reverse=True)
    
    return GPUDebtMarketSummary(
        total_market_size_b=GPU_DEBT_MARKET_OVERVIEW["total_market_size_b"],
        companies=companies,
        key_risks=GPU_DEBT_MARKET_OVERVIEW["risk_factors"],
        market_dynamics=GPU_DEBT_MARKET_OVERVIEW["market_dynamics"],
    )


# =============================================================================
# TICKER-SPECIFIC DEBT ANALYSIS
# =============================================================================

@dataclass
class TickerDebtAnalysis:
    """Debt analysis for a specific ticker."""
    ticker: str
    name: str
    total_debt_b: float
    total_equity_b: float
    debt_to_equity: float
    debt_to_ebitda: float
    interest_coverage: float
    current_ratio: float
    debt_rating: str
    key_maturities: list[dict]
    gpu_exposure: dict  # GPU-specific debt info if applicable
    risk_assessment: str
    key_risks: list[str]


def fetch_ticker_debt_analysis(ticker: str, settings) -> Optional[TickerDebtAnalysis]:
    """
    Fetch comprehensive debt analysis for any ticker.
    Uses FMP API for balance sheet and financial data.
    """
    from ai_options_trader.altdata.fmp import (
        fetch_balance_sheet, fetch_key_metrics, fetch_ratios, fetch_profile,
    )
    
    t = ticker.upper()
    
    # Check if we have GPU-specific data
    gpu_info = GPU_DEBT_COMPANIES.get(t, {})
    
    try:
        # Use centralized FMP client
        bs_list = fetch_balance_sheet(settings=settings, ticker=t, periods=1)
        bs_data = bs_list[0] if bs_list else {}
        km_data = fetch_key_metrics(settings=settings, ticker=t)
        ratios_data = fetch_ratios(settings=settings, ticker=t)
        profile_obj = fetch_profile(settings=settings, ticker=t)
        profile_data = profile_obj.__dict__ if profile_obj else {}
        
        # Calculate debt metrics
        total_debt = float(bs_data.get("totalDebt", 0) or 0) / 1e9
        total_equity = float(bs_data.get("totalStockholdersEquity", 0) or 0) / 1e9
        
        debt_to_equity = float(ratios_data.get("debtEquityRatioTTM", 0) or 0)
        debt_to_ebitda = float(km_data.get("debtToAssetsTTM", 0) or 0) * 5  # Rough approximation
        interest_coverage = float(ratios_data.get("interestCoverageTTM", 0) or 0)
        current_ratio = float(ratios_data.get("currentRatioTTM", 0) or 0)
        
        # Determine risk assessment
        risk_factors = []
        if debt_to_equity > 2:
            risk_factors.append(f"High leverage: D/E ratio {debt_to_equity:.1f}x")
        if interest_coverage < 3 and interest_coverage > 0:
            risk_factors.append(f"Weak interest coverage: {interest_coverage:.1f}x")
        if current_ratio < 1:
            risk_factors.append(f"Liquidity concern: Current ratio {current_ratio:.2f}")
        
        # GPU-specific risks
        if gpu_info:
            if gpu_info.get("gpu_backed_debt_b", 0) > 0:
                risk_factors.append(f"GPU-backed debt: ${gpu_info['gpu_backed_debt_b']:.1f}B")
                risk_factors.append(f"GPU collateral depreciation risk")
            if gpu_info.get("gpu_capex_b", 0) > 5:
                risk_factors.append(f"Heavy GPU CapEx: ${gpu_info['gpu_capex_b']:.0f}B+ committed")
            if gpu_info.get("key_risk"):
                risk_factors.append(gpu_info["key_risk"])
        
        # Risk assessment
        if debt_to_equity > 3 or interest_coverage < 2:
            risk_assessment = "HIGH RISK"
        elif debt_to_equity > 1.5 or interest_coverage < 4:
            risk_assessment = "ELEVATED"
        elif debt_to_equity > 0.5:
            risk_assessment = "MODERATE"
        else:
            risk_assessment = "LOW"
        
        # GPU exposure
        gpu_exposure = {}
        if gpu_info:
            gpu_exposure = {
                "has_gpu_debt": gpu_info.get("gpu_backed_debt_b", 0) > 0,
                "gpu_backed_debt_b": gpu_info.get("gpu_backed_debt_b", 0),
                "gpu_capex_b": gpu_info.get("gpu_capex_b", 0),
                "gpu_collateral_b": gpu_info.get("gpu_collateral_value_b", 0),
                "ltv_ratio": gpu_info.get("ltv_ratio", 0),
            }
        
        return TickerDebtAnalysis(
            ticker=t,
            name=profile_data.get("companyName", t),
            total_debt_b=total_debt,
            total_equity_b=total_equity,
            debt_to_equity=debt_to_equity,
            debt_to_ebitda=debt_to_ebitda,
            interest_coverage=interest_coverage,
            current_ratio=current_ratio,
            debt_rating=profile_data.get("rating", "N/A") or "N/A",
            key_maturities=[],  # Would need additional API for maturity schedule
            gpu_exposure=gpu_exposure,
            risk_assessment=risk_assessment,
            key_risks=risk_factors,
        )
        
    except Exception as e:
        return None

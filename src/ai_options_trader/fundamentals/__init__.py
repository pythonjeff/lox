"""
Fundamentals Analysis Module

CFA-level financial analysis organized into two categories:

FINANCIAL MODELING (sensitivity.py, valuation.py):
- Revenue/margin sensitivity models
- DCF valuation
- Scenario analysis
- Reverse DCF for implied growth

ECOSYSTEM ANALYSIS (nvda_ecosystem.py, openai_exposure.py, partnerships.py):
- NVDA ecosystem revenue exposure
- OpenAI partner analysis
- Broader partnership networks
"""

# Financial Modeling
from .sensitivity import (
    SensitivityModel,
    build_sensitivity_model,
    run_scenario_analysis,
)
from .valuation import (
    DCFModel,
    calculate_implied_growth,
    reverse_dcf,
)

# Ecosystem Analysis
from .nvda_ecosystem import (
    NVDA_ECOSYSTEM,
    NVDA_BASKETS,
    build_ecosystem_report,
    get_ecosystem_summary,
)
from .openai_exposure import (
    OPENAI_EXPOSURE,
    build_openai_exposure_report,
    get_openai_thesis_summary,
)
from .partnerships import (
    NVDA_PARTNERS,
    build_partner_health_report,
    analyze_demand_peak_signals,
)

__all__ = [
    # Financial Modeling
    "SensitivityModel",
    "build_sensitivity_model",
    "run_scenario_analysis",
    "DCFModel",
    "calculate_implied_growth",
    "reverse_dcf",
    # Ecosystem Analysis
    "NVDA_ECOSYSTEM",
    "NVDA_BASKETS",
    "build_ecosystem_report",
    "get_ecosystem_summary",
    "OPENAI_EXPOSURE",
    "build_openai_exposure_report",
    "get_openai_thesis_summary",
    "NVDA_PARTNERS",
    "build_partner_health_report",
    "analyze_demand_peak_signals",
]

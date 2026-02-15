"""
Lived Inflation Index — Category definitions, weight calculation, and debt overlay.

Each category maps a BLS CPI sub-index to:
  - freq_score:    how often consumers actually pay for it
  - essentiality:  whether they can stop paying (3=can't, 2=can reduce, 1=can cut)

The LII weight formula is:
    raw = cpi_weight × freq_score × essentiality_factor
    lii_weight = raw / sum(all raw)

Debt servicing categories (student loans, credit cards, auto loans) use FRED
data rather than BLS — CPI excludes all debt service costs.
Proxy calculations live in lii_calculator.py (compute_debt_cost_series).
"""
from __future__ import annotations

from copy import deepcopy

# ── Essentiality scaling ──────────────────────────────────────────────
# Multiplier applied to freq_score to capture "can you stop paying?"
# This widens the spread between non-discretionary monthly bills
# (rent, insurance, utilities) and deferrable purchases (vehicles, TVs).
#
#   3 = Non-discretionary — you literally cannot stop paying
#   2 = Semi-essential    — you can reduce but not eliminate
#   1 = Discretionary     — you can cut or defer indefinitely

ESSENTIALITY_MULTIPLIER: dict[int, float] = {
    3: 1.5,   # shelter, utilities, insurance, prescriptions, childcare
    2: 1.0,   # gas, healthcare visits, personal care, maintenance
    1: 0.5,   # dining out, apparel, vehicles, electronics, airfares
}

# ── Frequency score rationale (logarithmic, not linear) ──────────────
#   Daily (365x/yr)    → 10  : Groceries, gas — constant exposure
#   Weekly (52x/yr)    →  8  : Food shopping, fuel fills
#   2-3x/week          →  7  : Dining out, personal care, pet food
#   Monthly (12x/yr)   →  6  : Rent, insurance, utilities, subscriptions
#   Quarterly (4x/yr)  →  4  : Apparel, medical visits, maintenance
#   Semi-annual (2x/yr)→  2.5: Airfares, some services
#   Annually (1x/yr)   →  1.5: Furnishings, annual expenses
#   Every 2-5 years    →  1.0: Used cars, computers
#   Every 5+ years     →  0.5: New cars, TVs, appliances


# ── BLS CPI categories ───────────────────────────────────────────────
# cpi_weight: BLS relative importance (approximate, sums to ~95%)
# All series: CUUR0000 prefix = All Urban Consumers, US City Average, NSA

CATEGORIES: list[dict] = [
    # ─── Food ──────────────────────────────────────────────────────
    {"name": "Food at home",           "series_id": "CUUR0000SAF11",  "cpi_weight": 0.087, "freq_score": 8.0,  "freq_label": "Weekly",       "essentiality": 3},
    {"name": "Food away from home",    "series_id": "CUUR0000SEFV",   "cpi_weight": 0.056, "freq_score": 7.0,  "freq_label": "2-3x/week",    "essentiality": 1},
    {"name": "Alcoholic beverages",    "series_id": "CUUR0000SAF116", "cpi_weight": 0.010, "freq_score": 7.0,  "freq_label": "Weekly",       "essentiality": 1},

    # ─── Shelter ───────────────────────────────────────────────────
    {"name": "Rent of primary res.",   "series_id": "CUUR0000SEHA",   "cpi_weight": 0.076, "freq_score": 6.0,  "freq_label": "Monthly",      "essentiality": 3},
    {"name": "Owners' equiv. rent",    "series_id": "CUUR0000SEHC",   "cpi_weight": 0.268, "freq_score": 6.0,  "freq_label": "Monthly",      "essentiality": 3},
    {"name": "Homeowner insurance",    "series_id": "CUUR0000SEHH",   "cpi_weight": 0.008, "freq_score": 6.0,  "freq_label": "Monthly",      "essentiality": 3},  # was SEHA (bug), now SEHH
    {"name": "Household furnishings",  "series_id": "CUUR0000SAH3",   "cpi_weight": 0.036, "freq_score": 1.5,  "freq_label": "Annually",     "essentiality": 1},

    # ─── Utilities ─────────────────────────────────────────────────
    {"name": "Electricity",            "series_id": "CUUR0000SEHF01", "cpi_weight": 0.025, "freq_score": 6.0,  "freq_label": "Monthly",      "essentiality": 3},
    {"name": "Utility (piped gas)",    "series_id": "CUUR0000SEHF02", "cpi_weight": 0.008, "freq_score": 6.0,  "freq_label": "Monthly",      "essentiality": 3},
    {"name": "Water/sewer/trash",      "series_id": "CUUR0000SEHG",   "cpi_weight": 0.011, "freq_score": 6.0,  "freq_label": "Monthly",      "essentiality": 3},
    {"name": "Phone/internet",         "series_id": "CUUR0000SEED",   "cpi_weight": 0.034, "freq_score": 6.0,  "freq_label": "Monthly",      "essentiality": 3},

    # ─── Transportation ────────────────────────────────────────────
    {"name": "Gasoline",               "series_id": "CUUR0000SETB01", "cpi_weight": 0.032, "freq_score": 8.0,  "freq_label": "Weekly",       "essentiality": 2},
    {"name": "Auto insurance",         "series_id": "CUUR0000SETE",   "cpi_weight": 0.028, "freq_score": 6.0,  "freq_label": "Monthly",      "essentiality": 3},
    {"name": "New vehicles",           "series_id": "CUUR0000SETA01", "cpi_weight": 0.041, "freq_score": 0.5,  "freq_label": "Every 5-7 yrs","essentiality": 1},
    {"name": "Used vehicles",          "series_id": "CUUR0000SETA02", "cpi_weight": 0.023, "freq_score": 1.0,  "freq_label": "Every 4-5 yrs","essentiality": 1},
    {"name": "Car maintenance/repair", "series_id": "CUUR0000SETD",   "cpi_weight": 0.012, "freq_score": 4.0,  "freq_label": "Quarterly",    "essentiality": 2},  # NEW
    {"name": "Airfares",               "series_id": "CUUR0000SETG01", "cpi_weight": 0.006, "freq_score": 2.5,  "freq_label": "2x/year",      "essentiality": 1},

    # ─── Insurance & Healthcare ────────────────────────────────────
    {"name": "Health insurance",       "series_id": "CUUR0000SEME",   "cpi_weight": 0.008, "freq_score": 6.0,  "freq_label": "Monthly",      "essentiality": 3},
    {"name": "Medical care services",  "series_id": "CUUR0000SAM2",   "cpi_weight": 0.051, "freq_score": 4.0,  "freq_label": "Quarterly",    "essentiality": 2},
    {"name": "Prescription drugs",     "series_id": "CUUR0000SEMF01", "cpi_weight": 0.014, "freq_score": 6.0,  "freq_label": "Monthly",      "essentiality": 3},

    # ─── Education & Childcare ─────────────────────────────────────
    {"name": "Education (tuition)",    "series_id": "CUUR0000SAE1",   "cpi_weight": 0.015, "freq_score": 6.0,  "freq_label": "Monthly",      "essentiality": 2},  # split from old 0.030
    {"name": "Childcare/daycare",      "series_id": "CUUR0000SEEB",   "cpi_weight": 0.015, "freq_score": 6.0,  "freq_label": "Monthly",      "essentiality": 3},  # NEW — broken out

    # ─── Recreation & Personal ─────────────────────────────────────
    {"name": "Streaming/subscriptions","series_id": "CUUR0000SERF01", "cpi_weight": 0.010, "freq_score": 6.0,  "freq_label": "Monthly",      "essentiality": 1},  # NEW — cable/satellite/streaming
    {"name": "Personal care products", "series_id": "CUUR0000SAG1",   "cpi_weight": 0.026, "freq_score": 7.0,  "freq_label": "Weekly",       "essentiality": 2},
    {"name": "Pet care",               "series_id": "CUUR0000SS61031","cpi_weight": 0.012, "freq_score": 7.0,  "freq_label": "Weekly",       "essentiality": 2},  # NEW — 67% of US households
    {"name": "TVs",                    "series_id": "CUUR0000SERA01", "cpi_weight": 0.001, "freq_score": 0.5,  "freq_label": "Every 5+ yrs", "essentiality": 1},
    {"name": "Computers/peripherals",  "series_id": "CUUR0000SEEE01", "cpi_weight": 0.002, "freq_score": 1.0,  "freq_label": "Every 3-4 yrs","essentiality": 1},
    {"name": "Apparel",                "series_id": "CUUR0000SAA",    "cpi_weight": 0.025, "freq_score": 4.0,  "freq_label": "Quarterly",    "essentiality": 1},

    # ─── Home & Household ──────────────────────────────────────────
    {"name": "Home maintenance/repair","series_id": "CUUR0000SAH2",   "cpi_weight": 0.010, "freq_score": 4.0,  "freq_label": "Quarterly",    "essentiality": 2},  # NEW — HVAC, plumbing, etc.
]


# ── Scenario profiles ─────────────────────────────────────────────────
# Override frequency scores to model different consumer archetypes.
# Only override the categories that differ from the default.

SCENARIO_PROFILES: dict[str, dict] = {
    "default": {
        "label": "Median Consumer",
        "description": "Baseline frequency scores from the standard model.",
        "overrides": {},
    },
    "renter_no_car": {
        "label": "Renter, No Car",
        "description": "Zero out vehicle categories, boost rent, transit, food.",
        "overrides": {
            "CUUR0000SETA01": 0.0,   # New vehicles → 0
            "CUUR0000SETA02": 0.0,   # Used vehicles → 0
            "CUUR0000SETE":   0.0,   # Auto insurance → 0
            "CUUR0000SETB01": 0.0,   # Gasoline → 0
            "CUUR0000SETD":   0.0,   # Car maintenance → 0
            "CUUR0000SEHA":   8.0,   # Rent → higher
            "CUUR0000SAF11":  9.0,   # Food at home → higher
            "CUUR0000SETG01": 4.0,   # Airfares/transit → higher
            "CUUR0000SERF01": 7.0,   # Streaming → more (no car, more home time)
        },
    },
    "homeowner_family": {
        "label": "Homeowner with Family",
        "description": "Boost food, insurance, childcare, utilities, healthcare, pet care.",
        "overrides": {
            "CUUR0000SAF11":  9.0,   # Food at home
            "CUUR0000SETE":   7.0,   # Auto insurance
            "CUUR0000SEME":   7.0,   # Health insurance
            "CUUR0000SEEB":   8.0,   # Childcare/daycare
            "CUUR0000SEHF01": 7.0,   # Electricity
            "CUUR0000SEHG":   7.0,   # Water/sewer
            "CUUR0000SAM2":   5.0,   # Medical care
            "CUUR0000SS61031":8.0,   # Pet care → higher (family dog)
            "CUUR0000SAH2":   5.0,   # Home maintenance → higher
            "CUUR0000SEHH":   7.0,   # Homeowner insurance → higher
        },
    },
    "retiree": {
        "label": "Retiree",
        "description": "Boost healthcare, prescriptions, food, utilities; reduce tech, vehicles.",
        "overrides": {
            "CUUR0000SAM2":   7.0,   # Medical care services
            "CUUR0000SEMF01": 8.0,   # Prescription drugs
            "CUUR0000SEME":   8.0,   # Health insurance
            "CUUR0000SAF11":  9.0,   # Food at home
            "CUUR0000SEHF01": 7.0,   # Electricity
            "CUUR0000SAA":    2.0,   # Apparel → lower
            "CUUR0000SEEE01": 0.5,   # Computers → lower
            "CUUR0000SETA01": 0.3,   # New vehicles → lower
            "CUUR0000SS61031":8.0,   # Pet care → higher
            "CUUR0000SEEB":   0.0,   # Childcare → 0
        },
    },
    "single_urban": {
        "label": "Single Urban Professional",
        "description": "Boost food out, rent, transit, personal care, streaming.",
        "overrides": {
            "CUUR0000SEFV":   9.0,   # Food away from home
            "CUUR0000SEHA":   8.0,   # Rent
            "CUUR0000SAG1":   8.0,   # Personal care
            "CUUR0000SETG01": 4.0,   # Airfares → more frequent
            "CUUR0000SETA01": 0.3,   # New vehicles → rare
            "CUUR0000SETA02": 0.5,   # Used vehicles → rare
            "CUUR0000SERF01": 8.0,   # Streaming → higher
            "CUUR0000SEEB":   0.0,   # Childcare → 0
            "CUUR0000SS61031":6.0,   # Pet care → moderate
        },
    },
}


# ── Debt servicing categories (NOT in CPI) ─────────────────────────────
# Data sourced from FRED, not BLS. These represent monthly obligations
# that CPI excludes entirely: student loans, credit cards, auto loans.
#
# Proxy calculations in lii_calculator.py:
#   credit  → monthly_interest_cost = REVOLSL × (TERMCBCCALLNS / 12)
#   student → total_outstanding / est_borrowers / avg_term
#   auto    → standard amortization(MVLOASM / 85M borrowers, RIFLPBCIANM60NM, 72mo)

DEBT_CATEGORIES: list[dict] = [
    {
        "name": "Student loan payments",
        "key": "student",
        "weight": 0.035,           # 3.5% of household budget
        "freq_score": 6.0,
        "freq_label": "Monthly",
        "essentiality": 3,         # non-discretionary (income-driven, can't discharge)
        "fred_balance": "SLOAS",               # total outstanding (quarterly)
        "fred_rate": None,                      # use fixed proxy
        "is_debt": True,
        "note": "COVID forbearance Mar 2020–Oct 2023 zeroed payments; YoY proxy spikes on resumption.",
    },
    {
        "name": "Credit card debt service",
        "key": "credit",
        "weight": 0.025,           # 2.5%
        "freq_score": 6.0,
        "freq_label": "Monthly",
        "essentiality": 3,         # minimum payments are non-discretionary
        "fred_balance": "REVOLSL",             # revolving credit (monthly)
        "fred_rate": "TERMCBCCALLNS",          # CC interest rate (quarterly)
        "is_debt": True,
        "note": "Strongest signal — captures BOTH rising balances AND rising rates simultaneously.",
    },
    {
        "name": "Auto loan payments",
        "key": "auto",
        "weight": 0.020,           # 2.0%
        "freq_score": 6.0,
        "freq_label": "Monthly",
        "essentiality": 3,         # can't skip car payments
        "fred_balance": "MVLOASM",             # motor vehicle loans outstanding (monthly)
        "fred_rate": "RIFLPBCIANM60NM",       # 60-mo new auto rate (monthly)
        "is_debt": True,
    },
]

# All FRED series IDs needed for debt overlay
DEBT_FRED_SERIES = list({
    s for cat in DEBT_CATEGORIES
    for s in [cat.get("fred_balance"), cat.get("fred_rate")]
    if s
})


# ── Shelter alternatives ──────────────────────────────────────────────
# OER is the most criticized CPI component. It imputes "what would your
# house rent for?" — ignoring that actual mortgage payments exploded when
# rates went from 3% → 7%. These alternatives let users swap in reality.
#
# FRED series:
#   MORTGAGE30US  — 30-Year Fixed Mortgage Rate (weekly → resample monthly)
#   MSPUS         — Median Sales Price of Houses Sold (quarterly → interpolate)

OER_SERIES_ID = "CUUR0000SEHC"  # the category that gets swapped

SHELTER_MODES = {
    "oer": {
        "label": "BLS Standard (OER)",
        "description": "Owner's Equivalent Rent — BLS imputed rental value. Lags market by 12-18 months.",
    },
    "mdsp": {
        "label": "Mortgage Burden (MDSP)",
        "description": "Fed mortgage debt-service ratio — actual share of income going to mortgage payments across all US households. Captures refis, ARMs, and locked-in rates.",
    },
    "mortgage": {
        "label": "New-Purchase (Max Pain)",
        "description": "Uses higher of OER or new-purchase mortgage cost. Captures rate spike pain, never pretends shelter costs are falling.",
    },
}

SHELTER_FRED_SERIES = ["MORTGAGE30US", "MSPUS", "MDSP"]


# ── Helpers ───────────────────────────────────────────────────────────

def get_all_series_ids() -> list[str]:
    """Return all BLS series IDs needed for LII calculation."""
    return [c["series_id"] for c in CATEGORIES]


def calculate_lii_weights(
    categories: list[dict] | None = None,
    freq_overrides: dict[str, float] | None = None,
    use_essentiality: bool = True,
) -> list[dict]:
    """
    Calculate LII weights from CPI weights × frequency scores × essentiality.

    Returns a new list of category dicts with added 'lii_weight' field.
    Weights are normalized to sum to 1.0.

    Args:
        categories: Category list (defaults to CATEGORIES)
        freq_overrides: {series_id: new_freq_score} for scenario profiles
        use_essentiality: If True (default), apply essentiality multiplier.
            This widens the spread between non-discretionary monthly bills
            and deferrable purchases.
    """
    cats = deepcopy(categories or CATEGORIES)
    overrides = freq_overrides or {}

    # Apply overrides
    for cat in cats:
        sid = cat["series_id"]
        if sid in overrides:
            cat["freq_score"] = overrides[sid]

    # Calculate raw LII = cpi_weight × freq_score × essentiality_factor
    for cat in cats:
        ess = cat.get("essentiality", 2)
        ess_factor = ESSENTIALITY_MULTIPLIER.get(ess, 1.0) if use_essentiality else 1.0
        cat["raw_lii"] = cat["cpi_weight"] * cat["freq_score"] * ess_factor

    total_raw = sum(c["raw_lii"] for c in cats)

    # Normalize to sum to 1.0
    for cat in cats:
        cat["lii_weight"] = cat["raw_lii"] / total_raw if total_raw > 0 else 0.0

    return cats

"""
Congressional committee alignment — the actual source of edge in congressional trading.

Academic research (Ziobrowski et al.) shows Senate alpha is concentrated in
committee-aligned trades: lawmakers who sit on committees with oversight of the
industry they're trading in have access to non-public information.

A senator on Armed Services buying Lockheed = information asymmetry.
That same senator buying Netflix = rich person buying stocks.

Structure:
  MEMBER_COMMITTEES  — bioguide_id → list of committee keys
  COMMITTEE_SECTORS  — committee key → GICS sectors with oversight
  COMMITTEE_TICKERS  — committee key → specific tickers where oversight is direct

is_committee_aligned(bioguide_id, ticker, gics_sector) → (bool, committee_name | "")
"""
from __future__ import annotations

# ── Member → Committees ───────────────────────────────────────────────────────
# Updated for 119th Congress (2025–2027). Bioguide IDs are permanent.
# Committees listed as internal keys (see COMMITTEE_SECTORS below).

MEMBER_COMMITTEES: dict[str, list[str]] = {
    # ── Senate ────────────────────────────────────────────────────────────────
    "K000383": ["ARMED_SERVICES", "INTELLIGENCE", "ENERGY_NATURAL_RESOURCES"],   # Angus King (I-ME)
    "B001236": ["AGRICULTURE", "APPROPRIATIONS", "VETERANS_AFFAIRS"],             # John Boozman (R-AR)
    "M001190": ["ARMED_SERVICES", "ENERGY_NATURAL_RESOURCES", "HELP"],            # Markwayne Mullin (R-OK)
    "P000595": ["ARMED_SERVICES", "COMMERCE_SCIENCE_TRANSPORT", "HOMELAND_SECURITY"],  # Gary Peters (D-MI)
    "W000802": ["BUDGET", "FINANCE", "JUDICIARY"],                                 # Sheldon Whitehouse (D-RI)
    "C001047": ["APPROPRIATIONS", "COMMERCE_SCIENCE_TRANSPORT", "ENVIRONMENT"],    # Shelley Moore Capito (R-WV)
    "F000479": ["AGRICULTURE", "BANKING", "JOINT_ECONOMIC"],                       # John Fetterman (D-PA)
    "M000934": ["APPROPRIATIONS", "COMMERCE_SCIENCE_TRANSPORT", "VETERANS_AFFAIRS"],  # Jerry Moran (R-KS)
    "M000355": ["APPROPRIATIONS", "RULES_ADMIN"],                                  # Mitch McConnell (R-KY)
    "H000273": ["COMMERCE_SCIENCE_TRANSPORT", "ENERGY_NATURAL_RESOURCES", "HELP"], # John Hickenlooper (D-CO)
    "T000278": ["ARMED_SERVICES", "COMMERCE_SCIENCE_TRANSPORT", "AGRICULTURE"],    # Tommy Tuberville (R-AL)
    "M001243": ["ARMED_SERVICES", "BANKING"],                                      # David H. McCormick (R-PA)
    "T000490": ["ARMED_SERVICES", "JUDICIARY"],                                    # David J. Taylor (R-OH) — new member

    # ── House ─────────────────────────────────────────────────────────────────
    "C001123": ["ARMED_SERVICES"],                                                 # Gilbert Cisneros (D-CA)
    "M001232": ["FINANCIAL_SERVICES", "VETERANS_AFFAIRS"],                         # April McClain Delaney (D-MD)
    "S000168": ["FINANCIAL_SERVICES", "FOREIGN_AFFAIRS"],                          # Maria Elvira Salazar (R-FL)
    "G000583": ["FINANCIAL_SERVICES", "HOMELAND_SECURITY"],                        # Josh Gottheimer (D-NJ)
    "M001239": ["ARMED_SERVICES", "SCIENCE_SPACE_TECH"],                           # John McGuire (R-VA)
    "D000399": ["WAYS_AND_MEANS", "BUDGET"],                                       # Lloyd Doggett (D-TX)
    "M001218": ["FINANCIAL_SERVICES", "FOREIGN_AFFAIRS", "SCIENCE_SPACE_TECH"],    # Richard Dean McCormick (R-GA)
    "M001217": ["HOMELAND_SECURITY", "OVERSIGHT", "FOREIGN_AFFAIRS"],             # Jared Moskowitz (D-FL)
    "K000398": ["HOMELAND_SECURITY", "BUDGET"],                                    # Thomas Kean Jr (R-NJ)
    "F000110": ["TRANSPORTATION_INFRASTRUCTURE", "SMALL_BUSINESS"],               # Cleo Fields (D-LA)
    "M001236": ["ARMED_SERVICES", "FINANCIAL_SERVICES"],                           # Tim Moore (R-NC)
    "A000372": ["ARMED_SERVICES", "FINANCIAL_SERVICES"],                           # Richard Allen (R-GA)
    "M001231": ["INTELLIGENCE", "ARMED_SERVICES"],                                 # Rob Wittman (R-VA) — Armed Services chair subcom
    "W000804": ["ARMED_SERVICES", "NATURAL_RESOURCES"],                            # Robert Wittman (R-VA)
}

# ── Committee → GICS Sectors ─────────────────────────────────────────────────
# GICS sector names as returned by FMP /v3/profile → sector field.

COMMITTEE_SECTORS: dict[str, list[str]] = {
    "ARMED_SERVICES":            ["Industrials"],          # Aerospace & Defense is sub-industry of Industrials
    "INTELLIGENCE":              ["Industrials", "Information Technology"],
    "BANKING":                   ["Financials"],
    "FINANCIAL_SERVICES":        ["Financials"],
    "FINANCE":                   ["Financials", "Health Care"],   # Senate Finance = healthcare insurance + pharma
    "WAYS_AND_MEANS":            ["Health Care", "Financials"],
    "HELP":                      ["Health Care"],
    "ENERGY_NATURAL_RESOURCES":  ["Energy", "Utilities", "Materials"],
    "ENERGY_COMMERCE":           ["Energy", "Health Care", "Communication Services"],
    "COMMERCE_SCIENCE_TRANSPORT":["Information Technology", "Communication Services", "Industrials"],
    "AGRICULTURE":               ["Consumer Staples", "Materials"],
    "HOMELAND_SECURITY":         ["Information Technology", "Industrials"],
    "JUDICIARY":                 ["Information Technology", "Health Care"],
    "APPROPRIATIONS":            ["Industrials", "Information Technology"],  # gov contractors
    "FOREIGN_AFFAIRS":           ["Industrials"],
    "FOREIGN_RELATIONS":         ["Industrials"],
    "SCIENCE_SPACE_TECH":        ["Information Technology", "Industrials"],
    "TRANSPORTATION_INFRASTRUCTURE": ["Industrials"],
    "ENVIRONMENT":               ["Utilities", "Energy", "Materials"],
    "VETERANS_AFFAIRS":          ["Health Care"],
    "OVERSIGHT":                 ["Industrials", "Information Technology"],
    "RULES_ADMIN":               [],
    "BUDGET":                    [],
    "JOINT_ECONOMIC":            [],
    "SMALL_BUSINESS":            [],
    "NATURAL_RESOURCES":         ["Energy", "Materials"],
}

# ── Committee → High-value specific tickers ───────────────────────────────────
# For sectors where the GICS label is too broad (e.g., "Industrials" covers
# both defense AND airlines), we maintain specific ticker sets so an Armed
# Services member buying Delta doesn't get credit.

COMMITTEE_TICKERS: dict[str, set[str]] = {
    "ARMED_SERVICES": {
        # Prime contractors
        "LMT", "RTX", "NOC", "GD", "HII", "BA",
        # Mid-tier defense
        "BWXT", "LDOS", "SAIC", "L3H", "CACI", "BAH", "KTOS", "AVAV",
        "HEICO", "TDG", "VSEC", "AXON", "MOOG", "ROLL", "DRS", "MRCY",
        "SARO", "CDRE", "FLIR", "PLTR", "ACVA", "ATRO",
        # Semiconductors for defense
        "MCHP",
    },
    "INTELLIGENCE": {
        # Defense overlap
        "LMT", "RTX", "NOC", "SAIC", "BAH", "PLTR", "LDOS", "CACI",
        # Cybersecurity (intelligence community primary buyer)
        "PANW", "CRWD", "FEYE", "S", "ZS", "OKTA", "CYBR", "NET",
        "TELOS", "BOOZ", "MANT",
    },
    "HOMELAND_SECURITY": {
        # Cybersecurity + border tech
        "PANW", "CRWD", "FEYE", "S", "ZS", "OKTA", "CYBR", "NET",
        "AXON", "TELOS", "SAIC", "BAH", "LDOS",
    },
    "BANKING": {
        "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW",
        "USB", "PNC", "TFC", "FITB", "HBAN", "KEY", "RF", "CFG",
        "MTB", "ZION", "CMA", "SIVB",
    },
    "FINANCIAL_SERVICES": {
        "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW",
        "V", "MA", "AXP", "COF", "SYF", "DFS", "ALLY",
        "USB", "PNC", "TFC", "FITB", "HBAN", "KEY", "RF", "CFG",
        # Crypto/fintech — Financial Services has jurisdiction
        "COIN", "HOOD", "SOFI", "SQ", "PYPL", "VOYG", "MSTR",
    },
    "COMMERCE_SCIENCE_TRANSPORT": {
        # Telecom (Commerce Committee primary)
        "T", "VZ", "CMCSA", "CHTR", "TMUS", "DISH",
        # Airlines
        "DAL", "UAL", "LUV", "AAL", "ALK",
        # Big tech (Commerce has oversight of internet/consumer tech)
        "AAPL", "MSFT", "GOOGL", "GOOG", "META", "AMZN", "NFLX",
        "NVDA", "AMD", "INTC", "QCOM", "AVGO",
    },
    "SCIENCE_SPACE_TECH": {
        # Space & tech
        "AAPL", "MSFT", "GOOGL", "GOOG", "META", "AMZN", "NVDA",
        "AMD", "INTC", "QCOM", "AVGO", "TXN", "ADI", "MCHP",
        # Space
        "RKLB", "ASTS", "PL", "MAXR", "BWXT",
    },
    "ENERGY_NATURAL_RESOURCES": {
        "XOM", "CVX", "COP", "EOG", "PXD", "DVN", "MPC", "PSX", "VLO",
        "OXY", "SLB", "HAL", "BKR",
        # Utilities
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "PCG",
        # Materials / mining
        "NEM", "FCX", "ALB", "MP", "VALE", "RIO",
        # Renewables
        "ENPH", "FSLR", "SEDG", "PLUG", "TSLA",
    },
    "FINANCE": {
        # Healthcare insurance / pharma (Senate Finance has Medicare/Medicaid jurisdiction)
        "UNH", "CVS", "CI", "HUM", "MOH", "CNC", "ELV",
        "PFE", "MRK", "LLY", "ABBV", "AMGN", "GILD", "BMY", "REGN",
        "MCK", "ABC", "CAH",
    },
    "WAYS_AND_MEANS": {
        # Same as Finance (House equivalent)
        "UNH", "CVS", "CI", "HUM", "MOH", "CNC", "ELV",
        "PFE", "MRK", "LLY", "ABBV", "AMGN", "GILD", "BMY", "REGN",
    },
    "HELP": {
        "UNH", "CVS", "CI", "HUM", "ISRG", "MDT", "SYK", "BSX",
        "EW", "ABT", "ZBH", "HOLX", "DXCM",
        # Education
        "APEI", "STRA", "PRDO",
    },
    "AGRICULTURE": {
        "ADM", "BG", "CTVA", "SAFM", "TSN", "CAG", "CPB",
        "K", "GIS", "MKC", "SJM", "HRL", "MOS", "NTR",
    },
    "JUDICIARY": {
        # Antitrust oversight of big tech + pharma IP
        "AAPL", "GOOGL", "GOOG", "META", "AMZN", "MSFT",
        "PFE", "MRK", "LLY", "ABBV", "AMGN",
    },
    "APPROPRIATIONS": {
        # Government contractors — follow the federal budget
        "LDOS", "SAIC", "CACI", "BAH", "ACN", "CSCO", "MSFT",
        "LMT", "RTX", "NOC", "GD", "HII",
    },
}


# ── Alignment check ───────────────────────────────────────────────────────────

def is_committee_aligned(
    bioguide_id: str,
    ticker: str,
    gics_sector: str = "",
) -> tuple[bool, str]:
    """
    Returns (is_aligned, matching_committee_name).

    Alignment if:
    1. Ticker appears in the committee's specific ticker list, OR
    2. The company's GICS sector is in the committee's sector list
       AND the committee has meaningful sector-level oversight (not just "Budget").
    """
    committees = MEMBER_COMMITTEES.get(bioguide_id, [])
    ticker_upper = (ticker or "").strip().upper()
    sector_lower = (gics_sector or "").strip()

    for committee in committees:
        # Check specific ticker list first (higher precision)
        if ticker_upper in COMMITTEE_TICKERS.get(committee, set()):
            return True, committee

        # Check sector alignment (broader but still meaningful)
        sectors = COMMITTEE_SECTORS.get(committee, [])
        if sectors and sector_lower and sector_lower in sectors:
            return True, committee

    return False, ""


def get_member_committees(bioguide_id: str, api_key: str | None = None) -> list[str]:
    """
    Return committee keys for a member.
    Tries static map first; if not found and api_key is provided, fetches live
    from Congress.gov and updates the static map in-memory for the session.
    """
    bio = bioguide_id or ""
    if not bio:
        return []
    if bio in MEMBER_COMMITTEES:
        return MEMBER_COMMITTEES[bio]
    if api_key:
        fetched = _fetch_committees_from_congress_gov(bio, api_key)
        if fetched:
            MEMBER_COMMITTEES[bio] = fetched
            return fetched
    return []


# ── Congress.gov live lookup ──────────────────────────────────────────────────

# Map Congress.gov committee names → our internal committee keys
_CONGRESS_GOV_NAME_MAP: dict[str, str] = {
    "armed services": "ARMED_SERVICES",
    "defense": "ARMED_SERVICES",
    "intelligence": "INTELLIGENCE",
    "select committee on intelligence": "INTELLIGENCE",
    "banking": "BANKING",
    "banking, housing, and urban affairs": "BANKING",
    "financial services": "FINANCIAL_SERVICES",
    "finance": "FINANCE",
    "ways and means": "WAYS_AND_MEANS",
    "health, education, labor": "HELP",
    "help": "HELP",
    "energy and natural resources": "ENERGY_NATURAL_RESOURCES",
    "energy and commerce": "ENERGY_COMMERCE",
    "commerce, science, and transportation": "COMMERCE_SCIENCE_TRANSPORT",
    "commerce, science": "COMMERCE_SCIENCE_TRANSPORT",
    "agriculture": "AGRICULTURE",
    "agriculture, nutrition": "AGRICULTURE",
    "homeland security": "HOMELAND_SECURITY",
    "judiciary": "JUDICIARY",
    "appropriations": "APPROPRIATIONS",
    "foreign affairs": "FOREIGN_AFFAIRS",
    "foreign relations": "FOREIGN_RELATIONS",
    "science, space, and technology": "SCIENCE_SPACE_TECH",
    "science, space": "SCIENCE_SPACE_TECH",
    "transportation and infrastructure": "TRANSPORTATION_INFRASTRUCTURE",
    "veterans' affairs": "VETERANS_AFFAIRS",
    "environment and public works": "ENVIRONMENT",
    "oversight": "OVERSIGHT",
    "natural resources": "NATURAL_RESOURCES",
    "budget": "BUDGET",
    "rules and administration": "RULES_ADMIN",
    "joint economic": "JOINT_ECONOMIC",
    "small business": "SMALL_BUSINESS",
}


def _congress_gov_name_to_key(name: str) -> str | None:
    name_lower = name.lower()
    for pattern, key in _CONGRESS_GOV_NAME_MAP.items():
        if pattern in name_lower:
            return key
    return None


_CONGRESS_GOV_CACHE: dict[str, list[str]] = {}


def _fetch_committees_from_congress_gov(bioguide_id: str, api_key: str) -> list[str]:
    """Fetch committee assignments for a member from Congress.gov API."""
    if bioguide_id in _CONGRESS_GOV_CACHE:
        return _CONGRESS_GOV_CACHE[bioguide_id]
    try:
        import requests
        r = requests.get(
            f"https://api.congress.gov/v3/member/{bioguide_id}/committee-assignment",
            params={"format": "json", "limit": 50, "api_key": api_key},
            timeout=10,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        assignments = data.get("committeeAssignments", [])
        keys: list[str] = []
        for assignment in assignments:
            committee_name = (
                assignment.get("committee", {}).get("name", "")
                or assignment.get("name", "")
            )
            key = _congress_gov_name_to_key(committee_name)
            if key and key not in keys:
                keys.append(key)
        _CONGRESS_GOV_CACHE[bioguide_id] = keys
        return keys
    except Exception:
        return []


def alignment_note(committee: str) -> str:
    """One-line human-readable note about why this committee matters."""
    notes = {
        "ARMED_SERVICES":            "Armed Services — defense contractor oversight",
        "INTELLIGENCE":              "Intelligence — classified procurement briefings",
        "BANKING":                   "Banking — bank regulation & stress tests",
        "FINANCIAL_SERVICES":        "Financial Services — bank/fintech/crypto regulation",
        "FINANCE":                   "Finance — Medicare/Medicaid/pharma pricing",
        "WAYS_AND_MEANS":            "Ways & Means — healthcare + tax legislation",
        "HELP":                      "HELP — healthcare & biotech regulation",
        "ENERGY_NATURAL_RESOURCES":  "Energy & Natural Resources — oil/gas/mining/utilities",
        "ENERGY_COMMERCE":           "Energy & Commerce — energy + health + telecom",
        "COMMERCE_SCIENCE_TRANSPORT":"Commerce — telecom, big tech, airlines",
        "SCIENCE_SPACE_TECH":        "Science, Space & Technology — tech + space",
        "AGRICULTURE":               "Agriculture — crop policy & food companies",
        "HOMELAND_SECURITY":         "Homeland Security — cybersecurity + border tech",
        "JUDICIARY":                 "Judiciary — antitrust (tech + pharma)",
        "APPROPRIATIONS":            "Appropriations — federal budget + contractors",
        "FOREIGN_AFFAIRS":           "Foreign Affairs — defense + geopolitical",
        "FOREIGN_RELATIONS":         "Foreign Relations — defense + geopolitical",
        "TRANSPORTATION_INFRASTRUCTURE": "Transportation — infrastructure + industrials",
        "VETERANS_AFFAIRS":          "Veterans' Affairs — VA healthcare",
        "ENVIRONMENT":               "Environment — utilities + energy transition",
        "OVERSIGHT":                 "Oversight — government contractor accountability",
        "NATURAL_RESOURCES":         "Natural Resources — energy + materials",
    }
    return notes.get(committee, committee)

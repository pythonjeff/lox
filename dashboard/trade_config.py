"""
Trade indicator configuration and evaluation for the LOX FUND Dashboard.

Per-position indicator definitions (target / invalidation / direction)
and the logic to evaluate indicator status (green / yellow / red).
"""

import re

from dashboard.quotes import get_live_source_value

# ═══════════════════════════════════════════════════════════════
# TRADE_INDICATORS_CONFIG
# Key format: "{underlying}_{strike}_{opt_type}_{expiry}" for options,
# or raw symbol for equity/crypto positions.
# ═══════════════════════════════════════════════════════════════

TRADE_INDICATORS_CONFIG = {
    # ── EQUITY / ETF POSITIONS ──

    "AMZN": [
        {"name": "AMZN Price", "current_value": "$204.65", "target_value": ">$220", "invalidation_value": "<$180", "direction": "above"},
        {"name": "10Y Yield", "current_value": "4.5%", "target_value": "<4.0%", "invalidation_value": ">5.0%", "direction": "below"},
        {"name": "Consumer Confidence", "current_value": "104", "target_value": ">100", "invalidation_value": "<90", "direction": "above"},
        {"name": "AWS Rev Growth YoY", "current_value": "19%", "target_value": ">15%", "invalidation_value": "<10%", "direction": "above"},
    ],

    "BE": [
        {"name": "BE Price", "current_value": "$154.29", "target_value": ">$170", "invalidation_value": "<$120", "direction": "above"},
        {"name": "Nat Gas (Henry Hub)", "current_value": "$3.50", "target_value": ">$3.00", "invalidation_value": "<$2.00", "direction": "above"},
        {"name": "AI Capex Trend", "current_value": "$65B/qtr", "target_value": ">$60B", "invalidation_value": "<$40B", "direction": "above"},
        {"name": "IRA Policy Risk", "current_value": "Low", "target_value": "Supportive", "invalidation_value": "Hostile", "direction": "above"},
    ],

    "SPXU": [
        {"name": "SPX Level", "current_value": "6050", "target_value": "<5500", "invalidation_value": ">6500", "direction": "below"},
        {"name": "VIX", "current_value": "15.5", "target_value": ">25", "invalidation_value": "<12", "direction": "above"},
        {"name": "ISM Mfg PMI", "current_value": "49.3", "target_value": "<47", "invalidation_value": ">53", "direction": "below"},
        {"name": "Initial Jobless Claims", "current_value": "219K", "target_value": ">260K", "invalidation_value": "<200K", "direction": "above"},
    ],

    "VIXY": [
        {"name": "VIX", "current_value": "15.5", "target_value": ">25", "invalidation_value": "<12", "direction": "above"},
        {"name": "VIXY Price", "current_value": "$26.31", "target_value": ">$35", "invalidation_value": "<$20", "direction": "above"},
        {"name": "VIX Futures Contango", "current_value": "5%", "target_value": "<0%", "invalidation_value": ">8%", "direction": "below"},
        {"name": "Realized Vol 20d", "current_value": "12%", "target_value": ">20%", "invalidation_value": "<8%", "direction": "above"},
    ],

    # ── OPTIONS: BEARISH EQUITY ──

    "SPY_550_P_2026-06-18": [
        {"name": "SPY Price", "current_value": "$605", "target_value": "<$550", "invalidation_value": ">$630", "direction": "below"},
        {"name": "VIX", "current_value": "15.5", "target_value": ">25", "invalidation_value": "<12", "direction": "above"},
        {"name": "Unemployment Rate", "current_value": "4.1%", "target_value": ">4.5%", "invalidation_value": "<3.8%", "direction": "above"},
        {"name": "ISM Mfg PMI", "current_value": "49.3", "target_value": "<47", "invalidation_value": ">53", "direction": "below"},
        {"name": "Fwd EPS Growth", "current_value": "10%", "target_value": "<5%", "invalidation_value": ">15%", "direction": "below"},
    ],

    # ── OPTIONS: CREDIT STRESS ──

    "HYG_74_P_2026-07-17": [
        {"name": "HY OAS Spread", "current_value": "310bp", "target_value": ">500bp", "invalidation_value": "<250bp", "direction": "above"},
        {"name": "HYG Price", "current_value": "$76.50", "target_value": "<$74", "invalidation_value": ">$79", "direction": "below"},
        {"name": "US Default Rate", "current_value": "2.8%", "target_value": ">4%", "invalidation_value": "<2%", "direction": "above"},
        {"name": "BBB-BB Spread", "current_value": "120bp", "target_value": ">200bp", "invalidation_value": "<80bp", "direction": "above"},
        {"name": "Fed Funds Rate", "current_value": "5.33%", "target_value": ">5.25%", "invalidation_value": "<4.75%", "direction": "above"},
    ],

    # ── OPTIONS: VOLATILITY ──

    "VIXY_30_C_2026-03-20": [
        {"name": "VIX", "current_value": "15.5", "target_value": ">25", "invalidation_value": "<12", "direction": "above"},
        {"name": "VIXY Price", "current_value": "$26.31", "target_value": ">$30", "invalidation_value": "<$22", "direction": "above"},
        {"name": "VIX Futures Contango", "current_value": "5%", "target_value": "<0%", "invalidation_value": ">8%", "direction": "below"},
        {"name": "MOVE Index", "current_value": "90", "target_value": ">110", "invalidation_value": "<70", "direction": "above"},
    ],

    # ── OPTIONS: SECTOR SHORTS ──

    "TAN_32_P_2026-06-18": [
        {"name": "TAN Price", "current_value": "$45", "target_value": "<$32", "invalidation_value": ">$55", "direction": "below"},
        {"name": "10Y Yield", "current_value": "4.5%", "target_value": ">5%", "invalidation_value": "<3.5%", "direction": "above"},
        {"name": "Solar Install Growth", "current_value": "8%", "target_value": "<5%", "invalidation_value": ">15%", "direction": "below"},
        {"name": "IRA Subsidy Risk", "current_value": "Moderate", "target_value": "Cuts", "invalidation_value": "Expansion", "direction": "above"},
    ],

    "CRWV_45_P_2026-08-21": [
        {"name": "CRWV Price", "current_value": "$48", "target_value": "<$45", "invalidation_value": ">$60", "direction": "below"},
        {"name": "GPU Rental Margin", "current_value": "50%", "target_value": "<40%", "invalidation_value": ">60%", "direction": "below"},
        {"name": "Customer Concentration", "current_value": "62%", "target_value": ">50%", "invalidation_value": "<30%", "direction": "above"},
        {"name": "Debt/EBITDA", "current_value": "8x", "target_value": ">10x", "invalidation_value": "<5x", "direction": "above"},
    ],

    "SNDK_150_P_2026-03-20": [
        {"name": "SNDK Price", "current_value": "$158", "target_value": "<$150", "invalidation_value": ">$170", "direction": "below"},
        {"name": "NAND Flash ASP", "current_value": "$0.08/GB", "target_value": "<$0.06", "invalidation_value": ">$0.10", "direction": "below"},
        {"name": "Memory Cycle Phase", "current_value": "Late peak", "target_value": "Downturn", "invalidation_value": "Upturn", "direction": "above"},
    ],

    # ── OPTIONS: SILVER SHORTS ──

    "SLV_65.5_P_2026-02-20": [
        {"name": "SLV Price", "current_value": "$74.77", "target_value": "<$65.50", "invalidation_value": ">$82", "direction": "below"},
        {"name": "Silver Spot (LBMA)", "current_value": "$86.10", "target_value": "<$72", "invalidation_value": ">$95", "direction": "below", "source": "silver_spot"},
        {"name": "Gold/Silver Ratio", "current_value": "88", "target_value": ">95", "invalidation_value": "<80", "direction": "above"},
        {"name": "DXY (USD Index)", "current_value": "106", "target_value": ">110", "invalidation_value": "<100", "direction": "above"},
        {"name": "Real Rates (10Y TIPS)", "current_value": "2.0%", "target_value": ">2.5%", "invalidation_value": "<1.5%", "direction": "above"},
    ],

    "SLV_65_P_2026-03-13": [
        {"name": "SLV Price", "current_value": "$74.77", "target_value": "<$65", "invalidation_value": ">$82", "direction": "below"},
        {"name": "Silver Spot (LBMA)", "current_value": "$86.10", "target_value": "<$72", "invalidation_value": ">$95", "direction": "below", "source": "silver_spot"},
        {"name": "Gold/Silver Ratio", "current_value": "88", "target_value": ">95", "invalidation_value": "<80", "direction": "above"},
        {"name": "DXY (USD Index)", "current_value": "106", "target_value": ">110", "invalidation_value": "<100", "direction": "above"},
        {"name": "Global Mfg PMI", "current_value": "50.1", "target_value": "<49", "invalidation_value": ">52", "direction": "below"},
    ],

    "SLV_52_P_2026-09-18": [
        {"name": "SLV Price", "current_value": "$74.77", "target_value": "<$52", "invalidation_value": ">$90", "direction": "below"},
        {"name": "Silver Spot (LBMA)", "current_value": "$86.10", "target_value": "<$57", "invalidation_value": ">$100", "direction": "below", "source": "silver_spot"},
        {"name": "Gold/Silver Ratio", "current_value": "88", "target_value": ">100", "invalidation_value": "<75", "direction": "above"},
        {"name": "DXY (USD Index)", "current_value": "106", "target_value": ">112", "invalidation_value": "<98", "direction": "above"},
        {"name": "Real Rates (10Y TIPS)", "current_value": "2.0%", "target_value": ">3.0%", "invalidation_value": "<1.0%", "direction": "above"},
        {"name": "COMEX Silver Inventory", "current_value": "290Moz", "target_value": ">350Moz", "invalidation_value": "<250Moz", "direction": "above"},
    ],

    "TLT_86_P_2026-10-17": [
        {"name": "CPI YoY", "current_value": "3.2%", "target_value": ">3.5%", "invalidation_value": "<2.5%", "direction": "above"},
        {"name": "10Y Auction Tail", "current_value": "5.7bp", "target_value": ">3bp", "invalidation_value": "<1bp", "direction": "above"},
        {"name": "Deficit/GDP", "current_value": "5.25%", "target_value": ">5%", "invalidation_value": "<4%", "direction": "above"},
        {"name": "Fed Funds", "current_value": "5.33%", "target_value": "hold/hike", "invalidation_value": "cuts", "direction": "above"},
        {"name": "TLT Price", "current_value": "88.08", "target_value": "<84", "invalidation_value": ">92", "direction": "below"},
    ],
}


# ═══════════════════════════════════════════════════════════════
# Indicator helpers
# ═══════════════════════════════════════════════════════════════

def _parse_numeric(value_str):
    """Extract a float from strings like '3.2%', '5.7bp', '88.08', '<84'. Returns None if not parseable."""
    if value_str is None or not isinstance(value_str, str):
        return None
    s = value_str.strip().replace(",", "")
    s = re.sub(r"^[<>=~]+\s*", "", s)
    s = s.replace("%", "").replace("bp", "")
    try:
        return float(s)
    except ValueError:
        return None


def position_indicator_key(position_dict):
    """Build lookup key for TRADE_INDICATORS_CONFIG from position dict."""
    opt_info = position_dict.get("opt_info")
    if not opt_info:
        return None
    u = (opt_info.get("underlying") or "").upper()
    s = opt_info.get("strike")
    t = (opt_info.get("opt_type") or opt_info.get("type") or "").upper()
    t = "P" if "P" in t else "C"
    e = opt_info.get("expiry", "")
    if not u or s is None or not e:
        return None
    s_val = float(s)
    s_str = str(int(s_val)) if s_val == int(s_val) else f"{s_val:g}"
    return f"{u}_{s_str}_{t}_{e}"


def indicator_status(indicator):
    """
    Compute how well an indicator supports the trade thesis.

    Maps current value onto a 0-1 scale between invalidation (0) and target (1):
        >= 0.7  -> green  (thesis working)
        0.3-0.7 -> yellow (neutral)
        < 0.3   -> red    (thesis under pressure)
    """
    direction = (indicator.get("direction") or "above").lower()
    curr = _parse_numeric(indicator.get("current_value") or indicator.get("current"))
    tgt = _parse_numeric(indicator.get("target_value") or indicator.get("target"))
    inv = _parse_numeric(indicator.get("invalidation_value") or indicator.get("invalidation"))

    if curr is None or tgt is None or inv is None:
        return "neutral"

    if direction == "above":
        if curr >= tgt:
            return "green"
        if curr <= inv:
            return "red"
        span = tgt - inv
        if span <= 0:
            return "neutral"
        pct = (curr - inv) / span
    else:
        if curr <= tgt:
            return "green"
        if curr >= inv:
            return "red"
        span = inv - tgt
        if span <= 0:
            return "neutral"
        pct = (inv - curr) / span

    if pct >= 0.7:
        return "green"
    if pct >= 0.3:
        return "yellow"
    return "red"


def get_indicators_for_position(position_dict):
    """
    Return indicators list with status for a position.
    Auto-updates current_value from live sources where configured.
    """
    symbols = [position_dict.get("symbol")]
    key = position_indicator_key(position_dict)
    if key:
        symbols.append(key)

    raw = None
    for k in symbols:
        if k and k in TRADE_INDICATORS_CONFIG:
            raw = TRADE_INDICATORS_CONFIG[k]
            break
    if not raw:
        return []

    out = []
    for ind in raw:
        i = dict(ind)
        i["current_value"] = i.get("current_value") or i.get("current")
        i["target_value"] = i.get("target_value") or i.get("target")
        i["invalidation_value"] = i.get("invalidation_value") or i.get("invalidation")
        # Auto-update from live source if configured
        source = i.get("source")
        if source:
            live_val = get_live_source_value(source)
            if live_val:
                i["current_value"] = live_val
        i["status"] = indicator_status(i)
        out.append(i)
    return out

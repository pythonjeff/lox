"""
Regime domain utilities for the LOX FUND Dashboard.
Handles classification of market regimes across multiple domains.
"""
import logging
import requests
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Regime detail cache (shared across all domains)
# ─────────────────────────────────────────────────────────────────────────────
_REGIME_DETAIL_CACHE = {
    "state": None,
    "prev_state": None,          # previous state for delta computation
    "mc_params": None,
    "fiscal_scorecard": None,
    "timestamp": None,
}
_REGIME_DETAIL_LOCK = threading.Lock()
_REGIME_DETAIL_TTL = 600  # 10 minutes — macro data changes daily at most

REGIME_NAMES = [
    "growth", "inflation", "volatility", "credit", "rates", "liquidity",
    "consumer", "fiscal", "usd", "commodities",
]

REGIME_DISPLAY_NAMES = {
    "growth": "Growth",
    "inflation": "Inflation",
    "volatility": "Volatility",
    "credit": "Credit",
    "rates": "Rates",
    "liquidity": "Liquidity",
    "consumer": "Consumer",
    "fiscal": "Fiscal",
    "usd": "USD",
    "commodities": "Commodities",
}

# Classification label → severity for color mapping.
# Severity levels: "low" (green), "moderate" (blue), "elevated" (amber), "high" (red)
REGIME_LABEL_SEVERITY = {
    # Growth (threshold-derived)
    "Boom": "low", "Accelerating": "low", "Stable Growth": "moderate",
    "Slowing": "elevated", "Contraction": "high",
    # Inflation (threshold-derived)
    "Deflationary": "elevated", "Below Target": "low", "At Target": "low",
    "Elevated": "moderate", "Above Target": "elevated", "Hot Inflation": "high",
    # Volatility (threshold-derived)
    "Low Volatility": "low", "Normal Volatility": "low",
    "Elevated Volatility": "elevated", "Vol Shock": "high", "Vol Crisis": "high",
    # Volatility (legacy classifier labels — keep for backward compat)
    "Normal volatility (baseline)": "low",
    "Elevated volatility (fragile risk)": "elevated",
    "Vol shock / stress (hedging bid)": "high",
    # Credit (threshold-derived)
    "Credit Euphoria": "moderate", "Credit Calm": "low", "Credit Neutral": "low",
    "Credit Widening": "elevated", "Credit Stress": "high", "Credit Crisis": "high",
    # Rates (threshold-derived)
    "Accommodative": "low", "Neutral Rates": "low", "Restrictive": "moderate",
    "Rate Shock": "elevated", "Rates Crisis": "high",
    # Rates (legacy classifier labels)
    "Neutral rates backdrop": "low",
    "Steep curve (risk-on / reflation-ish)": "low",
    "Rates shock lower (duration tailwind)": "moderate",
    "Rates shock higher (duration headwind)": "elevated",
    "Inverted curve (growth scare)": "high",
    # Liquidity (threshold-derived)
    "Flush Liquidity": "low", "Ample Funding": "low", "Normal Funding": "moderate",
    "Tightening": "elevated", "Structural Tightening": "elevated", "Funding Stress": "high",
    # Liquidity (legacy classifier labels)
    "Flush liquidity": "low", "Ample funding": "low", "Normal funding": "moderate",
    "Tightening / balance-sheet constraint": "elevated",
    "Structural tightening": "elevated", "Funding stress": "high",
    # Consumer (threshold-derived)
    "Consumer Boom": "low", "Consumer Expanding": "low", "Consumer Stable": "moderate",
    "Consumer Weakening": "elevated", "Consumer Stress": "high",
    # Fiscal (threshold-derived)
    "Benign": "low", "Elevated Funding": "moderate", "Stress Building": "elevated",
    "Fiscal Stress": "high", "Fiscal Dominance Risk": "high",
    # USD (threshold-derived)
    "Dollar Plunge": "high", "Weak Dollar": "moderate", "Neutral Dollar": "low",
    "Strong Dollar": "moderate", "Dollar Surge": "high",
    # Commodities (threshold-derived)
    "Disinflation": "low", "Neutral Commodities": "low",
    "Reflation": "elevated", "Commodity Shock": "high", "Energy Crisis": "high",
    # Commodities (legacy classifier labels)
    "Commodity disinflation (pressure easing)": "moderate",
    "Neutral commodities backdrop": "low",
    "Commodity reflation (inflation pressure)": "elevated",
    "Energy shock (inflation impulse)": "high",
}

# Actual score thresholds per regime (only for continuous-scoring classifiers).
# Each entry: list of (threshold_score, label_below, label_above)
REGIME_SCORE_THRESHOLDS = {
    "growth": [
        (30, "Boom", "Accelerating"), (45, "Accelerating", "Stable Growth"),
        (60, "Stable Growth", "Slowing"), (75, "Slowing", "Contraction"),
    ],
    "inflation": [
        (20, "Deflationary", "Below Target"), (35, "Below Target", "At Target"),
        (50, "At Target", "Elevated"), (65, "Elevated", "Above Target"),
        (80, "Above Target", "Hot Inflation"),
    ],
    "credit": [
        (25, "Credit Euphoria", "Credit Calm"), (40, "Credit Calm", "Credit Neutral"),
        (55, "Credit Neutral", "Credit Widening"), (65, "Credit Widening", "Credit Stress"),
        (80, "Credit Stress", "Credit Crisis"),
    ],
    "liquidity": [
        (30, "Flush Liquidity", "Ample Funding"), (45, "Ample Funding", "Normal Funding"),
        (55, "Normal Funding", "Tightening"), (65, "Tightening", "Structural Tightening"),
        (75, "Structural Tightening", "Funding Stress"),
    ],
    "consumer": [
        (30, "Consumer Boom", "Consumer Expanding"), (45, "Consumer Expanding", "Consumer Stable"),
        (55, "Consumer Stable", "Consumer Weakening"), (70, "Consumer Weakening", "Consumer Stress"),
    ],
    "fiscal": [
        (25, "Benign", "Elevated Funding"), (45, "Elevated Funding", "Stress Building"),
        (65, "Stress Building", "Fiscal Stress"), (80, "Fiscal Stress", "Fiscal Dominance Risk"),
    ],
    "volatility": [
        (30, "Low Volatility", "Normal Volatility"), (50, "Normal Volatility", "Elevated Volatility"),
        (70, "Elevated Volatility", "Vol Shock"), (85, "Vol Shock", "Vol Crisis"),
    ],
    "rates": [
        (30, "Accommodative", "Neutral Rates"), (50, "Neutral Rates", "Restrictive"),
        (65, "Restrictive", "Rate Shock"), (80, "Rate Shock", "Rates Crisis"),
    ],
    "usd": [
        (20, "Dollar Plunge", "Weak Dollar"), (40, "Weak Dollar", "Neutral Dollar"),
        (60, "Neutral Dollar", "Strong Dollar"), (80, "Strong Dollar", "Dollar Surge"),
    ],
    "commodities": [
        (30, "Disinflation", "Neutral Commodities"), (50, "Neutral Commodities", "Reflation"),
        (70, "Reflation", "Commodity Shock"), (85, "Commodity Shock", "Energy Crisis"),
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Professional regime blurbs (shown in hero zone)
# ─────────────────────────────────────────────────────────────────────────────
REGIME_BLURBS = {
    "growth": "How fast is the economy growing? Tracks jobs, factories, and early warning signs of a slowdown. When companies stop hiring and factories go quiet, a recession usually isn't far behind.",
    "inflation": "Are prices rising too fast? Watches the cost of living, what producers charge, and whether markets expect it to last. High inflation forces the Fed to raise rates, which hurts stocks and bonds.",
    "volatility": "Measures fear and its term structure. Tracks the VIX level, VX futures curve (contango vs backwardation), spot-to-futures basis, and near-term acceleration. When the curve flips to backwardation and spot trades above futures, it signals acute panic.",
    "credit": "Can companies still borrow cheaply? Tracks the extra interest that risky borrowers pay over safe government bonds. When that gap widens, it means investors are worried about defaults.",
    "rates": "What's happening with interest rates? Watches Treasury yields from 3-month bills to 10-year bonds and the shape of the curve. When short rates top long rates, it's a classic recession warning.",
    "liquidity": "Is the financial plumbing working and how much cash is in the system? Tracks overnight rates, the Fed's balance sheet, bank reserves, and the reverse repo facility. When liquidity tightens, asset prices feel it fast.",
    "consumer": "How is the average household doing? Tracks confidence surveys, spending, mortgage costs, and whether people are falling behind on loans. Consumer health drives about 70% of GDP.",
    "fiscal": "Can the government keep spending at this pace? Watches the deficit, debt interest costs, and how easily the Treasury can sell new bonds. Rising fiscal pressure pushes long-term rates higher.",
    "usd": "How strong is the dollar? Uses the FRED broad trade-weighted index (26 partners) with a 3-year z-score window. A strong dollar hurts U.S. exporters and squeezes EM borrowers. Big moves in either direction create global ripple effects.",
    "commodities": "What are raw materials telling us? Gold rises on fear, oil reflects supply and demand, copper tracks global factory activity, and silver follows both industrial use and safe-haven flows.",
}


def _build_regime_cache(settings, refresh=False):
    """Build or return cached unified regime state + MC params."""
    with _REGIME_DETAIL_LOCK:
        if _REGIME_DETAIL_CACHE["state"] and _REGIME_DETAIL_CACHE["timestamp"]:
            age = (datetime.now(timezone.utc) - _REGIME_DETAIL_CACHE["timestamp"]).total_seconds()
            if age < _REGIME_DETAIL_TTL and not refresh:
                return (
                    _REGIME_DETAIL_CACHE["state"],
                    _REGIME_DETAIL_CACHE["mc_params"],
                    _REGIME_DETAIL_CACHE["fiscal_scorecard"],
                )

    from lox.regimes.features import build_unified_regime_state

    state = build_unified_regime_state(settings=settings, refresh=refresh)
    mc_params = state.to_monte_carlo_params()

    fiscal_scorecard = None
    try:
        from lox.fiscal.signals import build_fiscal_state
        from lox.fiscal.scoring import score_fiscal_regime
        fiscal_state = build_fiscal_state(settings=settings, start_date="2011-01-01", refresh=refresh)
        fiscal_scorecard = score_fiscal_regime(fiscal_state.inputs)
    except Exception as e:
        logger.warning(f"Failed to build fiscal scorecard: {e}")

    with _REGIME_DETAIL_LOCK:
        # Save previous state for per-metric delta computation
        if _REGIME_DETAIL_CACHE["state"] is not None:
            _REGIME_DETAIL_CACHE["prev_state"] = _REGIME_DETAIL_CACHE["state"]
        _REGIME_DETAIL_CACHE["state"] = state
        _REGIME_DETAIL_CACHE["mc_params"] = mc_params
        _REGIME_DETAIL_CACHE["fiscal_scorecard"] = fiscal_scorecard
        _REGIME_DETAIL_CACHE["timestamp"] = datetime.now(timezone.utc)

    return state, mc_params, fiscal_scorecard


def _extract_mc_drivers(mc_params, domain):
    """Extract MC impact entries relevant to a specific domain from _drivers."""
    drivers = mc_params.get("_drivers", {})
    domain_title = REGIME_DISPLAY_NAMES.get(domain, domain.title())

    impact = {}
    domain_drivers = []
    for param_key, driver_list in drivers.items():
        if param_key.startswith("_"):
            continue
        for driver_str in driver_list:
            if domain_title.lower() in driver_str.lower() or domain in driver_str.lower():
                impact[param_key] = mc_params.get(param_key, 0)
                domain_drivers.append(driver_str)

    if not impact:
        return {}

    return {
        **{k: v for k, v in impact.items()},
        "drivers": domain_drivers,
    }


def _parse_raw_value(value):
    """Extract a numeric value from a metric value (string or number) for visual bars."""
    if isinstance(value, (int, float)):
        return float(value)
    import re
    s = str(value).strip()
    s = s.replace(",", "").replace("$", "").replace("T", "").replace("B", "").replace("K", "")
    match = re.search(r'[+-]?\d+\.?\d*', s)
    if match:
        try:
            return float(match.group())
        except (ValueError, TypeError):
            return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Unified metric spec: exactly 9 metrics per regime.
# Each entry: key (classifier lookup), name, weight, fmt (None=pre-formatted),
# plus range context (healthy/stressed endpoints, extreme, inverted).
# ─────────────────────────────────────────────────────────────────────────────
def _pct(v):  return f"{v:.1f}%"
def _pct2(v): return f"{v:.2f}%"
def _pctS(v): return f"{v:+.1f}%"
def _bp(v):   return f"{v:.0f}bp"
def _bpS(v):  return f"{v:+.0f}bp"
def _dec1(v): return f"{v:.1f}"
def _dec2(v): return f"{v:.2f}"
def _k(v):    return f"{v:+,.0f}K"
def _idx(v):  return f"{v:.0f}"


_REGIME_METRICS_SPEC = {
    # ── Growth (9 core metrics) ─────────────────────────────────────────────
    "growth": [
        {"key": "Payrolls 3m", "name": "Job Growth (3mo pace)", "weight": 0.15, "fmt": None,
         "desc": "Monthly hiring trend, annualized",
         "healthy": 300, "stressed": -150, "healthy_label": "+300K", "stressed_label": "-150K",
         "inverted": True},
        {"key": "ISM Mfg", "name": "Factory Activity (ISM)", "weight": 0.14, "fmt": None,
         "desc": "Above 50 = factories expanding",
         "healthy": 55, "stressed": 42, "healthy_label": "55", "stressed_label": "42",
         "inverted": True},
        {"key": "Claims 4wk", "name": "Jobless Claims (4wk avg)", "weight": 0.12, "fmt": None,
         "desc": "New unemployment filings per week",
         "healthy": 210000, "stressed": 350000, "healthy_label": "210K", "stressed_label": "350K",
         "inverted": False},
        {"key": "IndProd YoY", "name": "Industrial Output YoY", "weight": 0.11, "fmt": None,
         "desc": "Factory and mine production growth",
         "healthy": 3.0, "stressed": -3.0, "healthy_label": "+3%", "stressed_label": "-3%",
         "inverted": True},
        {"key": "Retail MoM", "name": "Core Retail Sales MoM", "weight": 0.10, "fmt": None,
         "desc": "Consumer spending ex-autos and gas",
         "healthy": 0.5, "stressed": -0.5, "healthy_label": "+0.5%", "stressed_label": "-0.5%",
         "inverted": True},
        {"key": "UNRATE", "name": "Unemployment Rate", "weight": 0.10, "fmt": None,
         "desc": "Share of workforce without a job",
         "healthy": 3.8, "stressed": 6.0, "healthy_label": "3.8%", "stressed_label": "6.0%",
         "inverted": False},
        {"key": "ISM New Ord", "name": "Factory New Orders (ISM)", "weight": 0.10, "fmt": None,
         "desc": "Forward-looking demand for goods",
         "healthy": 55, "stressed": 42, "healthy_label": "55", "stressed_label": "42",
         "inverted": True},
        {"key": "2s10s", "name": "Yield Curve (2s/10s)", "weight": 0.10, "fmt": None,
         "desc": "Negative = classic recession signal",
         "healthy": 1.0, "stressed": -0.5, "healthy_label": "+1.0%", "stressed_label": "-0.5%",
         "inverted": True},
        {"key": "Mich Expect", "name": "Consumer Expectations", "weight": 0.08, "fmt": None,
         "desc": "U of Michigan forward-looking sentiment",
         "healthy": 90, "stressed": 45, "healthy_label": "90", "stressed_label": "45",
         "inverted": True},
    ],

    # ── Inflation (9 core metrics) ──────────────────────────────────────────
    "inflation": [
        {"key": "Core PCE", "name": "Core PCE (Fed's preferred)", "weight": 0.16, "fmt": None,
         "desc": "Prices ex-food/energy — the Fed's #1 gauge",
         "healthy": 2.0, "stressed": 4.0, "healthy_label": "2.0%", "stressed_label": "4.0%",
         "inverted": False},
        {"key": "CPI YoY", "name": "Consumer Prices (CPI)", "weight": 0.14, "fmt": None,
         "desc": "Headline cost-of-living increase",
         "healthy": 2.0, "stressed": 5.0, "healthy_label": "2.0%", "stressed_label": "5.0%",
         "inverted": False},
        {"key": "Core CPI", "name": "Core CPI (ex-food/energy)", "weight": 0.12, "fmt": None,
         "desc": "Underlying price trend without volatile items",
         "healthy": 2.0, "stressed": 5.0, "healthy_label": "2.0%", "stressed_label": "5.0%",
         "inverted": False},
        {"key": "Trimmed PCE", "name": "Trimmed Mean PCE", "weight": 0.10, "fmt": None,
         "desc": "Strips outliers for a cleaner signal",
         "healthy": 2.0, "stressed": 4.0, "healthy_label": "2.0%", "stressed_label": "4.0%",
         "inverted": False},
        {"key": "PPI YoY", "name": "Producer Prices (PPI)", "weight": 0.10, "fmt": None,
         "desc": "Wholesale costs — leads consumer prices",
         "healthy": 1.5, "stressed": 6.0, "healthy_label": "+1.5%", "stressed_label": "+6.0%",
         "inverted": False},
        {"key": "CPI 3m Ann", "name": "CPI Trend (3mo pace)", "weight": 0.10, "fmt": None,
         "desc": "Recent momentum — is inflation accelerating?",
         "healthy": 2.0, "stressed": 5.0, "healthy_label": "2.0%", "stressed_label": "5.0%",
         "inverted": False},
        {"key": "5Y BE", "name": "5Y Inflation Expectations", "weight": 0.10, "fmt": None,
         "desc": "What bond markets expect inflation to average",
         "healthy": 2.2, "stressed": 3.5, "healthy_label": "2.20%", "stressed_label": "3.50%",
         "inverted": False},
        {"key": "5Y5Y Fwd", "name": "Long-Run Expectations (5Y5Y)", "weight": 0.10, "fmt": None,
         "desc": "Are long-term expectations still anchored?",
         "healthy": 2.2, "stressed": 3.0, "healthy_label": "2.20%", "stressed_label": "3.00%",
         "inverted": False},
        {"key": "Oil YoY", "name": "Oil Price Change YoY", "weight": 0.08, "fmt": None,
         "desc": "Energy costs feed into everything else",
         "healthy": 0.0, "stressed": 30.0, "healthy_label": "0%", "stressed_label": "+30%",
         "inverted": False},
    ],

    # ── Volatility (9 metrics) ──────────────────────────────────────────────
    "volatility": [
        {"key": "VIX", "name": "Fear Index (VIX)", "weight": 0.15, "fmt": None,
         "desc": "Expected stock market volatility over 30 days",
         "healthy": 12, "healthy_label": "Calm", "stressed": 35, "stressed_label": "Fear",
         "extreme": 82.69, "extreme_date": "Mar 2020", "extreme_event": "COVID crash", "inverted": False},
        {"key": "VIX z", "name": "VIX vs History", "weight": 0.12, "fmt": None,
         "desc": "How unusual is the current VIX level?",
         "healthy": -0.5, "healthy_label": "Below avg", "stressed": 2.5, "stressed_label": "Extreme",
         "extreme": 4.0, "extreme_date": "Mar 2020", "extreme_event": "COVID", "inverted": False},
        {"key": "VX Contango", "name": "Futures Curve Shape", "weight": 0.14, "fmt": None,
         "desc": "Positive = normal, negative = panic (backwardation)",
         "healthy": 5.0, "healthy_label": "Contango", "stressed": -3.0, "stressed_label": "Backwardation",
         "extreme": -15.0, "extreme_date": "Mar 2020", "extreme_event": "COVID VX inversion", "inverted": True},
        {"key": "Spot Basis", "name": "Spot vs Futures Gap", "weight": 0.12, "fmt": None,
         "desc": "VIX above futures = acute fear premium",
         "healthy": -5.0, "healthy_label": "Discount", "stressed": 5.0, "stressed_label": "Spot premium",
         "extreme": 25.0, "extreme_date": "Mar 2020", "extreme_event": "COVID panic premium", "inverted": False},
        {"key": "9d/VIX Ratio", "name": "Near-Term Fear Ratio", "weight": 0.10, "fmt": None,
         "desc": "Short-term vol outpacing 30-day = imminent risk",
         "healthy": 0.90, "healthy_label": "Calm", "stressed": 1.15, "stressed_label": "Near-term panic",
         "extreme": 1.50, "extreme_date": "Aug 2024", "extreme_event": "Yen carry unwind", "inverted": False},
        {"key": "Term Spread", "name": "Vol Term Structure", "weight": 0.10, "fmt": None,
         "desc": "VIX above 3-month VIX = inverted (bearish)",
         "healthy": -2.0, "healthy_label": "Contango", "stressed": 3.0, "stressed_label": "Inverted",
         "extreme": 10.0, "extreme_date": "Aug 2024", "extreme_event": "Yen carry unwind", "inverted": False},
        {"key": "VIX 5d Chg", "name": "VIX 5-Day Spike", "weight": 0.09, "fmt": None, "abs_value": True,
         "desc": "How fast is fear rising this week?",
         "healthy": 0, "healthy_label": "Stable", "stressed": 30, "stressed_label": "Spiking",
         "extreme": 100, "extreme_date": "Mar 2020", "extreme_event": "COVID crash", "inverted": False},
        {"key": "Spike 20d", "name": "Spike Persistence (20d)", "weight": 0.09, "fmt": None,
         "desc": "How often has VIX spiked recently?",
         "healthy": 0, "healthy_label": "No spikes", "stressed": 40, "stressed_label": "Repeated spikes",
         "extreme": 80, "extreme_date": "Mar 2020", "extreme_event": "COVID volatility cluster", "inverted": False},
        {"key": "Vol Pressure", "name": "Overall Vol Pressure", "weight": 0.09, "fmt": None,
         "desc": "Composite score of all vol signals",
         "healthy": -1.0, "healthy_label": "Suppressed", "stressed": 2.0, "stressed_label": "Elevated",
         "extreme": 4.0, "extreme_date": "Mar 2020", "extreme_event": "COVID", "inverted": False},
    ],

    # ── Credit (9 metrics) ──────────────────────────────────────────────────
    "credit": [
        {"key": "HY OAS", "name": "Junk Bond Spread (HY)", "weight": 0.18, "fmt": None,
         "desc": "Extra yield investors demand for risky debt",
         "healthy": 300, "healthy_label": "Tight", "stressed": 600, "stressed_label": "Stress",
         "extreme": 1100, "extreme_date": "Mar 2020", "extreme_event": "COVID credit freeze", "inverted": False},
        {"key": "BBB OAS", "name": "Investment Grade Spread (BBB)", "weight": 0.12, "fmt": None,
         "desc": "Lowest-rung investment grade — first to crack",
         "healthy": 100, "healthy_label": "Tight", "stressed": 250, "stressed_label": "Widening",
         "extreme": 401, "extreme_date": "Mar 2020", "extreme_event": "Fallen-angel fears", "inverted": False},
        {"key": "HY-IG", "name": "Junk vs IG Gap", "weight": 0.11, "fmt": None,
         "desc": "Widening = investors fleeing risky debt",
         "healthy": 200, "healthy_label": "Compressed", "stressed": 400, "stressed_label": "Quality flight",
         "extreme": 700, "extreme_date": "Mar 2020", "extreme_event": "COVID", "inverted": False},
        {"key": "CCC OAS", "name": "Distressed Debt Spread (CCC)", "weight": 0.10, "fmt": None,
         "desc": "Near-default borrowers — early warning canary",
         "healthy": 600, "healthy_label": "Risk-on", "stressed": 1200, "stressed_label": "Distress",
         "extreme": 2000, "extreme_date": "Mar 2020", "extreme_event": "COVID", "inverted": False},
        {"key": "BB OAS", "name": "Mid-Grade Junk Spread (BB)", "weight": 0.09, "fmt": None,
         "desc": "Largest junk bond segment by volume",
         "healthy": 200, "healthy_label": "Tight", "stressed": 400, "stressed_label": "Widening",
         "extreme": 700, "extreme_date": "Mar 2020", "extreme_event": "COVID", "inverted": False},
        {"key": "AAA OAS", "name": "Safest Corporate Spread (AAA)", "weight": 0.08, "fmt": None,
         "desc": "Even top-rated debt widens in a panic",
         "healthy": 40, "healthy_label": "Tight", "stressed": 120, "stressed_label": "Flight to quality",
         "extreme": 175, "extreme_date": "Mar 2020", "extreme_event": "Liquidity crisis", "inverted": False},
        {"key": "HY 30d Chg", "name": "Junk Spread Momentum (30d)", "weight": 0.10, "fmt": None,
         "desc": "How fast are spreads widening?",
         "healthy": 0, "healthy_label": "Stable", "stressed": 50, "stressed_label": "Rapid widening",
         "extreme": 350, "extreme_date": "Mar 2020", "extreme_event": "COVID panic", "inverted": False},
        {"key": "SLOOS", "name": "Bank Lending Standards", "weight": 0.12, "fmt": None,
         "desc": "Are banks making it harder to borrow?",
         "healthy": -10, "healthy_label": "Easing", "stressed": 40, "stressed_label": "Tight",
         "extreme": 70, "extreme_date": "Q1 2023", "extreme_event": "SVB banking stress", "inverted": False},
        {"key": "CC Delinq", "name": "Credit Card Delinquency", "weight": 0.10, "fmt": None,
         "desc": "Consumers falling behind on payments",
         "healthy": 2.5, "healthy_label": "Healthy", "stressed": 5.0, "stressed_label": "Stress",
         "extreme": 6.77, "extreme_date": "Q1 2010", "extreme_event": "Post-GFC peak", "inverted": False},
    ],

    # ── Rates (9 metrics) ───────────────────────────────────────────────────
    "rates": [
        {"key": "10Y", "name": "10-Year Treasury Yield", "weight": 0.16, "fmt": None,
         "desc": "Benchmark rate for mortgages and corporate debt",
         "healthy": 2.5, "healthy_label": "Neutral", "stressed": 5.0, "stressed_label": "Restrictive",
         "extreme": 5.02, "extreme_date": "Oct 2023", "extreme_event": "Term premium surge", "inverted": False},
        {"key": "2s10s", "name": "Yield Curve (2s/10s)", "weight": 0.14, "fmt": None,
         "desc": "Inverted curve = classic recession warning",
         "healthy": 100, "healthy_label": "Normal", "stressed": -80, "stressed_label": "Deep inversion",
         "extreme": -108, "extreme_date": "Jul 2023", "extreme_event": "Deepest inversion since '81", "inverted": True},
        {"key": "2Y", "name": "2-Year Treasury Yield", "weight": 0.12, "fmt": None,
         "desc": "Reflects market's view on near-term Fed policy",
         "healthy": 2.0, "healthy_label": "Neutral", "stressed": 5.0, "stressed_label": "Restrictive",
         "extreme": 5.12, "extreme_date": "Jul 2023", "extreme_event": "Peak rate pricing", "inverted": False},
        {"key": "10Y 20d Chg", "name": "10Y Rate Momentum (20d)", "weight": 0.10, "fmt": None, "abs_value": True,
         "desc": "How fast are long-term rates moving?",
         "healthy": 0, "healthy_label": "Stable", "stressed": 40, "stressed_label": "Rapid repricing",
         "extreme": 80, "extreme_date": "Oct 2023", "extreme_event": "Term premium shock", "inverted": False},
        {"key": "3M", "name": "3-Month T-Bill Rate", "weight": 0.10, "fmt": None,
         "desc": "Short-term rate set by Fed policy",
         "healthy": 2.0, "healthy_label": "Neutral", "stressed": 5.5, "stressed_label": "Restrictive",
         "extreme": 5.56, "extreme_date": "Jul 2023", "extreme_event": "Peak Fed tightening", "inverted": False},
        {"key": "3M10Y", "name": "Yield Curve (3mo/10Y)", "weight": 0.10, "fmt": None,
         "desc": "Fed's preferred recession indicator",
         "healthy": 150, "healthy_label": "Normal", "stressed": -100, "stressed_label": "Recession signal",
         "extreme": -190, "extreme_date": "May 2023", "extreme_event": "Deepest 3M10Y inversion", "inverted": True},
        {"key": "2s10s 20d Chg", "name": "Curve Momentum (20d)", "weight": 0.09, "fmt": None, "abs_value": True,
         "desc": "Is the yield curve steepening or flattening fast?",
         "healthy": 0, "healthy_label": "Stable", "stressed": 30, "stressed_label": "Curve whiplash",
         "extreme": 60, "extreme_date": "Mar 2020", "extreme_event": "COVID curve steepening", "inverted": False},
        {"key": "10Y z", "name": "10Y Rate vs History", "weight": 0.10, "fmt": None,
         "desc": "How extreme is the current 10Y yield?",
         "healthy": 0, "healthy_label": "Normal", "stressed": 2.0, "stressed_label": "Extreme",
         "extreme": 3.0, "extreme_date": "Oct 2023", "extreme_event": "Rate shock", "inverted": False},
        {"key": "Real 10Y", "name": "Real Interest Rate (10Y)", "weight": 0.09, "fmt": None,
         "desc": "Inflation-adjusted borrowing cost",
         "healthy": 0.5, "healthy_label": "Neutral", "stressed": 2.5, "stressed_label": "Restrictive",
         "extreme": 2.51, "extreme_date": "Oct 2023", "extreme_event": "Real rate shock", "inverted": False},
    ],

    # ── Liquidity (9 core metrics) ──────────────────────────────────────────
    "liquidity": [
        {"key": "SOFR", "name": "Overnight Funding Rate (SOFR)", "weight": 0.14, "fmt": None,
         "desc": "Cost of borrowing cash overnight",
         "healthy": 2.0, "healthy_label": "Accommodative", "stressed": 5.5, "stressed_label": "Restrictive",
         "inverted": False},
        {"key": "EFFR", "name": "Fed Funds Rate (EFFR)", "weight": 0.12, "fmt": None,
         "desc": "Rate banks charge each other overnight",
         "healthy": 2.0, "healthy_label": "Accommodative", "stressed": 5.5, "stressed_label": "Restrictive",
         "inverted": False},
        {"key": "Corridor", "name": "Rate Corridor (SOFR-EFFR)", "weight": 0.10, "fmt": None,
         "desc": "Gap signals plumbing stress in money markets",
         "healthy": 0, "healthy_label": "Normal", "stressed": 10, "stressed_label": "Plumbing stress",
         "inverted": False},
        {"key": "Reserves", "name": "Bank Reserves ($T)", "weight": 0.14, "fmt": None,
         "desc": "Cash banks hold at the Fed — lifeblood of lending",
         "healthy": 3.5, "healthy_label": "Ample", "stressed": 2.5, "stressed_label": "Scarce",
         "inverted": True},
        {"key": "Reserve z", "name": "Reserves vs History", "weight": 0.10, "fmt": None,
         "desc": "Are reserves unusually low or high?",
         "healthy": 0.5, "healthy_label": "Ample", "stressed": -1.5, "stressed_label": "Scarce",
         "inverted": True},
        {"key": "RRP", "name": "Reverse Repo Balance ($B)", "weight": 0.10, "fmt": None,
         "desc": "Excess cash parked at the Fed — draining = tighter",
         "healthy": 500, "healthy_label": "Buffer intact", "stressed": 0, "stressed_label": "Depleted",
         "inverted": True},
        {"key": "TGA", "name": "Treasury Cash Balance ($B)", "weight": 0.10, "fmt": None,
         "desc": "Government's checking account — low = debt ceiling risk",
         "healthy": 800, "healthy_label": "Healthy", "stressed": 200, "stressed_label": "Running low",
         "inverted": True},
        {"key": "Fed Assets", "name": "Fed Balance Sheet ($T)", "weight": 0.12, "fmt": None,
         "desc": "QE adds liquidity, QT removes it",
         "healthy": 7.0, "healthy_label": "Normalizing", "stressed": 4.5, "stressed_label": "Tight",
         "inverted": True},
        {"key": "Fed Assets z", "name": "Fed Balance Sheet vs History", "weight": 0.08, "fmt": None,
         "desc": "How unusual is the current Fed balance sheet?",
         "healthy": 0, "healthy_label": "Normal", "stressed": -2.0, "stressed_label": "Contracting",
         "inverted": True},
    ],

    # ── Consumer (9 metrics) ────────────────────────────────────────────────
    "consumer": [
        {"key": "Michigan", "name": "Consumer Confidence", "weight": 0.15, "fmt": None,
         "desc": "How households feel about the economy right now",
         "healthy": 90, "healthy_label": "Confident", "stressed": 60, "stressed_label": "Pessimistic",
         "extreme": 50.0, "extreme_date": "Jun 2022", "extreme_event": "Inflation shock", "inverted": True},
        {"key": "30Y Mortgage", "name": "Mortgage Rate (30Y)", "weight": 0.13, "fmt": None,
         "desc": "Cost of buying a home — weighs on spending",
         "healthy": 4.0, "healthy_label": "Affordable", "stressed": 7.5, "stressed_label": "Crushing",
         "extreme": 7.79, "extreme_date": "Oct 2023", "extreme_event": "Rate shock", "inverted": False},
        {"key": "Retail MoM", "name": "Retail Spending MoM", "weight": 0.12, "fmt": None,
         "desc": "Monthly change in consumer spending",
         "healthy": 0.3, "healthy_label": "Growing", "stressed": -0.5, "stressed_label": "Falling",
         "extreme": -8.2, "extreme_date": "Apr 2020", "extreme_event": "COVID lockdown", "inverted": True},
        {"key": "Mich Expect", "name": "Consumer Outlook", "weight": 0.11, "fmt": None,
         "desc": "Where do households think the economy is heading?",
         "healthy": 80, "healthy_label": "Optimistic", "stressed": 50, "stressed_label": "Fearful",
         "extreme": 46.8, "extreme_date": "Jun 2022", "extreme_event": "Inflation angst", "inverted": True},
        {"key": "CC Debt YoY", "name": "Consumer Borrowing YoY", "weight": 0.11, "fmt": None,
         "desc": "Growing = confident spending, shrinking = pullback",
         "healthy": 5.0, "healthy_label": "Healthy", "stressed": -2.0, "stressed_label": "Deleveraging",
         "extreme": -5.0, "extreme_date": "2010", "extreme_event": "Post-GFC contraction", "inverted": True},
        {"key": "Savings Rate", "name": "Personal Savings Rate", "weight": 0.10, "fmt": None,
         "desc": "Low savings = households stretched thin",
         "healthy": 7.0, "healthy_label": "Buffer", "stressed": 3.0, "stressed_label": "Depleted",
         "extreme": 2.2, "extreme_date": "Jun 2022", "extreme_event": "Inflation squeeze", "inverted": True},
        {"key": "Home Px YoY", "name": "Home Prices YoY", "weight": 0.10, "fmt": None,
         "desc": "Wealth effect — rising prices boost spending",
         "healthy": 5.0, "healthy_label": "Appreciating", "stressed": -5.0, "stressed_label": "Declining",
         "extreme": -18.0, "extreme_date": "Feb 2009", "extreme_event": "GFC crash", "inverted": True},
        {"key": "Delinquency", "name": "Loan Delinquency Rate", "weight": 0.09, "fmt": None,
         "desc": "Share of consumer loans past due",
         "healthy": 2.0, "healthy_label": "Current", "stressed": 5.0, "stressed_label": "Rising defaults",
         "extreme": 7.4, "extreme_date": "Q4 2009", "extreme_event": "GFC default wave", "inverted": False},
        {"key": "Auto Tighten", "name": "Auto Loan Standards", "weight": 0.09, "fmt": None,
         "desc": "Banks tightening = harder to buy a car",
         "healthy": -10, "healthy_label": "Easing", "stressed": 30, "stressed_label": "Tight",
         "extreme": 60, "extreme_date": "Q1 2023", "extreme_event": "SVB banking stress", "inverted": False},
    ],

    # ── Fiscal (9 metrics) ──────────────────────────────────────────────────
    "fiscal": [
        {"key": "Deficit 12m", "name": "Annual Deficit ($T)", "weight": 0.16, "fmt": None,
         "desc": "How much more the government spends than it earns",
         "healthy": 0.5, "healthy_label": "Manageable", "stressed": 2.5, "stressed_label": "Unsustainable",
         "extreme": 3.13, "extreme_date": "2020", "extreme_event": "COVID spending", "inverted": False},
        {"key": "Deficit/Receipts", "name": "Deficit as % of Revenue", "weight": 0.14, "fmt": None,
         "desc": "Gap between spending and tax receipts",
         "healthy": 10, "healthy_label": "Low", "stressed": 40, "stressed_label": "Heavy",
         "extreme": 80, "extreme_date": "2020", "extreme_event": "COVID", "inverted": False},
        {"key": "z Deficit", "name": "Deficit vs History", "weight": 0.10, "fmt": None,
         "desc": "How unusual is the current deficit?",
         "healthy": -1.0, "healthy_label": "Normal", "stressed": 2.0, "stressed_label": "Extreme",
         "extreme": 3.5, "extreme_date": "2020", "extreme_event": "COVID", "inverted": False},
        {"key": "Deficit Trend", "name": "Deficit Trajectory", "weight": 0.10, "fmt": None,
         "desc": "Slope of 12-month deficit trend — rising = worsening",
         "healthy": -0.5, "healthy_label": "Improving", "stressed": 0.5, "stressed_label": "Worsening",
         "extreme": 1.5, "extreme_date": "2020", "extreme_event": "COVID spike", "inverted": False},
        {"key": "Interest YoY", "name": "Interest Cost Growth YoY", "weight": 0.12, "fmt": None,
         "desc": "How fast is debt interest eating the budget?",
         "healthy": 0, "healthy_label": "Stable", "stressed": 20, "stressed_label": "Surging",
         "extreme": 40, "extreme_date": "2024", "extreme_event": "Rate pass-through", "inverted": False},
        {"key": "TGA", "name": "Treasury Cash Balance ($B)", "weight": 0.10, "fmt": None,
         "desc": "Government's checking account — low = debt ceiling risk",
         "healthy": 800, "healthy_label": "Healthy", "stressed": 200, "stressed_label": "Running low",
         "extreme": 50, "extreme_date": "Oct 2015", "extreme_event": "Debt ceiling crisis", "inverted": True},
        {"key": "TGA 28d Chg", "name": "Treasury Cash Change (28d)", "weight": 0.08, "fmt": None,
         "desc": "Is the government's cash balance growing or shrinking?",
         "healthy": 50, "healthy_label": "Building", "stressed": -100, "stressed_label": "Draining",
         "extreme": -200, "extreme_date": "2023", "extreme_event": "Debt ceiling drawdown", "inverted": True},
        {"key": "Auction Tail", "name": "Auction Demand (Tail)", "weight": 0.10, "fmt": None,
         "desc": "Higher tail = Treasury paid more to sell bonds",
         "healthy": 0.0, "healthy_label": "Strong", "stressed": 4.0, "stressed_label": "Weak",
         "extreme": 6.0, "extreme_date": "2023", "extreme_event": "Term premium surge", "inverted": False},
        {"key": "Dealer Take", "name": "Primary Dealer Takedown (%)", "weight": 0.10, "fmt": None,
         "desc": "High = dealers stuck with bonds no one else wants",
         "healthy": 10, "healthy_label": "Low", "stressed": 25, "stressed_label": "Forced buying",
         "extreme": 35, "extreme_date": "2020", "extreme_event": "COVID auction stress", "inverted": False},
    ],


    # ── USD (9 metrics) ─────────────────────────────────────────────────────
    "usd": [
        {"key": "DXY", "name": "Trade-Weighted Dollar", "weight": 0.16, "fmt": None,
         "desc": "FRED broad index — 26 trading partners, not the DXY",
         "healthy": 108, "healthy_label": "Balanced", "stressed": 128, "stressed_label": "Strong headwind",
         "extreme": 130, "extreme_date": "Sep 2022", "extreme_event": "Dollar wrecking ball", "inverted": False},
        {"key": "YoY Chg", "name": "Dollar Change YoY", "weight": 0.14, "fmt": None,
         "desc": "Year-over-year shift — captures sustained trends",
         "healthy": 0, "healthy_label": "Stable", "stressed": 8, "stressed_label": "Major trend",
         "extreme": 15, "extreme_date": "Sep 2022", "extreme_event": "Dollar surge", "abs_value": True, "inverted": False},
        {"key": "60d Chg", "name": "Dollar Trend (60d)", "weight": 0.12, "fmt": None,
         "desc": "Medium-term direction — are we strengthening or weakening?",
         "healthy": 0, "healthy_label": "Stable", "stressed": 5, "stressed_label": "Trend move",
         "extreme": 10, "extreme_date": "Sep 2022", "extreme_event": "Dollar surge", "abs_value": True, "inverted": False},
        {"key": "20d Chg", "name": "Dollar Momentum (20d)", "weight": 0.10, "fmt": None,
         "desc": "Short-term speed of dollar moves",
         "healthy": 0, "healthy_label": "Stable", "stressed": 3, "stressed_label": "Rapid move",
         "extreme": 6, "extreme_date": "Mar 2020", "extreme_event": "Flight to safety", "abs_value": True, "inverted": False},
        {"key": "USD z", "name": "Dollar Level vs 3yr History", "weight": 0.12, "fmt": None,
         "desc": "Is the dollar unusually high or low vs 3yr avg?",
         "healthy": 0, "healthy_label": "Normal", "stressed": 2.0, "stressed_label": "Extreme",
         "extreme": 3.0, "extreme_date": "Sep 2022", "extreme_event": "Dollar spike", "inverted": False},
        {"key": "Strength", "name": "Dollar Strength Score", "weight": 0.10, "fmt": None,
         "desc": "Composite of level + momentum z-scores",
         "healthy": 0, "healthy_label": "Neutral", "stressed": 1.5, "stressed_label": "Strong $",
         "extreme": 2.5, "extreme_date": "Sep 2022", "extreme_event": "Dollar wrecking ball", "inverted": False},
        {"key": "200d MA Dist", "name": "Gap from 200-Day Avg", "weight": 0.10, "fmt": None,
         "desc": "Far from average = mean-reversion risk",
         "healthy": 0, "healthy_label": "At trend", "stressed": 3, "stressed_label": "Stretched",
         "extreme": 6, "extreme_date": "Sep 2022", "extreme_event": "Well above trend", "abs_value": True, "inverted": False},
        {"key": "60d z", "name": "Recent Move vs 3yr History", "weight": 0.08, "fmt": None,
         "desc": "How unusual is the recent 60d move?",
         "healthy": 0, "healthy_label": "Normal", "stressed": 2.0, "stressed_label": "Extreme",
         "extreme": 3.0, "extreme_date": "Sep 2022", "extreme_event": "Dollar surge", "abs_value": True, "inverted": False},
        {"key": "90d RVol", "name": "Dollar Volatility (90d)", "weight": 0.08, "fmt": None,
         "desc": "How choppy are FX markets?",
         "healthy": 4, "healthy_label": "Calm", "stressed": 8, "stressed_label": "Volatile",
         "extreme": 12, "extreme_date": "Mar 2020", "extreme_event": "COVID volatility", "inverted": False},
    ],

    # ── Commodities (9 metrics) ─────────────────────────────────────────────
    "commodities": [
        {"key": "Gold", "name": "Gold Price", "weight": 0.14, "fmt": None,
         "desc": "Safe-haven demand — rises on fear and dollar weakness",
         "healthy": 50, "healthy_label": "Normal", "stressed": 120, "stressed_label": "Fear bid",
         "extreme": 150, "extreme_date": "Oct 2024", "extreme_event": "De-dollarization bid", "inverted": False},
        {"key": "Silver", "name": "Silver Price", "weight": 0.10, "fmt": None,
         "desc": "Part industrial, part safe haven",
         "healthy": 25, "healthy_label": "Normal", "stressed": 100, "stressed_label": "Risk bid",
         "extreme": 130, "extreme_date": "Feb 2021", "extreme_event": "Precious metals rally", "inverted": False},
        {"key": "WTI", "name": "Oil Price (WTI Crude)", "weight": 0.14, "fmt": None,
         "desc": "Energy cost driver — spikes feed into inflation",
         "healthy": 55, "healthy_label": "Balanced", "stressed": 95, "stressed_label": "Supply shock",
         "extreme": 124, "extreme_date": "Jun 2022", "extreme_event": "Ukraine war", "inverted": False},
        {"key": "Copper", "name": "Copper Price (Dr. Copper)", "weight": 0.12, "fmt": None,
         "desc": "Industrial bellwether — falling = global slowdown",
         "healthy": 35, "healthy_label": "Normal", "stressed": 22, "stressed_label": "Recession signal",
         "extreme": 18, "extreme_date": "Mar 2020", "extreme_event": "COVID demand collapse", "inverted": True},
        {"key": "Broad 60d", "name": "Commodity Index Move (60d)", "weight": 0.10, "fmt": None,
         "desc": "Broad basket — rising = supply disruption risk",
         "healthy": 0, "healthy_label": "Stable", "stressed": 10, "stressed_label": "Supply disruption",
         "extreme": 25, "extreme_date": "Mar 2022", "extreme_event": "Ukraine commodity shock", "inverted": False},
        {"key": "Gold 20d Ret", "name": "Gold Momentum (20d)", "weight": 0.09, "fmt": None, "abs_value": True,
         "desc": "Speed of gold price changes",
         "healthy": 0, "healthy_label": "Stable", "stressed": 5, "stressed_label": "Big move",
         "extreme": 15, "extreme_date": "Mar 2020", "extreme_event": "Flight to safety", "inverted": False},
        {"key": "WTI 20d Ret", "name": "Oil Momentum (20d)", "weight": 0.10, "fmt": None, "abs_value": True,
         "desc": "Speed of oil price changes",
         "healthy": 0, "healthy_label": "Stable", "stressed": 15, "stressed_label": "Supply shock",
         "extreme": 30, "extreme_date": "Mar 2020", "extreme_event": "COVID oil crash", "inverted": False},
        {"key": "Copper 60d Ret", "name": "Copper Trend (60d)", "weight": 0.10, "fmt": None, "abs_value": True,
         "desc": "Copper direction signals global growth outlook",
         "healthy": 0, "healthy_label": "Stable", "stressed": 10, "stressed_label": "Growth signal",
         "extreme": 25, "extreme_date": "2021", "extreme_event": "Recovery surge", "inverted": False},
        {"key": "Pressure", "name": "Commodity Pressure Score", "weight": 0.11, "fmt": None,
         "desc": "Composite of all commodity stress signals",
         "healthy": -0.5, "healthy_label": "Calm", "stressed": 1.5, "stressed_label": "Inflationary",
         "extreme": 3.0, "extreme_date": "Jun 2022", "extreme_event": "Broad commodity surge", "inverted": False},
    ],
}


def _enrich_metrics(regime_result, domain, prev_regime_result=None):
    """Build metric rows from _REGIME_METRICS_SPEC, pulling live values from
    the classifier's output where available.  Computes per-metric deltas from
    the previous state for trend display."""
    spec = _REGIME_METRICS_SPEC.get(domain, [])
    if not spec:
        return []

    raw_data = {}
    if regime_result and regime_result.metrics:
        raw_data = regime_result.metrics

    prev_raw_data = {}
    if prev_regime_result and prev_regime_result.metrics:
        prev_raw_data = prev_regime_result.metrics

    metrics = []
    for m in spec:
        key = m["key"]
        value = raw_data.get(key)

        raw = _parse_raw_value(value) if value is not None else None

        # For metrics where both directions = stress (e.g. rate moves), use absolute value for positioning
        if raw is not None and m.get("abs_value"):
            raw = abs(raw)

        if value is not None and m.get("fmt"):
            try:
                formatted = m["fmt"](float(value)) if isinstance(value, (int, float)) else str(value)
            except (ValueError, TypeError):
                formatted = str(value)
        elif value is not None:
            formatted = str(value)
        else:
            formatted = "—"

        # Compute delta from previous reading
        prev_value = prev_raw_data.get(key)
        prev_raw = _parse_raw_value(prev_value) if prev_value is not None else None
        if prev_raw is not None and m.get("abs_value"):
            prev_raw = abs(prev_raw)

        delta = None
        if raw is not None and prev_raw is not None:
            delta = round(raw - prev_raw, 4)

        entry = {
            "name": m["name"],
            "desc": m.get("desc", ""),
            "value": formatted,
            "weight": m["weight"],
            "raw_value": raw,
            "prev_raw": prev_raw,
            "delta": delta,
            "range": {
                "healthy": m["healthy"],
                "healthy_label": m["healthy_label"],
                "stressed": m["stressed"],
                "stressed_label": m["stressed_label"],
                "extreme": m.get("extreme"),
                "extreme_date": m.get("extreme_date"),
                "extreme_event": m.get("extreme_event"),
                "inverted": m.get("inverted", False),
            },
        }
        metrics.append(entry)
    return metrics


def _derive_label_from_score(domain, score_val):
    """Derive the threshold-based classification label from a score."""
    thresholds = REGIME_SCORE_THRESHOLDS.get(domain, [])
    if score_val is None or not thresholds:
        return None
    label = thresholds[0][1]
    for t_score, t_below, t_above in thresholds:
        if score_val < t_score:
            label = t_below
            break
        label = t_above
    return label


def _build_all_regimes_summary(state):
    """Build a summary array of all regime scores for the context strip."""
    summary = []
    for domain in REGIME_NAMES:
        regime = getattr(state, domain, None)
        if regime is not None:
            score_val = regime.score if hasattr(regime, "score") else None
            # Derive label from score + thresholds for consistency
            derived = _derive_label_from_score(domain, score_val)
            classification = derived or regime.label or regime.name
            severity = REGIME_LABEL_SEVERITY.get(classification, "moderate")
            summary.append({
                "name": domain,
                "domain": REGIME_DISPLAY_NAMES.get(domain, domain.title()),
                "score": score_val,
                "classification": classification,
                "severity": severity,
            })
        else:
            summary.append({
                "name": domain,
                "domain": REGIME_DISPLAY_NAMES.get(domain, domain.title()),
                "score": None,
                "classification": "N/A",
                "severity": "moderate",
            })
    return summary


def get_regime_summary(settings, refresh=False):
    """Lightweight summary of all regimes for the main dashboard.

    Returns dict with: regimes (list), overall_risk_score, overall_category,
    macro_quadrant, as_of.
    """
    try:
        state, mc_params, fiscal_scorecard = _build_regime_cache(settings, refresh=refresh)
    except Exception as e:
        logger.error(f"Failed to build regime cache for summary: {e}")
        return {"error": str(e), "regimes": []}

    return {
        "regimes": _build_all_regimes_summary(state),
        "overall_risk_score": state.overall_risk_score,
        "overall_category": state.overall_category,
        "macro_quadrant": state.macro_quadrant,
        "as_of": state.asof,
    }


def get_regime_detail(settings, regime_name, refresh=False):
    """
    Get detailed regime data for a single domain, normalized for the dashboard API.

    Returns a dict with: regime_name, domain, as_of, classification, description,
    composite_score, metrics, pillars, mc_impact, flags.
    """
    if regime_name not in REGIME_NAMES:
        return {"error": f"Unknown regime: {regime_name}"}

    try:
        state, mc_params, fiscal_scorecard = _build_regime_cache(settings, refresh=refresh)
    except Exception as e:
        logger.error(f"Failed to build regime cache: {e}")
        return {"error": str(e)}

    regime_result = getattr(state, regime_name, None)
    if regime_result is None:
        return {
            "regime_name": regime_name,
            "domain": REGIME_DISPLAY_NAMES.get(regime_name, regime_name.title()),
            "as_of": state.asof,
            "classification": "Unavailable",
            "description": "Regime data could not be computed.",
            "composite_score": None,
            "metrics": [],
            "pillars": [],
            "mc_impact": {},
            "flags": [],
            "all_regimes": _build_all_regimes_summary(state),
            "overall_risk_score": state.overall_risk_score,
            "overall_category": state.overall_category,
            "macro_quadrant": state.macro_quadrant,
        }

    # Get previous state for delta computation
    prev_regime_result = None
    with _REGIME_DETAIL_LOCK:
        prev_state = _REGIME_DETAIL_CACHE.get("prev_state")
        if prev_state is not None:
            prev_regime_result = getattr(prev_state, regime_name, None)

    metrics = _enrich_metrics(regime_result, regime_name, prev_regime_result=prev_regime_result)

    pillars = []
    if regime_name == "fiscal" and fiscal_scorecard:
        try:
            for s in fiscal_scorecard.sub_scores:
                pillars.append({
                    "name": s.name,
                    "score": round(s.score, 1),
                    "weight": s.weight,
                    "weighted_contribution": round(s.score * s.weight, 1),
                    "percentile": round(s.percentile, 1) if s.percentile is not None else None,
                })
        except Exception as e:
            logger.warning(f"Failed to extract fiscal pillars: {e}")

    prev_score = None
    try:
        from lox.data.regime_history import load_history
        history = load_history()
        prev_data = history.get("domains", {}).get(regime_name, {})
        prev_score = prev_data.get("score")
    except Exception:
        pass

    # Derive the classification from the score + dashboard thresholds so the
    # label always matches where the marker sits on the spectrum bar.
    score_val = regime_result.score if hasattr(regime_result, "score") else None
    derived = _derive_label_from_score(regime_name, score_val)
    classification = derived or regime_result.label or regime_result.name

    severity = REGIME_LABEL_SEVERITY.get(classification, "moderate")

    thresholds = []
    for t_score, t_below, t_above in REGIME_SCORE_THRESHOLDS.get(regime_name, []):
        thresholds.append({"score": t_score, "label_below": t_below, "label_above": t_above})

    _result = {
        "regime_name": regime_name,
        "domain": REGIME_DISPLAY_NAMES.get(regime_name, regime_name.title()),
        "blurb": REGIME_BLURBS.get(regime_name, ""),
        "as_of": state.asof,
        "classification": classification,
        "severity": severity,
        "thresholds": thresholds,
        "description": regime_result.description or "",
        "composite_score": regime_result.score if hasattr(regime_result, "score") else None,
        "prev_score": prev_score,
        "metrics": metrics,
        "pillars": pillars,
        "flags": list(regime_result.tags) if hasattr(regime_result, "tags") else [],
        "all_regimes": _build_all_regimes_summary(state),
        "overall_risk_score": state.overall_risk_score,
        "overall_category": state.overall_category,
        "macro_quadrant": state.macro_quadrant,
    }

    return _result


def get_regime_domains_data(settings):
    """Fetch regime status for each domain using available modules."""
    domains = {}
    
    if not settings:
        return {"domains": domains, "error": "Settings not available"}
    
    # Funding regime
    domains["funding"] = _get_funding_regime(settings)
    
    # USD regime (DXY)
    domains["usd"] = _get_usd_regime(settings)
    
    # Commodities regime (Gold price)
    domains["commod"] = _get_commodities_regime(settings)
    
    # Volatility regime
    domains["volatility"] = _get_volatility_regime(settings)
    
    # Housing regime (30Y mortgage rate)
    domains["housing"] = _get_housing_regime(settings)
    
    # Crypto regime (BTC price)
    domains["crypto"] = _get_crypto_regime(settings)
    
    return {
        "domains": domains,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _get_funding_regime(settings):
    """Get funding/liquidity regime status."""
    try:
        from lox.funding.signals import build_funding_state
        from lox.funding.regime import classify_funding_regime
        
        state = build_funding_state(settings=settings, start_date="2020-01-01", refresh=False)
        regime = classify_funding_regime(state.inputs)
        label = regime.label or regime.name
        
        color = "green" if any(x in label.lower() for x in ["normal", "easy", "benign"]) else \
                "red" if any(x in label.lower() for x in ["stress", "tight", "crisis"]) else "yellow"
        
        return {"label": label.upper(), "color": color}
    except Exception as e:
        print(f"[Regimes] Funding error: {e}")
        return {"label": "N/A", "color": "gray"}


def _get_usd_regime(settings):
    """
    Get USD regime based on DXY (Dollar Index).
    Post-2020 context: DXY ranged 89 (Jan 2021) to 114 (Sep 2022)
    """
    try:
        if not getattr(settings, 'FMP_API_KEY', None):
            return {"label": "N/A", "color": "gray"}
        
        url = "https://financialmodelingprep.com/api/v3/quote/DX-Y.NYB"
        resp = requests.get(url, params={"apikey": settings.FMP_API_KEY}, timeout=10)
        data = resp.json()
        
        if not (isinstance(data, list) and data):
            return {"label": "N/A", "color": "gray"}
        
        dxy = data[0].get("price", 0)
        
        # Post-2020 DXY thresholds
        if dxy >= 108:
            color = "red"  # Extreme = headwind for EM/commodities
        elif dxy >= 103:
            color = "yellow"
        elif dxy >= 95:
            color = "yellow"
        else:
            color = "yellow"  # Weak USD is neither universally good nor bad
        
        return {"label": f"DXY {dxy:.1f}", "color": color}
    except Exception as e:
        print(f"[Regimes] USD error: {e}")
        return {"label": "N/A", "color": "gray"}


def _get_commodities_regime(settings):
    """
    Get commodities regime based on gold price.
    Post-2020: Gold ranged $1,680 (Mar 2020) to ATH $2,700+ (Oct 2024)
    """
    try:
        from lox.commodities.signals import build_commodities_state
        
        state = build_commodities_state(settings=settings, start_date="2020-01-01", refresh=False)
        gold_price = state.inputs.gold
        
        if gold_price is not None:
            # Post-2020 gold thresholds
            if gold_price >= 2400:
                color = "red"  # Historic highs, inflation/fear bid
            elif gold_price >= 2000:
                color = "yellow"  # Elevated but not extreme
            else:
                color = "green"  # Relatively low, deflationary
            
            return {"label": f"GOLD ${gold_price:,.0f}", "color": color}
        
        # Fallback to regime classifier
        from lox.commodities.regime import classify_commodities_regime
        regime = classify_commodities_regime(state.inputs)
        label = regime.label or regime.name
        color = "red" if any(x in label.lower() for x in ["spike", "inflation"]) else \
                "green" if "disinflation" in label.lower() else "yellow"
        
        return {"label": label.upper(), "color": color}
    except Exception as e:
        print(f"[Regimes] Commodities error: {e}")
        return {"label": "N/A", "color": "gray"}


def _get_volatility_regime(settings):
    """Get volatility regime status."""
    try:
        from lox.volatility.signals import build_volatility_state
        from lox.volatility.regime import classify_volatility_regime
        
        state = build_volatility_state(settings=settings, start_date="2020-01-01", refresh=False)
        regime = classify_volatility_regime(state.inputs)
        label = regime.label or regime.name
        
        color = "green" if any(x in label.lower() for x in ["low", "calm", "complacent"]) else \
                "red" if any(x in label.lower() for x in ["high", "stress", "spike", "crisis"]) else "yellow"
        
        return {"label": label.upper(), "color": color}
    except Exception as e:
        print(f"[Regimes] Volatility error: {e}")
        return {"label": "N/A", "color": "gray"}


def _get_housing_regime(settings):
    """
    Get housing regime based on 30Y mortgage rate.
    Post-2020 thresholds: <5% green, 5-6.5% yellow, >6.5% red
    """
    try:
        from lox.housing.signals import build_housing_state
        
        state = build_housing_state(settings=settings, start_date="2020-01-01", refresh=False)
        mortgage_rate = state.inputs.mortgage_30y
        
        if mortgage_rate is not None:
            if mortgage_rate < 5.0:
                color, status = "green", "LOW"
            elif mortgage_rate < 6.5:
                color, status = "yellow", "MODERATE"
            else:
                color, status = "red", "ELEVATED"
            
            return {"label": f"{status} ({mortgage_rate:.2f}%)", "color": color}
        
        # Fallback to mortgage spread
        spread = state.inputs.mortgage_spread
        if spread is not None:
            if spread > 2.5:
                return {"label": f"STRESSED ({spread:.1f}% sprd)", "color": "red"}
            elif spread < 1.8:
                return {"label": f"HEALTHY ({spread:.1f}% sprd)", "color": "green"}
            else:
                return {"label": f"NORMAL ({spread:.1f}% sprd)", "color": "yellow"}
        
        return {"label": "N/A", "color": "gray"}
    except Exception as e:
        print(f"[Regimes] Housing error: {e}")
        return {"label": "N/A", "color": "gray"}


def _get_crypto_regime(settings):
    """
    Get crypto regime based on BTC price.
    Post-2024 thresholds (ATH ~$108K): >$100K green, $70-100K yellow, <$70K red
    """
    try:
        if not getattr(settings, 'FMP_API_KEY', None):
            return {"label": "N/A", "color": "gray"}
        
        url = "https://financialmodelingprep.com/api/v3/quote/BTCUSD"
        resp = requests.get(url, params={"apikey": settings.FMP_API_KEY}, timeout=10)
        data = resp.json()
        
        if not (isinstance(data, list) and data):
            return {"label": "N/A", "color": "gray"}
        
        btc_price = data[0].get("price", 0)
        change_pct = data[0].get("changesPercentage", 0)
        
        # Daily momentum override for big moves
        if change_pct > 5:
            color = "green"
        elif change_pct < -5:
            color = "red"
        elif btc_price >= 100000:
            color = "green"  # ATH zone
        elif btc_price >= 70000:
            color = "yellow"  # Consolidation
        elif btc_price >= 50000:
            color = "red"  # Pullback
        else:
            color = "red"  # Bear market
        
        return {"label": f"BTC ${btc_price/1000:.0f}K", "color": color}
    except Exception as e:
        print(f"[Regimes] Crypto error: {e}")
        return {"label": "N/A", "color": "gray"}


def get_regime_label(vix_val, hy_val):
    """Determine overall market regime from VIX and HY spreads."""
    if vix_val is None:
        return "UNKNOWN"
    if vix_val > 25 or (hy_val and hy_val > 400):
        return "RISK-OFF"
    elif vix_val > 18 or (hy_val and hy_val > 350):
        return "CAUTIOUS"
    else:
        return "RISK-ON"

"""
Cross-regime scenario triggers — actionable macro trade signals.

Transforms generic regime outputs into named macro trade setups by
evaluating conditions across 2+ pillars simultaneously.

Each scenario produces:
- Conviction level (HIGH/MEDIUM/LOW)
- One-line thesis
- Concrete trade expressions with sizing hints
- Key trigger metrics
- Primary risk / invalidation

Usage:
    from lox.regimes.scenarios import evaluate_scenarios
    active = evaluate_scenarios(unified_state)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from lox.regimes.features import UnifiedRegimeState

from lox.regimes.base import RegimeResult

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ScenarioTrade:
    """A single trade expression within a scenario."""

    direction: str       # "LONG", "SHORT", "REDUCE", "HEDGE"
    ticker: str          # e.g. "TLT", "GLD", "HYG"
    instrument: str      # "equity", "calls", "puts", "spreads"
    rationale: str       # one-line why
    sizing_hint: str     # "core" | "tactical" | "hedge" | "starter"


@dataclass(frozen=True)
class PillarCondition:
    """
    A single condition on one regime pillar.

    Evaluated as: score_check(pillar.score) OR name_check(pillar.name/label).
    Both are optional; at least one must be provided.
    If both are provided, either passing counts as a match (OR logic)
    so the condition is robust to classifier refactors.
    """

    domain: str
    score_min: float | None = None
    score_max: float | None = None
    name_contains: tuple[str, ...] | None = None
    label_contains: tuple[str, ...] | None = None
    required: bool = True

    def evaluate(self, regime: RegimeResult | None) -> bool:
        if regime is None:
            return False

        has_score = self.score_min is not None or self.score_max is not None
        has_name = bool(self.name_contains or self.label_contains)

        score_ok = False
        if has_score:
            above = self.score_min is None or regime.score >= self.score_min
            below = self.score_max is None or regime.score <= self.score_max
            score_ok = above and below

        name_ok = False
        if self.name_contains:
            low = regime.name.lower()
            name_ok = any(n.lower() in low for n in self.name_contains)
        if self.label_contains and not name_ok:
            low = regime.label.lower()
            name_ok = any(lb.lower() in low for lb in self.label_contains)

        if has_score and has_name:
            return score_ok or name_ok
        if has_score:
            return score_ok
        if has_name:
            return name_ok
        return False


@dataclass(frozen=True)
class ScenarioResult:
    """Result of evaluating one scenario against current regime state."""

    scenario_id: str
    name: str
    conviction: str                         # "HIGH", "MEDIUM", "LOW"
    thesis: str
    trades: tuple[ScenarioTrade, ...]
    trigger_metrics: dict[str, str]
    primary_risk: str
    conditions_met: int
    conditions_total: int
    conditions_required: int
    required_met: int
    domains_involved: tuple[str, ...]


@dataclass
class ScenarioDefinition:
    """
    A named macro trade setup defined by conditions across 2+ pillars.

    Conviction logic:
    - HIGH: all required + at least 1 optional met
    - MEDIUM: all required met (no optional or none met)
    - Not activated if any required condition fails
    """

    scenario_id: str
    name: str
    thesis_template: str
    conditions: list[PillarCondition]
    trades: list[ScenarioTrade]
    primary_risk: str


# ═════════════════════════════════════════════════════════════════════════════
# Scenario Definitions
# ═════════════════════════════════════════════════════════════════════════════

SCENARIOS: list[ScenarioDefinition] = [

    # ── 1. STAGFLATION SQUEEZE ──────────────────────────────────────────
    ScenarioDefinition(
        scenario_id="stagflation_squeeze",
        name="STAGFLATION SQUEEZE",
        thesis_template=(
            "Growth decelerating ({growth_label}, score {growth_score:.0f}) "
            "while inflation remains sticky ({inflation_label}, score {inflation_score:.0f}) "
            "— classic stagflation setup pressuring both bonds and equities."
        ),
        conditions=[
            PillarCondition(
                domain="growth", score_min=60,
                label_contains=("Slowing", "Contraction"), required=True,
            ),
            PillarCondition(
                domain="inflation", score_min=50,
                label_contains=("Elevated", "Above Target", "Hot"), required=True,
            ),
            PillarCondition(
                domain="rates",
                name_contains=("bear_steepener", "rates_shock_up"),
                label_contains=("Bear steepener", "Rates shock higher"), required=True,
            ),
            PillarCondition(
                domain="consumer", score_min=55,
                label_contains=("Weakening", "Stress"), required=False,
            ),
        ],
        trades=[
            ScenarioTrade("SHORT", "TLT", "puts", "Duration risk in stagflation; real yields rising", "core"),
            ScenarioTrade("LONG", "GLD", "calls", "Gold benefits from negative real rate expectations + uncertainty", "core"),
            ScenarioTrade("LONG", "XLE", "equity", "Energy as inflation hedge with commodity pass-through", "tactical"),
            ScenarioTrade("SHORT", "XLY", "puts", "Consumer discretionary crushed by sticky prices + slowing growth", "tactical"),
            ScenarioTrade("LONG", "TIP", "equity", "TIPS outperform nominals when breakevens widen", "hedge"),
        ],
        primary_risk="Inflation breaks lower on demand destruction, turning setup into pure risk-off.",
    ),

    # ── 2. LIQUIDITY CRUNCH ─────────────────────────────────────────────
    ScenarioDefinition(
        scenario_id="liquidity_crunch",
        name="LIQUIDITY CRUNCH",
        thesis_template=(
            "Funding markets tightening ({liquidity_label}, score {liquidity_score:.0f}) "
            "with credit spreads widening ({credit_label}, score {credit_score:.0f}) "
            "— systemic liquidity withdrawal accelerating."
        ),
        conditions=[
            PillarCondition(
                domain="liquidity", score_min=55,
                name_contains=("tightening", "structural_tightening", "funding_stress"),
                required=True,
            ),
            PillarCondition(
                domain="credit", score_min=55,
                label_contains=("Widening", "Stress", "Crisis"), required=True,
            ),
            PillarCondition(
                domain="volatility", score_min=55,
                name_contains=("elevated", "shock"), required=True,
            ),
            PillarCondition(
                domain="fiscal", score_min=45,
                name_contains=("stress_building", "fiscal_stress", "fiscal_dominance"),
                required=False,
            ),
        ],
        trades=[
            ScenarioTrade("REDUCE", "SPY", "equity", "De-risk equity; liquidity withdrawal hits risk assets first", "core"),
            ScenarioTrade("LONG", "UVXY", "equity", "Vol benefits from liquidity-driven selloffs", "tactical"),
            ScenarioTrade("SHORT", "HYG", "puts", "Credit spreads widen sharply in funding stress", "core"),
            ScenarioTrade("LONG", "TLT", "calls", "Flight to quality as funding stress escalates", "tactical"),
            ScenarioTrade("LONG", "BIL", "equity", "Short-duration safety during funding dislocations", "hedge"),
        ],
        primary_risk="Fed intervenes with emergency facilities (repo, swap lines), compressing spreads rapidly.",
    ),

    # ── 3. GOLDILOCKS UNWIND ────────────────────────────────────────────
    ScenarioDefinition(
        scenario_id="goldilocks_unwind",
        name="GOLDILOCKS UNWIND",
        thesis_template=(
            "Benign macro ({growth_label} + {inflation_label}) with compressed credit "
            "({credit_label}, score {credit_score:.0f}) and low vol ({volatility_label}) "
            "— complacency at extremes, mean reversion risk elevated."
        ),
        conditions=[
            PillarCondition(
                domain="growth", score_max=55,
                label_contains=("Stable", "Accelerating", "Boom"), required=True,
            ),
            PillarCondition(
                domain="inflation", score_max=50,
                label_contains=("At Target", "Below Target"), required=True,
            ),
            PillarCondition(
                domain="credit", score_max=40,
                label_contains=("Calm", "Euphoria"), required=True,
            ),
            PillarCondition(
                domain="volatility", score_max=45,
                name_contains=("normal",), required=True,
            ),
        ],
        trades=[
            ScenarioTrade("LONG", "VXX", "calls", "Vol at floor; asymmetric upside in tail hedges", "hedge"),
            ScenarioTrade("LONG", "UVXY", "calls", "Leveraged vol for convexity on any unwind", "starter"),
            ScenarioTrade("SHORT", "HYG", "puts", "Cheap put protection when credit spreads at tights", "hedge"),
            ScenarioTrade("LONG", "GLD", "calls", "Cheap optionality on regime shift", "starter"),
        ],
        primary_risk="Goldilocks persists longer than expected; carry bleeds hedges.",
    ),

    # ── 4. DURATION REGIME SHIFT ────────────────────────────────────────
    ScenarioDefinition(
        scenario_id="duration_regime_shift",
        name="DURATION REGIME SHIFT",
        thesis_template=(
            "Rates dislocating ({rates_label}) with fiscal pressure building "
            "({fiscal_label}, score {fiscal_score:.0f}) and USD weakening "
            "({usd_label}) — term premium repricing underway."
        ),
        conditions=[
            PillarCondition(
                domain="rates", score_min=60,
                name_contains=("rates_shock_up", "bear_steepener"), required=True,
            ),
            PillarCondition(
                domain="fiscal", score_min=45,
                name_contains=("stress_building", "fiscal_stress", "fiscal_dominance"),
                required=True,
            ),
            PillarCondition(
                domain="usd", score_max=40,
                name_contains=("usd_weak", "usd_plunge"), required=True,
            ),
            PillarCondition(
                domain="commodities",
                name_contains=("commodity_reflation", "energy_shock"), required=False,
            ),
        ],
        trades=[
            ScenarioTrade("SHORT", "TLT", "puts", "Long end under supply + term premium pressure", "core"),
            ScenarioTrade("LONG", "TIP", "equity", "TIPS benefit from rising breakevens + real yield repricing", "core"),
            ScenarioTrade("LONG", "GLD", "calls", "Gold rallies on fiscal credibility concerns + weak USD", "core"),
            ScenarioTrade("LONG", "DBC", "equity", "Broad commodities benefit from weak USD + fiscal expansion", "tactical"),
            ScenarioTrade("SHORT", "UUP", "equity", "USD weakening as foreign buyers retreat from Treasuries", "tactical"),
        ],
        primary_risk="Fed steps in with yield curve control or Treasury announces buyback program.",
    ),

    # ── 5. RISK-OFF CASCADE ─────────────────────────────────────────────
    ScenarioDefinition(
        scenario_id="risk_off_cascade",
        name="RISK-OFF CASCADE",
        thesis_template=(
            "Multi-pillar stress: growth deteriorating ({growth_label}, score {growth_score:.0f}), "
            "credit widening ({credit_label}), vol spiking ({volatility_label}), "
            "consumer weakening ({consumer_label}) — cascading risk-off event."
        ),
        conditions=[
            PillarCondition(
                domain="growth", score_min=60,
                label_contains=("Slowing", "Contraction"), required=True,
            ),
            PillarCondition(
                domain="credit", score_min=55,
                label_contains=("Widening", "Stress", "Crisis"), required=True,
            ),
            PillarCondition(
                domain="volatility", score_min=60,
                name_contains=("elevated", "shock"), required=True,
            ),
            PillarCondition(
                domain="consumer", score_min=55,
                label_contains=("Weakening", "Stress"), required=True,
            ),
            PillarCondition(
                domain="liquidity", score_min=55,
                name_contains=("tightening", "funding_stress"), required=False,
            ),
        ],
        trades=[
            ScenarioTrade("LONG", "TLT", "calls", "Flight to safety; rate cuts being priced aggressively", "core"),
            ScenarioTrade("LONG", "GLD", "calls", "Safe haven demand spikes in multi-asset stress", "core"),
            ScenarioTrade("SHORT", "HYG", "puts", "Credit spreads blow out in risk-off cascade", "core"),
            ScenarioTrade("SHORT", "SPY", "puts", "Broad equity downside protection", "tactical"),
            ScenarioTrade("LONG", "XLP", "equity", "Defensive rotation into staples", "hedge"),
        ],
        primary_risk="Coordinated central bank intervention (rate cuts, QE restart) reverses cascade quickly.",
    ),

    # ── 6. REFLATION TRADE ──────────────────────────────────────────────
    ScenarioDefinition(
        scenario_id="reflation_trade",
        name="REFLATION TRADE",
        thesis_template=(
            "Growth accelerating ({growth_label}, score {growth_score:.0f}) with rising "
            "inflation ({inflation_label}) and commodity strength ({commodities_label}) "
            "— reflationary impulse favoring real assets over bonds."
        ),
        conditions=[
            PillarCondition(
                domain="growth", score_max=45,
                label_contains=("Accelerating", "Boom"), required=True,
            ),
            PillarCondition(
                domain="inflation", score_min=50,
                label_contains=("Elevated", "Above Target", "Hot"), required=True,
            ),
            PillarCondition(
                domain="commodities",
                name_contains=("commodity_reflation", "energy_shock"), required=True,
            ),
            PillarCondition(
                domain="usd",
                name_contains=("usd_weak", "usd_plunge"), required=False,
            ),
        ],
        trades=[
            ScenarioTrade("LONG", "XLE", "calls", "Energy producers benefit from rising commodity prices", "core"),
            ScenarioTrade("LONG", "DBC", "equity", "Broad commodity exposure in reflationary environment", "core"),
            ScenarioTrade("LONG", "EEM", "calls", "EM equities leveraged to reflation + weak USD", "tactical"),
            ScenarioTrade("SHORT", "TLT", "puts", "Duration risk as rates reprice higher for growth + inflation", "core"),
            ScenarioTrade("LONG", "XLI", "equity", "Cyclicals/industrials benefit from capex cycle", "tactical"),
        ],
        primary_risk="Growth stalls while inflation stays elevated, converting reflation into stagflation.",
    ),

    # ── 7. FED PIVOT SETUP ──────────────────────────────────────────────
    ScenarioDefinition(
        scenario_id="fed_pivot_setup",
        name="FED PIVOT SETUP",
        thesis_template=(
            "Growth slowing ({growth_label}, score {growth_score:.0f}) with inflation "
            "receding ({inflation_label}, score {inflation_score:.0f}) and credit widening "
            "({credit_label}) — conditions ripe for dovish Fed pivot."
        ),
        conditions=[
            PillarCondition(
                domain="growth", score_min=55,
                label_contains=("Slowing", "Contraction"), required=True,
            ),
            PillarCondition(
                domain="inflation", score_max=50,
                label_contains=("At Target", "Below Target", "Deflationary"),
                required=True,
            ),
            PillarCondition(
                domain="credit", score_min=50,
                label_contains=("Widening", "Stress"), required=True,
            ),
            PillarCondition(
                domain="rates",
                name_contains=("inverted_curve", "bull_steepener"), required=True,
            ),
            PillarCondition(
                domain="usd",
                name_contains=("usd_weak", "usd_plunge"), required=False,
            ),
        ],
        trades=[
            ScenarioTrade("LONG", "TLT", "calls", "Duration rally as market prices aggressive rate cuts", "core"),
            ScenarioTrade("LONG", "QQQ", "calls", "Growth/tech benefits most from falling real yields", "core"),
            ScenarioTrade("SHORT", "UUP", "equity", "Dollar weakens on rate differential compression", "tactical"),
            ScenarioTrade("LONG", "XLF", "calls", "Banks benefit from curve re-steepening post-pivot", "tactical"),
            ScenarioTrade("LONG", "GLD", "equity", "Gold benefits from falling real yields + easing cycle", "hedge"),
        ],
        primary_risk="Inflation re-accelerates, forcing the Fed to stay restrictive longer than priced.",
    ),

    # ── 8. CREDIT CRUNCH ────────────────────────────────────────────────
    ScenarioDefinition(
        scenario_id="credit_crunch",
        name="CREDIT CRUNCH",
        thesis_template=(
            "Credit markets seizing ({credit_label}, score {credit_score:.0f}) with "
            "funding stress ({liquidity_label}) and consumer deterioration "
            "({consumer_label}) — credit transmission mechanism breaking down."
        ),
        conditions=[
            PillarCondition(
                domain="credit", score_min=65,
                label_contains=("Stress", "Crisis"), required=True,
            ),
            PillarCondition(
                domain="liquidity", score_min=55,
                name_contains=("tightening", "structural_tightening", "funding_stress"),
                required=True,
            ),
            PillarCondition(
                domain="consumer", score_min=55,
                label_contains=("Weakening", "Stress"), required=True,
            ),
            PillarCondition(
                domain="growth", score_min=55,
                label_contains=("Slowing", "Contraction"), required=False,
            ),
        ],
        trades=[
            ScenarioTrade("SHORT", "HYG", "puts", "HY spreads blow out in credit crunch", "core"),
            ScenarioTrade("LONG", "LQD", "equity", "Flight to quality within credit (IG over HY)", "tactical"),
            ScenarioTrade("LONG", "TLT", "calls", "Treasuries rally as credit stress forces rate cuts", "core"),
            ScenarioTrade("SHORT", "KRE", "puts", "Regional banks most exposed to credit deterioration", "core"),
            ScenarioTrade("LONG", "GLD", "equity", "Safe haven allocation during credit events", "hedge"),
        ],
        primary_risk="Fed backstops credit markets directly (corporate bond buying, TALF restart).",
    ),
]


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation Engine
# ═════════════════════════════════════════════════════════════════════════════

# Priority metrics per domain — first non-None match wins
_PRIORITY_KEYS: dict[str, list[str]] = {
    "growth": ["ISM Mfg", "Payrolls 3m", "Claims 4wk", "UNRATE"],
    "inflation": ["CPI YoY", "Core PCE", "10Y BE", "Oil YoY"],
    "volatility": ["VIX", "VIX z", "Vol Pressure"],
    "credit": ["HY OAS", "HY 30d Chg", "BBB OAS"],
    "rates": ["10Y", "2s10s", "10Y 20d Chg"],
    "liquidity": ["SOFR", "Corridor", "Reserves"],
    "consumer": ["Michigan", "Retail MoM", "30Y Mortgage"],
    "fiscal": ["Deficit 12m", "z Deficit", "Auction Tail", "FPI"],
    "usd": ["DXY", "20d Chg", "Strength"],
    "commodities": ["WTI", "Gold", "Broad 60d"],
}


def _build_thesis(defn: ScenarioDefinition, state: UnifiedRegimeState) -> str:
    """Substitute live regime values into the thesis template."""
    tvars: dict[str, Any] = {}
    for domain in ("growth", "inflation", "volatility", "credit", "rates",
                    "liquidity", "consumer", "fiscal", "usd", "commodities"):
        regime = getattr(state, domain, None)
        if regime:
            tvars[f"{domain}_label"] = regime.label
            tvars[f"{domain}_score"] = regime.score
            tvars[f"{domain}_name"] = regime.name
        else:
            tvars[f"{domain}_label"] = "N/A"
            tvars[f"{domain}_score"] = 50.0
            tvars[f"{domain}_name"] = "unknown"
    try:
        return defn.thesis_template.format(**tvars)
    except KeyError:
        return defn.thesis_template


def _extract_trigger_metrics(
    state: UnifiedRegimeState,
    met_domains: set[str],
) -> dict[str, str]:
    """Pull the most relevant metric from each triggered pillar."""
    out: dict[str, str] = {}
    for domain in sorted(met_domains):
        regime = getattr(state, domain, None)
        if not regime or not regime.metrics:
            continue
        for key in _PRIORITY_KEYS.get(domain, []):
            val = regime.metrics.get(key)
            if val is not None:
                out[domain] = f"{key}: {val}"
                break
        else:
            for k, v in regime.metrics.items():
                if v is not None:
                    out[domain] = f"{k}: {v}"
                    break
    return out


def evaluate_scenario(
    defn: ScenarioDefinition,
    state: UnifiedRegimeState,
) -> ScenarioResult | None:
    """
    Evaluate one scenario. Returns ScenarioResult if active, None otherwise.

    Conviction:
    - HIGH  = all required met + at least 1 optional met
    - MEDIUM = all required met
    """
    required = [c for c in defn.conditions if c.required]
    optional = [c for c in defn.conditions if not c.required]

    req_met = 0
    opt_met = 0
    met_domains: set[str] = set()

    for c in required:
        if c.evaluate(getattr(state, c.domain, None)):
            req_met += 1
            met_domains.add(c.domain)

    for c in optional:
        if c.evaluate(getattr(state, c.domain, None)):
            opt_met += 1
            met_domains.add(c.domain)

    if req_met < len(required):
        return None

    conviction = "HIGH" if opt_met > 0 else "MEDIUM"

    return ScenarioResult(
        scenario_id=defn.scenario_id,
        name=defn.name,
        conviction=conviction,
        thesis=_build_thesis(defn, state),
        trades=tuple(defn.trades),
        trigger_metrics=_extract_trigger_metrics(state, met_domains),
        primary_risk=defn.primary_risk,
        conditions_met=req_met + opt_met,
        conditions_total=len(required) + len(optional),
        conditions_required=len(required),
        required_met=req_met,
        domains_involved=tuple(sorted(met_domains)),
    )


def evaluate_scenarios(
    state: UnifiedRegimeState,
    scenarios: list[ScenarioDefinition] | None = None,
) -> list[ScenarioResult]:
    """
    Evaluate all scenarios against current regime state.

    Returns active ScenarioResults sorted by conviction then conditions met.
    """
    if scenarios is None:
        scenarios = SCENARIOS

    results: list[ScenarioResult] = []
    for defn in scenarios:
        try:
            result = evaluate_scenario(defn, state)
            if result is not None:
                results.append(result)
        except Exception as e:
            logger.warning("Scenario '%s' evaluation failed: %s", defn.scenario_id, e)

    conv_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    results.sort(key=lambda r: (conv_order.get(r.conviction, 3), -r.conditions_met))
    return results


# ═════════════════════════════════════════════════════════════════════════════
# LLM Formatter
# ═════════════════════════════════════════════════════════════════════════════

def format_scenarios_for_llm(scenarios: list[ScenarioResult]) -> str:
    """Format active scenarios as markdown for LLM system prompt injection."""
    if not scenarios:
        return ""

    lines = [
        "## Active Macro Scenarios",
        "",
        "The following cross-regime scenario triggers are currently active.",
        "Reference these scenarios when discussing trades and positioning.",
        "",
    ]

    for s in scenarios:
        lines.append(f"### {s.name} ({s.conviction} conviction)")
        lines.append(f"**Thesis:** {s.thesis}")
        lines.append(f"**Conditions met:** {s.conditions_met}/{s.conditions_total}")
        if s.trigger_metrics:
            triggers = ", ".join(f"{d}: {m}" for d, m in s.trigger_metrics.items())
            lines.append(f"**Triggers:** {triggers}")
        lines.append("**Trades:**")
        for t in s.trades:
            lines.append(f"  - {t.direction} {t.ticker} ({t.instrument}) — {t.rationale} [{t.sizing_hint}]")
        lines.append(f"**Primary risk:** {s.primary_risk}")
        lines.append("")

    return "\n".join(lines)

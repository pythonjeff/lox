"""
Cross-pillar inconsistency & dislocation detection.

Surfaces contradictions between regime pillars that create exploitable
market dislocations — the kind of signal conflicts a PM would flag in
a morning risk meeting.

12 rules in 4 categories:
  1. Cross-Asset Divergence (credit-vol, growth-credit, rates-growth, usd-commodities)
  2. Hidden Stress (funding-credit, consumer-growth, fiscal-rates)
  3. Momentum Conflicts (trend reversal, stale regime)
  4. Complacency Signals (broad complacency, vol-credit euphoria, inflation-rate disconnect)

Usage:
    from lox.regimes.inconsistencies import detect_inconsistencies
    dislocations = detect_inconsistencies(unified_state)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lox.regimes.features import UnifiedRegimeState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketDislocation:
    """A detected cross-pillar inconsistency or market dislocation."""

    dislocation_id: str
    name: str
    severity: str               # "HIGH" | "MEDIUM"
    thesis: str                 # one-line explanation
    trade_implication: str      # how to capitalize
    domains_involved: tuple[str, ...]
    metrics_snapshot: dict[str, str]


# ═════════════════════════════════════════════════════════════════════════════
# Helper
# ═════════════════════════════════════════════════════════════════════════════

def _metric_snap(state: "UnifiedRegimeState", domains: list[str]) -> dict[str, str]:
    """Pull score + label for each domain into a compact dict."""
    out: dict[str, str] = {}
    for d in domains:
        regime = getattr(state, d, None)
        if regime:
            out[d] = f"{regime.label} ({regime.score:.0f})"
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Detection Rules
# ═════════════════════════════════════════════════════════════════════════════

def _check_credit_vol_divergence(state: "UnifiedRegimeState") -> MarketDislocation | None:
    """Credit calm but vol elevated, or vice versa. Credit typically leads vol."""
    cr = state.credit
    vo = state.volatility
    if not cr or not vo:
        return None

    if cr.score < 35 and vo.score > 55:
        return MarketDislocation(
            dislocation_id="credit_vol_divergence",
            name="CREDIT-VOL DIVERGENCE",
            severity="HIGH",
            thesis=f"Vol elevated ({vo.label}, {vo.score:.0f}) but credit calm ({cr.label}, {cr.score:.0f}) — credit typically leads, either vol is overreacting or credit is lagging.",
            trade_implication="If credit is right: sell vol (short VIX futures, sell straddles). If vol is right: buy HY protection before spreads widen.",
            domains_involved=("credit", "volatility"),
            metrics_snapshot=_metric_snap(state, ["credit", "volatility"]),
        )

    if vo.score < 35 and cr.score > 55:
        return MarketDislocation(
            dislocation_id="credit_vol_divergence",
            name="CREDIT-VOL DIVERGENCE",
            severity="HIGH",
            thesis=f"Credit widening ({cr.label}, {cr.score:.0f}) but vol suppressed ({vo.label}, {vo.score:.0f}) — vol hasn't repriced credit deterioration.",
            trade_implication="Buy vol (VIX calls, SPX puts) while cheap — credit stress typically transmits to equity vol with a lag.",
            domains_involved=("credit", "volatility"),
            metrics_snapshot=_metric_snap(state, ["credit", "volatility"]),
        )

    return None


def _check_growth_credit_disconnect(state: "UnifiedRegimeState") -> MarketDislocation | None:
    """Growth deteriorating but credit still calm."""
    gr = state.growth
    cr = state.credit
    if not gr or not cr:
        return None

    if gr.score > 60 and cr.score < 40:
        return MarketDislocation(
            dislocation_id="growth_credit_disconnect",
            name="GROWTH-CREDIT DISCONNECT",
            severity="HIGH",
            thesis=f"Growth deteriorating ({gr.label}, {gr.score:.0f}) but credit calm ({cr.label}, {cr.score:.0f}) — credit hasn't priced in the slowdown.",
            trade_implication="Buy HY put protection (HYG puts) while spreads are tight — credit repricing is asymmetric when growth confirms weakness.",
            domains_involved=("growth", "credit"),
            metrics_snapshot=_metric_snap(state, ["growth", "credit"]),
        )

    return None


def _check_rates_growth_mismatch(state: "UnifiedRegimeState") -> MarketDislocation | None:
    """Rates pricing recession while growth is strong, or vice versa."""
    ra = state.rates
    gr = state.growth
    if not ra or not gr:
        return None

    if ra.score < 35 and gr.score < 40:
        return MarketDislocation(
            dislocation_id="rates_growth_mismatch",
            name="RATES-GROWTH MISMATCH",
            severity="MEDIUM",
            thesis=f"Rates calm/rallying ({ra.label}, {ra.score:.0f}) with growth strong ({gr.label}, {gr.score:.0f}) — bond market may be pricing a slowdown that hasn't arrived.",
            trade_implication="Pay rates (short TLT) if growth data confirms strength — bonds are too expensive for the macro backdrop.",
            domains_involved=("rates", "growth"),
            metrics_snapshot=_metric_snap(state, ["rates", "growth"]),
        )

    return None


def _check_usd_commodities_divergence(state: "UnifiedRegimeState") -> MarketDislocation | None:
    """USD strengthening with commodities rallying — normally inversely correlated."""
    us = state.usd
    co = state.commodities
    if not us or not co:
        return None

    if us.score > 60 and co.score > 55:
        return MarketDislocation(
            dislocation_id="usd_commodities_divergence",
            name="USD-COMMODITIES DIVERGENCE",
            severity="MEDIUM",
            thesis=f"USD strong ({us.label}, {us.score:.0f}) but commodities rallying ({co.label}, {co.score:.0f}) — unusual divergence, one will revert.",
            trade_implication="If supply-driven commodity rally: hedge with long USD. If USD is overshooting: short UUP and ride commodity momentum.",
            domains_involved=("usd", "commodities"),
            metrics_snapshot=_metric_snap(state, ["usd", "commodities"]),
        )

    return None


def _check_funding_credit_lag(state: "UnifiedRegimeState") -> MarketDislocation | None:
    """Funding stress without credit market response."""
    lq = state.liquidity
    cr = state.credit
    if not lq or not cr:
        return None

    if lq.score > 55 and cr.score < 40:
        return MarketDislocation(
            dislocation_id="funding_credit_lag",
            name="FUNDING STRESS WITHOUT CREDIT RESPONSE",
            severity="HIGH",
            thesis=f"Funding markets stressed ({lq.label}, {lq.score:.0f}) but credit calm ({cr.label}, {cr.score:.0f}) — plumbing stress hasn't transmitted yet.",
            trade_implication="Front-run the transmission: buy HY protection (HYG puts), reduce levered positions. Funding stress → credit widening lag is typically 2-4 weeks.",
            domains_involved=("liquidity", "credit"),
            metrics_snapshot=_metric_snap(state, ["liquidity", "credit"]),
        )

    return None


def _check_consumer_growth_lag(state: "UnifiedRegimeState") -> MarketDislocation | None:
    """Consumer deterioration not yet reflected in headline growth."""
    co = state.consumer
    gr = state.growth
    if not co or not gr:
        return None

    if co.score > 60 and gr.score < 45:
        return MarketDislocation(
            dislocation_id="consumer_growth_lag",
            name="CONSUMER DETERIORATION IGNORED",
            severity="MEDIUM",
            thesis=f"Consumer weakening ({co.label}, {co.score:.0f}) but growth still benign ({gr.label}, {gr.score:.0f}) — consumer is 70% of GDP, this gap will close.",
            trade_implication="Position for growth downgrade: long TLT calls, short XLY (consumer discretionary), reduce cyclical exposure.",
            domains_involved=("consumer", "growth"),
            metrics_snapshot=_metric_snap(state, ["consumer", "growth"]),
        )

    return None


def _check_fiscal_rates_lag(state: "UnifiedRegimeState") -> MarketDislocation | None:
    """Fiscal pressure building but rates haven't repriced term premium."""
    fi = state.fiscal
    ra = state.rates
    if not fi or not ra:
        return None

    if fi.score > 55 and ra.score < 45:
        return MarketDislocation(
            dislocation_id="fiscal_rates_lag",
            name="FISCAL PRESSURE BUILDING QUIETLY",
            severity="MEDIUM",
            thesis=f"Fiscal stress building ({fi.label}, {fi.score:.0f}) but rates calm ({ra.label}, {ra.score:.0f}) — bond market hasn't repriced term premium.",
            trade_implication="Short long-end duration (TLT puts, steepener trades). Fiscal deterioration → term premium repricing can be sudden and violent.",
            domains_involved=("fiscal", "rates"),
            metrics_snapshot=_metric_snap(state, ["fiscal", "rates"]),
        )

    return None


def _check_trend_reversal_warnings(state: "UnifiedRegimeState") -> list[MarketDislocation]:
    """Any pillar with fast-worsening momentum but still benign score."""
    results: list[MarketDislocation] = []
    trends = state.trends or {}

    for domain in ("growth", "inflation", "volatility", "credit", "rates", "liquidity"):
        regime = getattr(state, domain, None)
        trend = trends.get(domain)
        if not regime or not trend:
            continue
        if trend.momentum_z is not None and trend.momentum_z > 1.5 and regime.score < 35:
            results.append(MarketDislocation(
                dislocation_id=f"trend_reversal_{domain}",
                name=f"TREND REVERSAL WARNING: {domain.upper()}",
                severity="HIGH",
                thesis=f"{domain.title()} score benign ({regime.score:.0f}) but deteriorating fast (momentum z={trend.momentum_z:+.1f}) — regime shift likely imminent.",
                trade_implication=f"Reduce {domain} complacency trades. Score lags reality — momentum confirms directional shift before score crosses threshold.",
                domains_involved=(domain,),
                metrics_snapshot={domain: f"{regime.label} ({regime.score:.0f}), momo z={trend.momentum_z:+.1f}"},
            ))

    return results


def _check_stale_regimes(state: "UnifiedRegimeState") -> list[MarketDislocation]:
    """Extended neutral regimes suggest building pressure for a breakout."""
    results: list[MarketDislocation] = []
    trends = state.trends or {}

    for domain in ("growth", "inflation", "volatility", "credit", "rates", "liquidity"):
        regime = getattr(state, domain, None)
        trend = trends.get(domain)
        if not regime or not trend:
            continue
        if trend.days_in_regime is not None and trend.days_in_regime > 60 and 40 <= regime.score <= 60:
            results.append(MarketDislocation(
                dislocation_id=f"stale_regime_{domain}",
                name=f"STALE REGIME: {domain.upper()}",
                severity="MEDIUM",
                thesis=f"{domain.title()} stuck in neutral ({regime.label}, {regime.score:.0f}) for {trend.days_in_regime}d — extended range-bound regimes resolve with directional breaks.",
                trade_implication=f"Buy {domain} straddles/strangles. Implied vol for {domain}-sensitive assets likely underpriced after extended calm.",
                domains_involved=(domain,),
                metrics_snapshot={domain: f"{regime.label} ({regime.score:.0f}), {trend.days_in_regime}d in regime"},
            ))

    return results


def _check_broad_complacency(state: "UnifiedRegimeState") -> MarketDislocation | None:
    """4+ core pillars benign simultaneously = max mean-reversion risk."""
    core = ["growth", "inflation", "volatility", "credit", "rates", "liquidity"]
    benign_count = 0
    benign_domains: list[str] = []

    for d in core:
        regime = getattr(state, d, None)
        if regime and regime.score < 35:
            benign_count += 1
            benign_domains.append(d)

    if benign_count >= 4:
        return MarketDislocation(
            dislocation_id="broad_complacency",
            name="BROAD COMPLACENCY",
            severity="HIGH",
            thesis=f"{benign_count}/6 core pillars benign ({', '.join(benign_domains)}) — everything looks calm, max mean-reversion risk. Markets are most fragile when consensus says 'nothing can go wrong.'",
            trade_implication="Buy tail hedges (OTM SPX puts, VIX calls) while cheap. Skew is flat when complacency is high — asymmetric upside in protection.",
            domains_involved=tuple(benign_domains),
            metrics_snapshot=_metric_snap(state, benign_domains),
        )

    return None


def _check_vol_credit_euphoria(state: "UnifiedRegimeState") -> MarketDislocation | None:
    """Both vol and credit at extreme calm = tail risk severely underpriced."""
    vo = state.volatility
    cr = state.credit
    if not vo or not cr:
        return None

    if vo.score < 30 and cr.score < 30:
        return MarketDislocation(
            dislocation_id="vol_credit_euphoria",
            name="VOL-CREDIT EUPHORIA",
            severity="HIGH",
            thesis=f"Both vol ({vo.label}, {vo.score:.0f}) and credit ({cr.label}, {cr.score:.0f}) at extreme calm — tail risk severely underpriced.",
            trade_implication="Cheapest time to buy protection: VIX call spreads, HYG put spreads, SPX put butterflies. Convexity is maximum when both are suppressed.",
            domains_involved=("volatility", "credit"),
            metrics_snapshot=_metric_snap(state, ["volatility", "credit"]),
        )

    return None


def _check_inflation_rate_disconnect(state: "UnifiedRegimeState") -> MarketDislocation | None:
    """Inflation elevated but rates haven't repriced."""
    inf = state.inflation
    ra = state.rates
    if not inf or not ra:
        return None

    if inf.score > 55 and ra.score < 40:
        return MarketDislocation(
            dislocation_id="inflation_rate_disconnect",
            name="INFLATION-RATE DISCONNECT",
            severity="MEDIUM",
            thesis=f"Inflation elevated ({inf.label}, {inf.score:.0f}) but rates calm ({ra.label}, {ra.score:.0f}) — bond market hasn't priced persistent inflation.",
            trade_implication="Short duration (TLT puts). Buy TIPS over nominals. Breakevens should widen as rates catch up to inflation reality.",
            domains_involved=("inflation", "rates"),
            metrics_snapshot=_metric_snap(state, ["inflation", "rates"]),
        )

    return None


# ═════════════════════════════════════════════════════════════════════════════
# Main Engine
# ═════════════════════════════════════════════════════════════════════════════

_SINGLE_CHECKS = [
    _check_credit_vol_divergence,
    _check_growth_credit_disconnect,
    _check_rates_growth_mismatch,
    _check_usd_commodities_divergence,
    _check_funding_credit_lag,
    _check_consumer_growth_lag,
    _check_fiscal_rates_lag,
    _check_broad_complacency,
    _check_vol_credit_euphoria,
    _check_inflation_rate_disconnect,
]

_LIST_CHECKS = [
    _check_trend_reversal_warnings,
    _check_stale_regimes,
]


def detect_inconsistencies(state: "UnifiedRegimeState") -> list[MarketDislocation]:
    """
    Detect cross-pillar inconsistencies and market dislocations.

    Returns list of MarketDislocation sorted by severity (HIGH first).
    """
    results: list[MarketDislocation] = []

    for check in _SINGLE_CHECKS:
        try:
            d = check(state)
            if d is not None:
                results.append(d)
        except Exception as e:
            logger.warning("Dislocation check '%s' failed: %s", check.__name__, e)

    for check in _LIST_CHECKS:
        try:
            results.extend(check(state))
        except Exception as e:
            logger.warning("Dislocation check '%s' failed: %s", check.__name__, e)

    severity_order = {"HIGH": 0, "MEDIUM": 1}
    results.sort(key=lambda d: severity_order.get(d.severity, 2))
    return results


# ═════════════════════════════════════════════════════════════════════════════
# LLM Formatter
# ═════════════════════════════════════════════════════════════════════════════

def format_dislocations_for_llm(dislocations: list[MarketDislocation]) -> str:
    """Format detected dislocations as markdown for LLM system prompt injection."""
    if not dislocations:
        return ""

    lines = [
        "## Market Dislocations & Inconsistencies",
        "",
        "The following cross-pillar conflicts have been detected.",
        "These represent potential mispricings or lagging adjustments between markets.",
        "Explain which are most actionable and how a PM should position.",
        "",
    ]

    for d in dislocations:
        lines.append(f"### {d.name} ({d.severity})")
        lines.append(f"**Signal:** {d.thesis}")
        lines.append(f"**Trade:** {d.trade_implication}")
        if d.metrics_snapshot:
            metrics = ", ".join(f"{k}: {v}" for k, v in d.metrics_snapshot.items())
            lines.append(f"**Metrics:** {metrics}")
        lines.append("")

    return "\n".join(lines)

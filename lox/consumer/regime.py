"""
Consumer regime classifier — NEW domain (absorbs Housing).

Monitors consumer health: sentiment, spending, credit stress, housing drag.
Score 0 = consumer boom → 100 = consumer stress.
"""
from __future__ import annotations

from lox.regimes.base import RegimeResult


def classify_consumer(
    michigan_sentiment: float | None,
    michigan_expectations: float | None,
    retail_sales_control_mom: float | None,
    personal_spending_mom: float | None,
    credit_card_debt_yoy_chg: float | None,
    mortgage_30y: float | None,
) -> RegimeResult:
    """Classify the Consumer regime.

    Args:
        michigan_sentiment: University of Michigan Consumer Sentiment (long-run avg ~85).
        michigan_expectations: Michigan Expectations sub-index (leads sentiment).
        retail_sales_control_mom: Retail Sales Control Group MoM %.
        personal_spending_mom: Personal Spending MoM %.
        credit_card_debt_yoy_chg: YoY % change in credit card balances.
        mortgage_30y: 30-year fixed mortgage rate (%).
    """
    score = 50

    # ── Michigan Sentiment ────────────────────────────────────────────────
    if michigan_sentiment is not None:
        if michigan_sentiment > 95:
            score -= 15
        elif michigan_sentiment > 80:
            score -= 5
        elif michigan_sentiment < 60:
            score += 20
        elif michigan_sentiment < 70:
            score += 10

    # ── Expectations sub-index (LEADS sentiment) ──────────────────────────
    if michigan_expectations is not None:
        if michigan_expectations < 60:
            score += 10
        elif michigan_expectations > 85:
            score -= 5

    # ── Retail Sales Control Group MoM ────────────────────────────────────
    if retail_sales_control_mom is not None:
        if retail_sales_control_mom < -0.3:
            score += 10
        elif retail_sales_control_mom > 0.5:
            score -= 5

    # ── Credit Card Debt Acceleration ─────────────────────────────────────
    if credit_card_debt_yoy_chg is not None:
        if credit_card_debt_yoy_chg > 12:
            score += 10  # credit-fueled spending = fragile
        elif credit_card_debt_yoy_chg < 0:
            score += 5   # deleveraging = also weak demand

    # ── Mortgage Rate (housing drag on wealth effect) ─────────────────────
    if mortgage_30y is not None:
        if mortgage_30y > 7.5:
            score += 10
        elif mortgage_30y < 5.0:
            score -= 10

    # ── Divergence Signal ─────────────────────────────────────────────────
    sentiment_weak = michigan_sentiment is not None and michigan_sentiment < 70
    spending_ok = retail_sales_control_mom is not None and retail_sales_control_mom > 0
    if sentiment_weak and spending_ok:
        tags = ["consumer", "divergence", "fragile"]
        desc_suffix = " | ⚠️ Sentiment/spending divergence"
    else:
        tags = ["consumer", "demand"]
        desc_suffix = ""

    score = max(0, min(100, score))

    if score >= 70:
        label = "Consumer Stress"
    elif score >= 55:
        label = "Consumer Weakening"
    elif score >= 45:
        label = "Consumer Stable"
    elif score >= 30:
        label = "Consumer Expanding"
    else:
        label = "Consumer Boom"

    parts: list[str] = []
    if michigan_sentiment is not None:
        parts.append(f"Michigan: {michigan_sentiment:.0f}")
    if michigan_expectations is not None:
        parts.append(f"Expect: {michigan_expectations:.0f}")
    if retail_sales_control_mom is not None:
        parts.append(f"Retail Control MoM: {retail_sales_control_mom:+.1f}%")
    if mortgage_30y is not None:
        parts.append(f"30Y Mort: {mortgage_30y:.2f}%")

    return RegimeResult(
        name="consumer",
        label=label,
        description=(" | ".join(parts) if parts else "Insufficient data") + desc_suffix,
        score=score,
        domain="consumer",
        tags=tags,
        metrics={
            "michigan_sentiment": michigan_sentiment,
            "michigan_expectations": michigan_expectations,
            "retail_sales_control_mom": retail_sales_control_mom,
            "mortgage_30y": mortgage_30y,
        },
    )

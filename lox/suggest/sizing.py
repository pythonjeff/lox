"""
Position sizing, signal tension detection, and portfolio impact for `lox suggest`.

Pure function module — no API calls, no side effects. All inputs passed in.

Sizing method: vol-target — size each position so its standalone vol contribution
equals a per-trade risk budget (default 1% of NAV annualized).
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Conviction -> sizing scale factor
CONVICTION_SCALE = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}

# Default annualized vol if no data (reasonable ETF assumption)
DEFAULT_VOL = 0.25


@dataclass
class PortfolioImpact:
    """Aggregate impact of all sized ideas on the portfolio."""

    total_notional: float = 0.0
    total_pct_nav: float = 0.0
    current_net_delta: float = 0.0
    projected_net_delta: float = 0.0
    blended_exp_pnl: float = 0.0
    naive_diversified_var: float = 0.0
    n_ideas: int = 0


def compute_realized_vols(
    price_panel,
    tickers: list[str],
    window: int = 60,
) -> dict[str, float]:
    """Compute annualized realized vol from daily returns.

    Uses the shared price panel already fetched for playbook scoring —
    no additional API calls.

    Parameters
    ----------
    price_panel : DataFrame
        Daily close prices (date index × ticker columns).
    tickers : list[str]
        Tickers to compute vol for.
    window : int
        Lookback window in trading days.

    Returns
    -------
    dict[str, float]
        ticker -> annualized realized vol (e.g. 0.20 = 20%).
    """
    if price_panel is None or price_panel.empty:
        return {}

    rets = price_panel.pct_change().dropna()
    out: dict[str, float] = {}
    for ticker in tickers:
        if ticker not in rets.columns:
            continue
        series = rets[ticker].dropna()
        tail = series.tail(window)
        if len(tail) < 30:
            out[ticker] = DEFAULT_VOL
            continue
        daily_std = float(tail.std())
        if daily_std <= 0 or not np.isfinite(daily_std):
            out[ticker] = DEFAULT_VOL
            continue
        out[ticker] = daily_std * math.sqrt(252)
    return out


def size_positions(
    scored: list,
    realized_vols: dict[str, float],
    account_equity: float,
    vol_budget: float = 0.01,
    max_pct: float = 0.05,
) -> None:
    """Vol-target position sizing — mutates ScoredCandidate objects in-place.

    Parameters
    ----------
    scored : list[ScoredCandidate]
        Candidates with composite_score, conviction, exp_return already set.
    realized_vols : dict[str, float]
        ticker -> annualized realized vol.
    account_equity : float
        Account NAV in dollars. If <= 0, sizing is skipped.
    vol_budget : float
        Target annualized vol contribution per trade (default 1% = 0.01).
    max_pct : float
        Maximum single position as fraction of NAV (default 5%).
    """
    if account_equity <= 0:
        return

    for cand in scored:
        vol = realized_vols.get(cand.ticker, DEFAULT_VOL)
        if vol <= 0:
            vol = DEFAULT_VOL

        cand.realized_vol = vol

        # Base notional from vol targeting
        base_notional = (vol_budget * account_equity) / vol

        # Scale by conviction
        scale = CONVICTION_SCALE.get(cand.conviction, 0.5)
        notional = base_notional * scale

        # Cap at max single position
        max_notional = max_pct * account_equity
        notional = min(notional, max_notional)

        cand.notional = round(notional, 0)
        cand.pct_nav = notional / account_equity
        cand.exp_pnl = round(notional * cand.exp_return, 2)

        # Sizing label
        if scale >= 0.8:
            cand.sizing_label = "core"
        elif scale >= 0.5:
            cand.sizing_label = "tactical"
        else:
            cand.sizing_label = "starter"


def detect_signal_flags(
    candidate,
    pb_direction: str | None = None,
    raw_corr: float | None = None,
) -> list[str]:
    """Detect signal tensions and conflicts for a scored candidate.

    Returns human-readable warning strings for display.
    """
    flags: list[str] = []

    # 1. Low hit rate carrying a high composite
    if candidate.hit_rate > 0 and candidate.hit_rate < 0.55 and candidate.composite_score >= 65:
        flags.append("Low hit rate — scenario driving score")

    # 2. Playbook / scenario direction conflict
    if pb_direction is not None:
        pb_long = pb_direction.lower() in ("bullish", "long")
        idea_long = candidate.direction == "LONG"
        if pb_long != idea_long:
            flags.append("Playbook/scenario direction conflict")

    # 3. Outsized tail risk vs expected return
    if candidate.exp_return != 0 and candidate.var_5 != 0:
        if abs(candidate.var_5) > 3 * abs(candidate.exp_return):
            flags.append("Outsized tail risk vs expected return")

    # 4. High benchmark correlation for longs
    if raw_corr is not None and candidate.direction == "LONG" and raw_corr > 0.75:
        flags.append("High benchmark correlation")

    return flags


def apply_smart_conviction(candidate, signal_flags: list[str]) -> str:
    """Override conviction based on multi-signal agreement.

    Rules applied in order:
    1. hit_rate < 0.50 → cap at MEDIUM
    2. hit_rate >= 0.65 AND sharpe >= 1.0 AND scenario_score >= 60 → HIGH
    3. Fallback: existing composite-threshold logic
    """
    # Cap conviction when playbook is a coin flip
    if 0 < candidate.hit_rate < 0.50:
        if candidate.conviction == "HIGH":
            return "MEDIUM"
        return candidate.conviction

    # Upgrade when all signals align
    if (
        candidate.hit_rate >= 0.65
        and candidate.sharpe_est >= 1.0
        and candidate.scenario_score >= 60
        and len(signal_flags) == 0
    ):
        return "HIGH"

    # Reconcile conviction with final composite score
    # (portfolio adjustments can move scores after initial conviction assignment)
    if candidate.composite_score >= 70 and candidate.conviction != "HIGH":
        return "HIGH"
    if candidate.composite_score >= 45 and candidate.conviction == "LOW":
        return "MEDIUM"

    return candidate.conviction


def compute_portfolio_impact(
    scored: list,
    current_net_delta: float = 0.0,
    account_equity: float = 0.0,
) -> PortfolioImpact:
    """Compute aggregate portfolio impact from all sized ideas.

    ETF positions are approximated as delta=1 per dollar of notional.
    """
    if not scored or account_equity <= 0:
        return PortfolioImpact()

    total_notional = 0.0
    delta_change = 0.0
    blended_pnl = 0.0
    var_sq_sum = 0.0

    for cand in scored:
        n = cand.notional
        if n <= 0:
            continue
        total_notional += n
        sign = 1.0 if cand.direction == "LONG" else -1.0
        delta_change += n * sign
        blended_pnl += cand.exp_pnl
        # Per-idea VaR in $ terms
        idea_var = n * abs(cand.var_5)
        var_sq_sum += idea_var ** 2

    return PortfolioImpact(
        total_notional=round(total_notional, 0),
        total_pct_nav=total_notional / account_equity if account_equity > 0 else 0.0,
        current_net_delta=current_net_delta,
        projected_net_delta=current_net_delta + delta_change,
        blended_exp_pnl=round(blended_pnl, 2),
        naive_diversified_var=round(math.sqrt(var_sq_sum), 2) if var_sq_sum > 0 else 0.0,
        n_ideas=len([c for c in scored if c.notional > 0]),
    )

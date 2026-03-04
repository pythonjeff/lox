"""
Composite scoring engine for `lox suggest`.

Combines multiple signal sources into a single 0-100 conviction score
per candidate ticker. Weights adapt based on which signals are available.

Default mode (3 signals):
    playbook (0.40) + scenario (0.35) + correlation (0.25)

Deep mode (4 signals, --deep):
    playbook (0.30) + MC (0.25) + scenario (0.25) + correlation (0.20)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Weight presets
WEIGHTS_DEFAULT = {"playbook": 0.40, "scenario": 0.35, "correlation": 0.25}
WEIGHTS_DEEP = {"playbook": 0.30, "mc": 0.25, "scenario": 0.25, "correlation": 0.20}


@dataclass
class ScoredCandidate:
    """A candidate with composite conviction score and breakdown."""

    ticker: str
    direction: str              # "LONG" or "SHORT"
    composite_score: float      # 0-100
    conviction: str             # "HIGH", "MEDIUM", "LOW"
    thesis: str

    # Score breakdown (each 0-100)
    playbook_score: float = 0.0
    scenario_score: float = 0.0
    correlation_score: float = 0.0
    mc_score: float = 0.0

    # Playbook stats
    exp_return: float = 0.0
    hit_rate: float = 0.0
    var_5: float = 0.0
    n_analogs: int = 0
    sharpe_est: float = 0.0

    # MC stats (--deep only)
    mc_median_return: float | None = None
    mc_var_5: float | None = None

    # Source metadata
    source: str = ""            # "scenario", "quadrant", "default"
    scenario_conviction: str = ""


def compute_composite_scores(
    *,
    candidates: list[dict],
    playbook_ideas: dict | None = None,
    mc_scores: dict | None = None,
    correlation_scores: dict | None = None,
    active_scenarios: list | None = None,
    macro_quadrant: str = "",
    deep: bool = False,
) -> list[ScoredCandidate]:
    """
    Score and rank candidates using all available signals.

    Parameters
    ----------
    candidates : list[dict]
        Raw candidates from cross_asset suggest (ticker, direction, thesis, source, conviction).
    playbook_ideas : dict | None
        ticker -> PlaybookIdea from rank_macro_playbook.
    mc_scores : dict | None
        ticker -> MCScore from score_candidates_mc (only when --deep).
    correlation_scores : dict | None
        ticker -> float correlation vs benchmark.
    active_scenarios : list | None
        List of ScenarioResult from evaluate_scenarios.
    macro_quadrant : str
        Current macro quadrant string.
    deep : bool
        Whether MC scores are available.

    Returns
    -------
    list[ScoredCandidate]
        Sorted by composite_score descending.
    """
    playbook_ideas = playbook_ideas or {}
    mc_scores = mc_scores or {}
    correlation_scores = correlation_scores or {}
    active_scenarios = active_scenarios or []

    weights = WEIGHTS_DEEP if (deep and mc_scores) else WEIGHTS_DEFAULT

    # Build set of tickers in active scenarios for quick lookup
    scenario_tickers: dict[str, str] = {}  # ticker -> conviction
    for s in active_scenarios:
        for t in s.trades:
            scenario_tickers[t.ticker.upper()] = s.conviction

    scored: list[ScoredCandidate] = []

    for cand in candidates:
        ticker = cand["ticker"].upper()
        direction = cand.get("direction", "LONG")

        # --- Playbook component (0-100) ---
        pb = playbook_ideas.get(ticker)
        pb_score = 50.0  # neutral default
        if pb is not None:
            # Scale expected return to 0-100: +5% = 100, -5% = 0, 0% = 50
            raw = pb.exp_return * 100  # convert to percentage
            if direction == "SHORT":
                raw = -raw  # for shorts, negative expected return is good
            pb_score = float(np.clip(50 + raw * 10, 0, 100))

        # --- Scenario alignment component (0-100) ---
        sc_score = 0.0
        sc_conviction = ""
        cand_source = cand.get("source", "")
        cand_conv = cand.get("conviction", "")
        if ticker in scenario_tickers:
            sc_conviction = scenario_tickers[ticker]
            base = 80.0
            if sc_conviction == "HIGH":
                sc_score = min(100.0, base * 1.2)
            else:
                sc_score = base
        elif cand_source == "scenario":
            # Candidate was sourced from a scenario but scenarios weren't passed
            sc_score = 75.0 if cand_conv == "HIGH" else 65.0
            sc_conviction = cand_conv
        elif cand_source == "quadrant":
            sc_score = 50.0
        else:
            sc_score = 20.0

        # --- Correlation component (0-100) ---
        corr_val = correlation_scores.get(ticker, 0.5)
        if direction == "LONG":
            # Lower correlation = more diversifying = higher score
            corr_score = float(np.clip((1 - corr_val) * 50 + 25, 0, 100))
        else:
            # Higher correlation = better hedge = higher score
            corr_score = float(np.clip(corr_val * 50 + 25, 0, 100))

        # --- MC component (0-100, only in deep mode) ---
        mc_s = 0.0
        mc_median = None
        mc_v5 = None
        mc_data = mc_scores.get(ticker)
        if mc_data is not None:
            raw_mc = mc_data.median_return * 100
            if direction == "SHORT":
                raw_mc = -raw_mc
            mc_s = float(np.clip(50 + raw_mc * 5, 0, 100))
            mc_median = mc_data.median_return
            mc_v5 = mc_data.p5_return

        # --- Weighted composite ---
        composite = (
            weights.get("playbook", 0) * pb_score
            + weights.get("scenario", 0) * sc_score
            + weights.get("correlation", 0) * corr_score
            + weights.get("mc", 0) * mc_s
        )
        composite = float(np.clip(composite, 0, 100))

        # Conviction label
        if composite >= 70:
            conviction = "HIGH"
        elif composite >= 45:
            conviction = "MEDIUM"
        else:
            conviction = "LOW"

        scored.append(ScoredCandidate(
            ticker=ticker,
            direction=direction,
            composite_score=round(composite, 1),
            conviction=conviction,
            thesis=cand.get("thesis", ""),
            playbook_score=round(pb_score, 1),
            scenario_score=round(sc_score, 1),
            correlation_score=round(corr_score, 1),
            mc_score=round(mc_s, 1) if deep else 0.0,
            exp_return=pb.exp_return if pb else 0.0,
            hit_rate=pb.hit_rate if pb else 0.0,
            var_5=pb.var_5 if pb else 0.0,
            n_analogs=pb.n_analogs if pb else 0,
            sharpe_est=pb.sharpe_est if pb else 0.0,
            mc_median_return=mc_median,
            mc_var_5=mc_v5,
            source=cand.get("source", ""),
            scenario_conviction=sc_conviction,
        ))

    scored.sort(key=lambda x: x.composite_score, reverse=True)
    return scored

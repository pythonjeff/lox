"""Tests for the composite scoring engine and playbook integration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lox.suggest.scoring import compute_composite_scores, ScoredCandidate


def _make_candidates(*tickers):
    return [
        {"ticker": t, "direction": "LONG", "thesis": f"Test {t}", "source": "scenario", "conviction": "HIGH"}
        for t in tickers
    ]


class TestCompositeScoring:
    def test_returns_scored_candidates(self):
        cands = _make_candidates("XLE", "GLD", "TLT")
        result = compute_composite_scores(candidates=cands)
        assert len(result) == 3
        assert all(isinstance(r, ScoredCandidate) for r in result)

    def test_sorted_by_composite_descending(self):
        cands = _make_candidates("XLE", "GLD", "TLT")
        result = compute_composite_scores(candidates=cands)
        scores = [r.composite_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_scenario_alignment_boosts_score(self):
        """Candidates from scenarios should score higher than defaults."""
        scenario_cand = [{"ticker": "XLE", "direction": "LONG", "thesis": "t", "source": "scenario", "conviction": "HIGH"}]
        default_cand = [{"ticker": "XLF", "direction": "LONG", "thesis": "t", "source": "default", "conviction": "LOW"}]
        scored_s = compute_composite_scores(candidates=scenario_cand)
        scored_d = compute_composite_scores(candidates=default_cand)
        assert scored_s[0].scenario_score > scored_d[0].scenario_score

    def test_playbook_positive_return_boosts_score(self):
        from lox.ideas.macro_playbook import PlaybookIdea
        cands = _make_candidates("XLE")
        pb = {
            "XLE": PlaybookIdea(
                ticker="XLE", direction="bullish", exp_return=0.03,
                hit_rate=0.65, sharpe_est=1.2, var_5=-0.02,
                n_analogs=100, fwd_returns=np.array([0.03]),
            )
        }
        scored = compute_composite_scores(candidates=cands, playbook_ideas=pb)
        assert scored[0].playbook_score > 50  # positive return should be above neutral

    def test_playbook_negative_return_lowers_score_for_long(self):
        from lox.ideas.macro_playbook import PlaybookIdea
        cands = [{"ticker": "XLE", "direction": "LONG", "thesis": "t", "source": "scenario", "conviction": "MEDIUM"}]
        pb = {
            "XLE": PlaybookIdea(
                ticker="XLE", direction="bearish", exp_return=-0.04,
                hit_rate=0.35, sharpe_est=-0.8, var_5=-0.08,
                n_analogs=100, fwd_returns=np.array([-0.04]),
            )
        }
        scored = compute_composite_scores(candidates=cands, playbook_ideas=pb)
        assert scored[0].playbook_score < 50

    def test_deep_mode_uses_mc_weights(self):
        from lox.suggest.mc_scoring import MCScore
        cands = _make_candidates("XLE")
        mc = {
            "XLE": MCScore(
                ticker="XLE", current_price=80.0, median_return=0.05,
                p5_return=-0.10, p25_return=-0.02, p75_return=0.08,
                upside_ratio=4.0, realized_vol=25.0,
            )
        }
        scored_deep = compute_composite_scores(candidates=cands, mc_scores=mc, deep=True)
        scored_normal = compute_composite_scores(candidates=cands, deep=False)
        # Deep mode should produce a different score (MC weight > 0)
        assert scored_deep[0].mc_score > 0
        assert scored_normal[0].mc_score == 0

    def test_conviction_labels(self):
        cands = _make_candidates("A", "B", "C")
        result = compute_composite_scores(candidates=cands)
        for r in result:
            assert r.conviction in ("HIGH", "MEDIUM", "LOW")

    def test_correlation_helps_longs_with_low_corr(self):
        cands = [
            {"ticker": "GLD", "direction": "LONG", "thesis": "t", "source": "scenario", "conviction": "HIGH"},
            {"ticker": "QQQ", "direction": "LONG", "thesis": "t", "source": "scenario", "conviction": "HIGH"},
        ]
        corrs = {"GLD": -0.3, "QQQ": 0.9}  # GLD much more diversifying
        scored = compute_composite_scores(candidates=cands, correlation_scores=corrs)
        gld = next(s for s in scored if s.ticker == "GLD")
        qqq = next(s for s in scored if s.ticker == "QQQ")
        assert gld.correlation_score > qqq.correlation_score


class TestPlaybookIntegration:
    def test_playbook_with_trending_data(self):
        from lox.ideas.macro_playbook import rank_macro_playbook
        idx = pd.date_range("2018-01-01", periods=600, freq="B")
        X = pd.DataFrame({"trend": np.linspace(0, 1, 600)}, index=idx)
        px = pd.DataFrame(index=idx)
        px["WINNER"] = 100 * np.exp(np.cumsum(np.random.RandomState(42).normal(0.001, 0.01, 600)))
        px["LOSER"] = 100 * np.exp(np.cumsum(np.random.RandomState(99).normal(-0.001, 0.01, 600)))

        ideas = rank_macro_playbook(features=X, prices=px, tickers=["WINNER", "LOSER"], k=100, min_matches=30)
        assert len(ideas) >= 1
        for idea in ideas:
            assert hasattr(idea, "exp_return")
            assert hasattr(idea, "var_5")
            assert idea.n_analogs >= 30

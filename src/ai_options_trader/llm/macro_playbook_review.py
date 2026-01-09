from __future__ import annotations

import json
from typing import Any

from ai_options_trader.config import Settings
from ai_options_trader.llm.macro_recommendation import llm_macro_recommendation


def llm_macro_playbook_review(
    *,
    settings: Settings,
    asof: str,
    regime_features: dict[str, Any],
    playbook_ideas: list[dict[str, Any]],
    positions: list[dict[str, Any]] | None = None,
    account: dict[str, Any] | None = None,
    model: str | None = None,
    temperature: float = 0.2,
) -> str:
    """
    LLM summary + execution-style prompt for the macro playbook.

    This does NOT execute trades; it produces a decision memo suitable for a human to confirm.
    """
    # Back-compat wrapper: treat playbook ideas as generic candidates.
    candidates = []
    for it in playbook_ideas:
        d = dict(it)
        d.setdefault("source", "playbook")
        candidates.append(d)
    return llm_macro_recommendation(
        settings=settings,
        asof=asof,
        regime_features=regime_features,
        candidates=candidates,
        positions=positions,
        account=account,
        risk_watch=None,
        news=None,
        require_decision_line=False,
        model=model,
        temperature=temperature,
    )



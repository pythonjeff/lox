from __future__ import annotations

import json
from typing import Any

from ai_options_trader.config import Settings


def llm_macro_recommendation(
    *,
    settings: Settings,
    asof: str,
    regime_features: dict[str, Any],
    candidates: list[dict[str, Any]],
    budgeted_recommendations: list[dict[str, Any]] | None = None,
    positions: list[dict[str, Any]] | None = None,
    account: dict[str, Any] | None = None,
    risk_watch: dict[str, Any] | None = None,
    news: dict[str, Any] | None = None,
    require_decision_line: bool = False,
    model: str | None = None,
    temperature: float = 0.2,
) -> str:
    """
    LLM decision memo over either:
      - playbook candidates (regime-conditioned forward return stats), and/or
      - ML candidates (prob_up + exp_return predictions)

    This does NOT execute trades; it produces a human-readable recommendation stream.
    """
    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment / .env")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("openai package is not installed. Try: pip install -e .") from e

    client = OpenAI(api_key=settings.openai_api_key)
    chosen_model = model or settings.openai_model

    payload = {
        "asof": asof,
        "regime_features": regime_features,
        "candidates": candidates,
        "budgeted_recommendations": budgeted_recommendations,
        "account": account,
        "positions": positions,
        "risk_watch": risk_watch or {},
        "news": news or {},
    }
    payload_json = json.dumps(payload, indent=2, default=str)

    decision_line = "0) DECISION: GO|HOLD|NEEDS_REVIEW (one line)\n" if require_decision_line else ""
    prompt = f"""
You are the "macro trade captain" for a small account.
You are given JSON containing:
- the latest regime feature row (macro/funding/usd/rates/vol/commod and/or fci)
- a ranked list of trade candidates sourced from either a playbook and/or an ML model
- optional: a list of **budgeted_recommendations** (the only trades that fit budget/constraints and match the CLI table)
- optional: current account + open positions
- optional: risk_watch (concrete tracker values + upcoming economic calendar events)
- optional: news (headlines with URLs + sentiment summaries)

Task:
Write a concise decision memo that helps a human decide what to do today.

If the inputs are ambiguous, it is acceptable to recommend NO NEW TRADE and ask to verify first.

Output format:
{decision_line}1) TL;DR (2-5 bullets)
2) Ticker briefs (for each ticker in budgeted_recommendations; if none, use top 5 candidates):
   - TICKER: <ticker> (and OPTION <symbol> if present)
   - NEWS SENTIMENT (LLM): bullish|bearish|neutral + confidence 0..1
     - Evidence: include 1-3 URLs from `news.items_by_ticker[ticker]` (or `news.items`) if available; otherwise say "no headlines provided"
   - REGIME/TRACKER LINK: 2-4 bullets tying specific trackers/events to this ticker (use risk_watch.trackers + risk_watch.events)
   - WHAT WOULD CHANGE (ticker-specific): 2-4 bullets with concrete triggers (specific calendar events, specific trackers, or specific headlines)
   - READING (ticker-specific): 1-3 headline links (URLs) for this ticker (or "none provided")
3) Position review (if positions provided): 4-10 bullets, but each bullet must name the relevant symbol/ticker
4) Proposed actions (NOT advice):
   - Recommend at most 3 actions, each in one line (or output NONE):
     - OPEN_OPTION <symbol>
     - OPEN_SHARES <ticker>
     - CLOSE <symbol>

Rules:
- Use ONLY the JSON provided.
- Do not hallucinate prices, events, earnings, or macro facts not present.
- If budgeted_recommendations is provided, Proposed actions MUST be chosen from that list (it already satisfies budget/constraints).
- Otherwise, prefer candidates that already include an option_leg (budget/constraints more likely satisfied).
- If candidates disagree (e.g., ML bullish but playbook bearish), say so and pick a conservative action set.
- If require_decision_line is true, the first line MUST be exactly: "DECISION: GO" or "DECISION: HOLD" or "DECISION: NEEDS_REVIEW".
- Every NEWS SENTIMENT must be ticker-specific. Do NOT output a global sentiment line.
- Any mention of an "event" must come from risk_watch.events (or be explicitly marked as "not provided").
- Any mention of "geopolitics" or "major developments" must be directly supported by a provided headline URL; otherwise omit it.

JSON:
{payload_json}
""".strip()

    resp = client.chat.completions.create(
        model=chosen_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=float(temperature),
    )
    return (resp.choices[0].message.content or "").strip()



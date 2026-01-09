from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from ai_options_trader.config import Settings


def llm_event_trade_ideas(
    *,
    settings: Settings,
    event_text: str,
    event_url: str | None,
    user_thesis: str,
    focus: str = "treasuries",
    direction: str = "short",
    universe: list[str],
    max_trades: int = 3,
    max_premium_usd: float = 150.0,
    model: str | None = None,
    temperature: float = 0.2,
) -> str:
    """
    Given an event description + optional source text, propose trades.
    Output is human-readable markdown (NOT execution).
    """
    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment / .env")

    client = OpenAI(api_key=settings.openai_api_key)
    chosen_model = model or settings.openai_model

    payload: dict[str, Any] = {
        "focus": focus,
        "direction": direction,
        "user_thesis": user_thesis,
        "event_url": event_url,
        "event_text": (event_text or "")[:12000],
        "universe": universe,
        "constraints": {
            "max_trades": int(max_trades),
            "prefer_options": True,
            "max_premium_usd_per_contract": float(max_premium_usd),
            "defined_risk_preferred": True,
        },
    }

    prompt = f"""
You are an options-focused macro trader for a small account.

You are given JSON about a potential event and a tradable universe. Your job:
- Generate at most {int(max_trades)} SHORT/hedge trade ideas aligned with the thesis.
- Prefer defined-risk options structures (debit spreads) when practical.
- Keep each option leg premium <= ${float(max_premium_usd):.0f} per contract (if you mention a leg).

Rules:
- Use ONLY the JSON provided (no extra facts).
- If the event text is missing/too thin, say so and give conditional plans ("if X then Y").
- Stick to the given universe symbols for underlyings.
- Output clean markdown with sections:
  1) Event summary (2-4 bullets)
  2) Market path assumptions (3-6 bullets)
  3) Trade ideas (max {int(max_trades)}), each with:
     - Underlying
     - Direction (e.g., bearish duration / bearish equities / long vol)
     - Options structure suggestion (or shares/ETF alternative)
     - Why it matches the thesis
     - Key risk / invalidation
  4) What data to check at the open (3-6 bullets)

JSON:
{json.dumps(payload, indent=2)}
""".strip()

    resp = client.chat.completions.create(
        model=chosen_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=float(temperature),
    )
    return (resp.choices[0].message.content or "").strip()


def _extract_json_object(text: str) -> str:
    t = (text or "").strip()
    if not t:
        raise ValueError("Empty LLM response")
    # Remove fenced wrappers
    t = t.replace("```json", "```")
    t = t.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM did not return a JSON object. Got: {t[:200]!r}")
    return t[start : end + 1]


def llm_event_trade_ideas_json(
    *,
    settings: Settings,
    event_text: str,
    event_url: str | None,
    user_thesis: str,
    focus: str = "treasuries",
    direction: str = "short",
    universe: list[str],
    max_trades: int = 3,
    max_premium_usd: float = 150.0,
    model: str | None = None,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """
    Same as llm_event_trade_ideas, but returns structured JSON for execution workflows.
    """
    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment / .env")

    client = OpenAI(api_key=settings.openai_api_key)
    chosen_model = model or settings.openai_model

    payload: dict[str, Any] = {
        "focus": focus,
        "direction": direction,
        "user_thesis": user_thesis,
        "event_url": event_url,
        "event_text": (event_text or "")[:12000],
        "universe": universe,
        "constraints": {
            "max_trades": int(max_trades),
            "max_premium_usd_per_contract": float(max_premium_usd),
            "prefer_defined_risk": True,
        },
    }

    prompt = f"""
You are an options-focused macro trader for a small account.

Using ONLY the JSON below, return ONLY JSON with schema:
{{
  "event_summary": ["..."],  // 2-5 bullets
  "assumptions": ["..."],    // 3-8 bullets
  "trades": [
    {{
      "priority": 1,
      "underlying": "TLT",
      "action": "BUY_PUT" | "BUY_CALL" | "BUY_SHARES",
      "rationale": "...",
      "risk": "...",
      "target_dte_days": 60,              // integer, optional
      "max_premium_usd": {float(max_premium_usd):.0f}  // per contract cap, optional
    }}
  ]
}}

Rules:
- Trades MUST use an underlying from the provided universe list.
- Propose at most {int(max_trades)} trades.
- Prefer options for hedges (BUY_PUT/BUY_CALL) unless impossible; BUY_SHARES is allowed for inverse/hedge ETFs.
- If direction is "short", prefer bearish-duration expressions (e.g. puts on duration ETFs) for treasuries focus.
- Keep it realistic: if unsure, output fewer trades with clear conditions.

JSON INPUT:
{json.dumps(payload, indent=2)}
""".strip()

    resp = client.chat.completions.create(
        model=chosen_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=float(temperature),
    )
    raw = (resp.choices[0].message.content or "").strip()
    obj = json.loads(_extract_json_object(raw))
    return obj if isinstance(obj, dict) else {"event_summary": [], "assumptions": [], "trades": []}


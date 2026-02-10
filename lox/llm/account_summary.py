from __future__ import annotations

import json
from typing import Any

from lox.config import Settings


def llm_account_summary(
    *,
    settings: Settings,
    asof: str,
    account: dict[str, Any],
    positions: list[dict[str, Any]],
    recent_orders: list[dict[str, Any]] | None = None,
    risk_watch: dict[str, Any] | None = None,
    news: dict[str, Any] | None = None,
    model: str | None = None,
    temperature: float = 0.2,
) -> str:
    """
    LLM: summarize account/positions into "trades, themes, and risks".

    Grounded strictly in the provided JSON (no news, no extra facts).
    """
    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment / .env")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("openai package is not installed. Try: pip install -e .") from e

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.OPENAI_BASE_URL)
    chosen_model = model or settings.openai_model

    payload = {
        "asof": asof,
        "account": account,
        "positions": positions,
        "recent_orders": recent_orders or [],
        "risk_watch": risk_watch or {},
        "news": news or {},
    }
    payload_json = json.dumps(payload, indent=2, default=str)

    prompt = f"""
You are a trading operations + risk assistant for a small account.

You are given JSON containing:
- account balances (cash/equity/buying power)
- open positions (symbols, qty, avg entry, current price, unrealized P/L)
- optional: recent orders (if provided)
- optional: risk_watch (upcoming economic calendar events + tracker values)
- optional: news (headline lists with URLs, plus sentiment summaries)

Task:
Write a concise briefing so the operator knows what's going on and what to watch.

Output format:
1) Trades / exposures right now (4-10 bullets)
2) Themes (2-6 bullets): infer themes ONLY from symbols and exposures in positions/orders
3) Market risk watch (concrete) (5-12 bullets):
   - Prefer the provided calendar events + tracker series; be specific (e.g. "CPI release on YYYY-MM-DD HH:MMZ")
   - Tie each risk to an exposure in the positions if possible (e.g., rates risk vs long-duration proxies)
4) Articles / reading list (3-8 bullets):
   - Use ONLY URLs from the provided news items; include the URL next to the title
5) Checklist for the next session (3-8 bullets): what to review before placing new trades

Rules:
- Use ONLY the JSON provided. Do NOT introduce news, earnings, macro claims, or price levels not present.
- If you can't infer something, say so.
- Call out option positions explicitly (convexity, theta/decay) when symbols look like options.
- Keep it short (<= 250 words).

JSON:
{payload_json}
""".strip()

    resp = client.chat.completions.create(
        model=chosen_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=float(temperature),
    )
    return str(resp.choices[0].message.content or "").strip()


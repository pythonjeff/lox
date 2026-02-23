"""
Interactive regime chat — drop into a conversation with the LLM
pre-loaded with the regime snapshot (and optional ticker context).

Called from print_llm_regime_analysis() when --llm is active.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from rich.console import Console

from lox.config import Settings

logger = logging.getLogger(__name__)


def _fetch_ticker_context(settings: Settings, ticker: str) -> dict[str, Any]:
    """Fetch FMP profile + quote for *ticker*. Returns a dict (possibly partial)."""
    import requests

    t = ticker.upper()
    data: dict[str, Any] = {"ticker": t}

    if not settings.fmp_api_key:
        return data

    try:
        resp = requests.get(
            f"https://financialmodelingprep.com/api/v3/profile/{t}",
            params={"apikey": settings.fmp_api_key},
            timeout=10,
        )
        if resp.ok:
            items = resp.json()
            if items and isinstance(items, list):
                p = items[0]
                data["profile"] = {
                    "name": p.get("companyName"),
                    "sector": p.get("sector"),
                    "industry": p.get("industry"),
                    "market_cap": p.get("mktCap"),
                    "beta": p.get("beta"),
                    "description": (p.get("description") or "")[:500],
                }
    except Exception:
        logger.debug("Failed to fetch profile for %s", t, exc_info=True)

    try:
        resp = requests.get(
            f"https://financialmodelingprep.com/api/v3/quote/{t}",
            params={"apikey": settings.fmp_api_key},
            timeout=10,
        )
        if resp.ok:
            items = resp.json()
            if items and isinstance(items, list):
                q = items[0]
                data["quote"] = {
                    "price": q.get("price"),
                    "change": q.get("change"),
                    "change_pct": q.get("changesPercentage"),
                    "year_high": q.get("yearHigh"),
                    "year_low": q.get("yearLow"),
                    "pe": q.get("pe"),
                    "volume": q.get("volume"),
                }
    except Exception:
        logger.debug("Failed to fetch quote for %s", t, exc_info=True)

    return data


def _build_system_prompt(
    domain: str,
    snapshot: dict[str, Any] | Any,
    regime_label: str | None,
    regime_description: str | None,
    ticker_context: dict[str, Any] | None,
    diff_context: str | None = None,
    book_impact_context: str | None = None,
    scenario_context: str | None = None,
) -> str:
    snapshot_json = json.dumps(
        snapshot if isinstance(snapshot, dict) else {"data": str(snapshot)},
        indent=2,
        default=str,
    )

    today = datetime.now().strftime("%Y-%m-%d")

    parts = [
        "You are a PM-level macro strategist at a top-tier hedge fund. "
        "You write like a Goldman Sachs morning note or a Bridgewater daily observation — "
        "dense, data-heavy, opinionated, and directly actionable.",
        f"\nToday: {today}",
        f"\n## {domain.title()} Regime",
        f"Classification: **{regime_label or 'N/A'}**",
    ]
    if regime_description:
        parts.append(f"Summary: {regime_description}")
    parts.append(f"\nLive data:\n```json\n{snapshot_json}\n```")

    if diff_context:
        parts.append(f"\n{diff_context}")

    if ticker_context:
        tc_json = json.dumps(ticker_context, indent=2, default=str)
        parts.append(f"\n## Ticker: {ticker_context.get('ticker', '')}\n```json\n{tc_json}\n```")

    # Core instructions that apply to every response
    parts.append(f"""
## Your Operating Rules

**FORMAT:**
- Lead with your conclusion / trade idea in the first sentence. No preamble.
- Use short paragraphs (2-3 sentences max). No walls of text.
- Bold the key numbers: prices, levels, percentages, dates.
- When referencing a data point, cite the exact value from the snapshot.
- End with a concrete risk/reward: entry, target, stop, and what invalidates.

**CONTENT — what you MUST do:**
- Ground every claim in a specific number from the data provided. If the RSI is 69, say "RSI at **69** — approaching overbought." Don't say "RSI indicates it's near overbought territory."
- Name specific catalysts with dates when possible: "Feb 12 CPI print", "March FOMC", "Q4 earnings on Jan 28" — not "upcoming economic data."
- When quant scenarios exist in the data, frame your view around them: "Base case **$87.32** implies X% from here; bull case requires Y."
- For ETFs/bonds: discuss duration, real yields, curve positioning, and flow data. TLT is not a stock — don't analyze it like one.
- State your conviction: high/medium/low. Take a side.

**CONTENT — what you must NEVER do:**
- Never use phrases like "it's important to note", "investors should consider", "could potentially", "it remains to be seen."
- Never give both sides equally ("could go up or could go down"). Have a view.
- Never use generic headers like "Actionable Insights" or "Key Takeaways."
- Never speculate about catalysts you can't tie to specific data or known events.
- Never pad with filler. If you've made your point, stop.""")

    if diff_context:
        parts.append("""
**MEMORY / CHANGES:**
When prior session data is available, open with what moved and why:
- Cite the exact metric deltas (e.g., "VIX dropped from **22.1** to **18.5**, a **-16%** move").
- Attribute each move to a specific catalyst (name the event, date if known).
- Flag any metric that moved in a direction inconsistent with the regime classification.
- Keep it to the 3-4 most important changes, not an exhaustive list.""")

    if book_impact_context:
        parts.append(f"\n{book_impact_context}")
        parts.append("""
**BOOK IMPACT RULES:**
- When positions are provided, ALWAYS discuss which are most exposed to regime shifts.
- For puts: remember they PROFIT when the underlying drops. A regime that hurts the underlying is a TAILWIND for puts.
- For calls: they profit when the underlying rises. A regime that benefits the underlying is a TAILWIND for calls.
- Flag the 2-3 most at-risk positions and explain specifically what would need to change for the thesis to break.
- If asked about trades, prioritize positions already in the book before suggesting new ones.""")

    if scenario_context:
        parts.append(f"\n{scenario_context}")
        parts.append("""
**SCENARIO RULES:**
- When active scenarios are provided, frame your analysis through them.
- Reference the specific conditions that triggered each scenario.
- For each trade expression, give your conviction and adjust sizing based on how many conditions are met.
- If a scenario contradicts your independent analysis, explain why and which signal you trust more.
- Flag when a scenario is close to activating or deactivating based on metrics near thresholds.""")

    return "\n".join(parts)


def _chat_stream(
    settings: Settings,
    messages: list[dict[str, str]],
    system_prompt: str,
    console: Console,
) -> str:
    """Stream the LLM response token-by-token so the user sees output immediately."""
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.OPENAI_BASE_URL,
        timeout=90.0,
    )
    model = settings.openai_model

    full = [{"role": "system", "content": system_prompt}] + messages
    stream = client.chat.completions.create(
        model=model,
        messages=full,
        temperature=0.4,
        max_tokens=3000,
        stream=True,
    )

    console.print(f"\n[bold green]Analyst:[/bold green]")
    chunks: list[str] = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            chunks.append(delta)
            console.print(delta, end="", highlight=False)

    console.print()
    return "".join(chunks).strip()


def start_regime_chat(
    *,
    settings: Any,
    domain: str,
    snapshot: dict[str, Any] | Any,
    regime_label: str | None = None,
    regime_description: str | None = None,
    ticker: str = "",
    console: Console | None = None,
    book_impacts: list | None = None,
    active_scenarios: list | None = None,
) -> None:
    """Launch an interactive chat session with the regime snapshot as context."""
    from lox.cli_commands.shared.regime_memory import (
        build_diff_context,
        load_previous_session,
        save_session,
    )

    c = console or Console()

    if not getattr(settings, "openai_api_key", None):
        c.print("[red]Error:[/red] OPENAI_API_KEY not set in .env — cannot start chat.")
        return

    # ── Memory: load previous session and build diff ────────────────────
    diff_context: str | None = None
    previous = load_previous_session(domain)
    snap_dict = snapshot if isinstance(snapshot, dict) else {}

    if previous:
        prev_ts = previous.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(prev_ts.replace("Z", "+00:00"))
            prev_str = dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            prev_str = prev_ts
        prev_label = previous.get("regime_label", "?")
        c.print(f"\n[dim]Last review: {prev_str}  (regime: {prev_label})[/dim]")

        diff_context = build_diff_context(snap_dict, regime_label, previous)
    else:
        c.print(f"\n[dim]First run for {domain} — no prior session to compare.[/dim]")

    # ── Save current session as the new baseline ────────────────────────
    save_session(domain, snap_dict, regime_label, regime_description)

    ticker_context: dict[str, Any] | None = None
    if ticker.strip():
        t = ticker.strip().upper()
        c.print(f"\n[cyan]Loading context for {t}...[/cyan]")
        try:
            ticker_context = _fetch_ticker_context(settings, t)
            name = ticker_context.get("profile", {}).get("name", t)
            price = ticker_context.get("quote", {}).get("price")
            if price is not None:
                c.print(f"[dim]  {name}  ${price}[/dim]")
        except Exception as exc:
            c.print(f"[yellow]Warning: Could not load ticker context: {exc}[/yellow]")

    # ── Book impact context ────────────────────────────────────────────
    book_impact_context: str | None = None
    if book_impacts:
        from lox.cli_commands.shared.book_impact import format_book_impact_for_llm
        book_impact_context = format_book_impact_for_llm(book_impacts)
        if book_impact_context:
            c.print(f"[dim]Book impact loaded — the analyst knows your open positions.[/dim]")

    # ── Scenario context ────────────────────────────────────────────────
    scenario_context: str | None = None
    if active_scenarios:
        from lox.regimes.scenarios import format_scenarios_for_llm
        scenario_context = format_scenarios_for_llm(active_scenarios)
        if scenario_context:
            c.print(f"[dim]{len(active_scenarios)} active scenario(s) loaded into analyst context.[/dim]")

    system_prompt = _build_system_prompt(
        domain=domain,
        snapshot=snapshot,
        regime_label=regime_label,
        regime_description=regime_description,
        ticker_context=ticker_context,
        diff_context=diff_context,
        book_impact_context=book_impact_context,
        scenario_context=scenario_context,
    )

    c.print("\n[green]Chat started.[/green] Type your questions (or 'quit' to exit):\n")
    if previous:
        c.print(f"[dim]Memory loaded — the analyst knows what changed since your last review.[/dim]")
    if ticker.strip():
        c.print(f"[dim]Context: {domain.title()} regime + {ticker.strip().upper()}[/dim]\n")

    messages: list[dict[str, str]] = []

    while True:
        try:
            user_input = c.input("[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            c.print("\n[yellow]Chat ended.[/yellow]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            c.print("[yellow]Chat ended.[/yellow]")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            response = _chat_stream(settings, messages, system_prompt, c)
            messages.append({"role": "assistant", "content": response})
            c.print()

        except Exception as exc:
            c.print(f"\n[red]Error:[/red] {exc}")
            messages.pop()

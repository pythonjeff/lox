"""
Interactive regime chat — drop into a conversation with the LLM
pre-loaded with the regime snapshot (and optional ticker context).

Called from print_llm_regime_analysis() when --llm is active.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from lox.config import Settings


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
        pass

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
        pass

    return data


def _build_system_prompt(
    domain: str,
    snapshot: dict[str, Any] | Any,
    regime_label: str | None,
    regime_description: str | None,
    ticker_context: dict[str, Any] | None,
    diff_context: str | None = None,
) -> str:
    snapshot_json = json.dumps(
        snapshot if isinstance(snapshot, dict) else {"data": str(snapshot)},
        indent=2,
        default=str,
    )

    parts = [
        "You are a senior macro research analyst at a hedge fund.",
        f"\n## {domain.title()} Regime Context",
        f"Regime: {regime_label or 'N/A'}",
    ]
    if regime_description:
        parts.append(f"Description: {regime_description}")
    parts.append(f"\nData snapshot:\n```json\n{snapshot_json}\n```")

    if diff_context:
        parts.append(f"\n{diff_context}")
        parts.append("""
When the user asks a question (or as your opening analysis), you MUST:
1. Highlight the most significant metric changes since the last review
2. Explain what news, catalysts, or macro developments likely drove each change
   (e.g., FOMC decisions, CPI prints, earnings, geopolitical events, fiscal policy)
3. Assess whether the regime shift (or stability) is consistent with the data moves
4. Note any divergences or surprises that warrant attention""")

    if ticker_context:
        tc_json = json.dumps(ticker_context, indent=2, default=str)
        parts.append(f"\n## Ticker Focus: {ticker_context.get('ticker', '')}\n```json\n{tc_json}\n```")

    parts.append(f"""
Guidelines:
- Be concise but thorough
- Reference specific data points from the context
- Mention specific tickers, levels, and actionable insights
- Use professional hedge fund language
- When discussing positions, be specific about risk/reward

Current date: {datetime.now().strftime('%Y-%m-%d %H:%M')}""")

    return "\n".join(parts)


def _chat_once(
    settings: Settings,
    messages: list[dict[str, str]],
    system_prompt: str,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.OPENAI_BASE_URL)
    model = settings.openai_model

    full = [{"role": "system", "content": system_prompt}] + messages
    resp = client.chat.completions.create(
        model=model,
        messages=full,
        temperature=0.3,
        max_tokens=2500,
    )
    return (resp.choices[0].message.content or "").strip()


def start_regime_chat(
    *,
    settings: Any,
    domain: str,
    snapshot: dict[str, Any] | Any,
    regime_label: str | None = None,
    regime_description: str | None = None,
    ticker: str = "",
    console: Console | None = None,
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

    system_prompt = _build_system_prompt(
        domain=domain,
        snapshot=snapshot,
        regime_label=regime_label,
        regime_description=regime_description,
        ticker_context=ticker_context,
        diff_context=diff_context,
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
            with c.status("[cyan]Thinking...[/cyan]"):
                response = _chat_once(settings, messages, system_prompt)

            messages.append({"role": "assistant", "content": response})
            c.print(f"\n[bold green]Analyst:[/bold green]")
            c.print(Markdown(response))
            c.print()

        except Exception as exc:
            c.print(f"[red]Error:[/red] {exc}")
            messages.pop()

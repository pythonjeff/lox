"""
Interactive Research Chat - Conversation with the LLM analyst.

Usage:
    lox chat                           # Start fresh with portfolio context
    lox chat -c fiscal                 # Pre-load fiscal snapshot
    lox chat -c vol                    # Pre-load volatility snapshot
    lox chat -c macro                  # Pre-load macro dashboard
    lox chat -c regimes                # Pre-load all regime data
    lox chat -c fiscal -c funding      # Multiple contexts
    lox chat -c fiscal,funding,monetary  # Comma-separated contexts
    lox chat -t AAPL                   # Ticker-focused chat

After the context loads, you can ask follow-up questions like:
    "How does this affect my IWM position?"
    "What's the risk to my portfolio if VIX spikes to 25?"
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from lox.config import load_settings, Settings


# ── Context loaders ────────────────────────────────────────────────────────

def _get_context_data(context: str, settings: Settings, refresh: bool = False) -> tuple[str, dict]:
    """Load context data for a given domain."""

    if context in ("vol", "volatility"):
        from lox.volatility.signals import build_volatility_state
        from lox.volatility.regime import classify_volatility_regime

        state = build_volatility_state(settings=settings, start_date="2020-01-01", refresh=refresh)
        regime = classify_volatility_regime(state.inputs)
        inp = state.inputs
        return f"Volatility Regime: {regime.label}", {
            "domain": "volatility",
            "regime": regime.label,
            "description": getattr(regime, "description", ""),
            "vix": inp.vix,
            "vix_z": inp.z_vix,
            "vix_chg_5d_pct": inp.vix_chg_5d_pct,
            "term_spread": inp.vix_term_spread,
        }

    elif context == "macro":
        from lox.macro.signals import build_macro_state
        state = build_macro_state(settings=settings, start_date="2020-01-01", refresh=refresh)
        inp = state.inputs
        return "Macro State", {
            "domain": "macro",
            "cpi_yoy": inp.cpi_yoy,
            "core_cpi_yoy": inp.core_cpi_yoy,
            "payrolls_mom": inp.payrolls_mom,
            "unemployment_rate": inp.unemployment_rate,
            "vix": inp.vix,
            "breakeven_5y": inp.breakeven_5y,
        }

    elif context == "fiscal":
        from lox.fiscal.signals import build_fiscal_deficit_page_data
        from lox.fiscal.regime import classify_fiscal_regime_snapshot

        data = build_fiscal_deficit_page_data(settings=settings, lookback_years=5, refresh=refresh)
        net = data.get("net_issuance") if isinstance(data.get("net_issuance"), dict) else None
        tga = data.get("tga") if isinstance(data.get("tga"), dict) else None
        regime = classify_fiscal_regime_snapshot(
            deficit_pct_gdp=data.get("deficit_pct_gdp"),
            deficit_impulse_pct_gdp=data.get("deficit_impulse_pct_gdp"),
            long_duration_issuance_share=net.get("long_share") if net else None,
            tga_z_d_4w=tga.get("z_d_4w") if tga else None,
        )
        return f"Fiscal Regime: {regime.label}", {
            "domain": "fiscal",
            "regime": regime.label,
            "description": regime.description,
            "snapshot": data,
        }

    elif context in ("commod", "commodities"):
        from lox.commodities.signals import build_commodities_state
        from lox.commodities.regime import classify_commodities_regime

        state = build_commodities_state(settings=settings, start_date="2020-01-01", refresh=refresh)
        regime = classify_commodities_regime(state.inputs)
        return f"Commodities Regime: {regime.label}", {
            "domain": "commodities",
            "regime": regime.label,
            "snapshot": state.inputs.__dict__ if hasattr(state.inputs, "__dict__") else {},
        }

    elif context == "rates":
        from lox.rates.signals import build_rates_state
        from lox.rates.regime import classify_rates_regime

        state = build_rates_state(settings=settings, start_date="2020-01-01", refresh=refresh)
        regime = classify_rates_regime(state.inputs)
        return f"Rates Regime: {regime.label}", {
            "domain": "rates",
            "regime": regime.label,
            "ust_10y": state.inputs.ust_10y,
            "snapshot": state.inputs.__dict__ if hasattr(state.inputs, "__dict__") else {},
        }

    elif context == "funding":
        from lox.funding.signals import build_funding_state
        from lox.funding.regime import classify_funding_regime

        state = build_funding_state(settings=settings, start_date="2020-01-01", refresh=refresh)
        regime = classify_funding_regime(state.inputs)
        inp = state.inputs
        return f"Funding Regime: {regime.label}", {
            "domain": "funding",
            "regime": regime.label,
            "description": getattr(regime, "description", ""),
            "snapshot": {
                "effr": getattr(inp, "effr", None),
                "sofr": getattr(inp, "sofr", None),
                "on_rrp_usd_bn": getattr(inp, "on_rrp_usd_bn", None),
                "bank_reserves_usd_bn": getattr(inp, "bank_reserves_usd_bn", None),
            },
        }

    elif context == "monetary":
        from lox.monetary.signals import build_monetary_page_data
        from lox.monetary.regime import classify_monetary_regime
        from lox.monetary.models import MonetaryInputs

        data = build_monetary_page_data(settings=settings, lookback_years=5, refresh=refresh)
        inputs = MonetaryInputs(
            reserves_level=data.get("reserves_level"),
            reserves_z_level=data.get("reserves_z_level"),
            reserves_z_d_4w=data.get("reserves_z_d_4w"),
            on_rrp_level=data.get("on_rrp_level"),
            on_rrp_z_level=data.get("on_rrp_z_level"),
            on_rrp_z_d_4w=data.get("on_rrp_z_d_4w"),
            fed_bs_level=data.get("fed_bs_level"),
            fed_bs_z_d_12w=data.get("fed_bs_z_d_12w"),
            asof=data.get("asof"),
        )
        regime = classify_monetary_regime(inputs)
        return f"Monetary Regime: {regime.label}", {
            "domain": "monetary",
            "regime": regime.label,
            "description": getattr(regime, "description", ""),
            "snapshot": data,
        }

    elif context in ("regimes", "all"):
        contexts = {}
        for ctx in ("vol", "macro", "fiscal", "commodities", "rates", "funding", "monetary"):
            try:
                _, data = _get_context_data(ctx, settings, refresh=refresh)
                contexts[ctx] = data
            except Exception as e:
                contexts[ctx] = {"error": str(e)}
        return "All Regimes Summary", {"domains": contexts}

    elif context in ("portfolio", ""):
        from lox.data.alpaca import make_clients

        trading, _ = make_clients(settings)
        acct = trading.get_account()
        positions = trading.get_all_positions()
        positions_list = [
            {
                "symbol": getattr(p, "symbol", ""),
                "qty": float(getattr(p, "qty", 0)),
                "market_value": float(getattr(p, "market_value", 0)),
                "unrealized_pl": float(getattr(p, "unrealized_pl", 0)),
                "unrealized_plpc": float(getattr(p, "unrealized_plpc", 0)),
            }
            for p in positions
        ]
        return "Portfolio Context", {
            "domain": "portfolio",
            "nav": float(getattr(acct, "equity", 0)),
            "cash": float(getattr(acct, "cash", 0)),
            "positions": positions_list,
        }

    else:
        available = "vol, macro, fiscal, commodities, rates, funding, monetary, regimes, portfolio"
        raise ValueError(f"Unknown context: {context}. Available: {available}")


# ── Multi-context helpers ──────────────────────────────────────────────────

def _parse_contexts(context_input: list[str]) -> list[str]:
    """Parse context input which may contain comma-separated values."""
    contexts: list[str] = []
    for c in context_input:
        parts = [p.strip().lower() for p in c.split(",") if p.strip()]
        contexts.extend(parts)
    return contexts if contexts else ["portfolio"]


def _load_multiple_contexts(
    contexts: list[str], settings: Settings, refresh: bool, console: Console,
) -> tuple[str, dict]:
    """Load and merge multiple contexts."""
    if len(contexts) == 1:
        return _get_context_data(contexts[0], settings, refresh=refresh)

    merged: dict[str, Any] = {"domains": {}}
    titles: list[str] = []
    for ctx in contexts:
        try:
            title, data = _get_context_data(ctx, settings, refresh=refresh)
            titles.append(ctx.capitalize())
            merged["domains"][data.get("domain", ctx)] = data
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load {ctx}: {e}[/yellow]")
            merged["domains"][ctx] = {"error": str(e)}
    return " + ".join(titles) + " Context", merged


# ── Display helpers ────────────────────────────────────────────────────────

def _format_context_display(title: str, data: dict, console: Console) -> None:
    """Print a summary of loaded context."""
    if "domains" in data:
        table = Table(title="Regime Summary", show_header=True, box=None, padding=(0, 2))
        table.add_column("Domain", style="cyan")
        table.add_column("Regime", style="bold")
        for name, info in data["domains"].items():
            if "error" in info:
                table.add_row(name, "[red]Error[/red]")
            else:
                table.add_row(name, info.get("regime", "N/A"))
        console.print(table)

    elif "positions" in data:
        console.print(Panel(
            f"[bold]NAV:[/bold] ${data['nav']:,.0f}\n"
            f"[bold]Cash:[/bold] ${data['cash']:,.0f}\n"
            f"[bold]Positions:[/bold] {len(data['positions'])}",
            title="Portfolio Summary",
            border_style="cyan",
        ))
        if data["positions"]:
            table = Table(show_header=True, box=None, padding=(0, 2))
            table.add_column("Symbol", style="cyan")
            table.add_column("Value", justify="right")
            table.add_column("P&L", justify="right")
            for p in data["positions"]:
                style = "green" if p["unrealized_pl"] >= 0 else "red"
                table.add_row(
                    p["symbol"],
                    f"${p['market_value']:,.0f}",
                    f"[{style}]${p['unrealized_pl']:+,.0f}[/{style}]",
                )
            console.print(table)
    else:
        console.print(Panel(
            f"[bold]Regime:[/bold] {data.get('regime', 'N/A')}\n"
            f"[dim]{data.get('description', '')}[/dim]",
            title=f"{title}",
            border_style="cyan",
        ))


# ── Ticker context ─────────────────────────────────────────────────────────

def _fetch_ticker_context(settings: Settings, ticker: str) -> tuple[str, dict]:
    """Fetch quote + profile for a ticker via FMP."""
    import requests

    t = ticker.upper()
    data: dict[str, Any] = {"ticker": t, "domain": "ticker"}

    if settings.fmp_api_key:
        try:
            url = f"https://financialmodelingprep.com/api/v3/profile/{t}"
            resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=10)
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
            url = f"https://financialmodelingprep.com/api/v3/quote/{t}"
            resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=10)
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

    title = f"Ticker: {t}"
    if "profile" in data:
        title = f"{data['profile'].get('name', t)} ({t})"
    return title, data


# ── LLM chat ──────────────────────────────────────────────────────────────

def _chat_with_llm(
    settings: Settings,
    messages: list[dict],
    context_data: dict,
    model: str | None = None,
    temperature: float = 0.3,
) -> str:
    """Send chat to LLM with context."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.OPENAI_BASE_URL)
    chosen_model = model or settings.openai_model

    context_json = json.dumps(context_data, indent=2, default=str)

    system_message = f"""You are a senior macro research analyst at a hedge fund.
You have access to the following market data and regime context:

{context_json}

Guidelines:
- Be concise but thorough
- Reference specific data points from the context
- Consider portfolio implications when relevant
- Mention specific tickers, levels, and actionable insights
- Use professional hedge fund language
- When discussing positions, be specific about risk/reward

Current date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""

    full_messages = [{"role": "system", "content": system_message}] + messages

    resp = client.chat.completions.create(
        model=chosen_model,
        messages=full_messages,
        temperature=temperature,
        max_tokens=2500,
    )
    return (resp.choices[0].message.content or "").strip()


# ── CLI command ────────────────────────────────────────────────────────────

def register(app: typer.Typer) -> None:
    """Register the chat command."""

    @app.command("chat")
    def chat_cmd(
        context: list[str] = typer.Option(
            ["portfolio"],
            "--context", "-c",
            help="Context(s): vol, macro, fiscal, commodities, rates, funding, monetary, regimes, portfolio",
        ),
        ticker: str = typer.Option("", "--ticker", "-t", help="Ticker for focused analysis"),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh data"),
        model: str = typer.Option("", "--model", "-m", help="Override LLM model"),
    ):
        """
        Interactive research chat with LLM analyst.

        Load market context and have a conversation about trading.

        Examples:
            lox research chat                        # Portfolio context
            lox research chat -t AAPL                # Ticker chat
            lox research chat -c fiscal              # Fiscal regime
            lox research chat -c fiscal -c funding   # Multiple contexts
            lox research chat -c regimes             # All regimes
        """
        console = Console()
        settings = load_settings()

        if not settings.openai_api_key:
            console.print("[red]Error:[/red] OPENAI_API_KEY not set in .env")
            raise typer.Exit(1)

        # Load ticker context if specified
        ticker_context: dict | None = None
        if ticker.strip():
            try:
                console.print(f"\n[cyan]Loading context for {ticker.upper()}...[/cyan]")
                title, ticker_context = _fetch_ticker_context(settings, ticker)
                _format_context_display(title, ticker_context, console)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load ticker context: {e}[/yellow]")

        # Load domain contexts
        parsed = _parse_contexts(context)
        console.print(f"\n[cyan]Loading context(s): {', '.join(parsed)}...[/cyan]")

        try:
            title, context_data = _load_multiple_contexts(parsed, settings, refresh, console)
        except Exception as e:
            console.print(f"[red]Error loading context:[/red] {e}")
            raise typer.Exit(1)

        # Merge ticker context
        if ticker_context:
            context_data["ticker_focus"] = ticker_context
            title = f"{ticker.upper()} + {title}"

        _format_context_display(title, context_data, console)

        # Chat loop
        console.print("\n[green]Chat started.[/green] Type your questions (or 'quit' to exit):\n")
        if ticker.strip():
            console.print(f"[dim]Focused on {ticker.upper()}. Ask about outlook, risks, price targets, etc.[/dim]\n")

        messages: list[dict] = []
        chosen_model = model.strip() or None

        while True:
            try:
                user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Chat ended.[/yellow]")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[yellow]Chat ended.[/yellow]")
                break

            messages.append({"role": "user", "content": user_input})

            try:
                with console.status("[cyan]Thinking...[/cyan]"):
                    response = _chat_with_llm(
                        settings=settings,
                        messages=messages,
                        context_data=context_data,
                        model=chosen_model,
                    )

                messages.append({"role": "assistant", "content": response})
                console.print(f"\n[bold green]Analyst:[/bold green]")
                console.print(Markdown(response))
                console.print()

            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                messages.pop()  # Remove failed user message

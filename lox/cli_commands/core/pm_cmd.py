"""
PM Morning Report — startup hedge fund daily briefing.

Single command that pulls macro regime state, portfolio Greeks,
positions, and active scenarios into a dense, actionable morning note.

Usage:
    lox pm              # Full report with LLM briefing (default)
    lox pm --no-llm     # Data only, no LLM
    lox pm --json       # Machine-readable JSON output

Author: Lox Capital Research
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lox.config import load_settings, Settings

logger = logging.getLogger(__name__)

# ── Display helpers ───────────────────────────────────────────────────

SCORE_LOW = 35
SCORE_HIGH = 65


def _score_color(score: float) -> str:
    if score < SCORE_LOW:
        return "green"
    if score < SCORE_HIGH:
        return "yellow"
    return "red"


def _bar(score: float, width: int = 10) -> str:
    filled = int((score / 100) * width)
    empty = width - filled
    c = _score_color(score)
    return f"[{c}]{'█' * filled}{'░' * empty}[/{c}]"


def _safe_float(x) -> float:
    try:
        return float(x) if x is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def _fmt_metrics(metrics: dict | None, max_items: int = 3) -> str:
    if not metrics:
        return "[dim]—[/dim]"
    parts = []
    for k, v in metrics.items():
        if v is not None and len(parts) < max_items:
            parts.append(f"{k} {v}")
    return ", ".join(parts) if parts else "[dim]—[/dim]"


def _trend_arrow(trend) -> str:
    if trend is None:
        return "[dim]—[/dim]"
    return f"[{trend.trend_color}]{trend.trend_arrow}[/{trend.trend_color}]"


def _delta7(trend) -> str:
    if trend is None:
        return "[dim]—[/dim]"
    val = trend.score_chg_7d
    if val is None:
        return "[dim]—[/dim]"
    sign = "+" if val > 0 else ""
    if abs(val) < 0.5:
        return f"[dim]{sign}{val:.0f}[/dim]"
    c = "red" if val > 0 else "green"
    return f"[{c}]{sign}{val:.0f}[/{c}]"


# ── Data fetching ─────────────────────────────────────────────────────

PM_PILLARS = [
    ("Growth",    "growth"),
    ("Inflation", "inflation"),
    ("Volatility","volatility"),
    ("Credit",    "credit"),
    ("Rates",     "rates"),
    ("Liquidity", "liquidity"),
    ("Consumer",  "consumer"),
    ("Fiscal",    "fiscal"),
    ("Earnings",  "earnings"),
    ("Oil/Cmdty", "commodities"),
]


def _fetch_all_data(settings: Settings) -> dict[str, Any]:
    """Parallel fetch: regime state, portfolio greeks, raw positions."""
    results: dict[str, Any] = {"state": None, "greeks": None, "positions": [], "account": {}}

    def _regime():
        from lox.regimes import build_unified_regime_state
        return build_unified_regime_state(settings=settings, start_date="2020-01-01", refresh=False)

    def _greeks():
        from lox.risk.greeks import compute_portfolio_greeks
        return compute_portfolio_greeks(settings)

    def _positions():
        from lox.data.alpaca import make_clients
        trading, _ = make_clients(settings)
        acct = trading.get_account()
        positions = trading.get_all_positions()
        return acct, positions

    tasks = {"regime": _regime, "greeks": _greeks, "positions": _positions}

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if name == "regime":
                    results["state"] = result
                elif name == "greeks":
                    results["greeks"] = result
                elif name == "positions":
                    acct, positions = result
                    results["positions"] = positions
                    import os
                    total_capital = float(os.environ.get("FUND_TOTAL_CAPITAL", 950))
                    try:
                        from lox.nav.investors import read_investor_flows
                        flows = read_investor_flows()
                        if flows:
                            total_capital = sum(f.amount for f in flows if float(f.amount) > 0)
                    except Exception:
                        pass
                    equity = _safe_float(getattr(acct, "equity", 0))
                    cash = _safe_float(getattr(acct, "cash", 0))
                    results["account"] = {
                        "equity": equity,
                        "cash": cash,
                        "total_capital": total_capital,
                        "pnl": equity - total_capital,
                        "pnl_pct": ((equity - total_capital) / total_capital * 100) if total_capital > 0 else 0,
                        "cash_pct": (cash / equity * 100) if equity > 0 else 0,
                        "mode": "PAPER" if settings.alpaca_paper else "LIVE",
                    }
            except Exception:
                logger.warning("PM fetch '%s' failed", name, exc_info=True)

    return results


# ── Rendering ─────────────────────────────────────────────────────────

def _render_header(console: Console, state, account: dict) -> None:
    if state:
        risk = state.overall_risk_score
        cat = state.overall_category.upper()
        quad = state.macro_quadrant
        rc = _score_color(risk)
        risk_line = f"Risk: [{rc}]{risk:.0f}/100[/{rc}] ({cat})  Quadrant: {quad}"
    else:
        risk_line = "[yellow]Regime data unavailable[/yellow]"

    nav = account.get("equity", 0)
    pnl = account.get("pnl", 0)
    pnl_pct = account.get("pnl_pct", 0)
    mode = account.get("mode", "PAPER")
    pnl_c = "green" if pnl >= 0 else "red"

    console.print(Panel(
        f"{risk_line}\n"
        f"NAV: ${nav:,.0f}  P&L: [{pnl_c}]${pnl:+,.0f} ({pnl_pct:+.1f}%)[/{pnl_c}]  [{mode}]",
        title=f"[bold]LOX CAPITAL — PM MORNING REPORT[/bold]  [dim]{datetime.now().strftime('%b %-d %Y')}[/dim]",
        border_style="cyan",
        padding=(0, 2),
    ))


def _render_macro(console: Console, state) -> None:
    if state is None:
        console.print("[yellow]  Macro data unavailable[/yellow]")
        return

    trends = state.trends or {}
    tbl = Table(
        show_header=True, header_style="bold",
        box=None, padding=(0, 2),
    )
    tbl.add_column("Pillar", style="bold", min_width=10, no_wrap=True)
    tbl.add_column("", min_width=10, no_wrap=True)  # bar
    tbl.add_column("Score", justify="center", min_width=5, no_wrap=True)
    tbl.add_column("", justify="center", min_width=3, no_wrap=True)  # arrow
    tbl.add_column("Δ7d", justify="right", min_width=4, no_wrap=True)
    tbl.add_column("Regime", min_width=16, no_wrap=True)
    tbl.add_column("Key Metric", ratio=1)

    for display_name, domain_key in PM_PILLARS:
        regime = getattr(state, domain_key, None)
        if regime:
            c = _score_color(regime.score)
            t = trends.get(domain_key)
            tbl.add_row(
                display_name,
                _bar(regime.score),
                f"[{c}]{regime.score:.0f}[/{c}]",
                _trend_arrow(t),
                _delta7(t),
                f"[{c}]{regime.label}[/{c}]",
                _fmt_metrics(regime.metrics),
            )
        else:
            tbl.add_row(display_name, _bar(0), "[dim]—[/dim]", "[dim]—[/dim]", "[dim]—[/dim]", "[dim]No data[/dim]", "")

    console.print("\n[bold cyan][1] MACRO ENVIRONMENT[/bold cyan]")
    console.print(tbl)


def _render_scenarios(console: Console, state) -> None:
    console.print("\n[bold cyan][2] SCENARIOS[/bold cyan]")
    if state is None or not state.active_scenarios:
        console.print("  [dim]No scenarios active.[/dim]")
        return

    for s in state.active_scenarios:
        style = "bold red" if s.conviction == "HIGH" else ("bold yellow" if s.conviction == "MEDIUM" else "dim")
        top_trade = ""
        if s.trades:
            t = s.trades[0]
            tc = "green" if t.direction.upper() == "LONG" else "red"
            top_trade = f" → [{tc}]{t.direction}[/{tc}] {t.ticker}"
        console.print(
            f"  [{style}]{s.conviction:>6}[/{style}]  {s.name} "
            f"({s.conditions_met}/{s.conditions_total}){top_trade}"
        )


def _render_portfolio(console: Console, greeks, positions, account: dict) -> None:
    console.print("\n[bold cyan][3] PORTFOLIO[/bold cyan]")

    # NAV line
    nav = account.get("equity", 0)
    cash_pct = account.get("cash_pct", 0)
    parts = [f"NAV: ${nav:,.0f}", f"Cash: {cash_pct:.1f}%"]

    if greeks:
        parts.extend([
            f"Delta: {greeks.net_delta:+.0f}",
            f"Theta: ${greeks.net_theta:+.0f}/day",
            f"Vega: {greeks.net_vega:+.0f}",
        ])

    console.print(f"  {' '.join(f'[bold]{p}[/bold]' for p in parts)}")

    # Position highlights
    if positions:
        pos_data = []
        for p in positions:
            sym = str(getattr(p, "symbol", ""))
            pnl = _safe_float(getattr(p, "unrealized_pl", 0))
            pnl_pc = _safe_float(getattr(p, "unrealized_plpc", 0)) * 100
            mv = _safe_float(getattr(p, "market_value", 0))
            pos_data.append((sym, pnl, pnl_pc, mv))

        # Bleeders (down >15%)
        bleeders = [(s, pl, ppc, mv) for s, pl, ppc, mv in pos_data if ppc < -15]
        bleeders.sort(key=lambda x: x[2])
        if bleeders:
            console.print()
            for sym, pnl, ppc, _ in bleeders[:3]:
                console.print(f"  [red]Bleeding:[/red] {sym}  ${pnl:+,.0f} ({ppc:+.1f}%) → Cut or roll")

        # Winners (up >20%)
        winners = [(s, pl, ppc, mv) for s, pl, ppc, mv in pos_data if ppc > 20]
        winners.sort(key=lambda x: -x[2])
        if winners:
            if not bleeders:
                console.print()
            for sym, pnl, ppc, _ in winners[:3]:
                console.print(f"  [green]Winner:[/green]   {sym}  ${pnl:+,.0f} ({ppc:+.1f}%) → Trail stop")

    # Risk signals from Greeks
    if greeks and greeks.risk_signals:
        console.print()
        for sig in greeks.risk_signals[:3]:
            console.print(f"  [yellow]⚠ {sig}[/yellow]")

    # Theta burn context
    if greeks and nav > 0:
        theta_bp = abs(greeks.net_theta) / nav * 10000
        if theta_bp > 5:
            console.print(f"\n  [yellow]⚠ Theta burn ${abs(greeks.net_theta):.0f}/day vs ${nav:,.0f} NAV = {theta_bp:.0f}bp/day[/yellow]")


def _build_pm_system_prompt(state, greeks, positions, account: dict) -> str:
    """Build CIO-grade system prompt with all data injected."""
    today = datetime.now().strftime("%Y-%m-%d")

    # Macro summary
    macro_lines = []
    if state:
        for display_name, domain_key in PM_PILLARS:
            regime = getattr(state, domain_key, None)
            if regime:
                t = (state.trends or {}).get(domain_key)
                delta_str = f"Δ7d {t.score_chg_7d:+.0f}" if t and t.score_chg_7d is not None else ""
                metrics_str = ", ".join(f"{k}: {v}" for k, v in (regime.metrics or {}).items() if v is not None)
                macro_lines.append(f"  {display_name}: {regime.score:.0f}/100 {regime.label} {delta_str}  [{metrics_str}]")

    macro_block = "\n".join(macro_lines) if macro_lines else "  Data unavailable"

    # Scenarios
    scenario_lines = []
    if state and state.active_scenarios:
        for s in state.active_scenarios:
            trades_str = ", ".join(f"{t.direction} {t.ticker}" for t in s.trades[:2])
            scenario_lines.append(f"  {s.conviction} {s.name} ({s.conditions_met}/{s.conditions_total}) → {trades_str}")
    scenario_block = "\n".join(scenario_lines) if scenario_lines else "  None active"

    # Portfolio
    port_lines = []
    nav = account.get("equity", 0)
    pnl = account.get("pnl", 0)
    pnl_pct = account.get("pnl_pct", 0)
    cash_pct = account.get("cash_pct", 0)
    port_lines.append(f"  NAV: ${nav:,.0f}  Fund P&L: ${pnl:+,.0f} ({pnl_pct:+.1f}%)  Cash: {cash_pct:.1f}%")

    if greeks:
        port_lines.append(f"  Greeks: Delta {greeks.net_delta:+.0f}  Gamma {greeks.net_gamma:+.2f}  Theta ${greeks.net_theta:+.0f}/day  Vega {greeks.net_vega:+.0f}")

    if positions:
        for p in positions:
            sym = str(getattr(p, "symbol", ""))
            qty = _safe_float(getattr(p, "qty", 0))
            pnl_p = _safe_float(getattr(p, "unrealized_pl", 0))
            pnl_pc = _safe_float(getattr(p, "unrealized_plpc", 0)) * 100
            mv = _safe_float(getattr(p, "market_value", 0))
            port_lines.append(f"  {sym}: qty={qty:+.1f} mv=${mv:,.0f} P&L=${pnl_p:+,.0f} ({pnl_pc:+.1f}%)")

    if greeks and greeks.risk_signals:
        for sig in greeks.risk_signals:
            port_lines.append(f"  ⚠ {sig}")

    portfolio_block = "\n".join(port_lines)

    # Overall
    overall = ""
    if state:
        overall = f"Risk: {state.overall_risk_score:.0f}/100 ({state.overall_category.upper()})  Quadrant: {state.macro_quadrant}"

    return f"""You are the CIO of a startup hedge fund delivering the PM morning note to your portfolio manager.
Today: {today}

## Fund Overview
{overall}

## Macro Environment (10 pillars, 0-100 score, higher = more stress)
{macro_block}

## Active Scenarios
{scenario_block}

## Portfolio & Greeks
{portfolio_block}

## Your Instructions
You are writing a dense, opinionated 2-3 paragraph morning briefing. This is not a summary — it's a BRIEF.

RULES:
- Lead with THE thing that matters most today. No preamble.
- Cite exact numbers from the data: scores, deltas, P&L, Greeks. Don't paraphrase.
- Reference specific positions by name. Say what to do with them.
- State conviction: HIGH / MEDIUM / LOW.
- Give 1-2 concrete trade ideas with entry logic tied to the regime data.
- Flag the #1 risk to the book right now.
- If theta burn is high relative to NAV, flag it.
- If any pillar moved >5 points in 7 days, call it out.
- Max 250 words. Dense. Every sentence earns its place.
- End with what you're watching today (catalyst, data release, level).
- NEVER use phrases like "it's important to note" or "investors should consider".
- Take a side. Have a view. Be wrong sometimes — that's better than hedging every sentence."""


def _render_llm_briefing(console: Console, settings: Settings, state, greeks, positions, account: dict) -> None:
    """Stream LLM CIO briefing."""
    if not getattr(settings, "openai_api_key", None):
        console.print("\n[yellow]Set OPENAI_API_KEY in .env to enable LLM briefing.[/yellow]")
        return

    console.print("\n[bold cyan][4] PM BRIEFING[/bold cyan]")

    system_prompt = _build_pm_system_prompt(state, greeks, positions, account)

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.OPENAI_BASE_URL,
            timeout=90.0,
        )

        stream = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Deliver the PM morning note."},
            ],
            temperature=0.4,
            max_tokens=1000,
            stream=True,
        )

        console.print()
        chunks: list[str] = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                chunks.append(delta)
                console.print(delta, end="", highlight=False)

        console.print("\n")

    except Exception as e:
        console.print(f"\n[red]LLM briefing failed:[/red] {e}")


def _to_json(state, greeks, positions, account: dict) -> dict:
    """Machine-readable JSON output."""
    out: dict[str, Any] = {
        "asof": datetime.now().isoformat(),
        "account": account,
    }

    if state:
        out["macro"] = {
            "risk_score": state.overall_risk_score,
            "category": state.overall_category,
            "quadrant": state.macro_quadrant,
            "pillars": {},
        }
        for display_name, domain_key in PM_PILLARS:
            regime = getattr(state, domain_key, None)
            if regime:
                t = (state.trends or {}).get(domain_key)
                out["macro"]["pillars"][domain_key] = {
                    "display": display_name,
                    "score": regime.score,
                    "label": regime.label,
                    "delta_7d": t.score_chg_7d if t else None,
                    "metrics": regime.metrics,
                }
        out["scenarios"] = [
            {
                "name": s.name,
                "conviction": s.conviction,
                "conditions_met": s.conditions_met,
                "conditions_total": s.conditions_total,
                "trades": [{"direction": t.direction, "ticker": t.ticker, "instrument": t.instrument} for t in s.trades],
            }
            for s in (state.active_scenarios or [])
        ]

    if greeks:
        out["greeks"] = greeks.to_dict()

    if positions:
        out["positions"] = [
            {
                "symbol": str(getattr(p, "symbol", "")),
                "qty": _safe_float(getattr(p, "qty", 0)),
                "market_value": _safe_float(getattr(p, "market_value", 0)),
                "unrealized_pl": _safe_float(getattr(p, "unrealized_pl", 0)),
                "unrealized_plpc": _safe_float(getattr(p, "unrealized_plpc", 0)),
            }
            for p in positions
        ]

    return out


# ── Command registration ─────────────────────────────────────────────

def register_pm(app: typer.Typer) -> None:
    """Register the `lox pm` command on the main app."""

    @app.command("pm")
    def pm_cmd(
        llm: bool = typer.Option(True, "--llm/--no-llm", help="Include LLM CIO briefing (default: on)"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
    ):
        """
        PM Morning Report — your daily hedge fund briefing.

        Combines macro regime state (10 pillars), active scenarios,
        portfolio Greeks, and an opinionated LLM CIO brief.

        Examples:
            lox pm              # Full report with LLM
            lox pm --no-llm     # Data sections only
            lox pm --json       # JSON for programmatic use
        """
        from rich.progress import Progress, SpinnerColumn, TextColumn

        console = Console()
        settings = load_settings()

        # Fetch all data in parallel
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Building morning report...[/bold cyan]"),
            transient=True,
        ) as progress:
            progress.add_task("fetch", total=None)
            data = _fetch_all_data(settings)

        state = data["state"]
        greeks = data["greeks"]
        positions = data["positions"]
        account = data["account"]

        # JSON mode
        if json_out:
            console.print_json(json.dumps(_to_json(state, greeks, positions, account), default=str))
            return

        # Render sections
        _render_header(console, state, account)
        _render_macro(console, state)
        _render_scenarios(console, state)
        _render_portfolio(console, greeks, positions, account)

        # LLM briefing
        if llm:
            _render_llm_briefing(console, settings, state, greeks, positions, account)
        else:
            console.print("\n[dim]Add --llm (default) for CIO briefing[/dim]")

        console.print()

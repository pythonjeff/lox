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
    """Parallel fetch: regime state, portfolio greeks, raw positions, econ calendar."""
    results: dict[str, Any] = {"state": None, "greeks": None, "positions": [], "account": {}, "calendar": []}

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

    def _calendar():
        """Fetch next 5 days of US high-impact economic events."""
        import requests as _req
        from datetime import timedelta, timezone as _tz
        api_key = getattr(settings, "FMP_API_KEY", None)
        if not api_key:
            return []
        now = datetime.now(_tz.utc)
        from_d = now.strftime("%Y-%m-%d")
        to_d = (now + timedelta(days=5)).strftime("%Y-%m-%d")
        try:
            resp = _req.get(
                "https://financialmodelingprep.com/api/v3/economic_calendar",
                params={"from": from_d, "to": to_d, "apikey": api_key},
                timeout=15,
            )
            resp.raise_for_status()
            events = resp.json()
            if not isinstance(events, list):
                return []
            high_impact_kw = [
                "FOMC", "CPI", "PCE", "GDP", "NFP", "Jobless", "Fed", "Auction",
                "Employment", "Payroll", "Retail Sales", "ISM", "PPI", "Michigan",
                "Housing", "Durable", "Treasury",
            ]
            out = []
            cutoff = now + timedelta(days=5)
            for e in events:
                if not isinstance(e, dict):
                    continue
                date_str = e.get("date", "")
                try:
                    from datetime import timezone as _tz2
                    dt = datetime.fromisoformat(date_str.replace(" ", "T").replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=_tz2.utc)
                except Exception:
                    continue
                if dt < now or dt > cutoff:
                    continue
                country = (e.get("country") or "").upper()
                if country and country != "US":
                    continue
                name = e.get("event", "")
                is_high = any(kw.lower() in name.lower() for kw in high_impact_kw)
                out.append({
                    "date": dt.strftime("%a %b %-d"),
                    "time": dt.strftime("%-I:%M%p ET"),
                    "event": name,
                    "estimate": e.get("estimate"),
                    "previous": e.get("previous"),
                    "high_impact": is_high,
                })
            # Sort by date, high-impact first
            out.sort(key=lambda x: (not x["high_impact"], x["date"]))
            return out[:12]
        except Exception:
            return []

    tasks = {"regime": _regime, "greeks": _greeks, "positions": _positions, "calendar": _calendar}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if name == "regime":
                    results["state"] = result
                elif name == "greeks":
                    results["greeks"] = result
                elif name == "calendar":
                    results["calendar"] = result
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

        # Composite regime headline
        if state.composite:
            from lox.cli_commands.shared.composite_display import (
                _regime_style, _confidence_color,
            )
            from lox.regimes.composite import COMPOSITE_LABELS
            c = state.composite
            rs = _regime_style(c.regime)
            cc = _confidence_color(c.confidence)
            regime_line = f"Regime: [{rs}]{c.label}[/{rs}] [{cc}]({c.confidence:.0%} conf)[/{cc}]"
        else:
            regime_line = ""
    else:
        risk_line = "[yellow]Regime data unavailable[/yellow]"
        regime_line = ""

    nav = account.get("equity", 0)
    pnl = account.get("pnl", 0)
    pnl_pct = account.get("pnl_pct", 0)
    mode = account.get("mode", "PAPER")
    pnl_c = "green" if pnl >= 0 else "red"

    body = f"{risk_line}\n"
    if regime_line:
        body += f"{regime_line}\n"
    body += f"NAV: ${nav:,.0f}  P&L: [{pnl_c}]${pnl:+,.0f} ({pnl_pct:+.1f}%)[/{pnl_c}]  [{mode}]"

    console.print(Panel(
        body,
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


def _build_pm_system_prompt(state, greeks, positions, account: dict, calendar: list | None = None) -> str:
    """Build CIO-grade system prompt with all data injected."""
    today = datetime.now().strftime("%Y-%m-%d")

    # ── Macro: full pillar data with metrics ─────────────────────────
    macro_lines = []
    movers = []  # pillars that moved >5pts in 7d
    if state:
        for display_name, domain_key in PM_PILLARS:
            regime = getattr(state, domain_key, None)
            if regime:
                t = (state.trends or {}).get(domain_key)
                chg7 = t.score_chg_7d if t and t.score_chg_7d is not None else 0
                delta_str = f"Δ7d {chg7:+.0f}" if t and t.score_chg_7d is not None else ""
                trend_str = f"trend={t.trend_direction}" if t else ""
                metrics_str = ", ".join(f"{k}: {v}" for k, v in (regime.metrics or {}).items() if v is not None)
                macro_lines.append(f"  {display_name}: {regime.score:.0f}/100 [{regime.label}] {delta_str} {trend_str}  |  {metrics_str}")
                if abs(chg7) > 5:
                    direction = "UP" if chg7 > 0 else "DOWN"
                    movers.append(f"{display_name} moved {direction} {abs(chg7):.0f}pts → now {regime.score:.0f} ({regime.label})")

    macro_block = "\n".join(macro_lines) if macro_lines else "  Data unavailable"
    movers_block = "\n  ".join(movers) if movers else "None"

    # ── Scenarios with full trade expressions ────────────────────────
    scenario_lines = []
    if state and state.active_scenarios:
        for s in state.active_scenarios:
            scenario_lines.append(f"  {s.conviction} CONVICTION: {s.name} ({s.conditions_met}/{s.conditions_total} conditions)")
            scenario_lines.append(f"    Thesis: {s.thesis}")
            for t in s.trades:
                scenario_lines.append(f"    → {t.direction} {t.ticker} via {t.instrument} ({t.sizing_hint}) — {t.rationale}")
            scenario_lines.append(f"    Primary risk: {s.primary_risk}")
    scenario_block = "\n".join(scenario_lines) if scenario_lines else "  None active"

    # ── Portfolio: aggregates + per-underlying Greeks ─────────────────
    nav = account.get("equity", 0)
    pnl = account.get("pnl", 0)
    pnl_pct = account.get("pnl_pct", 0)
    cash_pct = account.get("cash_pct", 0)

    port_lines = [f"  NAV: ${nav:,.0f}  Fund P&L: ${pnl:+,.0f} ({pnl_pct:+.1f}%)  Cash: {cash_pct:.1f}%"]

    if greeks:
        port_lines.append(f"  Portfolio Greeks: Delta {greeks.net_delta:+.0f}  Gamma {greeks.net_gamma:+.2f}  Theta ${greeks.net_theta:+.0f}/day  Vega {greeks.net_vega:+.0f}")
        theta_bp = abs(greeks.net_theta) / nav * 10000 if nav > 0 else 0
        port_lines.append(f"  Theta burn: ${abs(greeks.net_theta):.0f}/day = {theta_bp:.0f}bp/day of NAV")

        # Per-underlying exposure
        if greeks.by_underlying:
            port_lines.append("\n  Exposure by underlying:")
            for u in greeks.by_underlying:
                port_lines.append(
                    f"    {u.underlying} (${u.underlying_price:,.2f}): "
                    f"delta={u.net_delta:+.0f} gamma={u.net_gamma:+.2f} "
                    f"theta=${u.net_theta:+.0f}/day vega={u.net_vega:+.0f}"
                )

    # Per-position detail
    if positions:
        port_lines.append("\n  Positions:")
        for p in positions:
            sym = str(getattr(p, "symbol", ""))
            qty = _safe_float(getattr(p, "qty", 0))
            pnl_p = _safe_float(getattr(p, "unrealized_pl", 0))
            pnl_pc = _safe_float(getattr(p, "unrealized_plpc", 0)) * 100
            mv = _safe_float(getattr(p, "market_value", 0))
            cost = _safe_float(getattr(p, "avg_entry_price", 0))
            current = _safe_float(getattr(p, "current_price", 0))
            side = getattr(p, "side", "")
            port_lines.append(
                f"    {sym}: {side} qty={qty:+.1f} entry=${cost:.2f} now=${current:.2f} "
                f"mv=${mv:,.0f} P&L=${pnl_p:+,.0f} ({pnl_pc:+.1f}%)"
            )

    if greeks and greeks.risk_signals:
        port_lines.append("\n  Risk signals:")
        for sig in greeks.risk_signals:
            port_lines.append(f"    ⚠ {sig}")

    portfolio_block = "\n".join(port_lines)

    # ── Overall ──────────────────────────────────────────────────────
    overall = ""
    if state:
        overall = f"Risk: {state.overall_risk_score:.0f}/100 ({state.overall_category.upper()})  Quadrant: {state.macro_quadrant}"
        if state.composite:
            c = state.composite
            overall += f"\nComposite Regime: {c.label} (confidence: {c.confidence:.0%})"
            overall += f"\n  {c.description}"

            # Top 3 transition probabilities
            sorted_trans = sorted(c.transition_outlook.items(), key=lambda x: -x[1])
            trans_str = ", ".join(f"{r}: {p:.0%}" for r, p in sorted_trans[:3])
            overall += f"\n  Transition outlook (30d): {trans_str}"

            # Playbook summary
            pb = c.playbook
            overall += (
                f"\n  Canonical playbook: Equity={pb.equity_stance} Duration={pb.duration_stance} "
                f"Credit={pb.credit_stance} Vol={pb.vol_stance} Cash={pb.cash_target_pct:.0f}% "
                f"Gross={pb.gross_exposure}"
            )

            # Top swing factors
            if c.swing_factors:
                from lox.regimes.composite import COMPOSITE_LABELS
                sf_lines = []
                for sf in c.swing_factors[:3]:
                    eta = f" (~{sf.days_to_flip:.0f}d)" if sf.days_to_flip else ""
                    sf_lines.append(
                        f"{sf.pillar.upper()} {sf.current_score:.0f}->{sf.target_score:.0f} "
                        f"for {COMPOSITE_LABELS[sf.target_regime]}{eta}"
                    )
                overall += f"\n  Swing factors: {'; '.join(sf_lines)}"

    # ── Economic Calendar ─────────────────────────────────────────────
    cal_lines = []
    if calendar:
        for ev in calendar:
            marker = "★" if ev.get("high_impact") else " "
            est = f"est {ev['estimate']}" if ev.get("estimate") else ""
            prev = f"prev {ev['previous']}" if ev.get("previous") else ""
            detail = ", ".join(filter(None, [est, prev]))
            cal_lines.append(f"  {marker} {ev['date']} {ev['time']}  {ev['event']}  {detail}")
    calendar_block = "\n".join(cal_lines) if cal_lines else "  No major events in next 5 days"

    return f"""You are the CIO of a $1M startup hedge fund. This is the PM morning note. Your PM reads this before markets open and needs to know exactly what to DO today.
Today: {today}

## Fund Overview
{overall}

## Macro Environment (10 pillars, 0-100 score, higher = more stress)
{macro_block}

## 7-Day Movers (pillars that moved >5 points)
  {movers_block}

## Active Scenarios
{scenario_block}

## Portfolio & Greeks
{portfolio_block}

## Upcoming Data Releases (next 5 days, ★ = high impact)
{calendar_block}

## OUTPUT FORMAT — follow this EXACTLY

**WHAT MOVED** (2-3 sentences)
Name the biggest pillar movers in the last 7 days. Cite the exact score, delta, and the key metric driving it. Example: "Vol spiked +16 to 67 — VIX at 23.8 with contango flipping negative at -7.3%, meaning the term structure is inverted and hedges are getting expensive."

**THE BOOK** (3-5 sentences)
Go position by position through the book. For each position:
- Is the current regime helping or hurting this position? Be specific.
- What's the P&L and is the thesis still intact?
- Give a VERB: hold, add, cut, roll, trail stop, take profit. No ambiguity.
For puts: they PROFIT when the underlying drops. Stress regimes are TAILWINDS for puts.
For calls: they PROFIT when the underlying rises. Risk-on regimes are TAILWINDS for calls.

**ACTION** (2-3 sentences per trade, max 2 trades)
New trade ideas that the regime data is SCREAMING for. For each trade:
- Ticker, direction, instrument (calls/puts/equity/spreads)
- Entry trigger: the specific price level, score threshold, or event that activates this trade. Example: "Enter on VIX > 25 or if CPI prints hot on Wed"
- Sizing: fraction of NAV (e.g., "1-2% of NAV", "starter 0.5%")
- Strike/expiry logic if options: near-the-money vs OTM, front-month vs 45-60 DTE, and why
- The EXACT regime score or metric that supports this trade — cite the number
If a scenario is active, your trades should reference the scenario trade expressions. Do NOT just repeat the scenario trades generically — add the entry trigger, sizing, and strike reasoning.

**RISK** (2 sentences)
The single biggest risk to the book right now.
- Name the specific event, level, or data print that would break the thesis (e.g., "SPX closes above 5,200" or "CPI prints below 3.0%")
- Which position in the book is most exposed and what's the approximate downside

**WATCHING** (1-2 sentences)
The specific catalyst you are watching TODAY or this week.
- Name the exact event and date/time from the calendar above (e.g., "CPI Wed 8:30am — consensus 3.1%")
- State the level or print that would change the view (e.g., "Core below 0.2% MoM flips us to duration-long")
- Which positions are most sensitive to this event
If no major events this week, name the technical level or flow dynamic you're monitoring instead.

## RULES
- Cite EXACT numbers from the data. Not "elevated" — say "VIX at 23.8, z-score +1.4".
- Every position in the book gets a verb. No position left unaddressed.
- Theta burn: if >20bp/day of NAV, it's a problem. Flag it with the exact numbers.
- Scenarios: if active, your trade ideas should align with or explicitly disagree with the scenario trades.
- NEVER use filler phrases: "it's worth noting", "investors should consider", "amidst rising", "given the current environment", "serves as a hedge". Write like a desk note, not a research report.
- NEVER say "monitor" or "keep an eye on" — say exactly what level or print changes the view.
- Max 450 words. Every sentence must contain a number, a ticker, or a verb.
- Take a side. State conviction. Be wrong sometimes."""


def _render_llm_briefing(console: Console, settings: Settings, state, greeks, positions, account: dict, calendar: list | None = None) -> None:
    """Stream LLM CIO briefing."""
    if not getattr(settings, "openai_api_key", None):
        console.print("\n[yellow]Set OPENAI_API_KEY in .env to enable LLM briefing.[/yellow]")
        return

    console.print("\n[bold cyan][4] PM BRIEFING[/bold cyan]")

    system_prompt = _build_pm_system_prompt(state, greeks, positions, account, calendar)

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
            temperature=0.5,
            max_tokens=2000,
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
        calendar = data.get("calendar", [])

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
            _render_llm_briefing(console, settings, state, greeks, positions, account, calendar)
        else:
            console.print("\n[dim]Add --llm (default) for CIO briefing[/dim]")

        console.print()

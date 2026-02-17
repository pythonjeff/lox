"""
Core CLI commands - Clean, elegant, hedge-fund grade.

Design principles:
1. Progressive disclosure - simple by default, advanced when needed
2. Single responsibility - each command does ONE thing well
3. Clear naming - commands describe WHAT, not HOW
4. Sensible defaults - works out of the box

Author: Lox Capital Research
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lox.config import load_settings
from lox.data.alpaca import make_clients


def _safe_float(x) -> float:
    """Safe float conversion with 0.0 default."""
    try:
        return float(x) if x is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def register_core(app: typer.Typer) -> None:
    """Register core commands directly on the main app."""
    
    @app.command("status")
    def status(
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Show position details"),
    ):
        """
        Portfolio health at a glance. Fast, no LLM.
        
        Shows: NAV, cash, positions summary, P&L, risk flags.
        
        Examples:
            lox status
            lox status -v
        """
        console = Console()
        settings = load_settings()
        trading, _ = make_clients(settings)
        
        # Account
        try:
            acct = trading.get_account()
        except Exception as e:
            console.print(f"[red]Connection failed:[/red] {e}")
            raise typer.Exit(1)
        
        mode = "PAPER" if settings.alpaca_paper else "LIVE"
        cash = _safe_float(getattr(acct, "cash", 0))
        equity = _safe_float(getattr(acct, "equity", 0))
        bp = _safe_float(getattr(acct, "buying_power", 0))
        
        # Positions
        positions = trading.get_all_positions()
        n_positions = len(positions)
        
        # Get total capital from investor flows (deposits only) or env var fallback
        total_capital = float(os.environ.get("FUND_TOTAL_CAPITAL", 950))
        try:
            from lox.nav.investors import read_investor_flows
            flows = read_investor_flows()
            if flows:
                total_capital = sum(f.amount for f in flows if float(f.amount) > 0)
        except Exception:
            pass  # Fall back to env var
        
        # Fund-level P&L (NAV - Total Capital)
        fund_pnl = equity - total_capital
        fund_pnl_pct = (fund_pnl / total_capital * 100) if total_capital > 0 else 0.0
        
        # Position-level unrealized P&L (for reference)
        position_pnl = sum(_safe_float(getattr(p, "unrealized_pl", 0)) for p in positions)
        
        # Risk flags
        losers = [p for p in positions if _safe_float(getattr(p, "unrealized_plpc", 0)) < -0.20]
        expiring = []  # Would need option parsing
        
        # Build status panel
        status_lines = [
            f"[bold]NAV:[/bold] ${equity:,.0f}",
            f"[bold]Cash:[/bold] ${cash:,.0f} ({cash/equity*100:.0f}% of NAV)" if equity > 0 else f"[bold]Cash:[/bold] ${cash:,.0f}",
            f"[bold]Positions:[/bold] {n_positions}",
            f"[bold]Fund P&L:[/bold] ${fund_pnl:+,.0f} ({fund_pnl_pct:+.1f}%)",
        ]
        
        if losers:
            status_lines.append(f"[yellow]⚠ {len(losers)} position(s) down >20%[/yellow]")
        
        console.print(Panel(
            "\n".join(status_lines),
            title=f"Lox Portfolio [{mode}]",
            subtitle=datetime.now().strftime("%Y-%m-%d %H:%M"),
            expand=False,
        ))
        
        # Verbose: show positions
        if verbose and positions:
            table = Table(title="Positions", show_header=True)
            table.add_column("Symbol", style="cyan")
            table.add_column("Qty", justify="right")
            table.add_column("Entry", justify="right")
            table.add_column("Current", justify="right")
            table.add_column("P&L", justify="right")
            table.add_column("P&L %", justify="right")
            
            for p in sorted(positions, key=lambda x: _safe_float(getattr(x, "unrealized_pl", 0))):
                sym = str(getattr(p, "symbol", ""))
                qty = _safe_float(getattr(p, "qty", 0))
                entry = _safe_float(getattr(p, "avg_entry_price", 0))
                current = _safe_float(getattr(p, "current_price", 0))
                pnl = _safe_float(getattr(p, "unrealized_pl", 0))
                pnl_pc = _safe_float(getattr(p, "unrealized_plpc", 0)) * 100
                
                pnl_style = "green" if pnl >= 0 else "red"
                
                table.add_row(
                    sym[:20],
                    f"{qty:+.1f}" if abs(qty) < 10 else f"{qty:+.0f}",
                    f"${entry:.2f}",
                    f"${current:.2f}",
                    f"[{pnl_style}]${pnl:+,.0f}[/{pnl_style}]",
                    f"[{pnl_style}]{pnl_pc:+.1f}%[/{pnl_style}]",
                )
            
            console.print(table)
    
    @app.command("report")
    def report(
        period: str = typer.Option("week", "--period", "-p", help="day|week|month"),
    ):
        """
        Generate portfolio report. Clean summary for investors.
        
        Examples:
            lox report
            lox report --period month
        """
        console = Console()
        settings = load_settings()
        trading, _ = make_clients(settings)
        
        # Get current data
        acct = trading.get_account()
        positions = trading.get_all_positions()
        
        mode = "PAPER" if settings.alpaca_paper else "LIVE"
        cash = _safe_float(getattr(acct, "cash", 0))
        equity = _safe_float(getattr(acct, "equity", 0))
        
        # P&L
        total_pnl = sum(_safe_float(getattr(p, "unrealized_pl", 0)) for p in positions)
        
        # Report header
        console.print(Panel(
            f"[bold]Lox Capital {period.title()}ly Report[/bold]\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Mode: {mode}",
            expand=False,
        ))
        
        # NAV Summary
        console.print("\n[bold cyan]Portfolio Summary[/bold cyan]")
        nav_table = Table(show_header=False)
        nav_table.add_column("Metric", style="bold")
        nav_table.add_column("Value", justify="right")
        nav_table.add_row("NAV (Equity)", f"${equity:,.0f}")
        nav_table.add_row("Cash", f"${cash:,.0f}")
        nav_table.add_row("Invested", f"${equity - cash:,.0f}")
        nav_table.add_row("Unrealized P&L", f"${total_pnl:+,.0f}")
        console.print(nav_table)
        
        # Positions
        if positions:
            console.print("\n[bold cyan]Current Positions[/bold cyan]")
            pos_table = Table(show_header=True)
            pos_table.add_column("Symbol", style="cyan")
            pos_table.add_column("Qty", justify="right")
            pos_table.add_column("Value", justify="right")
            pos_table.add_column("P&L", justify="right")
            pos_table.add_column("P&L %", justify="right")
            
            for p in sorted(positions, key=lambda x: -abs(_safe_float(getattr(x, "market_value", 0)))):
                sym = str(getattr(p, "symbol", ""))[:15]
                qty = _safe_float(getattr(p, "qty", 0))
                mv = _safe_float(getattr(p, "market_value", 0))
                pnl = _safe_float(getattr(p, "unrealized_pl", 0))
                pnl_pc = _safe_float(getattr(p, "unrealized_plpc", 0)) * 100
                
                style = "green" if pnl >= 0 else "red"
                pos_table.add_row(
                    sym,
                    f"{qty:+.1f}" if abs(qty) < 10 else f"{qty:+.0f}",
                    f"${abs(mv):,.0f}",
                    f"[{style}]${pnl:+,.0f}[/{style}]",
                    f"[{style}]{pnl_pc:+.1f}%[/{style}]",
                )
            
            console.print(pos_table)
        
        # For weekly report, also run the full report
        if period == "week":
            console.print("\n[dim]For detailed weekly report: lox weekly report[/dim]")
    
    @app.command("analyze")
    def analyze(
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Show all positions"),
    ):
        """
        Position-level risk scan with real tickers and P&L.

        Examples:
            lox analyze
            lox analyze -v
        """
        console = Console()
        settings = load_settings()
        trading, _ = make_clients(settings)

        acct = trading.get_account()
        positions = trading.get_all_positions()
        cash = _safe_float(getattr(acct, "cash", 0))
        equity = _safe_float(getattr(acct, "equity", 0))

        if not positions:
            console.print("[yellow]No open positions.[/yellow]")
            return

        # Classify every position
        values = [abs(_safe_float(getattr(p, "market_value", 0))) for p in positions]
        total_mv = sum(values) or 1.0

        losers, winners, concentrated, expiring_soon = [], [], [], []
        for p in positions:
            sym = str(getattr(p, "symbol", ""))
            pnl = _safe_float(getattr(p, "unrealized_pl", 0))
            pnl_pc = _safe_float(getattr(p, "unrealized_plpc", 0))
            mv = abs(_safe_float(getattr(p, "market_value", 0)))
            weight = mv / total_mv
            entry = _safe_float(getattr(p, "avg_entry_price", 0))
            current = _safe_float(getattr(p, "current_price", 0))

            row = {
                "sym": sym, "pnl": pnl, "pnl_pc": pnl_pc,
                "weight": weight, "entry": entry, "current": current, "mv": mv,
            }

            if pnl_pc < -0.15:
                losers.append(row)
            if pnl_pc > 0.20:
                winners.append(row)
            if weight > 0.25:
                concentrated.append(row)

        # Bleeding positions table
        if losers:
            losers.sort(key=lambda r: r["pnl_pc"])
            t = Table(title="[red]Bleeding Positions (> -15%)[/red]", show_header=True)
            t.add_column("Symbol", style="cyan")
            t.add_column("Entry", justify="right")
            t.add_column("Current", justify="right")
            t.add_column("P&L", justify="right")
            t.add_column("P&L %", justify="right")
            t.add_column("Weight", justify="right")
            t.add_column("Action", style="yellow")
            for r in losers:
                action = "Cut or roll" if r["pnl_pc"] < -0.30 else "Tighten stop"
                t.add_row(
                    r["sym"][:20],
                    f"${r['entry']:.2f}",
                    f"${r['current']:.2f}",
                    f"[red]${r['pnl']:+,.0f}[/red]",
                    f"[red]{r['pnl_pc']*100:+.1f}%[/red]",
                    f"{r['weight']*100:.1f}%",
                    action,
                )
            console.print(t)
        else:
            console.print("[green]No positions bleeding > -15%[/green]")

        # Concentration warnings
        if concentrated:
            lines = []
            for r in concentrated:
                lines.append(f"  [cyan]{r['sym']}[/cyan]: {r['weight']*100:.0f}% of portfolio")
            console.print(Panel(
                "\n".join(lines),
                title="[yellow]Concentration Risk (> 25%)[/yellow]",
                expand=False,
            ))

        # Winners to consider protecting
        if winners:
            t = Table(title="[green]Winners to Protect (> +20%)[/green]", show_header=True)
            t.add_column("Symbol", style="cyan")
            t.add_column("P&L", justify="right")
            t.add_column("P&L %", justify="right")
            t.add_column("Action", style="yellow")
            for r in sorted(winners, key=lambda x: -x["pnl_pc"]):
                action = "Trail stop or sell partial"
                t.add_row(
                    r["sym"][:20],
                    f"[green]${r['pnl']:+,.0f}[/green]",
                    f"[green]{r['pnl_pc']*100:+.1f}%[/green]",
                    action,
                )
            console.print(t)

        # Cash position
        cash_pct = (cash / equity * 100) if equity > 0 else 0
        cash_color = "green" if cash_pct > 15 else "yellow" if cash_pct > 5 else "red"
        console.print(f"\n[bold]Cash:[/bold] [{cash_color}]${cash:,.0f} ({cash_pct:.0f}%)[/{cash_color}]", end="")
        if cash_pct < 5:
            console.print(" — [red]dangerously low, limited ability to average down or hedge[/red]")
        elif cash_pct < 15:
            console.print(" — [yellow]light, consider freeing capital[/yellow]")
        else:
            console.print(" — adequate dry powder")

        if verbose:
            console.print()
            status(verbose=True)

    @app.command("suggest")
    def suggest(
        count: int = typer.Option(5, "--count", "-n", help="Number of ideas"),
    ):
        """
        Trade ideas based on your actual positions and current regime.

        Scans your portfolio for risks, then uses the regime state to
        generate specific hedges, income plays, and directional ideas.

        Examples:
            lox suggest
            lox suggest --count 3
        """
        console = Console()
        settings = load_settings()

        with console.status("[cyan]Loading positions & regime...[/cyan]"):
            trading, _ = make_clients(settings)
            positions = trading.get_all_positions()
            acct = trading.get_account()
            cash = _safe_float(getattr(acct, "cash", 0))
            equity = _safe_float(getattr(acct, "equity", 0))

            try:
                from lox.regimes import build_unified_regime_state
                regime_state = build_unified_regime_state(settings=settings, start_date="2020-01-01")
                overall = regime_state.overall_category
                risk_score = regime_state.overall_risk_score
                vol_label = regime_state.volatility.label if regime_state.volatility else "N/A"
                credit_label = regime_state.credit.label if regime_state.credit else "N/A"
                macro_quad = regime_state.macro_quadrant or "N/A"
            except Exception as e:
                console.print(f"[yellow]Regime data unavailable: {e}[/yellow]")
                overall = "UNKNOWN"
                risk_score = 50
                vol_label = "N/A"
                credit_label = "N/A"
                macro_quad = "N/A"

        # Regime context banner
        risk_color = "red" if risk_score >= 60 else "yellow" if risk_score >= 45 else "green"
        console.print(Panel(
            f"[bold]Regime:[/bold] {overall} ([{risk_color}]{risk_score:.0f}/100[/{risk_color}])\n"
            f"[bold]Macro:[/bold] {macro_quad}  |  "
            f"[bold]Vol:[/bold] {vol_label}  |  "
            f"[bold]Credit:[/bold] {credit_label}",
            title="Market Context",
            expand=False,
            border_style="cyan",
        ))

        # Parse positions into actionable data
        pos_data = []
        values = []
        for p in positions:
            sym = str(getattr(p, "symbol", ""))
            mv = abs(_safe_float(getattr(p, "market_value", 0)))
            values.append(mv)
            pos_data.append({
                "sym": sym,
                "qty": _safe_float(getattr(p, "qty", 0)),
                "mv": mv,
                "pnl_pc": _safe_float(getattr(p, "unrealized_plpc", 0)),
                "pnl": _safe_float(getattr(p, "unrealized_pl", 0)),
                "entry": _safe_float(getattr(p, "avg_entry_price", 0)),
                "current": _safe_float(getattr(p, "current_price", 0)),
            })

        total_mv = sum(values) or 1.0
        for p in pos_data:
            p["weight"] = p["mv"] / total_mv

        ideas = []

        # ── HEDGES: based on actual risk exposure ─────────────────────
        # Big losers: suggest cutting or hedging
        big_losers = [p for p in pos_data if p["pnl_pc"] < -0.15]
        for p in sorted(big_losers, key=lambda x: x["pnl_pc"])[:2]:
            underlying = p["sym"].split(" ")[0] if " " in p["sym"] else p["sym"]
            if p["pnl_pc"] < -0.30:
                ideas.append({
                    "type": "EXIT",
                    "ticker": p["sym"],
                    "action": f"Close position (down {p['pnl_pc']*100:.0f}%)",
                    "rationale": f"Cut loss at ${abs(p['pnl']):,.0f}. Redeploy capital.",
                })
            else:
                ideas.append({
                    "type": "HEDGE",
                    "ticker": underlying,
                    "action": f"Buy protective put or tighten stop",
                    "rationale": f"Down {p['pnl_pc']*100:.0f}%, ${abs(p['pnl']):,.0f} at risk.",
                })

        # Concentrated positions: suggest trimming
        concentrated = [p for p in pos_data if p["weight"] > 0.25]
        for p in concentrated[:2]:
            ideas.append({
                "type": "TRIM",
                "ticker": p["sym"],
                "action": f"Reduce to <25% (currently {p['weight']*100:.0f}%)",
                "rationale": "Concentration risk — single name > 25% of book.",
            })

        # High-risk regime: portfolio-level hedges
        if risk_score >= 55:
            ideas.append({
                "type": "HEDGE",
                "ticker": "SPY",
                "action": "Buy put spread (e.g. 30-45 DTE, -5% strike)",
                "rationale": f"Regime risk elevated ({risk_score:.0f}/100). Tail protection.",
            })

        # ── INCOME: covered calls on winners ──────────────────────────
        winners = [p for p in pos_data if p["pnl_pc"] > 0.10 and p["qty"] > 0]
        for p in sorted(winners, key=lambda x: -x["pnl_pc"])[:2]:
            underlying = p["sym"].split(" ")[0] if " " in p["sym"] else p["sym"]
            if p["qty"] >= 100 or (p["qty"] >= 1 and " " not in p["sym"]):
                ideas.append({
                    "type": "INCOME",
                    "ticker": underlying,
                    "action": f"Sell covered call (up {p['pnl_pc']*100:.0f}%)",
                    "rationale": f"Lock in gains, collect premium. P&L: ${p['pnl']:+,.0f}.",
                })

        # ── DIRECTIONAL: regime-aware ─────────────────────────────────
        if risk_score < 45:
            ideas.append({
                "type": "GROWTH",
                "ticker": "QQQ" if "GROWTH" in macro_quad.upper() else "XLF",
                "action": "Bull call spread or long shares",
                "rationale": f"Low-risk regime ({risk_score:.0f}). Lean into momentum.",
            })
        elif "STAGFLATION" in macro_quad.upper() or "HOT" in macro_quad.upper():
            ideas.append({
                "type": "ROTATE",
                "ticker": "XLE / GLD",
                "action": "Rotate into real assets / commodities",
                "rationale": f"Macro: {macro_quad}. Favor inflation beneficiaries.",
            })

        # Low cash warning
        cash_pct = (cash / equity * 100) if equity > 0 else 0
        if cash_pct < 10 and big_losers:
            ideas.append({
                "type": "FREE $",
                "ticker": big_losers[0]["sym"],
                "action": f"Close weakest position to raise cash",
                "rationale": f"Cash at {cash_pct:.0f}%. Need dry powder to act on ideas above.",
            })

        # Render ideas table
        if ideas:
            t = Table(title=f"Trade Ideas ({len(ideas)})", show_header=True)
            t.add_column("Type", style="bold cyan", width=8)
            t.add_column("Ticker", style="bold", width=22)
            t.add_column("Action", width=42)
            t.add_column("Rationale", style="dim")
            for idea in ideas[:count]:
                type_style = {
                    "EXIT": "[red]EXIT[/red]",
                    "HEDGE": "[yellow]HEDGE[/yellow]",
                    "TRIM": "[yellow]TRIM[/yellow]",
                    "INCOME": "[green]INCOME[/green]",
                    "GROWTH": "[cyan]GROWTH[/cyan]",
                    "ROTATE": "[magenta]ROTATE[/magenta]",
                    "FREE $": "[red]FREE $[/red]",
                }.get(idea["type"], idea["type"])
                t.add_row(
                    type_style,
                    idea["ticker"][:22],
                    idea["action"],
                    idea["rationale"][:60],
                )
            console.print(t)
        else:
            console.print("[green]Portfolio looks clean — no immediate action items.[/green]")

        console.print(f"\n[dim]Run [bold]lox analyze[/bold] for full position breakdown, "
                      f"[bold]lox regimes unified[/bold] for regime detail.[/dim]")

    @app.command("run")
    def run():
        """
        Morning briefing: portfolio status + risk scan + trade ideas.

        Combines status, analyze, and suggest into one clean output.

        Examples:
            lox run
        """
        console = Console()
        console.rule("[bold cyan]Lox Morning Briefing[/bold cyan]")

        # Step 1: Status
        console.print()
        status(verbose=True)

        # Step 2: Risk scan
        console.print()
        console.rule("[bold cyan]Risk Scan[/bold cyan]")
        analyze(verbose=False)

        # Step 3: Trade ideas
        console.print()
        console.rule("[bold cyan]Trade Ideas[/bold cyan]")
        suggest(count=5)

        console.print()
        console.rule("[dim]End of briefing[/dim]")

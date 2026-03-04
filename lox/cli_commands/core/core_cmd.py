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
        benchmark: str = typer.Option("SPY", "--benchmark", "-b", help="Benchmark for correlation (e.g. SPY)"),
        no_correlation: bool = typer.Option(False, "--no-correlation", help="Skip correlation fetch (faster)"),
        deep: bool = typer.Option(False, "--deep", help="Run per-candidate Monte Carlo (slower, ~15-20s)"),
    ):
        """
        Quant-level trade ideas: ticker + direction scored by playbook, scenarios & correlation.

        Scores candidates using regime-conditioned k-NN analog returns, active
        macro scenarios, and correlation vs benchmark. Portfolio-aware: penalises
        concentration and prefers diversifying exposure.

        Examples:
            lox suggest
            lox suggest --deep --count 8
            lox suggest --benchmark QQQ --count 5
        """
        from rich.text import Text

        _SPARK = "▁▂▃▄▅▆▇█"

        def _mini_spark(values):
            if not values or len(values) < 2:
                return ""
            lo, hi = min(values), max(values)
            span = hi - lo if hi != lo else 1.0
            return "".join(
                _SPARK[min(len(_SPARK) - 1, int((v - lo) / span * (len(_SPARK) - 1)))]
                for v in values[-20:]
            )

        console = Console()
        settings = load_settings()

        mode_label = "[bold magenta]DEEP[/bold magenta] " if deep else ""
        with console.status(f"[cyan]{mode_label}Scoring candidates (playbook + scenarios + correlation)...[/cyan]"):
            trading, _ = make_clients(settings)
            positions = trading.get_all_positions()

            try:
                from lox.regimes import build_unified_regime_state
                from lox.suggest import suggest_cross_asset

                regime_state = build_unified_regime_state(settings=settings, start_date="2020-01-01")
                overall = regime_state.overall_category
                risk_score = regime_state.overall_risk_score
                vol_label = regime_state.volatility.label if regime_state.volatility else "N/A"
                credit_label = regime_state.credit.label if regime_state.credit else "N/A"
                macro_quad = regime_state.macro_quadrant or "N/A"

                scored = suggest_cross_asset(
                    regime_state=regime_state,
                    positions=positions,
                    benchmark=benchmark,
                    count=count,
                    settings=settings,
                    use_correlation=not no_correlation,
                    deep=deep,
                )
            except Exception as e:
                console.print(f"[yellow]Regime/suggest failed: {e}[/yellow]")
                import traceback
                traceback.print_exc()
                overall = "UNKNOWN"
                risk_score = 50
                vol_label = "N/A"
                credit_label = "N/A"
                macro_quad = "N/A"
                scored = []

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

        if not scored:
            console.print("[green]No macro-aligned ideas — portfolio may already cover regime.[/green]")
            return

        # Main conviction table
        t = Table(title=f"Trade Ideas ({len(scored)})", show_header=True, expand=True)
        t.add_column("#", style="dim", width=3, justify="right")
        t.add_column("Dir", style="bold", width=6)
        t.add_column("Ticker", style="bold", width=7)
        t.add_column("Score", width=7, justify="right")
        t.add_column("Conv", width=6)
        t.add_column("E[R]", width=7, justify="right")
        t.add_column("Hit%", width=6, justify="right")
        t.add_column("VaR₅", width=7, justify="right")
        t.add_column("Analogs", width=8, justify="right")
        if deep:
            t.add_column("MC Med", width=7, justify="right")
        t.add_column("Thesis", ratio=1)

        for i, s in enumerate(scored, 1):
            dir_style = "[green]LONG[/green]" if s.direction == "LONG" else "[red]SHORT[/red]"

            # Score color
            if s.composite_score >= 70:
                sc_str = f"[bold green]{s.composite_score:.0f}[/bold green]"
            elif s.composite_score >= 45:
                sc_str = f"[yellow]{s.composite_score:.0f}[/yellow]"
            else:
                sc_str = f"[dim]{s.composite_score:.0f}[/dim]"

            # Conviction badge
            conv_map = {"HIGH": "[bold green]HIGH[/bold green]", "MEDIUM": "[yellow]MED[/yellow]", "LOW": "[dim]LOW[/dim]"}
            conv_str = conv_map.get(s.conviction, s.conviction)

            # Expected return (negate for SHORT — profit when asset drops)
            er = s.exp_return * 100
            if s.direction == "SHORT":
                er = -er
            er_str = f"[green]+{er:.1f}%[/green]" if er > 0 else f"[red]{er:.1f}%[/red]" if er < 0 else "0.0%"

            # Hit rate
            hr_str = f"{s.hit_rate * 100:.0f}%" if s.n_analogs > 0 else "—"

            # VaR
            var_val = s.var_5 * 100
            var_str = f"[red]{var_val:.1f}%[/red]" if var_val < 0 else f"{var_val:.1f}%"

            # Analog count + confidence
            if s.n_analogs >= 100:
                an_str = f"[green]{s.n_analogs}[/green]"
            elif s.n_analogs >= 50:
                an_str = f"[yellow]{s.n_analogs}[/yellow]"
            elif s.n_analogs > 0:
                an_str = f"[dim]{s.n_analogs}[/dim]"
            else:
                an_str = "—"

            row = [str(i), dir_style, s.ticker, sc_str, conv_str, er_str, hr_str, var_str, an_str]

            if deep:
                if s.mc_median_return is not None:
                    mc_val = s.mc_median_return * 100
                    mc_str = f"[green]+{mc_val:.1f}%[/green]" if mc_val > 0 else f"[red]{mc_val:.1f}%[/red]"
                else:
                    mc_str = "—"
                row.append(mc_str)

            thesis_short = (s.thesis or "")[:55]
            row.append(f"[dim]{thesis_short}[/dim]")

            t.add_row(*row)

        console.print(t)

        # Score breakdown for top pick
        top = scored[0]
        console.print(Panel(
            f"[bold]Playbook:[/bold] {top.playbook_score:.0f}  |  "
            f"[bold]Scenario:[/bold] {top.scenario_score:.0f}  |  "
            f"[bold]Correlation:[/bold] {top.correlation_score:.0f}"
            + (f"  |  [bold]MC:[/bold] {top.mc_score:.0f}" if deep else "")
            + f"\n[bold]Sharpe est:[/bold] {top.sharpe_est:.2f}  |  "
            f"[bold]Source:[/bold] {top.source}",
            title=f"Top Pick: {top.ticker} {top.direction}",
            expand=False,
            border_style="green" if top.conviction == "HIGH" else "yellow",
        ))

        # Active scenarios callout
        try:
            from lox.regimes.scenarios import evaluate_scenarios, SCENARIOS
            active = evaluate_scenarios(regime_state, SCENARIOS) if 'regime_state' in dir() else []
            if active:
                lines = []
                for s in active[:3]:
                    lines.append(f"[bold]{s.name}[/bold] ({s.conviction}) — {s.conditions_met}/{s.conditions_total} conditions")
                    lines.append(f"  [dim]Risk: {s.primary_risk[:70]}[/dim]")
                console.print(Panel(
                    "\n".join(lines),
                    title="Active Macro Scenarios",
                    expand=False,
                    border_style="cyan",
                ))
        except Exception:
            pass

        mode_note = " [magenta]--deep[/magenta] MC overlay active" if deep else " Add [bold]--deep[/bold] for Monte Carlo overlay"
        console.print(f"\n[dim]Benchmark: {benchmark} |{mode_note} | [bold]lox regime unified[/bold] for regime detail.[/dim]")

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
        suggest(count=3)

        console.print()
        console.rule("[dim]End of briefing[/dim]")

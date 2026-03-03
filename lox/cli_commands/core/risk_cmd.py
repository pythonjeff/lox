"""CLI command for the Portfolio Greeks Risk Dashboard.

Displays consolidated portfolio-level Greeks (delta, gamma, theta, vega),
per-underlying exposure decomposition, position-level detail, and
theta breakeven analysis.

Author: Lox Capital Research
"""
from __future__ import annotations

import json
import math

from rich.console import Console
from rich.panel import Panel
from rich.table import Table as RichTable

from lox.config import load_settings


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _greek_color(value: float, threshold: float = 0) -> str:
    """Color a Greek value: green for favorable, red for risky."""
    if value > threshold:
        return f"[green]{value:+.1f}[/green]"
    elif value < -threshold:
        return f"[red]{value:+.1f}[/red]"
    return f"[dim]{value:+.1f}[/dim]"


def _dollar(value: float) -> str:
    """Format a dollar value."""
    if abs(value) >= 1000:
        return f"${value:+,.0f}"
    return f"${value:+,.2f}"


def _delta_label(delta: float) -> str:
    """Human-readable delta interpretation."""
    if abs(delta) < 20:
        return "[green]neutral[/green]"
    if delta > 200:
        return "[yellow]strongly long[/yellow]"
    if delta > 0:
        return "[dim]net long[/dim]"
    if delta < -200:
        return "[yellow]strongly short[/yellow]"
    return "[dim]net short[/dim]"


def _theta_label(theta: float) -> str:
    """Human-readable theta interpretation."""
    if theta > 20:
        return "[green]earning from decay[/green]"
    if theta < -20:
        return "[yellow]bleeding decay[/yellow]"
    return "[dim]minimal[/dim]"


def _vega_label(vega: float) -> str:
    """Human-readable vega interpretation."""
    if vega > 50:
        return "[dim]long vol[/dim]"
    if vega < -50:
        return "[yellow]short vol[/yellow]"
    return "[dim]minimal[/dim]"


# ─────────────────────────────────────────────────────────────────────────────
# Display blocks
# ─────────────────────────────────────────────────────────────────────────────

def _show_account_header(console: Console, pg) -> None:
    """Show account snapshot."""
    opt_bp = f"${pg.options_bp:,.0f}" if pg.options_bp is not None else "n/a"
    lines = (
        f"[b]Equity:[/b]       ${pg.account_equity:,.0f}\n"
        f"[b]Buying Power:[/b] ${pg.buying_power:,.0f}\n"
        f"[b]Options BP:[/b]   {opt_bp}"
    )
    console.print(Panel.fit(lines, title="Account", border_style="cyan"))


def _show_greeks_summary(console: Console, pg) -> None:
    """Show portfolio-level Greeks in a compact panel."""
    lines = (
        f"  Net Delta:    [bold]{pg.net_delta:+.0f}[/bold]     {_delta_label(pg.net_delta)}\n"
        f"  Net Gamma:    [bold]{pg.net_gamma:+.2f}[/bold]\n"
        f"  Daily Theta:  [bold]{_dollar(pg.net_theta)}[/bold]/day  {_theta_label(pg.net_theta)}\n"
        f"  Net Vega:     [bold]{pg.net_vega:+.0f}[/bold]     {_vega_label(pg.net_vega)}"
    )
    console.print()
    console.print(Panel.fit(lines, title="Portfolio Greeks", border_style="bold cyan"))


def _show_underlying_exposure(console: Console, pg) -> None:
    """Show per-underlying delta decomposition."""
    if not pg.by_underlying:
        return

    table = RichTable(
        title="Exposure by Underlying",
        show_header=True,
        header_style="bold yellow",
        box=None,
        padding=(0, 1),
    )
    table.add_column("Underlying", style="bold", min_width=8)
    table.add_column("Equity Δ", justify="right")
    table.add_column("Options Δ", justify="right")
    table.add_column("Net Δ", justify="right")
    table.add_column("Gamma", justify="right")
    table.add_column("Theta/day", justify="right")
    table.add_column("Vega", justify="right")

    for u in pg.by_underlying:
        eq_d = f"{u.equity_delta:+.0f}" if u.equity_delta != 0 else "[dim]—[/dim]"
        opt_d = f"{u.options_delta:+.0f}" if u.options_delta != 0 else "[dim]—[/dim]"

        # Color net delta
        nd = u.net_delta
        if abs(nd) > 100:
            nd_str = f"[bold]{nd:+.0f}[/bold]"
        else:
            nd_str = f"{nd:+.0f}"

        g_str = f"{u.net_gamma:+.2f}" if u.net_gamma != 0 else "[dim]—[/dim]"
        t_str = _dollar(u.net_theta) if u.net_theta != 0 else "[dim]—[/dim]"
        v_str = f"{u.net_vega:+.0f}" if u.net_vega != 0 else "[dim]—[/dim]"

        table.add_row(u.underlying, eq_d, opt_d, nd_str, g_str, t_str, v_str)

    console.print()
    console.print(table)


def _show_position_detail(console: Console, pg) -> None:
    """Show every position with its Greek contribution."""
    if not pg.positions:
        return

    table = RichTable(
        title="Position Detail",
        show_header=True,
        header_style="bold yellow",
        box=None,
        padding=(0, 1),
    )
    table.add_column("Position", min_width=18)
    table.add_column("Qty", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("γ", justify="right")
    table.add_column("θ/day", justify="right")
    table.add_column("ν", justify="right")
    table.add_column("IV", justify="right", style="dim")
    table.add_column("P/L", justify="right")

    # Sort: equities first, then options grouped by underlying
    sorted_positions = sorted(
        pg.positions,
        key=lambda p: (
            0 if p.position_type in ("equity", "short_equity") else 1,
            p.underlying,
            p.symbol,
        ),
    )

    for p in sorted_positions:
        qty_str = f"{p.qty:+.0f}" if p.qty == int(p.qty) else f"{p.qty:+.1f}"

        # Delta
        if p.delta != 0:
            d_str = f"{p.delta:+.0f}"
        else:
            d_str = "[dim]—[/dim]"

        # Gamma, theta, vega (skip for equity)
        is_equity = p.position_type in ("equity", "short_equity")
        g_str = "[dim]—[/dim]" if is_equity else (f"{p.gamma:+.2f}" if p.gamma != 0 else "[dim]0[/dim]")
        t_str = "[dim]—[/dim]" if is_equity else (_dollar(p.theta) if p.theta != 0 else "[dim]$0[/dim]")
        v_str = "[dim]—[/dim]" if is_equity else (f"{p.vega:+.0f}" if p.vega != 0 else "[dim]0[/dim]")

        # IV
        iv_str = f"{p.iv:.0%}" if p.iv is not None else "[dim]—[/dim]"

        # P/L colored
        pl = p.unrealized_pl
        if pl > 0:
            pl_str = f"[green]{_dollar(pl)}[/green]"
        elif pl < 0:
            pl_str = f"[red]{_dollar(pl)}[/red]"
        else:
            pl_str = "[dim]$0[/dim]"

        table.add_row(
            p.display_name,
            qty_str,
            d_str,
            g_str,
            t_str,
            v_str,
            iv_str,
            pl_str,
        )

    console.print()
    console.print(table)


def _show_risk_signals(console: Console, pg) -> None:
    """Show auto-generated risk warnings."""
    if not pg.risk_signals:
        return

    console.print()
    console.print("[dim]─── Risk Signals ──────────────────────────────────────────────[/dim]")
    for signal in pg.risk_signals:
        console.print(f"  {signal}")


# ─────────────────────────────────────────────────────────────────────────────
# Theta breakeven analysis
# ─────────────────────────────────────────────────────────────────────────────

def _show_theta_breakeven(console: Console, pg) -> None:
    """Show theta breakeven analysis: how much does the market need to move to cover decay."""
    # Skip if no theta (no options, or theta-neutral)
    if pg.net_theta == 0:
        return

    # ── Per-underlying breakeven table ─────────────────────────────────
    # Only show underlyings that have theta exposure
    theta_underlyings = [u for u in pg.by_underlying if u.net_theta != 0]
    if not theta_underlyings:
        return

    console.print()
    table = RichTable(
        title="Theta Breakeven by Underlying",
        show_header=True,
        header_style="bold yellow",
        box=None,
        padding=(0, 1),
    )
    table.add_column("Underlying", style="bold", min_width=6)
    table.add_column("Price", justify="right")
    table.add_column("θ/day", justify="right")
    table.add_column("Δ B/E", justify="right")
    table.add_column("Δ B/E %", justify="right")
    table.add_column("γ Scalp %", justify="right")
    table.add_column("Status")

    for u in theta_underlyings:
        price_str = f"${u.underlying_price:,.0f}" if u.underlying_price > 0 else "[dim]—[/dim]"
        theta_str = _dollar(u.net_theta)
        abs_delta = abs(u.net_delta)

        # Delta breakeven: $ move needed to cover theta via delta P/L
        if u.net_theta < 0 and abs_delta > 0:
            be_dollar = abs(u.net_theta) / abs_delta
            be_dollar_str = f"${be_dollar:.2f}"
            if u.underlying_price > 0:
                be_pct = be_dollar / u.underlying_price * 100
                be_pct_str = f"{be_pct:.2f}%"
            else:
                be_pct_str = "[dim]—[/dim]"
        elif u.net_theta >= 0:
            be_dollar_str = "[green]earning[/green]"
            be_pct_str = "[green]—[/green]"
        else:
            be_dollar_str = "[dim]—[/dim]"
            be_pct_str = "[dim]—[/dim]"

        # Gamma scalp breakeven: daily move where gamma P/L = theta cost
        # P/L_gamma ≈ 0.5 × gamma × move² => move = sqrt(2 × |theta| / gamma)
        if u.net_theta < 0 and u.net_gamma > 0 and u.underlying_price > 0:
            gamma_be_move = math.sqrt(2 * abs(u.net_theta) / u.net_gamma)
            gamma_be_pct = gamma_be_move / u.underlying_price * 100
            gamma_be_str = f"{gamma_be_pct:.1f}%"
        elif u.net_theta >= 0:
            gamma_be_str = "[green]—[/green]"
        else:
            gamma_be_str = "[dim]n/a[/dim]"

        # Status label
        if u.net_theta > 0:
            status = "[green]earning[/green]"
        elif abs_delta == 0 and u.net_gamma <= 0:
            status = "[red]pure bleed[/red]"
        elif u.net_gamma > 0:
            status = "[dim]long γ[/dim]"
        else:
            status = "[yellow]needs move[/yellow]"

        table.add_row(
            u.underlying, price_str, theta_str,
            be_dollar_str, be_pct_str, gamma_be_str,
            status,
        )

    console.print(table)

    # ── Portfolio-level theta burn summary ─────────────────────────────
    daily = pg.net_theta
    weekly = daily * 5
    monthly = daily * 21
    annual = daily * 252

    lines = []

    # Time horizon projections
    if daily < 0:
        lines.append(f"  [b]Daily:[/b]     [red]{_dollar(daily)}[/red]")
        lines.append(f"  [b]Weekly:[/b]    [red]{_dollar(weekly)}[/red]  [dim](5 trading days)[/dim]")
        lines.append(f"  [b]Monthly:[/b]   [red]{_dollar(monthly)}[/red]  [dim](21 trading days)[/dim]")
    else:
        lines.append(f"  [b]Daily:[/b]     [green]{_dollar(daily)}[/green]")
        lines.append(f"  [b]Weekly:[/b]    [green]{_dollar(weekly)}[/green]  [dim](5 trading days)[/dim]")
        lines.append(f"  [b]Monthly:[/b]   [green]{_dollar(monthly)}[/green]  [dim](21 trading days)[/dim]")

    # Theta as % of equity
    if pg.account_equity > 0:
        annual_pct = abs(annual) / pg.account_equity * 100
        if daily < 0:
            lines.append(f"  [b]Annual:[/b]    [red]{_dollar(annual)}[/red]  = [bold]{annual_pct:.0f}% of equity[/bold]")
        else:
            lines.append(f"  [b]Annual:[/b]    [green]{_dollar(annual)}[/green]  = [bold]{annual_pct:.0f}% of equity[/bold]")

    lines.append("")

    # Portfolio-level delta breakeven
    if daily < 0 and abs(pg.net_delta) > 0:
        portfolio_be = abs(daily) / abs(pg.net_delta)
        direction = "in your favor" if pg.net_delta != 0 else ""
        if pg.net_delta < 0:
            direction = "↓ (your delta is short)"
        else:
            direction = "↑ (your delta is long)"
        lines.append(
            f"  [b]Delta breakeven:[/b] market needs to move "
            f"[bold]${portfolio_be:.2f}/day[/bold] {direction}"
        )

    # Portfolio-level gamma scalp breakeven
    if daily < 0 and pg.net_gamma > 0:
        gamma_be = math.sqrt(2 * abs(daily) / pg.net_gamma)
        lines.append(
            f"  [b]Gamma scalp B/E:[/b] needs [bold]${gamma_be:.2f}[/bold] daily move "
            f"for gamma P/L to cover theta"
        )

    # Days until 1% of equity is consumed by theta
    if daily < 0 and pg.account_equity > 0:
        one_pct = pg.account_equity * 0.01
        days_to_1pct = one_pct / abs(daily)
        lines.append(
            f"  [b]1% equity burn:[/b] theta costs 1% of account every "
            f"[bold]{days_to_1pct:.0f} trading days[/bold]"
        )

    console.print()
    console.print(Panel.fit("\n".join(lines), title="Theta Burn Analysis", border_style="bold yellow"))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def risk_dashboard(
    *,
    refresh: bool = False,
    json_out: bool = False,
) -> None:
    """Entry point for ``lox risk``."""
    settings = load_settings()
    console = Console()

    console.print("[dim]Fetching positions and Greeks...[/dim]")

    from lox.risk.greeks import compute_portfolio_greeks

    try:
        pg = compute_portfolio_greeks(settings)
    except Exception as e:
        console.print(f"[red]Error computing portfolio Greeks: {e}[/red]")
        return

    # ── JSON export ───────────────────────────────────────────────────
    if json_out:
        console.print_json(json.dumps(pg.to_dict(), default=str))
        return

    # ── Dashboard display ─────────────────────────────────────────────
    console.print()
    console.print(f"[bold cyan]LOX RISK DASHBOARD[/bold cyan]  [dim]{pg.asof}[/dim]")
    console.print()

    _show_account_header(console, pg)
    _show_greeks_summary(console, pg)
    _show_underlying_exposure(console, pg)
    _show_position_detail(console, pg)
    _show_risk_signals(console, pg)
    _show_theta_breakeven(console, pg)

    console.print()

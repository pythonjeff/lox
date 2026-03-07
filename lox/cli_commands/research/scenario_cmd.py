"""
LOX Research: Macro Scenario Simulator

Simulate the impact of macro shocks on any ticker using block-bootstrap
Monte Carlo with shock-to-MC translation.

Usage:
    lox research scenario SPY --oil 90 --cpi 4.5 --horizon 90
    lox research scenario QQQ --vix 35 --10y 5.5 --oil 110
    lox research scenario SPY --oil 90 --cpi 4.5 --put 630 --expiry 2026-06-18
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from lox.config import load_settings


def register(app: typer.Typer) -> None:
    """Register the scenario command."""

    @app.command("scenario")
    def scenario_cmd(
        symbol: str = typer.Argument(..., help="Ticker symbol (e.g., SPY, QQQ, XLE)"),
        oil: float | None = typer.Option(None, "--oil", help="WTI crude target price (e.g. 90)"),
        cpi: float | None = typer.Option(None, "--cpi", help="CPI YoY %% target (e.g. 4.5)"),
        vix: float | None = typer.Option(None, "--vix", help="VIX level target (e.g. 35)"),
        ten_y: float | None = typer.Option(None, "--10y", help="10Y Treasury yield target (e.g. 5.5)"),
        fed_funds: float | None = typer.Option(None, "--fed-funds", help="Fed funds rate target (e.g. 6.0)"),
        hy_spread: float | None = typer.Option(None, "--hy-spread", help="HY OAS target in bps (e.g. 600)"),
        dxy: float | None = typer.Option(None, "--dxy", help="Dollar index target (e.g. 110)"),
        gold: float | None = typer.Option(None, "--gold", help="Gold price target (e.g. 2500)"),
        horizon: int = typer.Option(90, "--horizon", help="Simulation horizon in calendar days"),
        sims: int = typer.Option(10_000, "--sims", "-n", help="Number of Monte Carlo simulations"),
        # Position flags
        put: float | None = typer.Option(None, "--put", help="Hypothetical put strike (e.g. 630)"),
        call: float | None = typer.Option(None, "--call", help="Hypothetical call strike (e.g. 700)"),
        expiry: str | None = typer.Option(None, "--expiry", help="Option expiry date (YYYY-MM-DD)"),
        qty: float = typer.Option(1, "--qty", help="Number of contracts for hypothetical position"),
        no_positions: bool = typer.Option(False, "--no-positions", help="Skip position auto-detection"),
    ):
        """
        Macro scenario simulator — what happens to a ticker under macro shocks.

        Specify target values for macro variables (not deltas). The engine
        fetches current values from FRED, computes the shock magnitude,
        translates to Monte Carlo parameter adjustments, and runs
        block-bootstrap simulation.

        Auto-detects open option positions from Alpaca. Use --put/--call
        with --expiry to test hypothetical positions instead.

        Examples:
            lox research scenario SPY --oil 90 --cpi 4.5
            lox research scenario SPY --oil 90 --put 630 --expiry 2026-06-18
            lox research scenario QQQ --vix 35 --10y 5.5
            lox research scenario XLE --oil 110 --horizon 60 --no-positions
        """
        console = Console()
        settings = load_settings()
        symbol = symbol.upper()

        if not settings.FRED_API_KEY:
            console.print("[red]Error:[/red] FRED_API_KEY required for scenario analysis")
            raise typer.Exit(1)

        if not settings.FMP_API_KEY:
            console.print("[red]Error:[/red] FMP_API_KEY required for price data")
            raise typer.Exit(1)

        # Build macro targets from CLI options
        macro_targets: dict[str, float] = {}
        if oil is not None:
            macro_targets["oil"] = oil
        if cpi is not None:
            macro_targets["cpi"] = cpi
        if vix is not None:
            macro_targets["vix"] = vix
        if ten_y is not None:
            macro_targets["10y"] = ten_y
        if fed_funds is not None:
            macro_targets["fed_funds"] = fed_funds
        if hy_spread is not None:
            macro_targets["hy_spread"] = hy_spread
        if dxy is not None:
            macro_targets["dxy"] = dxy
        if gold is not None:
            macro_targets["gold"] = gold

        if not macro_targets:
            console.print("[red]Error:[/red] Specify at least one macro variable (e.g. --oil 90, --vix 35)")
            console.print("[dim]See lox research scenario --help for all options[/dim]")
            raise typer.Exit(1)

        console.print()

        # Show inline reminder of last prediction for this symbol
        _show_last_prediction(console, symbol)

        # Resolve positions (auto-detect or manual)
        positions = _resolve_positions(
            console, settings, symbol,
            put_strike=put, call_strike=call, expiry=expiry, qty=qty,
            no_positions=no_positions,
        )

        # Run the scenario engine
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Running scenario analysis...[/bold cyan]"),
            transient=True,
        ) as progress:
            progress.add_task("scenario", total=None)

            try:
                from lox.scenarios.engine import run_scenario_analysis

                result = run_scenario_analysis(
                    settings=settings,
                    symbol=symbol,
                    macro_targets=macro_targets,
                    horizon_days=horizon,
                    n_sims=sims,
                    positions=positions,
                )
            except Exception as exc:
                console.print(f"[red]Scenario analysis failed:[/red] {exc}")
                raise typer.Exit(1)

        console.print(
            f"[bold cyan]LOX SCENARIO ANALYSIS[/bold cyan]  [bold]{symbol}[/bold]  "
            f"[bold]${result.current_price:,.2f}[/bold]  "
            f"[dim]({horizon}-day horizon, {sims:,} simulations)[/dim]"
        )
        console.print()

        _show_assumptions(console, result)
        _show_mc_adjustments(console, result)
        _show_price_distribution(console, result)
        _show_exit_analysis(console, result)
        _show_risk_metrics(console, result)
        _show_histogram(console, result)
        _show_multi_horizon(console, result)
        _show_factor_sensitivity(console, result)
        _show_analog_overlay(console, result)
        _show_percentile_strip(console, result)

        # Auto-log this prediction
        _auto_log(result, positions)

        console.print()

    return


# ── Position resolution ──────────────────────────────────────────────────


def _resolve_positions(
    console: Console,
    settings,
    symbol: str,
    put_strike: float | None,
    call_strike: float | None,
    expiry: str | None,
    qty: float,
    no_positions: bool,
) -> list[dict]:
    """Build the positions list from flags or auto-detection."""
    from datetime import date, timedelta

    positions: list[dict] = []

    # Manual hypothetical positions
    if put_strike is not None or call_strike is not None:
        if expiry is None:
            exp_date = date.today() + timedelta(days=90)
            expiry = exp_date.isoformat()

        if put_strike is not None:
            strike_str = f"{put_strike:.0f}" if put_strike == int(put_strike) else f"{put_strike}"
            positions.append({
                "display_name": f"{symbol} {strike_str}P {expiry}",
                "opt_type": "put",
                "strike": put_strike,
                "expiry": expiry,
                "qty": qty,
                "current_value": 0.0,
                "symbol": f"{symbol}_HYPO_P",
            })

        if call_strike is not None:
            strike_str = f"{call_strike:.0f}" if call_strike == int(call_strike) else f"{call_strike}"
            positions.append({
                "display_name": f"{symbol} {strike_str}C {expiry}",
                "opt_type": "call",
                "strike": call_strike,
                "expiry": expiry,
                "qty": qty,
                "current_value": 0.0,
                "symbol": f"{symbol}_HYPO_C",
            })

        for p in positions:
            cp = "P" if p["opt_type"] == "put" else "C"
            console.print(
                f"  [dim]Hypothetical:[/dim] [bold]{int(qty)}x {symbol} "
                f"{p['strike']:.0f}{cp} {expiry}[/bold]"
            )
        return positions

    # Auto-detect from Alpaca
    if no_positions:
        return []

    try:
        from lox.scenarios.engine import detect_option_positions

        detected = detect_option_positions(settings, symbol)
        if detected:
            for p in detected:
                cp = "P" if p["opt_type"] == "put" else "C"
                console.print(
                    f"  [dim]Detected:[/dim] [bold]{int(p['qty'])}x "
                    f"{p['display_name']}[/bold]  "
                    f"[dim](${p['current_value']:.2f}/contract)[/dim]"
                )
            return detected
    except Exception:
        pass

    return []


# ── Inline last prediction ───────────────────────────────────────────────


def _show_last_prediction(console: Console, symbol: str) -> None:
    """Show a one-liner about the last prediction for this symbol."""
    try:
        from lox.scenarios.tracking import get_last_prediction
        from datetime import date

        last = get_last_prediction(symbol)
        if last is None:
            return

        ts = last["timestamp"][:10]
        price = last["current_price"]
        p50 = last["full_distribution"]["p50"]
        days = last["horizon_days"]

        end_date = date.fromisoformat(last["horizon_end_date"])
        days_left = (end_date - date.today()).days

        if last.get("actual_price") is not None:
            actual = last["actual_price"]
            pct = last.get("actual_percentile", "?")
            scenario = last.get("actual_scenario", "?")
            scenario_label = scenario.replace("_", " ") if scenario else "?"
            console.print(
                f"  [dim]Previous: {ts} {symbol} ${price:,.0f} → "
                f"predicted p50 ${p50:,.0f}, actual ${actual:,.0f} (p{pct:.0f})[/dim]  "
                f"[green]✓ {scenario_label}[/green]"
            )
        elif days_left > 0:
            console.print(
                f"  [dim]Previous: {ts} {symbol} ${price:,.0f} → "
                f"predicted p50 ${p50:,.0f} ({days}d)[/dim]  "
                f"[yellow]⏳ {days_left} days left[/yellow]"
            )
        else:
            console.print(
                f"  [dim]Previous: {ts} {symbol} ${price:,.0f} → "
                f"predicted p50 ${p50:,.0f} ({days}d)[/dim]  "
                f"[cyan]⏳ awaiting scoring[/cyan]"
            )

        console.print()
    except Exception:
        pass


# ── Auto-log ─────────────────────────────────────────────────────────────


def _auto_log(result, positions: list[dict]) -> None:
    """Silently log this prediction."""
    try:
        from lox.scenarios.tracking import save_prediction

        pos_records = []
        for p in (positions or []):
            pos_records.append({
                "symbol": p.get("symbol", ""),
                "display_name": p.get("display_name", ""),
                "opt_type": p.get("opt_type", ""),
                "strike": p.get("strike", 0),
                "expiry": p.get("expiry", ""),
                "qty": p.get("qty", 0),
                "current_value": p.get("current_value", 0),
            })

        save_prediction(result, pos_records)
    except Exception:
        pass


# ── Display helpers ───────────────────────────────────────────────────────


def _fmt_shock(assumption) -> str:
    """Format a shock value with sign and percentage."""
    unit = assumption.unit
    shock = assumption.shock_abs
    pct = assumption.shock_pct
    sign = "+" if shock >= 0 else ""

    if unit == "$":
        return f"{sign}${shock:,.2f} ({sign}{pct:.0f}%)"
    elif unit == "%":
        return f"{sign}{shock:.1f}pp"
    elif unit == "bps":
        return f"{sign}{shock:.0f}bps ({sign}{pct:.0f}%)"
    elif unit == "pts":
        return f"{sign}{shock:.1f}pts ({sign}{pct:.0f}%)"
    return f"{sign}{shock:.2f}"


def _fmt_current(assumption) -> str:
    """Format the current value with appropriate unit."""
    unit = assumption.unit
    val = assumption.current

    if unit == "$":
        return f"${val:,.2f}"
    elif unit == "%":
        return f"{val:.1f}%"
    elif unit == "bps":
        return f"{val:.0f}bps"
    elif unit == "pts":
        return f"{val:.1f}"
    return f"{val:.2f}"


def _fmt_scenario_val(assumption) -> str:
    """Format the scenario target value."""
    unit = assumption.unit
    val = assumption.scenario

    if unit == "$":
        return f"${val:,.2f}"
    elif unit == "%":
        return f"{val:.1f}%"
    elif unit == "bps":
        return f"{val:.0f}bps"
    elif unit == "pts":
        return f"{val:.1f}"
    return f"{val:.2f}"


def _show_assumptions(console: Console, result) -> None:
    """Render the Scenario Assumptions panel."""
    from rich.console import Group
    from rich.text import Text

    table = Table(
        box=None,
        padding=(0, 2),
        show_header=True,
        header_style="bold dim",
    )
    table.add_column("Variable", style="bold", min_width=16)
    table.add_column("Current", justify="right", min_width=10)
    table.add_column("Scenario", justify="right", min_width=10)
    table.add_column("Shock", min_width=22)

    for a in result.assumptions:
        shock_str = _fmt_shock(a)
        shock_color = "red" if a.shock_abs > 0 else "green" if a.shock_abs < 0 else "dim"
        if a.variable == "gold":
            shock_color = "yellow" if a.shock_abs > 0 else "dim"

        table.add_row(
            a.display_name,
            _fmt_current(a),
            f"[bold]{_fmt_scenario_val(a)}[/bold]",
            f"[{shock_color}]{shock_str}[/{shock_color}]",
        )

    renderables = [table]
    if result.regime_effect:
        renderables.append(Text(""))
        renderables.append(
            Text.from_markup(
                f"  [bold yellow]Regime effect:[/bold yellow]  {result.regime_effect}"
            )
        )

    console.print(Panel(
        Group(*renderables),
        title="[bold]Scenario Assumptions[/bold]",
        border_style="cyan",
        padding=(0, 1),
    ))
    console.print()


def _show_mc_adjustments(console: Console, result) -> None:
    """Render the MC Parameter Adjustments panel."""
    mc = result.mc_adjustments
    bd = result.mc_breakdowns

    lines: list[str] = []

    # Equity drift
    drift = mc["equity_drift_adj"]
    drift_bd = bd.get("equity_drift_adj", {})
    drift_parts = [f"{k} {v:+.1%}" for k, v in drift_bd.items()]
    drift_detail = f"  ({', '.join(drift_parts)})" if drift_parts else ""
    drift_color = "red" if drift < 0 else "green" if drift > 0 else "dim"
    lines.append(
        f"  [bold]Equity drift:[/bold]   [{drift_color}]{drift:+.1%} ann.[/{drift_color}]{drift_detail}"
    )

    # Vol scaling
    vol = mc["equity_vol_adj"]
    vol_bd = bd.get("equity_vol_adj", {})
    vol_parts = [f"{k} {v:.2f}x" for k, v in vol_bd.items()]
    vol_detail = f"  ({', '.join(vol_parts)})" if vol_parts else ""
    vol_color = "red" if vol > 1.05 else "green" if vol < 0.95 else "dim"
    lines.append(
        f"  [bold]Vol scaling:[/bold]    [{vol_color}]{vol:.2f}x[/{vol_color}]{vol_detail}"
    )

    # Jump probability
    jump = mc["jump_prob_adj"]
    jump_bd = bd.get("jump_prob_adj", {})
    jump_parts = [f"{k} {v:.2f}x" for k, v in jump_bd.items()]
    jump_detail = f"  ({', '.join(jump_parts)})" if jump_parts else ""
    jump_color = "red" if jump > 1.1 else "dim"
    lines.append(
        f"  [bold]Jump prob:[/bold]      [{jump_color}]{jump:.2f}x[/{jump_color}]{jump_detail}"
    )

    console.print(Panel(
        "\n".join(lines),
        title="[bold]MC Parameter Adjustments[/bold]",
        border_style="blue",
        padding=(0, 1),
    ))
    console.print()


def _show_price_distribution(console: Console, result) -> None:
    """Render the price distribution scenario table with optional position P&L."""
    has_positions = bool(getattr(result, "position_estimates", None))

    table = Table(
        title=f"[bold]{result.horizon_days}-Day Price Distribution[/bold]",
        box=None,
        padding=(0, 1),
        show_header=True,
        header_style="bold",
    )
    table.add_column("Scenario", style="bold", min_width=10)
    table.add_column("Target", justify="right", min_width=7)
    table.add_column("Range", min_width=15)
    table.add_column("Return", justify="right", min_width=7)
    table.add_column("Prob", justify="right", min_width=5)

    if has_positions:
        for pe in result.position_estimates:
            cp = "P" if pe.opt_type == "put" else "C"
            table.add_column(f"{pe.strike:.0f}{cp} Best", justify="right", min_width=9)
            table.add_column(f"{pe.strike:.0f}{cp} Worst", justify="right", min_width=9)

    scenario_colors = {
        "BULL": "green",
        "BASE": "white",
        "BEAR": "red",
        "TAIL_RISK": "bright_red",
    }
    scenario_labels = {
        "BULL": "BULL",
        "BASE": "BASE",
        "BEAR": "BEAR",
        "TAIL_RISK": "TAIL RISK",
    }

    for key in ("BULL", "BASE", "BEAR", "TAIL_RISK"):
        s = result.scenarios[key]
        color = scenario_colors[key]
        label = scenario_labels[key]
        ret = s["return"]
        ret_color = "green" if ret > 0 else "red" if ret < 0 else "dim"

        row = [
            f"[{color}]{label}[/{color}]",
            f"[{color}]${s['target']:,.0f}[/{color}]",
            f"${s['range'][0]:,.0f} – ${s['range'][1]:,.0f}",
            f"[{ret_color}]{ret:+.1f}%[/{ret_color}]",
            f"{s['probability']}%",
        ]

        if has_positions:
            for pe in result.position_estimates:
                peak = pe.scenario_peak_pnl.get(key, 0) if pe.scenario_peak_pnl else pe.scenario_pnl.get(key, 0)
                peak_color = "green" if peak > 0 else "red" if peak < 0 else "dim"
                row.append(f"[{peak_color}]${peak:+,.0f}[/{peak_color}]")
                worst = pe.scenario_worst_pnl.get(key, 0) if pe.scenario_worst_pnl else pe.scenario_pnl.get(key, 0)
                worst_color = "green" if worst > 0 else "red" if worst < 0 else "dim"
                row.append(f"[{worst_color}]${worst:+,.0f}[/{worst_color}]")

        table.add_row(*row)

    console.print(table)
    console.print()


def _show_exit_analysis(console: Console, result) -> None:
    """Render the optimal exit analysis panel for each option position."""
    estimates = getattr(result, "position_estimates", None) or []
    if not estimates:
        return

    for pe in estimates:
        if pe.median_peak_pnl == 0 and pe.prob_profit == 0:
            continue

        cp = "P" if pe.opt_type == "put" else "C"
        title = (
            f"Optimal Exit Analysis: {int(pe.qty)}x {result.symbol} "
            f"{pe.strike:.0f}{cp} {pe.expiry}"
        )

        peak_color = "green" if pe.median_peak_pnl > 0 else "red"
        p25_color = "green" if pe.p25_peak_pnl > 0 else "red"
        p75_color = "green" if pe.p75_peak_pnl > 0 else "red"
        exit_color = "green" if pe.median_exit_price > pe.entry_value else "red"

        lines = [
            f"  [bold]Peak P&L (median):[/bold]   [{peak_color}]${pe.median_peak_pnl:+,.0f}[/{peak_color}]"
            f"   [dim](p25: ${pe.p25_peak_pnl:+,.0f}, p75: ${pe.p75_peak_pnl:+,.0f})[/dim]",
            f"  [bold]Optimal exit day:[/bold]    ~day {pe.median_peak_day}"
            f"   [dim](of {result.trading_days} trading days)[/dim]",
            f"  [bold]Option @ peak:[/bold]       [{exit_color}]${pe.median_exit_price:.2f}/contract[/{exit_color}]"
            f"   [dim](vs ${pe.entry_value:.2f} entry)[/dim]",
            "",
            "  [bold]Profit targets:[/bold]",
            f"    Any profit (>$0):   {_prob_bar(pe.prob_profit)}",
            f"    Double (+100%):     {_prob_bar(pe.prob_double)}",
            f"    Triple (+200%):     {_prob_bar(pe.prob_triple)}",
        ]

        console.print(Panel(
            "\n".join(lines),
            title=f"[bold]{title}[/bold]",
            border_style="green" if pe.median_peak_pnl > 0 else "red",
            padding=(0, 1),
        ))
        console.print()


def _prob_bar(pct: float) -> str:
    """Render a compact probability bar: '████░░░░░░  72%'."""
    filled = int(pct / 10)
    empty = 10 - filled
    color = "green" if pct >= 50 else "yellow" if pct >= 25 else "red"
    return f"[{color}]{'█' * filled}{'░' * empty}  {pct:.0f}%[/{color}]"


def _show_risk_metrics(console: Console, result) -> None:
    """Render the path-level risk metrics panel."""
    rm = getattr(result, "risk_metrics", None) or {}
    if not rm:
        return

    tail_table = Table(box=None, padding=(0, 2), show_header=False)
    tail_table.add_column("Metric", style="bold", min_width=20)
    tail_table.add_column("Value", justify="right", min_width=10)

    var_color = "red" if rm.get("var_95", 0) < -5 else "yellow" if rm.get("var_95", 0) < 0 else "green"
    tail_table.add_row("VaR (95%)", f"[{var_color}]{rm.get('var_95', 0):+.1f}%[/{var_color}]")
    tail_table.add_row("CVaR / Exp. Shortfall", f"[red]{rm.get('cvar_95', 0):+.1f}%[/red]")
    tail_table.add_row("Max Drawdown (median)", f"[red]{rm.get('max_drawdown_median', 0):.1f}%[/red]")
    tail_table.add_row("Max Drawdown (p95)", f"[bright_red]{rm.get('max_drawdown_p95', 0):.1f}%[/bright_red]")
    trough_day = rm.get("drawdown_trough_day_median", 0)
    tail_table.add_row("Drawdown trough (day)", f"[dim]~day {trough_day:.0f}[/dim]")

    ret_table = Table(box=None, padding=(0, 2), show_header=False)
    ret_table.add_column("Metric", style="bold", min_width=20)
    ret_table.add_column("Value", justify="right", min_width=10)

    er = rm.get("expected_return", 0)
    er_color = "green" if er > 0 else "red"
    ret_table.add_row("Expected return", f"[{er_color}]{er:+.1f}%[/{er_color}]")
    ret_table.add_row("Return std dev", f"{rm.get('return_std', 0):.1f}%")
    sortino = rm.get("sortino", 0)
    s_color = "green" if sortino > 1 else "yellow" if sortino > 0 else "red"
    ret_table.add_row("Sortino ratio (ann.)", f"[{s_color}]{sortino:.2f}[/{s_color}]")
    ret_table.add_row("", "")

    prob_table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold dim")
    prob_table.add_column("", min_width=6)
    prob_table.add_column("P(loss)", justify="center", min_width=8)
    prob_table.add_column(">5%", justify="center", min_width=6)
    prob_table.add_column(">10%", justify="center", min_width=6)
    prob_table.add_column(">20%", justify="center", min_width=6)

    pl = rm.get("prob_loss", 0)
    pl_color = "red" if pl > 60 else "yellow" if pl > 40 else "green"
    prob_table.add_row(
        "[red]Loss[/red]",
        f"[{pl_color}]{pl:.0f}%[/{pl_color}]",
        f"{rm.get('prob_loss_5pct', 0):.0f}%",
        f"{rm.get('prob_loss_10pct', 0):.0f}%",
        f"{rm.get('prob_loss_20pct', 0):.0f}%",
    )
    prob_table.add_row(
        "[green]Gain[/green]",
        f"{100 - pl:.0f}%",
        f"{rm.get('prob_gain_5pct', 0):.0f}%",
        f"{rm.get('prob_gain_10pct', 0):.0f}%",
        "",
    )

    from rich.columns import Columns
    from rich.console import Group

    console.print(Panel(
        Group(
            Columns([tail_table, ret_table], padding=4),
            prob_table,
        ),
        title="[bold]Risk Metrics[/bold]",
        border_style="red",
        padding=(0, 1),
    ))
    console.print()


def _show_histogram(console: Console, result) -> None:
    """Render an ASCII histogram of the terminal return distribution."""
    hist = getattr(result, "histogram", None) or {}
    if not hist:
        return

    bin_centers = hist.get("bin_centers", [])
    counts = hist.get("counts", [])
    median_ret = hist.get("median_return", 0)

    if not bin_centers or not counts:
        return

    max_count = max(counts) or 1
    bar_max_width = 40

    lines: list[str] = []
    for center, count in zip(bin_centers, counts):
        bar_len = int(count / max_count * bar_max_width)
        # Color: green for positive returns, red for negative
        if center > 0:
            bar = f"[green]{'█' * bar_len}[/green]"
        elif center < 0:
            bar = f"[red]{'█' * bar_len}[/red]"
        else:
            bar = f"[dim]{'█' * bar_len}[/dim]"

        # Mark median bin
        marker = " ◄ median" if abs(center - median_ret) < hist.get("bin_width", 999) else ""
        lines.append(f"  {center:+6.1f}% │{bar}{f'[dim]{marker}[/dim]' if marker else ''}")

    from rich.text import Text

    header = Text.from_markup(
        f"  [dim]Return distribution  |  "
        f"median: {median_ret:+.1f}%  |  "
        f"range: {hist['range'][0]:+.1f}% to {hist['range'][1]:+.1f}%[/dim]"
    )

    console.print(Panel(
        "\n".join([header.markup] + lines),
        title="[bold]Return Histogram[/bold]",
        border_style="cyan",
        padding=(0, 1),
    ))
    console.print()


def _show_multi_horizon(console: Console, result) -> None:
    """Render the multi-horizon term structure table (1M / 3M / 6M)."""
    mh = getattr(result, "multi_horizon", None) or []
    if not mh or len(mh) < 2:
        return

    table = Table(
        box=None,
        padding=(0, 2),
        show_header=True,
        header_style="bold dim",
    )
    table.add_column("Metric", style="bold", min_width=16)

    for h in mh:
        table.add_column(h["horizon_label"], justify="right", min_width=10)

    # E[R]
    er_cells = []
    for h in mh:
        v = h["expected_return"]
        c = "green" if v > 0 else "red"
        er_cells.append(f"[{c}]{v:+.1f}%[/{c}]")
    table.add_row("[bold]E[Return][/bold]", *er_cells)

    # p25 / p50 / p75
    for pct_key, label in [("p25", "Bear (p25)"), ("p50", "Base (p50)"), ("p75", "Bull (p75)")]:
        cells = []
        for h in mh:
            v = h.get(pct_key, 0)
            c = "green" if v > 0 else "red"
            cells.append(f"[{c}]{v:+.1f}%[/{c}]")
        table.add_row(f"[bold]{label}[/bold]", *cells)

    # VaR95
    var_cells = [f"[red]{h['var_95']:+.1f}%[/red]" for h in mh]
    table.add_row("[bold]VaR (95%)[/bold]", *var_cells)

    # CVaR95
    cvar_cells = [f"[red]{h['cvar_95']:+.1f}%[/red]" for h in mh]
    table.add_row("[bold]CVaR (95%)[/bold]", *cvar_cells)

    # Max Drawdown
    dd_cells = [f"[red]{h['max_drawdown_median']:.1f}%[/red]" for h in mh]
    table.add_row("[bold]Max DD (med)[/bold]", *dd_cells)

    # P(loss)
    pl_cells = []
    for h in mh:
        v = h["prob_loss"]
        c = "red" if v > 55 else "yellow" if v > 40 else "green"
        pl_cells.append(f"[{c}]{v:.0f}%[/{c}]")
    table.add_row("[bold]P(loss)[/bold]", *pl_cells)

    # Sortino
    sort_cells = []
    for h in mh:
        v = h["sortino"]
        c = "green" if v > 1 else "yellow" if v > 0 else "red"
        sort_cells.append(f"[{c}]{v:.2f}[/{c}]")
    table.add_row("[bold]Sortino[/bold]", *sort_cells)

    console.print(Panel(
        table,
        title="[bold]Risk Term Structure[/bold]",
        border_style="blue",
        padding=(0, 1),
    ))
    console.print()


def _show_analog_overlay(console: Console, result) -> None:
    """Render the historical analog overlay comparison panel."""
    ao = getattr(result, "analog_overlay", None)
    if ao is None:
        return

    signal_styles = {
        "ALIGNED": ("green", "MC and analogs agree"),
        "MC_OPTIMISTIC": ("yellow", "MC more bullish than history"),
        "MC_PESSIMISTIC": ("cyan", "MC more bearish than history"),
    }
    color, desc = signal_styles.get(ao.signal, ("dim", ""))

    table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold dim")
    table.add_column("", min_width=14)
    table.add_column("k-NN Analogs", justify="right", min_width=12)
    table.add_column("Monte Carlo", justify="right", min_width=12)
    table.add_column("Gap", justify="right", min_width=8)

    mc_med_color = "green" if ao.mc_median_return > 0 else "red"
    an_med_color = "green" if ao.analog_median_return > 0 else "red"
    gap_color = "yellow" if abs(ao.calibration_gap) > 2 else "dim"
    table.add_row(
        "[bold]Median return[/bold]",
        f"[{an_med_color}]{ao.analog_median_return:+.1f}%[/{an_med_color}]",
        f"[{mc_med_color}]{ao.mc_median_return:+.1f}%[/{mc_med_color}]",
        f"[{gap_color}]{ao.calibration_gap:+.1f}pp[/{gap_color}]",
    )

    mc_mn_color = "green" if ao.mc_mean_return > 0 else "red"
    an_mn_color = "green" if ao.analog_mean_return > 0 else "red"
    table.add_row(
        "[bold]Mean return[/bold]",
        f"[{an_mn_color}]{ao.analog_mean_return:+.1f}%[/{an_mn_color}]",
        f"[{mc_mn_color}]{ao.mc_mean_return:+.1f}%[/{mc_mn_color}]",
        "",
    )

    table.add_row(
        "[bold]p25 – p75[/bold]",
        f"{ao.analog_p25:+.1f}% – {ao.analog_p75:+.1f}%",
        "",
        "",
    )

    hr_color = "green" if ao.analog_hit_rate > 55 else "red" if ao.analog_hit_rate < 45 else "dim"
    table.add_row(
        "[bold]Hit rate[/bold]",
        f"[{hr_color}]{ao.analog_hit_rate:.0f}%[/{hr_color}]",
        "",
        "",
    )
    table.add_row(
        "[bold]VaR (5th pctl)[/bold]",
        f"[red]{ao.analog_var_5:+.1f}%[/red]",
        "",
        "",
    )

    from rich.console import Group
    from rich.text import Text

    signal_line = Text.from_markup(
        f"\n  [{color}]{ao.signal}[/{color}] — {desc}  "
        f"[dim]({ao.n_analogs} regime-similar days)[/dim]"
    )

    console.print(Panel(
        Group(table, signal_line),
        title="[bold]Historical Analog Overlay (k-NN vs MC)[/bold]",
        border_style="magenta",
        padding=(0, 1),
    ))
    console.print()


def _show_factor_sensitivity(console: Console, result) -> None:
    """Render the factor sensitivity tornado chart."""
    sensitivities = getattr(result, "factor_sensitivities", None) or []
    if not sensitivities:
        return

    table = Table(
        box=None,
        padding=(0, 2),
        show_header=True,
        header_style="bold dim",
    )
    table.add_column("Factor", style="bold", min_width=14)
    table.add_column("E[R] Impact", justify="right", min_width=12)
    table.add_column("Bar", min_width=26)
    table.add_column("VaR95 Impact", justify="right", min_width=12)
    table.add_column("Vol Impact", justify="right", min_width=10)
    table.add_column("Median $", justify="right", min_width=10)

    max_impact = max(abs(s.expected_return_delta) for s in sensitivities) or 1.0

    for s in sensitivities:
        er_color = "red" if s.expected_return_delta < -0.5 else "green" if s.expected_return_delta > 0.5 else "dim"
        var_color = "red" if s.var_95_delta < -0.5 else "green" if s.var_95_delta > 0.5 else "dim"
        vol_color = "red" if s.vol_delta > 0.5 else "green" if s.vol_delta < -0.5 else "dim"

        bar_width = 20
        bar_units = int(abs(s.expected_return_delta) / max_impact * bar_width)
        if s.expected_return_delta < 0:
            bar = " " * (bar_width - bar_units) + "[red]" + "█" * bar_units + "[/red]│"
        else:
            bar = " " * bar_width + "│[green]" + "█" * bar_units + "[/green]"

        table.add_row(
            s.display_name,
            f"[{er_color}]{s.expected_return_delta:+.1f}pp[/{er_color}]",
            bar,
            f"[{var_color}]{s.var_95_delta:+.1f}pp[/{var_color}]",
            f"[{vol_color}]{s.vol_delta:+.1f}pp[/{vol_color}]",
            f"${s.median_price:,.0f}",
        )

    console.print(Panel(
        table,
        title="[bold]Factor Sensitivity (vs no-shock baseline)[/bold]",
        border_style="yellow",
        padding=(0, 1),
    ))
    console.print()


def _show_percentile_strip(console: Console, result) -> None:
    """Render the percentile strip at the bottom."""
    dist = result.full_distribution
    labels = ["p5", "p10", "p25", "p50", "p75", "p90", "p95"]

    header = "  ".join(f"[bold]{lbl}[/bold]" for lbl in labels)
    values = "  ".join(f"${dist[lbl]:,.0f}" for lbl in labels)

    console.print(f"  {header}")
    console.print(f"  {values}")

"""
LOX Research: Factor Attribution Command

Fama-French regression on live positions — factor loadings, P&L attribution,
concentration analysis, and regime cross-reference.

Usage:
    lox research factors              # Full factor dashboard
    lox research factors --ticker SPY # Single position detail
    lox research factors --json       # Machine-readable
    lox research factors --llm        # LLM analysis
"""
from __future__ import annotations

import json as json_lib
import logging

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from lox.config import load_settings

logger = logging.getLogger(__name__)

# ── Regime × Factor historical expectations ──────────────────────────────────
# Based on empirical factor premia research (Asness, Ilmanen, Israel et al.)

REGIME_FACTOR_EXPECTATIONS: dict[str, dict[str, tuple[str, str]]] = {
    "RISK_ON": {
        "Mkt": ("TAILWIND", "equities rally in risk-on"),
        "SMB": ("TAILWIND", "small-caps outperform"),
        "HML": ("NEUTRAL", "no strong value/growth tilt"),
        "RMW": ("NEUTRAL", "quality less differentiated"),
        "CMA": ("NEUTRAL", "investment factor muted"),
        "Mom": ("TAILWIND", "trends persist"),
    },
    "REFLATION": {
        "Mkt": ("TAILWIND", "equities benefit from stimulus"),
        "SMB": ("TAILWIND", "small-caps leverage to recovery"),
        "HML": ("TAILWIND", "value outperforms as cyclicals lead"),
        "RMW": ("NEUTRAL", "quality neutral"),
        "CMA": ("HEADWIND", "aggressive investment rewarded"),
        "Mom": ("NEUTRAL", "trend regime shifting"),
    },
    "STAGFLATION": {
        "Mkt": ("HEADWIND", "equity underperforms in stagflation"),
        "SMB": ("HEADWIND", "small-caps vulnerable to margin squeeze"),
        "HML": ("TAILWIND", "value outperforms growth ~300bps historically"),
        "RMW": ("TAILWIND", "quality is defensive"),
        "CMA": ("NEUTRAL", "investment factor mixed"),
        "Mom": ("HEADWIND", "momentum crashes in regime transitions"),
    },
    "RISK_OFF": {
        "Mkt": ("HEADWIND", "equities sell off"),
        "SMB": ("HEADWIND", "flight to quality, large-cap safety"),
        "HML": ("NEUTRAL", "mixed — depends on recession depth"),
        "RMW": ("TAILWIND", "quality outperforms in downturns"),
        "CMA": ("TAILWIND", "conservative firms outperform"),
        "Mom": ("HEADWIND", "momentum crashes in panics"),
    },
    "TRANSITION": {
        "Mkt": ("NEUTRAL", "directionless, high dispersion"),
        "SMB": ("NEUTRAL", "no strong cap bias"),
        "HML": ("NEUTRAL", "style rotation underway"),
        "RMW": ("TAILWIND", "quality provides stability"),
        "CMA": ("NEUTRAL", "investment factor muted"),
        "Mom": ("HEADWIND", "momentum vulnerable to reversals"),
    },
}


def register(app: typer.Typer) -> None:
    """Register the factors command."""

    @app.command("factors")
    def factors_cmd(
        refresh: bool = typer.Option(False, "--refresh", help="Force re-download factor data and prices"),
        window: int = typer.Option(252, "--window", "-w", help="Rolling regression window (trading days)"),
        llm: bool = typer.Option(False, "--llm", help="LLM analysis of factor profile"),
        json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
        ticker: str = typer.Option("", "--ticker", "-t", help="Single position detail"),
    ):
        """Factor attribution — Fama-French regression on live positions."""
        console = Console()
        settings = load_settings()

        # ── Fetch positions ──────────────────────────────────────────────
        from lox.cli_commands.research.portfolio_cmd import _fetch_positions

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Fetching positions...[/bold cyan]"),
            transient=True,
        ) as prog:
            prog.add_task("fetch", total=None)
            positions = _fetch_positions(settings)

        if not positions:
            console.print("[yellow]No open positions found.[/yellow]")
            return

        # ── Fetch factor data ────────────────────────────────────────────
        from lox.factors.french import fetch_french_factors

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Loading Fama-French factors...[/bold cyan]"),
            transient=True,
        ) as prog:
            prog.add_task("ff", total=None)
            factor_df = fetch_french_factors(refresh=refresh)

        # ── Compute factor loadings ──────────────────────────────────────
        from lox.factors.loadings import compute_all_loadings

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Running factor regressions...[/bold cyan]"),
            transient=True,
        ) as prog:
            prog.add_task("ols", total=None)
            loadings = compute_all_loadings(
                settings=settings,
                positions=positions,
                factor_df=factor_df,
                window=window,
                refresh=refresh,
            )

        if not loadings:
            console.print("[yellow]Could not compute factor loadings for any position.[/yellow]")
            return

        # ── Build portfolio profile ──────────────────────────────────────
        from lox.factors.portfolio import build_portfolio_profile
        profile = build_portfolio_profile(loadings)

        # ── Compute attribution ──────────────────────────────────────────
        from lox.factors.attribution import compute_attribution
        attribution = compute_attribution(profile, factor_df)

        # ── Single ticker mode ───────────────────────────────────────────
        if ticker:
            ticker_upper = ticker.upper()
            matches = [l for l in loadings if l.ticker == ticker_upper]
            if not matches:
                console.print(f"[yellow]No position found for {ticker_upper}[/yellow]")
                return
            _render_single_position(console, matches[0], factor_df, window)
            return

        # ── JSON output ──────────────────────────────────────────────────
        if json_out:
            output = {
                "portfolio": profile.to_dict(),
                "attribution_20d": attribution.to_dict(),
            }
            console.print(json_lib.dumps(output, indent=2, default=str))
            return

        # ── Rich display ─────────────────────────────────────────────────
        console.print()
        console.print("[bold cyan]LOX RESEARCH[/bold cyan]  [bold]Factor Attribution[/bold]")
        console.print()

        _render_portfolio_profile(console, profile, window)
        _render_position_table(console, loadings)
        _render_attribution(console, attribution)
        _render_concentration(console, profile)
        _render_regime_cross_ref(console, profile, settings)

        if llm:
            _run_llm_analysis(console, settings, profile, attribution, ticker)


# ── Panel 1: Portfolio Factor Profile ─────────────────────────────────────────

def _beta_context(factor: str, beta: float) -> str:
    """Context string for a portfolio-level factor beta."""
    if factor == "Mkt":
        if beta > 1.15:
            return "above-market \u2014 directional equity risk"
        if beta > 0.85:
            return "near-market \u2014 benchmark-like exposure"
        if beta > 0.5:
            return "below-market \u2014 defensive positioning"
        if beta > 0:
            return "low beta \u2014 hedged"
        return "negative \u2014 net short equity"
    if factor == "SMB":
        if beta > 0.15:
            return "small-cap tilt \u2014 higher vol, cyclical"
        if beta < -0.15:
            return "large-cap tilt \u2014 quality/safety bias"
        return "near-neutral \u2014 no cap bias"
    if factor == "HML":
        if beta > 0.15:
            return "value tilt \u2014 cheap stocks, cyclical"
        if beta < -0.15:
            return "growth tilt \u2014 expensive/high-growth"
        return "near-neutral \u2014 no style bias"
    if factor == "RMW":
        if beta > 0.15:
            return "quality tilt \u2014 profitable, defensive"
        if beta < -0.15:
            return "junk tilt \u2014 speculative, fragile"
        return "near-neutral"
    if factor == "CMA":
        if beta > 0.15:
            return "conservative-investment firms"
        if beta < -0.15:
            return "aggressive-investment firms"
        return "near-neutral"
    if factor == "Mom":
        if beta > 0.15:
            return "momentum \u2014 riding trends, crash risk"
        if beta < -0.15:
            return "contrarian \u2014 mean-reversion bet"
        return "near-neutral"
    return ""


def _beta_color(beta: float) -> str:
    if abs(beta) > 0.3:
        return "bold yellow"
    if abs(beta) > 0.15:
        return "white"
    return "dim"


def _render_portfolio_profile(console: Console, profile: PortfolioFactorProfile, window: int) -> None:
    from lox.factors.loadings import FACTOR_NAMES

    table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold")
    table.add_column("Factor", min_width=10)
    table.add_column("Loading", justify="right", min_width=8)
    table.add_column("Context", ratio=2)

    for factor in FACTOR_NAMES:
        beta = profile.portfolio_betas.get(factor, 0.0)
        color = _beta_color(beta)
        ctx = _beta_context(factor, beta)
        table.add_row(
            f"[bold]{factor}[/bold]",
            f"[{color}]{beta:+.3f}[/{color}]",
            f"[dim]{ctx}[/dim]",
        )

    n_pos = len(profile.position_loadings)
    r2_pct = profile.portfolio_r_squared * 100
    subtitle = f"[dim]{n_pos} positions  |  Window: {window}d  |  R\u00b2: {r2_pct:.0f}%[/dim]"

    lines = [subtitle, "", table.__rich_console__(console, console.options).__next__() if False else ""]

    # Build panel content manually
    content_parts: list[str] = [
        subtitle,
        "",
    ]

    panel_group = Panel(
        table,
        title="[bold]Portfolio Factor Profile[/bold]",
        subtitle=subtitle,
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel_group)

    # Tilt description
    console.print(f"  [bold]Tilt:[/bold] {profile.tilt_description}")
    if profile.n_data_warnings > 0:
        console.print(f"  [yellow]{profile.n_data_warnings} position(s) with insufficient data[/yellow]")
    console.print()


# ── Panel 2: Position Factor Loadings ─────────────────────────────────────────

def _render_position_table(console: Console, loadings: list) -> None:
    from lox.factors.loadings import PositionLoadings

    table = Table(
        title="[bold]Position Factor Loadings[/bold]",
        box=None,
        padding=(0, 1),
        header_style="bold",
    )
    table.add_column("Position", style="bold", min_width=8, no_wrap=True)
    table.add_column("Wt%", justify="right", min_width=6)
    table.add_column("Mkt \u03b2", justify="right", min_width=6)
    table.add_column("SMB", justify="right", min_width=6)
    table.add_column("HML", justify="right", min_width=6)
    table.add_column("RMW", justify="right", min_width=6)
    table.add_column("CMA", justify="right", min_width=6)
    table.add_column("Mom", justify="right", min_width=6)
    table.add_column("R\u00b2", justify="right", min_width=5)
    table.add_column("Alpha", justify="right", min_width=7)

    for p in loadings:
        # Display name
        if p.position_type in ("call", "put"):
            display = f"{p.ticker} {p.position_type[0].upper()}"
        elif p.position_type == "short_equity":
            display = f"{p.ticker} S"
        else:
            display = p.ticker

        wt_color = "red" if p.weight_pct < 0 else "white"
        alpha_color = "green" if p.alpha_ann > 0 else "red" if p.alpha_ann < -0.05 else "dim"
        warn = " *" if p.data_warning else ""

        r2_color = "dim" if p.r_squared < 0.3 else "white"

        table.add_row(
            f"{display}{warn}",
            f"[{wt_color}]{p.weight_pct:+.1f}%[/{wt_color}]",
            f"{p.mkt_beta:+.2f}",
            f"{p.smb:+.2f}",
            f"{p.hml:+.2f}",
            f"{p.rmw:+.2f}",
            f"{p.cma:+.2f}",
            f"{p.mom:+.2f}",
            f"[{r2_color}]{p.r_squared:.2f}[/{r2_color}]",
            f"[{alpha_color}]{p.alpha_ann * 100:+.1f}%[/{alpha_color}]",
        )

    console.print(table)

    # Data warning footnote
    has_warnings = any(p.data_warning for p in loadings)
    if has_warnings:
        console.print("  [dim]* insufficient data \u2014 results may be unreliable[/dim]")
    console.print()


# ── Panel 3: P&L Attribution ─────────────────────────────────────────────────

def _render_attribution(console: Console, attr) -> None:
    from lox.factors.attribution import FactorAttribution

    console.print(f"[bold]P&L Attribution[/bold]  [dim]Last {attr.period_days} trading days[/dim]")
    console.print()

    total_color = "green" if attr.total_return_pct > 0 else "red" if attr.total_return_pct < 0 else "white"
    console.print(f"  [bold]Total Return:[/bold]  [{total_color}]{attr.total_return_pct:+.2f}%[/{total_color}]")
    console.print(f"  {'─' * 50}")

    # Bar chart for contributions
    max_abs = max((abs(fc.contribution_pct) for fc in attr.factor_contributions), default=1.0)
    max_abs = max(max_abs, abs(attr.alpha_residual_pct), 0.01)
    bar_width = 25

    for fc in attr.factor_contributions:
        bar_len = int(abs(fc.contribution_pct) / max_abs * bar_width)
        bar = "\u2588" * bar_len + "\u2591" * (bar_width - bar_len)
        color = "green" if fc.contribution_pct > 0.005 else "red" if fc.contribution_pct < -0.005 else "dim"
        pct_of_total = (fc.contribution_pct / attr.total_return_pct * 100) if abs(attr.total_return_pct) > 0.01 else 0
        console.print(
            f"  {fc.factor:<12} [{color}]{fc.contribution_pct:+.2f}%[/{color}]  "
            f"[dim]{bar}[/dim]  [dim]({pct_of_total:+.0f}%)[/dim]"
        )

    console.print(f"  {'─' * 50}")
    alpha_color = "green" if attr.alpha_residual_pct > 0.005 else "red" if attr.alpha_residual_pct < -0.005 else "dim"
    bar_len = int(abs(attr.alpha_residual_pct) / max_abs * bar_width)
    bar = "\u2588" * bar_len + "\u2591" * (bar_width - bar_len)
    console.print(
        f"  {'Alpha':<12} [{alpha_color}]{attr.alpha_residual_pct:+.2f}%[/{alpha_color}]  "
        f"[dim]{bar}[/dim]"
    )
    console.print()
    console.print(f"  [bold]Verdict:[/bold] {attr.verdict()}")
    console.print()


# ── Panel 4: Factor Concentration ─────────────────────────────────────────────

def _render_concentration(console: Console, profile: PortfolioFactorProfile) -> None:
    from lox.factors.loadings import FACTOR_NAMES

    console.print("[bold]Factor Concentration[/bold]")
    console.print()

    bar_width = 25
    sorted_factors = sorted(
        FACTOR_NAMES,
        key=lambda f: profile.factor_concentration.get(f, 0),
        reverse=True,
    )

    for factor in sorted_factors:
        conc = profile.factor_concentration.get(factor, 0)
        beta = profile.portfolio_betas.get(factor, 0)
        bar_len = int(conc / 100 * bar_width)
        bar = "\u2588" * bar_len + "\u2591" * (bar_width - bar_len)

        warn = ""
        if conc > 70:
            warn = "  [yellow]\u26a0 CONCENTRATED[/yellow]"
        elif conc > 50:
            warn = "  [yellow]\u26a0[/yellow]"

        console.print(
            f"  {factor:<12} {beta:+.3f}  [dim]{bar}[/dim]  {conc:4.0f}%{warn}"
        )

    if profile.concentration_warning:
        console.print()
        console.print(f"  [yellow]\u26a0 {profile.concentration_warning}[/yellow]")

    console.print()


# ── Panel 5: Regime × Factor Cross-Reference ─────────────────────────────────

def _render_regime_cross_ref(console: Console, profile: PortfolioFactorProfile, settings) -> None:
    """Cross-reference factor loadings with current composite regime."""
    try:
        from lox.regimes import build_unified_regime_state
        from lox.regimes.composite import classify_composite_regime

        state = build_unified_regime_state(settings=settings)
        composite = state.composite
        if composite is None:
            return

        regime_name = composite.regime if isinstance(composite.regime, str) else composite.regime.name
        expectations = REGIME_FACTOR_EXPECTATIONS.get(regime_name)
        if not expectations:
            return

        display_label = composite.label if hasattr(composite, "label") else regime_name
        console.print(
            f"[bold]Regime \u00d7 Factor Context[/bold]  "
            f"[dim]Current: {display_label} ({composite.confidence:.0f}% confidence)[/dim]"
        )
        console.print()

        table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold")
        table.add_column("Factor", min_width=8)
        table.add_column("Your Loading", justify="right", min_width=10)
        table.add_column("Regime Signal", min_width=12)
        table.add_column("Implication", ratio=2)

        from lox.factors.loadings import FACTOR_NAMES

        alignment_score = 0
        for factor in FACTOR_NAMES:
            beta = profile.portfolio_betas.get(factor, 0.0)
            factor_signal, desc = expectations.get(factor, ("NEUTRAL", ""))

            # The factor signal says what the FACTOR does in this regime.
            # Your LOADING SIGN determines if that's good or bad for YOUR book.
            # Short equity (beta < 0) + equity underperforms (HEADWIND) = TAILWIND for you.
            if factor_signal == "NEUTRAL" or abs(beta) < 0.05:
                book_signal = "NEUTRAL"
                sig_color = "dim"
                icon = "\u2014"
            else:
                # Factor tailwind + positive loading = book tailwind
                # Factor headwind + negative loading = book tailwind (you're short!)
                # Factor tailwind + negative loading = book headwind
                # Factor headwind + positive loading = book headwind
                factor_is_positive = factor_signal == "TAILWIND"
                loading_is_positive = beta > 0

                if factor_is_positive == loading_is_positive:
                    # Aligned: long into tailwind, or short into headwind
                    book_signal = "TAILWIND"
                    sig_color = "green"
                    icon = "\u2713"
                    alignment_score += 1
                else:
                    # Misaligned: long into headwind, or short into tailwind
                    book_signal = "HEADWIND"
                    sig_color = "red"
                    icon = "\u26a0"
                    alignment_score -= 1

            # Build implication that reflects book direction
            if book_signal != "NEUTRAL" and beta < -0.05 and factor_signal == "HEADWIND":
                implication = f"{desc}; [bold green]benefits your short[/bold green]"
            elif book_signal != "NEUTRAL" and beta < -0.05 and factor_signal == "TAILWIND":
                implication = f"{desc}; [bold red]hurts your short[/bold red]"
            else:
                implication = f"[dim]{desc}[/dim]"

            beta_color = _beta_color(beta)

            table.add_row(
                f"[bold]{factor}[/bold]",
                f"[{beta_color}]{beta:+.3f}[/{beta_color}]",
                f"[{sig_color}]{icon} {book_signal}[/{sig_color}]",
                implication,
            )

        console.print(table)

        # Summary
        if alignment_score >= 2:
            console.print("  [green]Your factor tilts are broadly aligned with the current regime.[/green]")
        elif alignment_score <= -2:
            console.print("  [red]Your factor tilts are hostile to the current regime \u2014 consider rotation.[/red]")
        else:
            console.print("  [yellow]Mixed alignment \u2014 some factor tilts conflict with regime expectations.[/yellow]")

        console.print()

    except Exception as e:
        logger.debug(f"Regime cross-reference unavailable: {e}")


# ── Single position detail ────────────────────────────────────────────────────

def _render_single_position(console: Console, loadings, factor_df, window: int) -> None:
    from lox.factors.loadings import FACTOR_NAMES

    console.print()
    console.print(f"[bold cyan]LOX RESEARCH[/bold cyan]  [bold]Factor Detail: {loadings.ticker}[/bold]")
    console.print()

    # Factor betas
    table = Table(box=None, padding=(0, 2), header_style="bold")
    table.add_column("Factor", min_width=10)
    table.add_column("Beta", justify="right", min_width=8)
    table.add_column("Context", ratio=2)

    for factor in FACTOR_NAMES:
        beta = loadings.betas_dict()[factor]
        color = _beta_color(beta)
        ctx = _beta_context(factor, beta)
        table.add_row(
            f"[bold]{factor}[/bold]",
            f"[{color}]{beta:+.4f}[/{color}]",
            f"[dim]{ctx}[/dim]",
        )

    console.print(Panel(
        table,
        title=f"[bold]{loadings.ticker} Factor Loadings[/bold]",
        subtitle=f"[dim]{loadings.position_type}  |  Weight: {loadings.weight_pct:+.1f}%  |  Window: {window}d[/dim]",
        border_style="cyan",
        padding=(1, 2),
    ))

    # Diagnostics
    console.print(f"  [bold]R\u00b2:[/bold] {loadings.r_squared:.4f}  ({loadings.r_squared * 100:.1f}% of returns explained by factors)")
    alpha_color = "green" if loadings.alpha_ann > 0 else "red"
    console.print(f"  [bold]Alpha (ann.):[/bold] [{alpha_color}]{loadings.alpha_ann * 100:+.2f}%[/{alpha_color}]")
    console.print(f"  [bold]Residual Vol:[/bold] {loadings.residual_vol * 100:.1f}%")
    console.print(f"  [bold]Observations:[/bold] {loadings.n_obs}")
    if loadings.data_warning:
        console.print(f"  [yellow]\u26a0 {loadings.data_warning}[/yellow]")
    console.print()


# ── LLM analysis ─────────────────────────────────────────────────────────────

def _run_llm_analysis(console: Console, settings, profile, attribution, ticker: str) -> None:
    """Run LLM chat with factor profile as context."""
    try:
        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis

        snapshot = {
            "portfolio_betas": profile.to_dict()["portfolio_betas"],
            "tilt_description": profile.tilt_description,
            "portfolio_r_squared": profile.portfolio_r_squared,
            "concentration_warning": profile.concentration_warning,
            "attribution_20d": attribution.to_dict(),
        }

        print_llm_regime_analysis(
            settings=settings,
            domain="factors",
            snapshot=snapshot,
            regime_label=profile.tilt_description,
            regime_description=(
                f"Factor profile: {profile.tilt_description}. "
                f"R\u00b2={profile.portfolio_r_squared:.2f}. "
                f"20d attribution verdict: {attribution.verdict()}"
            ),
            ticker=ticker,
        )
    except Exception as e:
        console.print(f"[dim]LLM analysis unavailable: {e}[/dim]")

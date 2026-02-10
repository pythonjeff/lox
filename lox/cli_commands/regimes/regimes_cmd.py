from __future__ import annotations

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lox.config import load_settings


def _traffic_light(score: float) -> str:
    """Return traffic-light emoji based on score."""
    if score >= 60:
        return "🔴"
    elif score >= 45:
        return "🟡"
    else:
        return "🟢"


def _score_color(score: float) -> str:
    """Rich color for score value."""
    if score >= 60:
        return "red"
    elif score >= 45:
        return "yellow"
    return "green"


def _fmt_key_inputs(regime) -> str:
    """Format the metrics dict into a compact 'Key Inputs' string."""
    if not regime or not regime.metrics:
        return regime.description[:60] if regime else "—"
    parts = []
    for k, v in regime.metrics.items():
        if v is not None:
            parts.append(f"{k}: {v}")
    return " | ".join(parts)[:65] if parts else regime.description[:60]


def register(app: typer.Typer) -> None:

    @app.command("unified")
    def unified_regimes(
        start: str = typer.Option("2020-01-01", "--start", help="Start date YYYY-MM-DD"),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh data"),
        json_output: bool = typer.Option(False, "--json", help="Output as JSON for ML"),
    ):
        """
        Show unified regime state across all 12 domains.

        This is the recommended view for understanding the current market regime
        and extracting ML-friendly features.
        """
        from lox.regimes import build_unified_regime_state
        from lox.regimes.features import CORE_DOMAINS, EXTENDED_DOMAINS
        from lox.data.regime_history import get_recent_changes
        import json as json_module

        console = Console()
        settings = load_settings()

        with console.status("[cyan]Building unified regime state...[/cyan]"):
            state = build_unified_regime_state(
                settings=settings,
                start_date=start,
                refresh=refresh,
            )

        if json_output:
            features = state.to_feature_dict()
            print(json_module.dumps(features, indent=2, default=str))
            return

        # ── Header Panel ──────────────────────────────────────────────────
        console.print()

        # Build macro quadrant detail line
        quadrant_detail = ""
        if state.growth and state.inflation:
            quadrant_detail = (
                f"\n│ Macro Quadrant: {state.macro_quadrant} "
                f"(Growth: {state.growth.label} {state.growth.score:.0f} + "
                f"Inflation: {state.inflation.label} {state.inflation.score:.0f})"
            )

        console.print(Panel(
            f"[bold]Overall: {state.overall_category}[/bold] (score: {state.overall_risk_score:.0f}/100)"
            f"{quadrant_detail}",
            title=f"🎯 Unified Regime State — {state.asof}",
            border_style="cyan",
        ))

        # ── Core Regimes Table ────────────────────────────────────────────
        core_table = Table(
            title="Core Regimes (Monte Carlo Inputs)",
            show_header=True,
            header_style="bold cyan",
        )
        core_table.add_column("Domain", style="bold")
        core_table.add_column("Regime")
        core_table.add_column("Score", justify="right")
        core_table.add_column("Key Inputs")

        for domain in CORE_DOMAINS:
            regime = getattr(state, domain, None)
            if regime:
                sc = _score_color(regime.score)
                tl = _traffic_light(regime.score)
                core_table.add_row(
                    f"{tl} {domain.title()}",
                    regime.label,
                    f"[{sc}]{regime.score:.0f}[/{sc}]",
                    regime.description[:65] + ("..." if len(regime.description) > 65 else ""),
                )
            else:
                core_table.add_row(f"  {domain.title()}", "N/A", "-", "Data unavailable")

        console.print(core_table)
        console.print()

        # ── Extended Regimes Table ────────────────────────────────────────
        ext_table = Table(
            title="Extended Regimes (Context)",
            show_header=True,
            header_style="bold magenta",
        )
        ext_table.add_column("Domain", style="bold")
        ext_table.add_column("Regime")
        ext_table.add_column("Score", justify="right")
        ext_table.add_column("Key Inputs")

        for domain in EXTENDED_DOMAINS:
            regime = getattr(state, domain, None)
            if regime:
                sc = _score_color(regime.score)
                tl = _traffic_light(regime.score)
                ext_table.add_row(
                    f"{tl} {domain.title()}",
                    regime.label,
                    f"[{sc}]{regime.score:.0f}[/{sc}]",
                    regime.description[:65] + ("..." if len(regime.description) > 65 else ""),
                )
            else:
                ext_table.add_row(f"  {domain.title()}", "N/A", "-", "Data unavailable")

        console.print(ext_table)
        console.print()

        # ── Regime Changes (30d) ──────────────────────────────────────────
        try:
            changes = get_recent_changes(days=30)
            if changes:
                console.print("[bold]Regime Changes (30d):[/bold]")
                for c in changes:
                    console.print(
                        f"  {c['domain'].title()}: "
                        f"{c.get('from_label', 'N/A')} → {c.get('to_label', 'N/A')} "
                        f"({c.get('date', '?')}) ⚠️"
                    )
                console.print()
        except Exception:
            pass

        # ── Monte Carlo Adjustments ───────────────────────────────────────
        mc_params = state.to_monte_carlo_params()
        drivers = mc_params.pop("_drivers", {})

        mc_table = Table(
            title="Monte Carlo Adjustments",
            show_header=True,
            header_style="bold green",
        )
        mc_table.add_column("Parameter")
        mc_table.add_column("Base", justify="right")
        mc_table.add_column("Adjusted", justify="right")
        mc_table.add_column("Driven By")

        def _mc_row(label: str, key: str, base: float, is_mult: bool = False):
            val = mc_params[key]
            driver_list = drivers.get(key, [])
            driver_str = ", ".join(driver_list) if driver_list else "—"

            if is_mult:
                base_str = f"{base:.1f}x"
                adj_str = f"{val:.2f}x"
                if val > base:
                    adj_str = f"[red]{adj_str}[/red]"
                elif val < base:
                    adj_str = f"[green]{adj_str}[/green]"
            else:
                base_str = f"{base*100:+.1f}%"
                adj_str = f"{val*100:+.1f}%"
                if val > base:
                    adj_str = f"[red]{adj_str}[/red]"
                elif val < base:
                    adj_str = f"[green]{adj_str}[/green]"

            mc_table.add_row(label, base_str, adj_str, driver_str[:55])

        _mc_row("Equity Drift", "equity_drift_adj", 0.0)
        _mc_row("Equity Vol", "equity_vol_adj", 1.0, is_mult=True)
        _mc_row("IV Drift", "iv_drift_adj", 0.0)
        _mc_row("Jump Prob", "jump_prob_adj", 1.0, is_mult=True)
        _mc_row("Spread Drift", "spread_drift_adj", 0.0)

        console.print(mc_table)
        console.print()

        # ── Portfolio Implication (LLM) ───────────────────────────────────
        try:
            from lox.config import load_settings as _ls
            s = _ls()
            if s.openai_api_key:
                import openai
                scores_dict = {
                    domain: getattr(state, domain).score
                    for domain in ["growth", "inflation", "volatility", "credit",
                                   "rates", "funding", "consumer", "fiscal",
                                   "positioning", "monetary", "usd", "commodities"]
                    if getattr(state, domain) is not None
                }
                scores_dict["macro_quadrant"] = state.macro_quadrant
                client = openai.OpenAI(api_key=s.openai_api_key)
                resp = client.chat.completions.create(
                    model=s.openai_model,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Given these regime scores (0=risk-on, 100=risk-off): {scores_dict}, "
                            "write ONE sentence on portfolio positioning implications. "
                            "Max 20 words. No hedging language."
                        ),
                    }],
                    max_tokens=60,
                    temperature=0.3,
                )
                implication = resp.choices[0].message.content.strip()
                console.print(f"[bold]Portfolio implication:[/bold] {implication}")
                console.print()
        except Exception:
            pass

        # ── Score Legend ───────────────────────────────────────────────────
        console.print(
            "[dim]Score guide: 0 = strong risk-on signal → 100 = strong risk-off signal[/dim]"
        )
        console.print(
            "[dim]Traffic lights: 🟢 <45 (favorable) | 🟡 45-59 (mixed) | 🔴 ≥60 (unfavorable)[/dim]"
        )
        console.print()
        console.print("[dim]Run with --json for ML-friendly feature export[/dim]")

    @app.command("transitions")
    def regime_transitions(
        domain: str = typer.Option("risk_category", "--domain", "-d", help="Regime domain"),
        horizon: int = typer.Option(21, "--horizon", help="Horizon in trading days (21=1mo, 63=3mo, 126=6mo)"),
        current: str = typer.Option("cautious", "--current", "-c", help="Current regime state"),
        adjust: bool = typer.Option(True, "--adjust/--no-adjust", help="Adjust for leading indicators"),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh regime data"),
    ):
        """
        Show regime transition probabilities for Monte Carlo.

        By default, adjusts probabilities based on leading indicators
        (yield curve, VIX, credit spreads, funding stress).

        Use --no-adjust for raw historical frequencies.
        """
        from lox.regimes import get_transition_matrix, build_unified_regime_state
        from lox.regimes.transitions import (
            get_adjusted_transition_matrix,
            LEADING_INDICATORS,
            LeadingIndicatorSignals,
        )

        console = Console()
        settings = load_settings()

        # Get adjusted or base matrix
        signals = LeadingIndicatorSignals()
        if adjust:
            with console.status("[cyan]Loading regime data for signal analysis...[/cyan]"):
                try:
                    unified_state = build_unified_regime_state(settings=settings, refresh=refresh)
                    matrix, signals = get_adjusted_transition_matrix(domain, horizon, unified_state)
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load signals ({e}), using base matrix[/yellow]")
                    matrix = get_transition_matrix(domain, horizon_days=horizon)
        else:
            matrix = get_transition_matrix(domain, horizon_days=horizon)

        # Header with clear explanation
        console.print()
        console.print(Panel(
            "[bold]What is this?[/bold]\n"
            "Probability of the market regime changing over the forecast horizon.\n"
            "Used to weight Monte Carlo scenarios by how likely each regime is.\n\n"
            "[bold]Methodology:[/bold] Historical frequency analysis\n"
            "• Based on empirical regime transitions from 2000-2024 market data\n"
            "• NOT ML/PCA - these are observed frequencies of regime changes\n"
            "• Longer horizons mean-revert toward equal probabilities\n\n"
            f"[bold]Horizon:[/bold] {horizon} trading days (~{horizon//21} month{'s' if horizon > 21 else ''})",
            title=f"📊 Regime Transition Matrix — {domain.replace('_', ' ').title()}",
            border_style="cyan",
        ))

        # Transition matrix with clear labels
        console.print()
        console.print("[bold]Transition Probabilities[/bold]")
        console.print("[dim]Read as: If TODAY is (row), what's the probability NEXT PERIOD is (column)?[/dim]")
        console.print()

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("If TODAY is...", style="bold")
        for s in matrix.states:
            table.add_column(f"→ {s.replace('_', ' ').title()}", justify="right")

        for i, from_state in enumerate(matrix.states):
            row = [from_state.replace("_", " ").title()]
            for j, to_state in enumerate(matrix.states):
                prob = matrix.matrix[i, j]
                if i == j:
                    row.append(f"[bold cyan]{prob:.0%}[/bold cyan] (stay)")
                elif prob > 0.2:
                    row.append(f"[yellow]{prob:.0%}[/yellow]")
                else:
                    row.append(f"[dim]{prob:.0%}[/dim]")
            table.add_row(*row)

        console.print(table)
        console.print()

        # Forecast from current state
        console.print(f"[bold]Your Forecast (starting from '{current}'):[/bold]")
        console.print(f"[dim]In {horizon} trading days (~{horizon//21} month{'s' if horizon > 21 else ''}), the market will likely be:[/dim]")
        console.print()

        probs = matrix.get_next_state_probs(current)
        for s, prob in sorted(probs.items(), key=lambda x: -x[1]):
            bar_len = int(prob * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)

            if s == current:
                interp = "(no change)"
            elif s == "risk_off":
                interp = "(deterioration)"
            elif s == "risk_on":
                interp = "(improvement)"
            else:
                interp = ""

            console.print(f"  {s.replace('_', ' ').title():12} {bar} [bold]{prob:.0%}[/bold] {interp}")

        # Show active leading indicators if adjusting
        active_signals = signals.active_signals()
        if adjust and active_signals:
            console.print()
            console.print("[bold yellow]⚠ Active Warning Signals (adjusting probabilities):[/bold yellow]")
            risk_off_mult, risk_on_mult = signals.risk_adjustment_factor()
            for sig in active_signals:
                indicator = LEADING_INDICATORS.get(sig, {})
                console.print(f"  • [yellow]{sig.replace('_', ' ').title()}[/yellow]: {indicator.get('description', '')}")
            console.print()
            console.print(f"  [dim]Combined adjustment: risk_off ×{risk_off_mult:.1f}, risk_on ×{risk_on_mult:.1f}[/dim]")
        elif adjust:
            console.print()
            console.print("[green]✓ No warning signals active - using base historical probabilities[/green]")

        console.print()
        console.print("[dim]Usage: These probabilities weight Monte Carlo scenarios.[/dim]")
        console.print("[dim]       Use --no-adjust to see raw historical frequencies.[/dim]")

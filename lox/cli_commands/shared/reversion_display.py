"""
Rich display panels for the mean-reversion screener (lox suggest).

Follows existing conventions: Table(box=None, padding=(0,2)),
Panel.fit(border_style="cyan"), score colors green<35 yellow<65 red>=65.
"""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lox.suggest.reversion import ReversionCandidate, ReversionScreenResult


def _score_style(score: float) -> str:
    if score >= 65:
        return "bold green"
    if score >= 45:
        return "yellow"
    return "dim"


def _ret_style(ret: float) -> str:
    return "green" if ret >= 0 else "red"


def _setup_label(signal: str) -> tuple[str, str]:
    """Return (label, style) showing the reversion trade direction."""
    if signal == "EXTENDED_UP":
        return "Overbought", "bold red"
    if signal == "EXTENDED_DOWN":
        return "Oversold", "bold green"
    return "—", "dim"


def _dir_style(direction: str) -> str:
    return "bold green" if direction == "LONG" else "bold red"


def _conv_style(conviction: str) -> str:
    if conviction == "HIGH":
        return "bold green"
    if conviction == "MEDIUM":
        return "yellow"
    return "dim"


def render_regime_context(console: Console, result: ReversionScreenResult) -> None:
    """One-line composite regime context header."""
    if not result.composite_regime:
        return
    regime = result.composite_regime
    conf = result.composite_confidence
    style = "green" if "RISK_ON" in regime else "red" if "RISK_OFF" in regime or "STAG" in regime else "yellow"
    console.print(
        f"  [{style}]{regime}[/{style}] ({conf:.0f}% confidence)"
        f"  |  {result.universe_size} scanned  |  {result.extended_count} extended",
        style="dim",
    )
    console.print()


def render_screener_table(console: Console, result: ReversionScreenResult) -> None:
    """Panel 1: Momentum screener — all extended tickers sorted by |z-score|."""
    extended = [s for s in result.all_scans if s.signal != "NEUTRAL"]
    if not extended:
        console.print("[dim]No extended tickers found at current threshold.[/dim]")
        return

    t = Table(box=None, padding=(0, 2), title="Reversion Screener", title_style="bold cyan")
    t.add_column("Ticker", style="bold")
    t.add_column("Name", style="dim", max_width=14, no_wrap=True, overflow="ellipsis")
    t.add_column("Price", justify="right")
    t.add_column("20d Ret", justify="right")
    t.add_column("60d Ret", justify="right")
    t.add_column("Z-Score", justify="right")
    t.add_column("RSI", justify="right")
    t.add_column("Setup", justify="right")

    for s in extended[:15]:  # cap display
        z_style = "bold red" if abs(s.zscore_20d) > 2.0 else "yellow" if abs(s.zscore_20d) > 1.5 else "dim"
        rsi_style = "red" if s.rsi_14 > 70 else "green" if s.rsi_14 < 30 else "dim"
        # Combined setup: "OB → SHORT" or "OS → LONG"
        if s.signal == "EXTENDED_UP":
            setup_str = "[bold red]OB → SHORT[/bold red]"
        elif s.signal == "EXTENDED_DOWN":
            setup_str = "[bold green]OS → LONG[/bold green]"
        else:
            setup_str = "[dim]—[/dim]"
        t.add_row(
            s.ticker,
            s.description[:14],
            f"${s.current_price:,.2f}",
            f"[{_ret_style(s.ret_20d)}]{s.ret_20d:+.1%}[/{_ret_style(s.ret_20d)}]",
            f"[{_ret_style(s.ret_60d)}]{s.ret_60d:+.1%}[/{_ret_style(s.ret_60d)}]",
            f"[{z_style}]{s.zscore_20d:+.1f}σ[/{z_style}]",
            f"[{rsi_style}]{s.rsi_14:.0f}[/{rsi_style}]",
            setup_str,
        )

    console.print(t)
    console.print()


def render_attribution_panel(console: Console, candidates: list[ReversionCandidate]) -> None:
    """Panel 2: Factor attribution for each candidate."""
    if not candidates:
        return

    t = Table(box=None, padding=(0, 2), title="Factor Attribution", title_style="bold cyan")
    t.add_column("Ticker", style="bold")
    t.add_column("Primary Factor")
    t.add_column("Loading", justify="right")
    t.add_column("Regime Driver")
    t.add_column("Rev Score", justify="right")
    t.add_column("Why", max_width=60)

    for c in candidates:
        attr = c.attribution
        assess = c.assessment

        # Compact regime driver: "commodities ↑ fast" or "growth ↓ slow"
        drivers_parts = []
        for domain, trend_dir, vel in attr.regime_drivers[:2]:
            arrow = "↑" if "DETER" in trend_dir or "WORSEN" in trend_dir else "↓" if "IMPROV" in trend_dir else "→"
            speed = "fast" if abs(vel) > 3 else "slow" if abs(vel) < 1 else ""
            drivers_parts.append(f"{domain} {arrow} {speed}".strip())
        drivers_str = ", ".join(drivers_parts) if drivers_parts else "—"

        rev_style = _score_style(assess.reversion_score)

        # Build a readable thesis: what moved, why, and whether factor is fading
        move_dir = "up" if c.scan.ret_20d > 0 else "down"
        thesis = f"{c.scan.ret_20d:+.1%} 20d ({c.scan.zscore_20d:+.1f}σ) on {attr.primary_factor}"
        if assess.factor_decelerating:
            thesis += " — factor fading"

        t.add_row(
            c.scan.ticker,
            attr.primary_factor,
            f"{attr.primary_loading:+.1f}",
            drivers_str,
            f"[{rev_style}]{assess.reversion_score:.0f}[/{rev_style}]",
            thesis,
        )

    console.print(t)
    console.print()


def render_trade_recommendations(console: Console, candidates: list[ReversionCandidate]) -> None:
    """Panel 3: Trade recommendations — instrument, sizing, conviction."""
    if not candidates:
        return

    t = Table(box=None, padding=(0, 2), title="Trade Recommendations", title_style="bold cyan")
    t.add_column("Ticker", style="bold")
    t.add_column("Dir")
    t.add_column("Instrument")
    t.add_column("DTE")
    t.add_column("Delta")
    t.add_column("IV Rank", justify="right")
    t.add_column("RV 20d", justify="right")
    t.add_column("Size", justify="right")
    t.add_column("% NAV", justify="right")
    t.add_column("Conv")

    for c in candidates:
        r = c.recommendation
        iv_str = f"{r.iv_rank:.0f}" if r.iv_rank is not None else "—"
        rv_str = f"{r.rv_20d:.0%}" if r.rv_20d is not None else "—"
        t.add_row(
            r.ticker,
            f"[{_dir_style(r.direction)}]{r.direction}[/{_dir_style(r.direction)}]",
            r.instrument,
            r.dte_target,
            r.delta_target,
            iv_str,
            rv_str,
            f"${r.notional:,.0f}",
            f"{r.pct_nav:.1%}",
            f"[{_conv_style(r.conviction)}]{r.conviction}[/{_conv_style(r.conviction)}]",
        )

    console.print(t)

    # Rationale footnotes
    for c in candidates:
        r = c.recommendation
        if r.rationale:
            console.print(f"  [dim]{r.ticker}: {r.rationale}[/dim]")
    console.print()


def render_reversion_dashboard(console: Console, result: ReversionScreenResult) -> None:
    """Full 4-panel dashboard."""
    console.print()
    console.print("  [bold cyan]LOX SUGGEST[/bold cyan]  Mean-Reversion Screener", style="bold")
    console.print("  " + "─" * 50)
    console.print()

    render_regime_context(console, result)
    render_screener_table(console, result)

    if result.candidates:
        render_attribution_panel(console, result.candidates)
        render_trade_recommendations(console, result.candidates)
    else:
        console.print("[dim]  No candidates met the reversion threshold.[/dim]")

    console.print(
        "  [dim italic]Regime-conditioned signals, not recommendations. "
        "Do your own analysis.[/dim italic]"
    )
    console.print()


def format_reversion_for_llm(result: ReversionScreenResult) -> str:
    """Format screener results as markdown for LLM system prompt context."""
    lines = [
        f"## Reversion Screener Results",
        f"Composite regime: {result.composite_regime} ({result.composite_confidence:.0f}% confidence)",
        f"Universe: {result.universe_size} tickers scanned, {result.extended_count} extended",
        "",
    ]

    if result.candidates:
        lines.append("### Top Reversion Candidates")
        for c in result.candidates:
            s = c.scan
            a = c.assessment
            r = c.recommendation
            lines.append(
                f"- **{s.ticker}** ({s.description}): "
                f"{s.ret_20d:+.1%} 20d (z={s.zscore_20d:+.1f}σ), RSI={s.rsi_14:.0f}, "
                f"reversion={a.reversion_score:.0f}/100"
            )
            lines.append(f"  Factor: {c.attribution.attribution_text}")
            lines.append(f"  Thesis: {a.thesis}")
            lines.append(
                f"  Reco: {r.direction} via {r.instrument}"
                f"{f', DTE {r.dte_target}, Δ {r.delta_target}' if r.instrument != 'equity' else ''}"
                f", ${r.notional:,.0f} ({r.pct_nav:.1%} NAV), {r.conviction} conviction"
            )
            if r.iv_rank is not None:
                lines.append(f"  IV rank: {r.iv_rank:.0f}, RV: {r.rv_20d:.0%}")
            lines.append("")
    else:
        lines.append("No extended tickers found at current threshold.")

    return "\n".join(lines)


def format_reversion_json(result: ReversionScreenResult) -> dict:
    """JSON serialization for --json flag."""
    return {
        "composite_regime": result.composite_regime,
        "composite_confidence": result.composite_confidence,
        "universe_size": result.universe_size,
        "extended_count": result.extended_count,
        "scan_timestamp": result.scan_timestamp,
        "candidates": [
            {
                "ticker": c.scan.ticker,
                "description": c.scan.description,
                "price": c.scan.current_price,
                "ret_5d": round(c.scan.ret_5d, 4),
                "ret_20d": round(c.scan.ret_20d, 4),
                "ret_60d": round(c.scan.ret_60d, 4),
                "zscore_20d": round(c.scan.zscore_20d, 2),
                "rsi_14": round(c.scan.rsi_14, 1),
                "dist_200d_pct": round(c.scan.dist_200d_pct, 4),
                "signal": c.scan.signal,
                "primary_factor": c.attribution.primary_factor,
                "primary_loading": c.attribution.primary_loading,
                "attribution": c.attribution.attribution_text,
                "reversion_score": round(c.assessment.reversion_score, 1),
                "playbook_reversion": c.assessment.playbook_reversion,
                "playbook_exp_return": round(c.assessment.playbook_exp_return, 4),
                "factor_decelerating": c.assessment.factor_decelerating,
                "thesis": c.assessment.thesis,
                "direction": c.recommendation.direction,
                "instrument": c.recommendation.instrument,
                "dte_target": c.recommendation.dte_target,
                "delta_target": c.recommendation.delta_target,
                "iv_rank": c.recommendation.iv_rank,
                "rv_20d": round(c.recommendation.rv_20d, 4) if c.recommendation.rv_20d else None,
                "conviction": c.recommendation.conviction,
                "notional": c.recommendation.notional,
                "pct_nav": round(c.recommendation.pct_nav, 4),
                "rationale": c.recommendation.rationale,
            }
            for c in result.candidates
        ],
        "all_extended": [
            {
                "ticker": s.ticker,
                "ret_20d": round(s.ret_20d, 4),
                "zscore_20d": round(s.zscore_20d, 2),
                "signal": s.signal,
            }
            for s in result.all_scans if s.signal != "NEUTRAL"
        ],
    }

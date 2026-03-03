"""CLI command for the Earnings regime (v2).

v2 features:
  - Sparkline momentum (beat rate + surprise trailing trends)
  - Sector heatmap (per-GICS beat rate, surprise, revision, signal)
  - Cross-regime signals (growth, credit, vol correlation)
  - Delta tracking (--delta 7d)
  - ML features export (--features) and JSON export (--json)
"""
from __future__ import annotations

from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table as RichTable

from lox.cli_commands.shared.regime_display import render_regime_panel
from lox.config import load_settings


# ─────────────────────────────────────────────────────────────────────────────
# Sparkline / trend helpers
# ─────────────────────────────────────────────────────────────────────────────

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo if hi != lo else 1.0
    return "".join(
        _SPARK_CHARS[min(len(_SPARK_CHARS) - 1, int((v - lo) / span * (len(_SPARK_CHARS) - 1)))]
        for v in values
    )


def _trend_label(values: list[float], higher_is_worse: bool = True) -> str:
    """Compare first half vs second half to determine trend direction."""
    if len(values) < 4:
        return ""
    mid = len(values) // 2
    old_avg = sum(values[:mid]) / mid
    new_avg = sum(values[mid:]) / (len(values) - mid)
    delta = new_avg - old_avg
    threshold = (max(values) - min(values)) * 0.1 if max(values) != min(values) else 0.01
    if abs(delta) < threshold:
        return "[dim]stable[/dim]"
    if higher_is_worse:
        return "[red]deteriorating[/red]" if delta < 0 else "[green]improving[/green]"
    return "[green]improving[/green]" if delta > 0 else "[red]deteriorating[/red]"


# ─────────────────────────────────────────────────────────────────────────────
# Context helpers
# ─────────────────────────────────────────────────────────────────────────────

def _beat_context(beat_rate):
    if beat_rate is None:
        return "—"
    if beat_rate > 80:
        return "blowout season"
    if beat_rate > 75:
        return "very strong"
    if beat_rate > 70:
        return "above average"
    if beat_rate > 65:
        return "normal"
    if beat_rate > 60:
        return "weakening"
    if beat_rate > 55:
        return "deteriorating"
    return "recessionary"


def _surprise_context(avg_surprise):
    if avg_surprise is None:
        return "—"
    if avg_surprise > 7:
        return "exceptional beats"
    if avg_surprise > 4:
        return "healthy beats"
    if avg_surprise > 1:
        return "modest beats"
    if avg_surprise > 0:
        return "barely beating"
    return "net misses"


def _revision_context(ratio):
    if ratio is None:
        return "—"
    if ratio > 0.20:
        return "broad upgrades"
    if ratio > 0.10:
        return "net positive"
    if ratio > 0:
        return "mildly positive"
    if ratio > -0.10:
        return "mildly negative"
    if ratio > -0.20:
        return "net negative"
    return "broad downgrades"


def _dispersion_context(disp):
    if disp is None:
        return "—"
    if disp > 25:
        return "very narrow leadership"
    if disp > 15:
        return "moderate divergence"
    if disp > 8:
        return "healthy spread"
    return "broad-based"


def _sector_signal(beat_rate):
    """Return a colored signal tag for a sector beat rate."""
    if beat_rate is None:
        return "[dim]—[/dim]"
    if beat_rate > 75:
        return "[green]strong[/green]"
    if beat_rate > 65:
        return "[green]healthy[/green]"
    if beat_rate > 55:
        return "[yellow]weakening[/yellow]"
    return "[red]stressed[/red]"


def _density_context(density):
    if density is None:
        return "—"
    if density > 200:
        return "peak season"
    if density > 100:
        return "mid-season"
    return "off-season"


# ─────────────────────────────────────────────────────────────────────────────
# Block 1: Earnings Momentum Sparklines
# ─────────────────────────────────────────────────────────────────────────────

def _show_earnings_sparklines(console: Console, inputs: dict) -> None:
    """Display beat rate and surprise trend sparklines."""
    beat_series = inputs.get("beat_rate_series") or []
    surprise_series = inputs.get("surprise_series") or []

    if not beat_series and not surprise_series:
        return

    lines: list[str] = []

    if beat_series:
        current = inputs.get("beat_rate")
        curr_str = f"{current:.0f}%" if current is not None else ""
        lines.append(
            f"  Beat Rate    {_sparkline(beat_series)}  "
            f"{curr_str}  {_trend_label(beat_series, higher_is_worse=False)}"
        )

    if surprise_series:
        current = inputs.get("avg_surprise_pct")
        curr_str = f"{current:+.1f}%" if current is not None else ""
        lines.append(
            f"  Avg Surprise {_sparkline(surprise_series)}  "
            f"{curr_str}  {_trend_label(surprise_series, higher_is_worse=False)}"
        )

    if lines:
        console.print()
        console.print("[dim]─── Earnings Momentum (90d weekly) ───────────────────────────────[/dim]")
        for ln in lines:
            console.print(ln)


# ─────────────────────────────────────────────────────────────────────────────
# Block 2: Sector Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def _show_sector_heatmap(console: Console, inputs: dict) -> None:
    """Display per-sector beat rate, surprise, revision, and signal."""
    sector_stats = inputs.get("sector_stats") or {}
    if not sector_stats:
        return

    # Filter to sectors with meaningful data (>= 5 reporters)
    qualified = [
        (name, stats)
        for name, stats in sector_stats.items()
        if stats.get("count", 0) >= 5
    ]
    if not qualified:
        return

    # Sort by beat rate descending
    qualified.sort(key=lambda x: x[1].get("beat_rate") or 0, reverse=True)

    table = RichTable(
        title="Sector Earnings Heatmap",
        show_header=True,
        header_style="bold yellow",
        box=None,
        padding=(0, 1),
    )
    table.add_column("Sector", min_width=22)
    table.add_column("Beat Rate", justify="right")
    table.add_column("Avg Surprise", justify="right")
    table.add_column("Revision", justify="right")
    table.add_column("N", justify="right", style="dim")
    table.add_column("Signal")

    for name, stats in qualified:
        br = stats.get("beat_rate")
        surprise = stats.get("avg_surprise")
        revision = stats.get("revision_ratio")
        count = stats.get("count", 0)

        # Color-code beat rate
        if br is not None:
            if br > 70:
                br_str = f"[green]{br:.0f}%[/green]"
            elif br > 55:
                br_str = f"[yellow]{br:.0f}%[/yellow]"
            else:
                br_str = f"[red]{br:.0f}%[/red]"
        else:
            br_str = "[dim]—[/dim]"

        surp_str = f"{surprise:+.1f}%" if surprise is not None else "[dim]—[/dim]"
        rev_str = f"{revision:+.2f}" if revision is not None else "[dim]—[/dim]"

        table.add_row(
            name,
            br_str,
            surp_str,
            rev_str,
            str(count),
            _sector_signal(br),
        )

    # Summary row
    disp = inputs.get("sector_dispersion")
    beating = inputs.get("sectors_beating", 0)
    total_rated = inputs.get("total_sectors_rated", 0)

    console.print()
    console.print(table)

    if disp is not None:
        console.print(
            f"  Dispersion: {disp:.0f}pp (best-worst) | "
            f"{beating}/{total_rated} sectors above 65% beat rate"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Block 3: Cross-Regime Signals
# ─────────────────────────────────────────────────────────────────────────────

def _show_cross_regime_signals(console: Console, earnings_score: float) -> None:
    """Show cross-regime confirmation/divergence signals for earnings."""
    lines: list[str] = []

    try:
        from lox.data.regime_history import get_score_series

        for domain, display in [("growth", "Growth"), ("credit", "Credit"), ("volatility", "Vol")]:
            series = get_score_series(domain)
            if not series:
                continue
            latest = series[-1]
            sc = latest.get("score")
            lb = latest.get("label", "")
            if not isinstance(sc, (int, float)):
                continue
            short_lb = lb.split("(")[0].strip() if "(" in lb else lb

            if domain == "growth":
                if sc > 65 and earnings_score > 50:
                    lines.append(
                        f"  {display} score {sc:.0f} ({short_lb}) + earnings weakening → "
                        f"[yellow]macro deterioration confirming earnings stress[/yellow]"
                    )
                elif sc > 65 and earnings_score < 40:
                    lines.append(
                        f"  {display} score {sc:.0f} ({short_lb}) but earnings still strong → "
                        f"[dim]earnings haven't caught up yet (lagging indicator)[/dim]"
                    )
                elif sc < 40 and earnings_score > 50:
                    lines.append(
                        f"  {display} score {sc:.0f} ({short_lb}) solid but earnings soft → "
                        f"[yellow]possible micro-level divergence[/yellow]"
                    )
                else:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [dim]neutral[/dim]")

            elif domain == "credit":
                if sc > 50 and earnings_score > 50:
                    lines.append(
                        f"  {display} score {sc:.0f} ({short_lb}) + earnings weakness → "
                        f"[red]fundamental + market stress aligned[/red]"
                    )
                elif sc > 50 and earnings_score < 35:
                    lines.append(
                        f"  {display} score {sc:.0f} ({short_lb}) widening but earnings strong → "
                        f"[yellow]credit leading or overreacting[/yellow]"
                    )
                elif sc < 35 and earnings_score > 50:
                    lines.append(
                        f"  {display} score {sc:.0f} ({short_lb}) tight + earnings weak → "
                        f"[yellow]market hasn't repriced earnings risk[/yellow]"
                    )
                else:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [dim]neutral[/dim]")

            elif domain == "volatility":
                if sc > 60 and earnings_score < 35:
                    lines.append(
                        f"  {display} score {sc:.0f} ({short_lb}) elevated but earnings strong → "
                        f"[yellow]market mispricing earnings strength[/yellow]"
                    )
                elif sc > 60 and earnings_score > 50:
                    lines.append(
                        f"  {display} score {sc:.0f} ({short_lb}) + earnings stress → "
                        f"[red]risk-off confirmed across assets[/red]"
                    )
                else:
                    lines.append(f"  {display} score {sc:.0f} ({short_lb}) — [dim]neutral[/dim]")

    except Exception:
        pass

    if lines:
        console.print()
        console.print("[dim]─── Cross-Regime Signals ──────────────────────────────────────[/dim]")
        for ln in lines:
            console.print(ln)


# ─────────────────────────────────────────────────────────────────────────────
# Block 4: Sector Stock Basket Drill-Down
# ─────────────────────────────────────────────────────────────────────────────

def _consensus_style(label: str) -> str:
    """Color-code analyst consensus labels."""
    if label in ("Strong Buy", "Buy"):
        return f"[green]{label}[/green]"
    if label == "Hold":
        return f"[yellow]{label}[/yellow]"
    if label in ("Sell", "Strong Sell"):
        return f"[red]{label}[/red]"
    return f"[dim]{label}[/dim]"


def _show_sector_basket(console: Console, data: dict, json_out: bool = False) -> None:
    """Display per-stock earnings basket for a single sector."""
    import json as json_mod

    sector = data["sector"]
    stocks = data["stocks"]
    summary = data["sector_summary"]
    sp500_br = data.get("sp500_beat_rate")

    if json_out:
        console.print_json(json_mod.dumps({
            "sector": sector,
            "sector_summary": summary,
            "sp500_beat_rate": sp500_br,
            "stocks": stocks,
        }, default=str))
        return

    count = len(stocks)

    # ── Sector summary header ─────────────────────────────────────────
    br = summary.get("beat_rate")
    surp = summary.get("avg_surprise")
    rev = summary.get("revision_ratio")

    header_lines: list[str] = []
    if br is not None:
        comparison = f" (vs {sp500_br:.0f}% S&P avg)" if sp500_br is not None else ""
        header_lines.append(f"  Beat Rate     [bold]{br:.0f}%[/bold]{comparison}")
    if surp is not None:
        header_lines.append(f"  Avg Surprise  [bold]{surp:+.1f}%[/bold]  {_surprise_context(surp)}")
    if rev is not None:
        header_lines.append(f"  Revision      [bold]{rev:+.2f}[/bold]  {_revision_context(rev)}")

    console.print()
    console.print(Panel.fit(
        "\n".join(header_lines),
        title=f"[bold]{sector}[/bold] Earnings Basket ({count} names)",
        subtitle="S&P 500",
        border_style="cyan",
    ))

    if not stocks:
        console.print("[dim]  No earnings data for this sector in trailing 90 days.[/dim]")
        return

    # ── Top Beats table ───────────────────────────────────────────────
    beats = [s for s in stocks if s.get("beat")]
    misses = [s for s in stocks if not s.get("beat")]

    if beats:
        top_n = beats[:15]  # top 15 beats
        table = RichTable(
            title=f"Top Beats ({len(beats)} names)",
            show_header=True,
            header_style="bold green",
            box=None,
            padding=(0, 1),
        )
        table.add_column("Ticker", style="bold", min_width=6)
        table.add_column("Name", min_width=20)
        table.add_column("Surprise", justify="right")
        table.add_column("EPS", justify="right", style="dim")
        table.add_column("Date", style="dim")
        table.add_column("Consensus")

        for s in top_n:
            surp_str = f"[green]+{s['surprise_pct']:.1f}%[/green]"
            eps_str = f"{s['actual_eps']:.2f} vs {s['estimated_eps']:.2f}"
            table.add_row(
                s["symbol"],
                _truncate(s["name"], 24),
                surp_str,
                eps_str,
                s.get("date", ""),
                _consensus_style(s.get("consensus", "—")),
            )

        console.print()
        console.print(table)

    # ── Worst Misses table ────────────────────────────────────────────
    if misses:
        worst_n = misses[-10:]  # worst 10 misses (already sorted desc)
        worst_n.reverse()  # worst first
        table = RichTable(
            title=f"Misses ({len(misses)} names)",
            show_header=True,
            header_style="bold red",
            box=None,
            padding=(0, 1),
        )
        table.add_column("Ticker", style="bold", min_width=6)
        table.add_column("Name", min_width=20)
        table.add_column("Surprise", justify="right")
        table.add_column("EPS", justify="right", style="dim")
        table.add_column("Date", style="dim")
        table.add_column("Consensus")

        for s in worst_n:
            surp_str = f"[red]{s['surprise_pct']:+.1f}%[/red]"
            eps_str = f"{s['actual_eps']:.2f} vs {s['estimated_eps']:.2f}"
            table.add_row(
                s["symbol"],
                _truncate(s["name"], 24),
                surp_str,
                eps_str,
                s.get("date", ""),
                _consensus_style(s.get("consensus", "—")),
            )

        console.print()
        console.print(table)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if too long."""
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


# ─────────────────────────────────────────────────────────────────────────────
# Core entry point
# ─────────────────────────────────────────────────────────────────────────────

def earnings_snapshot(
    *,
    llm: bool = False,
    ticker: str = "",
    refresh: bool = False,
    features: bool = False,
    json_out: bool = False,
    delta: str = "",
    sector: str = "",
) -> None:
    """Entry point for ``lox regime earnings``."""
    settings = load_settings()
    console = Console()

    # ── Sector drill-down (separate flow) ─────────────────────────────
    if sector:
        from lox.altdata.earnings_market import get_sector_stocks

        data = get_sector_stocks(sector, settings=settings, refresh=refresh)
        if data.get("error"):
            print(f"[yellow]⚠  {data['error']}[/yellow]")
            return
        _show_sector_basket(console, data, json_out=json_out)
        return

    # ── Normal earnings dashboard flow ────────────────────────────────
    from lox.cli_commands.shared.labs_utils import (
        handle_output_flags, parse_delta_period, show_delta_summary,
        get_delta_metrics, save_snapshot,
    )

    from lox.altdata.earnings_market import compute_earnings_regime_inputs

    inputs = compute_earnings_regime_inputs(settings=settings, refresh=refresh)

    if inputs.get("error"):
        print(f"[yellow]⚠  {inputs['error']}[/yellow]")

    from lox.earnings.regime import classify_earnings_regime

    # v2: extract sector-level params
    top = inputs.get("top_sectors") or []
    worst = inputs.get("worst_sectors") or []
    worst_br = worst[-1][1]["beat_rate"] if worst else None

    result = classify_earnings_regime(
        beat_rate=inputs["beat_rate"],
        avg_surprise_pct=inputs["avg_surprise_pct"],
        net_revision_ratio=inputs["net_revision_ratio"],
        reporting_density=inputs["reporting_density"],
        sector_dispersion=inputs.get("sector_dispersion"),
        worst_sector_beat_rate=worst_br,
        sectors_beating=inputs.get("sectors_beating"),
        total_sectors_rated=inputs.get("total_sectors_rated"),
        best_sector=top[0][0] if top else None,
        worst_sector=worst[-1][0] if worst else None,
    )

    # ── Build snapshot for delta / export ─────────────────────────────────
    total = inputs.get("total_sp500_surprises_90d", 0)
    season = "Peak Season" if (inputs.get("reporting_density") or 0) > 200 else "Off-Season"

    snapshot_data = {
        "beat_rate": inputs["beat_rate"],
        "avg_surprise_pct": inputs["avg_surprise_pct"],
        "net_revision_ratio": inputs["net_revision_ratio"],
        "reporting_density": inputs["reporting_density"],
        "sector_dispersion": inputs.get("sector_dispersion"),
        "sectors_beating": inputs.get("sectors_beating"),
        "total_sectors_rated": inputs.get("total_sectors_rated"),
        "total_sp500_surprises_90d": total,
        "season": season,
    }

    feature_dict = result.to_feature_dict()

    # Save snapshot for delta tracking
    save_snapshot("earnings", snapshot_data, result.label)

    # ── Handle --features and --json flags ────────────────────────────────
    if handle_output_flags(
        domain="earnings",
        snapshot=snapshot_data,
        features=feature_dict,
        regime=result.label,
        regime_description=result.description,
        asof=inputs.get("asof"),
        output_json=json_out,
        output_features=features,
    ):
        return

    # ── Handle --delta flag ───────────────────────────────────────────────
    if delta:
        delta_days = parse_delta_period(delta)
        metric_keys = [
            "Beat Rate:beat_rate:%",
            "Avg Surprise:avg_surprise_pct:%",
            "Net Revision:net_revision_ratio:",
            "Reporting:reporting_density:",
            "Sector Dispersion:sector_dispersion:pp",
            "Sectors Beating:sectors_beating:",
        ]
        metrics_for_delta, prev_regime = get_delta_metrics(
            "earnings", snapshot_data, metric_keys, delta_days,
        )
        show_delta_summary(
            "earnings", result.label, prev_regime, metrics_for_delta, delta_days,
        )
        if prev_regime is None:
            console.print(
                f"\n[dim]No cached data from {delta_days}d ago. "
                f"Run `lox regime earnings` daily to build history.[/dim]"
            )
        return

    # ── Build metrics table ───────────────────────────────────────────────
    def _v(x, fmt="{:.1f}"):
        return fmt.format(x) if x is not None else "n/a"

    disp = inputs.get("sector_dispersion")
    best_name = top[0][0] if top else "—"
    best_br = f"{top[0][1]['beat_rate']:.0f}%" if top else ""
    worst_name = worst[-1][0] if worst else "—"
    worst_br_str = f"{worst[-1][1]['beat_rate']:.0f}%" if worst else ""

    metrics = [
        {
            "name": "Beat Rate",
            "value": _v(inputs["beat_rate"], "{:.0f}%"),
            "context": _beat_context(inputs["beat_rate"]),
        },
        {
            "name": "Avg Surprise %",
            "value": _v(inputs["avg_surprise_pct"], "{:+.1f}%"),
            "context": _surprise_context(inputs["avg_surprise_pct"]),
        },
        {
            "name": "Net Revision Ratio",
            "value": _v(inputs["net_revision_ratio"], "{:+.2f}"),
            "context": _revision_context(inputs["net_revision_ratio"]),
        },
        {
            "name": "─── Sector Health ───",
            "value": "",
            "context": "",
        },
        {
            "name": "Sector Dispersion",
            "value": f"{disp:.0f}pp" if disp is not None else "n/a",
            "context": _dispersion_context(disp),
        },
        {
            "name": "Best Sector",
            "value": best_br,
            "context": best_name,
        },
        {
            "name": "Worst Sector",
            "value": worst_br_str,
            "context": worst_name,
        },
        {
            "name": "─── Coverage ───",
            "value": "",
            "context": "",
        },
        {
            "name": "Reporting (30d)",
            "value": str(inputs.get("reporting_density") or 0),
            "context": _density_context(inputs.get("reporting_density")),
        },
        {
            "name": "S&P 500 Coverage (90d)",
            "value": str(total),
            "context": f"of {inputs.get('sp500_count', 0)} constituents",
        },
    ]

    from lox.regimes.trend import get_domain_trend
    trend = get_domain_trend("earnings", result.score, result.label)

    print(render_regime_panel(
        domain="Earnings",
        asof=inputs.get("asof", "—"),
        regime_label=result.label,
        score=result.score,
        percentile=None,
        description=result.description,
        metrics=metrics,
        trend=trend,
    ))

    # ── Block 1: Earnings Momentum Sparklines ─────────────────────────────
    _show_earnings_sparklines(console, inputs)

    # ── Block 2: Sector Heatmap ───────────────────────────────────────────
    _show_sector_heatmap(console, inputs)

    # ── Block 3: Cross-Regime Signals ─────────────────────────────────────
    _show_cross_regime_signals(console, result.score)

    if llm:
        from lox.cli_commands.shared.regime_display import print_llm_regime_analysis

        print_llm_regime_analysis(
            settings=settings,
            domain="earnings",
            snapshot={
                **snapshot_data,
                "best_sector": best_name,
                "worst_sector": worst_name,
            },
            regime_label=result.label,
            regime_description=result.description,
            ticker=ticker,
        )

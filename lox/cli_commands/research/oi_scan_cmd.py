"""
OI Scanner — find options with outsized open interest across expiration dates.

Fetches the full options chain for a ticker, computes OI statistics per expiry,
and surfaces strikes where open interest is a statistical outlier relative to
sibling strikes on the same expiry.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class OIRow:
    expiry: str
    dte: int
    strike: float
    opt_type: str  # "Call" or "Put"
    oi: int
    volume: int
    iv: float | None
    bid: float
    ask: float
    last: float
    itm: bool
    # Computed
    oi_zscore: float = 0.0
    oi_vs_median: float = 0.0


# ── Data fetching ─────────────────────────────────────────────────────

def _fetch_chains(
    ticker: str,
    *,
    min_months: int = 0,
    max_months: int = 12,
) -> tuple[float, list[OIRow]]:
    """
    Fetch option chains for *ticker* across expirations between
    *min_months* and *max_months* out.  Returns (underlying_price, rows).
    """
    import yfinance as yf

    t = yf.Ticker(ticker)

    # Get underlying price
    info = t.info or {}
    price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
        or 0.0
    )

    expirations = t.options or []
    floor = date.today() + timedelta(days=min_months * 30)
    ceiling = date.today() + timedelta(days=max_months * 30)

    rows: list[OIRow] = []

    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        if exp_date < floor or exp_date > ceiling:
            continue
        dte = (exp_date - date.today()).days
        if dte < 0:
            continue

        try:
            chain = t.option_chain(exp_str)
        except Exception as e:
            logger.debug("Failed to fetch chain for %s %s: %s", ticker, exp_str, e)
            continue

        for opt_type_label, df in [("Call", chain.calls), ("Put", chain.puts)]:
            if df is None or df.empty:
                continue
            for _, r in df.iterrows():
                oi = int(r.get("openInterest") or 0)
                vol = r.get("volume")
                vol = int(vol) if vol is not None and not (isinstance(vol, float) and vol != vol) else 0
                iv_raw = r.get("impliedVolatility")
                iv = float(iv_raw) if iv_raw is not None and not (isinstance(iv_raw, float) and iv_raw != iv_raw) else None

                rows.append(OIRow(
                    expiry=exp_str,
                    dte=dte,
                    strike=float(r["strike"]),
                    opt_type=opt_type_label,
                    oi=oi,
                    volume=vol,
                    iv=iv,
                    bid=float(r.get("bid") or 0),
                    ask=float(r.get("ask") or 0),
                    last=float(r.get("lastPrice") or 0),
                    itm=bool(r.get("inTheMoney", False)),
                ))

    return float(price), rows


# ── Analysis ──────────────────────────────────────────────────────────

def _compute_outliers(
    rows: list[OIRow],
    *,
    min_oi: int = 50,
    zscore_threshold: float = 1.5,
) -> list[OIRow]:
    """
    For each (expiry, opt_type) group, compute z-scores of OI across strikes.
    Return rows whose OI stands out (z-score >= threshold AND oi >= min_oi),
    sorted by z-score descending.
    """
    from collections import defaultdict

    # Group by (expiry, opt_type)
    groups: dict[tuple[str, str], list[OIRow]] = defaultdict(list)
    for r in rows:
        groups[(r.expiry, r.opt_type)].append(r)

    outliers: list[OIRow] = []

    for key, group in groups.items():
        oi_vals = [r.oi for r in group]

        if len(oi_vals) < 3:
            continue

        # Filter zeros for stats so they don't drag mean/stdev down
        nonzero = [v for v in oi_vals if v > 0]
        if len(nonzero) < 2:
            continue

        mean_oi = statistics.mean(nonzero)
        stdev_oi = statistics.stdev(nonzero) if len(nonzero) > 1 else 1.0
        median_oi = statistics.median(nonzero)

        if stdev_oi == 0:
            stdev_oi = 1.0
        if median_oi == 0:
            median_oi = 1.0

        for r in group:
            if r.oi < min_oi:
                continue
            z = (r.oi - mean_oi) / stdev_oi
            r.oi_zscore = z
            r.oi_vs_median = r.oi / median_oi

            if z >= zscore_threshold:
                outliers.append(r)

    outliers.sort(key=lambda r: r.oi_zscore, reverse=True)
    return outliers


# ── Display ───────────────────────────────────────────────────────────

def _display_results(
    console: Console,
    ticker: str,
    price: float,
    outliers: list[OIRow],
    total_contracts: int,
    *,
    opt_filter: str = "all",
    range_label: str = "all",
):
    """Render results to terminal."""
    filter_label = {"calls": "calls only", "puts": "puts only"}.get(opt_filter, "calls + puts")
    console.print()
    console.print(f"[bold cyan]  OI SCANNER[/bold cyan]  [dim]{ticker}  @ ${price:,.2f}  |  {range_label}  |  {filter_label}[/dim]")
    console.print("[dim]  ─────────────────────────────────────────────[/dim]")

    if not outliers:
        console.print()
        console.print(f"[yellow]  No standout OI found across {total_contracts:,} contracts.[/yellow]")
        console.print("[dim]  Try lowering --min-oi or --zscore.[/dim]")
        console.print()
        return

    # Summary panel
    call_outliers = [r for r in outliers if r.opt_type == "Call"]
    put_outliers = [r for r in outliers if r.opt_type == "Put"]
    total_call_oi = sum(r.oi for r in call_outliers)
    total_put_oi = sum(r.oi for r in put_outliers)
    unique_expiries = len(set(r.expiry for r in outliers))

    summary = Table(show_header=False, box=None, padding=(0, 3), expand=True)
    summary.add_column(justify="center")
    summary.add_column(justify="center")
    summary.add_column(justify="center")
    summary.add_column(justify="center")
    summary.add_row(
        "[dim]SCANNED[/dim]",
        "[dim]OUTLIER STRIKES[/dim]",
        "[dim]CALL OI[/dim]",
        "[dim]PUT OI[/dim]",
    )
    summary.add_row(
        f"[bold]{total_contracts:,}[/bold]",
        f"[bold]{len(outliers)}[/bold] across {unique_expiries} dates",
        f"[bold green]{total_call_oi:,}[/bold green] ({len(call_outliers)})",
        f"[bold red]{total_put_oi:,}[/bold red] ({len(put_outliers)})",
    )
    console.print()
    console.print(Panel(summary, border_style="blue", padding=(1, 2)))

    # Group by expiry for display
    from collections import defaultdict
    by_expiry: dict[str, list[OIRow]] = defaultdict(list)
    for r in outliers:
        by_expiry[r.expiry].append(r)

    for exp_str in sorted(by_expiry.keys()):
        exp_rows = by_expiry[exp_str]
        dte = exp_rows[0].dte

        table = Table(
            title=f"[bold]{exp_str}[/bold] [dim]({dte}d)[/dim]",
            box=None,
            padding=(0, 1),
            show_edge=False,
            header_style="bold dim",
        )
        table.add_column("Type", justify="center")
        table.add_column("Strike", justify="right")
        table.add_column("Moneyness", justify="center")
        table.add_column("OI", justify="right")
        table.add_column("Volume", justify="right")
        table.add_column("OI z-score", justify="right")
        table.add_column("vs Median", justify="right")
        table.add_column("IV", justify="right")
        table.add_column("Bid", justify="right")
        table.add_column("Ask", justify="right")

        # Sort within expiry: calls first, then puts, each by strike
        exp_rows.sort(key=lambda r: (0 if r.opt_type == "Call" else 1, r.strike))

        for r in exp_rows:
            type_color = "green" if r.opt_type == "Call" else "red"
            type_str = f"[{type_color}]{r.opt_type[0]}[/{type_color}]"

            # Moneyness
            if price > 0:
                pct_from_spot = ((r.strike - price) / price) * 100
                if r.opt_type == "Put":
                    moneyness = f"{pct_from_spot:+.1f}%"
                else:
                    moneyness = f"{pct_from_spot:+.1f}%"
                if r.itm:
                    moneyness = f"[bold]{moneyness}[/bold] ITM"
                else:
                    moneyness = f"{moneyness} OTM"
            else:
                moneyness = "—"

            # Z-score color: higher = more unusual
            z = r.oi_zscore
            if z >= 3.0:
                z_str = f"[bold yellow]{z:.1f}[/bold yellow]"
            elif z >= 2.0:
                z_str = f"[yellow]{z:.1f}[/yellow]"
            else:
                z_str = f"{z:.1f}"

            med_str = f"{r.oi_vs_median:.1f}x"

            iv_str = f"{r.iv * 100:.0f}%" if r.iv else "—"

            table.add_row(
                type_str,
                f"${r.strike:.0f}" if r.strike == int(r.strike) else f"${r.strike:.2f}",
                moneyness,
                f"[bold]{r.oi:,}[/bold]",
                f"{r.volume:,}" if r.volume else "—",
                z_str,
                med_str,
                iv_str,
                f"${r.bid:.2f}" if r.bid else "—",
                f"${r.ask:.2f}" if r.ask else "—",
            )

        console.print()
        console.print(table)

    # Footer
    console.print()
    console.print("[dim]  ─────────────────────────────────────────────[/dim]")
    console.print("[dim]  z-score = how many std devs above the mean OI for that expiry/type[/dim]")
    console.print("[dim]  vs Median = OI as a multiple of the median OI for that group[/dim]")
    console.print()


# ── CLI registration ──────────────────────────────────────────────────

def register(app: typer.Typer) -> None:
    @app.command("oi-scan")
    def oi_scan(
        ticker: str = typer.Argument("XRT", help="Underlying ticker to scan"),
        months: int = typer.Option(0, "--months", "-m", help="Minimum months out (e.g. 5 = only July+ expirations)"),
        max_months: int = typer.Option(12, "--max", help="Maximum months out"),
        min_oi: int = typer.Option(50, "--min-oi", help="Minimum OI to consider as outlier"),
        zscore: float = typer.Option(1.5, "--zscore", "-z", help="Z-score threshold for outlier detection"),
        calls: bool = typer.Option(False, "--calls", "-c", help="Show calls only"),
        puts: bool = typer.Option(False, "--puts", "-p", help="Show puts only"),
        show_all: bool = typer.Option(False, "--all", "-a", help="Show all strikes with OI (not just outliers)"),
    ):
        """
        Scan options chains for unusual open interest concentrations.

        Fetches expirations between --months and --max months out, computes
        OI statistics per (expiry, call/put) group, and flags strikes where
        OI is a statistical outlier relative to sibling strikes.

        Examples:
            lox research oi-scan XRT                    # all available expirations
            lox research oi-scan XRT --months 5         # only 5+ months out
            lox research oi-scan SPY --months 2 --max 6 # 2-6 months out
            lox research oi-scan XRT --puts --months 3  # puts only, 3mo+ out
        """
        console = Console()

        range_label = f"{months}mo+" if months > 0 else "all"
        if max_months < 12:
            range_label = f"{months}-{max_months}mo" if months > 0 else f"<{max_months}mo"

        with console.status(f"[bold green]Scanning {ticker} options chain ({range_label})..."):
            price, rows = _fetch_chains(ticker, min_months=months, max_months=max_months)

        if not rows:
            console.print(f"[red]No options data found for {ticker}.[/red]")
            raise typer.Exit(1)

        # Filter by type if requested
        if calls and not puts:
            rows = [r for r in rows if r.opt_type == "Call"]
        elif puts and not calls:
            rows = [r for r in rows if r.opt_type == "Put"]

        total = len(rows)

        if show_all:
            display = _compute_outliers(rows, min_oi=1, zscore_threshold=-999)
        else:
            display = _compute_outliers(rows, min_oi=min_oi, zscore_threshold=zscore)

        opt_filter = "calls" if calls and not puts else "puts" if puts and not calls else "all"
        _display_results(console, ticker, price, display, total, opt_filter=opt_filter, range_label=range_label)

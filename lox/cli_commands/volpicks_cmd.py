"""
lox voltrades — high-volume options picker built on the validated
momentum-within-high-volatility factor (see `lox backtest vol-factors`).

Backtest evidence (S&P 500, top 20% by trailing 20d realized vol, 2023-09
to 2025-09, rebalanced walk-forward, no lookahead): within that volatile
slice, names with the strongest 20d/60d momentum beat the weakest by
+1.1 to +3.7pp of SPY-relative alpha depending on horizon (5-20 trading
days), t-stat 5-8 on the top bucket, monotonic across all five quintiles.
Buying the OVERSOLD names in that same slice showed no edge (t~1-3,
smaller and less consistent) — momentum persists, it doesn't mean-revert,
in this universe.

This command re-runs that same math live: ranks the current top-vol slice
of the universe by composite momentum, long-only (the backtest's "loser"
bucket was still net POSITIVE, just smaller — there's no validated basis
to short it), and — unless --no-options — attaches a live near-ATM call
suggestion per name via Alpaca's option chain.

Known limits, stated plainly:
- No historical single-name options/IV data exists anywhere in lox, so the
  backtest above scores the underlying's move, not option P&L. The option
  contract attached here is LIVE context (today's chain), not backtested —
  premium cost, IV crush after the move, and bid/ask spread are not
  validated by the numbers above.
- Universe is CURRENT S&P 500 membership applied across the whole backtest
  window (no point-in-time historical constituents available in lox), so
  there's mild survivorship bias in the backtest evidence.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

app = typer.Typer(add_completion=False, help="High-volume vol-momentum options picker")


def _pick_call_contract(candidates, target_dte: int, target_delta: float = 0.45):
    """Nearest-to-target-DTE, near-ATM call among live chain candidates."""
    calls = [c for c in candidates if c.opt_type == "call" and c.dte_days and c.dte_days > 0]
    if not calls:
        return None

    def _score(c):
        dte_penalty = abs(c.dte_days - target_dte)
        delta_penalty = abs((c.delta or 0.0) - target_delta) * 100.0
        return dte_penalty + delta_penalty

    calls.sort(key=_score)
    return calls[0]


@app.callback(invoke_without_command=True)
def voltrades(
    ctx: typer.Context,
    universe: str = typer.Option("sp500", "--universe", help="sp500 | dow30 | scan"),
    top_n: int = typer.Option(30, "--top-n", help="How many ranked picks to surface"),
    vol_top_pct: float = typer.Option(0.2, "--vol-top-pct", help="Fraction of universe kept as 'most volatile'"),
    min_price: float = typer.Option(10.0, "--min-price", help="Minimum stock price (options liquidity floor)"),
    target_dte: int = typer.Option(35, "--dte", help="Target days-to-expiry for the suggested call"),
    no_options: bool = typer.Option(False, "--no-options", help="Skip live option chain lookups (just rank stocks)"),
) -> None:
    """Rank the most volatile names in the universe by validated momentum; suggest a near-ATM call for each."""
    if ctx.invoked_subcommand is not None:
        return

    from lox.config import load_settings
    from lox.universe.sp500 import fetch_sp500_symbols, fetch_dow30_symbols, build_scan_universe
    from lox.backtest.vol_factors import fetch_universe_closes, realized_vol, momentum, rsi

    console = Console()
    settings = load_settings()

    if universe == "dow30":
        symbols = fetch_dow30_symbols(settings)
    elif universe == "scan":
        symbols = build_scan_universe(settings)
    else:
        symbols = fetch_sp500_symbols(settings)

    if not symbols:
        console.print("[red]Universe fetch returned no symbols (check FMP_API_KEY).[/red]")
        raise typer.Exit(1)

    lookback_start = date.today() - timedelta(days=200)  # buffer for 60d momentum + 20d vol window
    with console.status(f"Fetching {len(symbols)} tickers…"):
        closes = fetch_universe_closes(settings=settings, symbols=symbols, start=str(lookback_start))

    if closes.empty or len(closes) < 65:
        console.print("[red]Not enough price history returned.[/red]")
        raise typer.Exit(1)

    rv = realized_vol(closes, 20)
    mom20 = momentum(closes, 20)
    mom60 = momentum(closes, 60)
    r14 = rsi(closes, 14)

    vol_row = rv.iloc[-1]
    price_row = closes.iloc[-1]

    eligible = [s for s in vol_row.dropna().index if price_row.get(s, 0) >= min_price]
    if len(eligible) < 20:
        console.print("[red]Not enough eligible tickers after price filter.[/red]")
        raise typer.Exit(1)

    vol_sorted = vol_row[eligible].sort_values(ascending=False)
    cutoff = max(10, int(len(vol_sorted) * vol_top_pct))
    hi_vol_names = list(vol_sorted.index[:cutoff])

    # Composite momentum score = avg percentile rank of 20d + 60d momentum,
    # computed ONLY within the high-vol slice — matches the backtested factor.
    m20 = mom20.iloc[-1][hi_vol_names].dropna()
    m60 = mom60.iloc[-1][hi_vol_names].dropna()
    common = m20.index.intersection(m60.index)
    if len(common) < 10:
        console.print("[red]Not enough names with both momentum windows populated.[/red]")
        raise typer.Exit(1)
    m20r = m20[common].rank(pct=True)
    m60r = m60[common].rank(pct=True)
    score = ((m20r + m60r) / 2.0).sort_values(ascending=False)
    picks = list(score.head(top_n).items())

    console.print(Panel(
        Text(
            f"Universe: {universe} ({len(symbols)} tickers)  ·  Top {vol_top_pct:.0%} by realized vol "
            f"({len(hi_vol_names)} names)  ·  Ranked by 20d+60d momentum percentile  ·  Long-only",
            style="dim",
        ),
        title="[bold]LOX VOLTRADES — Vol-Momentum Picker[/bold]",
        border_style="bright_blue",
    ))
    console.print("[yellow]Backtested factor: underlying momentum within high-vol names (t-stat 5-8, see `lox backtest vol-factors`).[/yellow]")
    console.print("[yellow]NOT backtested: option contract economics (premium, IV crush, spread) — live chain context only.[/yellow]")
    console.print()

    # ── Live option chain lookups (optional) ─────────────────────────────────
    contracts: dict[str, object] = {}
    if not no_options:
        from lox.data.alpaca import make_clients, fetch_option_chain, to_candidates

        try:
            _, data_client = make_clients(settings)
        except Exception as exc:
            console.print(f"[yellow]Alpaca option client unavailable ({exc}) — showing rankings only.[/yellow]")
            console.print()
            data_client = None

        if data_client is not None:
            def _fetch(sym: str):
                try:
                    chain = fetch_option_chain(data_client, sym, feed=settings.alpaca_options_feed)
                    cands = list(to_candidates(chain, sym))
                    return sym, _pick_call_contract(cands, target_dte)
                except Exception:
                    return sym, None

            with console.status(f"Pulling live option chains for {len(picks)} names…"):
                with ThreadPoolExecutor(max_workers=6) as ex:
                    futures = [ex.submit(_fetch, sym) for sym, _ in picks]
                    for fut in as_completed(futures):
                        sym, contract = fut.result()
                        if contract is not None:
                            contracts[sym] = contract

    # ── Table ─────────────────────────────────────────────────────────────────
    table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold dim", expand=False)
    table.add_column("#",        width=3,  justify="right")
    table.add_column("Ticker",   width=8,  no_wrap=True)
    table.add_column("Score",    width=6,  justify="right")
    table.add_column("Mom20d",   width=8,  justify="right")
    table.add_column("Mom60d",   width=8,  justify="right")
    table.add_column("RSI14",    width=6,  justify="right")
    table.add_column("Price",    width=9,  justify="right")
    if not no_options:
        table.add_column("Suggested Call", width=28)

    for i, (sym, sc) in enumerate(picks, start=1):
        m20v = m20.get(sym)
        m60v = m60.get(sym)
        rsiv = r14.iloc[-1].get(sym)
        px = price_row.get(sym)

        row = [
            str(i),
            f"[bold]{sym}[/bold]",
            f"{sc:.2f}",
            f"{m20v*100:+.1f}%" if m20v is not None else "—",
            f"{m60v*100:+.1f}%" if m60v is not None else "—",
            f"{rsiv:.0f}" if rsiv is not None and rsiv == rsiv else "—",
            f"${px:.2f}" if px is not None else "—",
        ]

        if not no_options:
            c = contracts.get(sym)
            if c is not None:
                iv_str = f"IV {c.iv*100:.0f}%" if c.iv is not None else ""
                delta_str = f"Δ{c.delta:.2f}" if c.delta is not None else ""
                mid_str = f"${c.mid:.2f}" if c.mid is not None else (f"${c.last:.2f}" if c.last else "—")
                row.append(f"{sym} {c.expiry} ${c.strike:.0f}C  {delta_str} {iv_str} mid {mid_str}")
            else:
                row.append("[dim]no chain data[/dim]")

        table.add_row(*row)

    console.print(table)
    console.print()
    console.print(
        "[dim]Score = avg percentile rank of 20d/60d momentum within the high-vol slice. "
        "Higher = stronger validated edge per the backtest.[/dim]"
    )

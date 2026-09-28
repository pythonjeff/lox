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

import pandas as pd
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
    earnings_window: int = typer.Option(21, "--earnings-window", help="Flag upcoming earnings within this many days"),
    no_options: bool = typer.Option(False, "--no-options", help="Skip live option chain lookups (just rank stocks)"),
) -> None:
    """
    Rank the most volatile names in the universe by validated momentum, and
    make the FULL vol picture clear for each: is this name's vol already
    elevated or still compressed (about to move — see `backtest
    vol-compression`), and if it has earnings coming up, is the option
    market pricing more or less move than its own history (see `voltrades
    earnings`)? One command, three signals, so you're not juggling three.
    """
    if ctx.invoked_subcommand is not None:
        return

    from lox.config import load_settings
    from lox.universe.sp500 import fetch_sp500_symbols, fetch_dow30_symbols, build_scan_universe
    from lox.backtest.vol_factors import fetch_universe_closes, realized_vol, momentum, rsi, latest_vol_percentile
    from lox.backtest.earnings_vol import historical_earnings_moves, implied_move_pct, classify_richness

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

    # 430 calendar days ≈ 290 trading days: enough for the 252d own-history
    # vol-percentile window (latest_vol_percentile) plus the 20d RV window
    # it's built from, with a buffer for holidays/gaps.
    lookback_start = date.today() - timedelta(days=430)
    with console.status(f"Fetching {len(symbols)} tickers…"):
        closes = fetch_universe_closes(settings=settings, symbols=symbols, start=str(lookback_start))

    if closes.empty or len(closes) < 65:
        console.print("[red]Not enough price history returned.[/red]")
        raise typer.Exit(1)

    rv = realized_vol(closes, 20)
    mom20 = momentum(closes, 20)
    mom60 = momentum(closes, 60)
    r14 = rsi(closes, 14)
    vpr = latest_vol_percentile(closes, 20, 252)  # own-history vol regime, 0=compressed, 1=elevated

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
    pick_syms = [sym for sym, _ in picks]

    console.print(Panel(
        Text(
            f"Universe: {universe} ({len(symbols)} tickers)  ·  Top {vol_top_pct:.0%} by realized vol "
            f"({len(hi_vol_names)} names)  ·  Ranked by 20d+60d momentum percentile  ·  Long-only",
            style="dim",
        ),
        title="[bold]LOX VOLTRADES — Vol-Momentum Picker[/bold]",
        border_style="bright_blue",
    ))
    console.print("[yellow]Momentum (Score) is the validated directional edge (t-stat 5-8, `lox backtest vol-factors`).[/yellow]")
    console.print("[yellow]Vol Regime is descriptive, not directional (`lox backtest vol-compression`): COMPRESSED predicts bigger realized-vol swings ahead, not a bigger net move.[/yellow]")
    console.print("[yellow]Earnings richness is a live pricing check, not backtested (`lox voltrades earnings`) — no historical options/IV data exists in lox.[/yellow]")
    console.print()

    # ── Upcoming earnings for this pick list only (cheap — ~top_n tickers) ────
    upcoming_earnings: dict[str, date] = {}
    try:
        from lox.altdata.fmp import fetch_earnings_calendar
        today = date.today()
        to_date = today + timedelta(days=earnings_window)
        with console.status("Checking upcoming earnings for the pick list…"):
            cal_rows = fetch_earnings_calendar(settings=settings, tickers=pick_syms, from_date=str(today), to_date=str(to_date))
        for r in cal_rows:
            sym = str(r.get("symbol") or "").strip().upper()
            d = r.get("date")
            if not sym or not d:
                continue
            try:
                edate = pd.Timestamp(str(d)[:10]).date()
            except Exception:
                continue
            if sym not in upcoming_earnings or edate < upcoming_earnings[sym]:
                upcoming_earnings[sym] = edate
    except Exception:
        pass

    # ── Live option chain lookups (optional) ─────────────────────────────────
    contracts: dict[str, object] = {}
    earnings_labels: dict[str, str] = {}
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
                except Exception:
                    cands = []
                contract = _pick_call_contract(cands, target_dte) if cands else None

                earn_label = None
                if sym in upcoming_earnings and cands:
                    edate = upcoming_earnings[sym]
                    spot = price_row.get(sym)
                    trailing_vol = vol_row.get(sym)
                    try:
                        hist_moves = historical_earnings_moves(settings=settings, ticker=sym, closes=closes[sym])
                    except Exception:
                        hist_moves = []
                    if len(hist_moves) >= 4 and spot:
                        med = sorted(hist_moves)[len(hist_moves) // 2]
                        _, _, implied = implied_move_pct(
                            cands, float(spot), edate,
                            daily_vol_annualized=float(trailing_vol) if trailing_vol else None,
                        )
                        if implied is not None and med > 0:
                            ratio = implied / med
                            earn_label = f"{edate.strftime('%m/%d')} {classify_richness(ratio)} ({ratio:.1f}×)"
                    if earn_label is None:
                        earn_label = f"{edate.strftime('%m/%d')} (no hist. data)"

                return sym, contract, earn_label

            with console.status(f"Pulling live option chains for {len(picks)} names…"):
                with ThreadPoolExecutor(max_workers=6) as ex:
                    futures = [ex.submit(_fetch, sym) for sym in pick_syms]
                    for fut in as_completed(futures):
                        sym, contract, earn_label = fut.result()
                        if contract is not None:
                            contracts[sym] = contract
                        if earn_label is not None:
                            earnings_labels[sym] = earn_label

    # ── Table ─────────────────────────────────────────────────────────────────
    table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold dim", expand=False)
    table.add_column("#",          width=3,  justify="right")
    table.add_column("Ticker",     width=8,  no_wrap=True)
    table.add_column("Score",      width=6,  justify="right")
    table.add_column("Vol Regime", width=17, no_wrap=True)
    table.add_column("RSI14",      width=6,  justify="right")
    table.add_column("Price",      width=9,  justify="right")
    table.add_column("Earnings",   width=20, no_wrap=True)
    if not no_options:
        table.add_column("Suggested Call", width=28)

    for i, (sym, sc) in enumerate(picks, start=1):
        rsiv = r14.iloc[-1].get(sym)
        px = price_row.get(sym)
        vpr_v = vpr.get(sym)

        if vpr_v is None or vpr_v != vpr_v:
            vol_regime = "—"
            vol_style = "dim"
        elif vpr_v < 0.3:
            vol_regime = f"COMPRESSED {vpr_v*100:.0f}%ile"
            vol_style = "bold cyan"
        elif vpr_v > 0.7:
            vol_regime = f"ELEVATED {vpr_v*100:.0f}%ile"
            vol_style = "bold yellow"
        else:
            vol_regime = f"NEUTRAL {vpr_v*100:.0f}%ile"
            vol_style = "white"

        earn_str = earnings_labels.get(sym, "—")
        earn_style = "white"
        if "RICH" in earn_str:
            earn_style = "bold red"
        elif "CHEAP" in earn_str:
            earn_style = "bold green"

        row = [
            str(i),
            f"[bold]{sym}[/bold]",
            f"{sc:.2f}",
            Text(vol_regime, style=vol_style),
            f"{rsiv:.0f}" if rsiv is not None and rsiv == rsiv else "—",
            f"${px:.2f}" if px is not None else "—",
            Text(earn_str, style=earn_style),
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
        "[dim]Score = momentum percentile within the high-vol slice (validated edge). "
        "Vol Regime = this name's realized vol vs its OWN trailing 252d history — "
        "COMPRESSED (<30%ile) historically saw vol expand 1.4× next, ELEVATED (>70%ile) faded to 0.76×. "
        "Earnings = live implied move vs historical earnings move, only shown if reporting within "
        f"{earnings_window}d.[/dim]"
    )


# ── Earnings vol-pricing checker ────────────────────────────────────────────

@app.command("earnings")
def voltrades_earnings(
    universe: str = typer.Option("sp500", "--universe", help="sp500 | dow30 | scan"),
    days_ahead: int = typer.Option(10, "--days-ahead", help="Look for earnings within this many calendar days"),
    min_history: int = typer.Option(4, "--min-history", help="Minimum past earnings prints required to trust the historical move stat"),
) -> None:
    """
    "Don't overpay for vol" checker: for names reporting earnings soon,
    compares the LIVE option-implied move against each ticker's own
    historical earnings-day moves. No historical options/IV data exists in
    lox, so this uses realized price history as the "was this fair"
    reference instead — a live decision aid, not a backtested edge.
    """
    from lox.config import load_settings
    from lox.universe.sp500 import fetch_sp500_symbols, fetch_dow30_symbols, build_scan_universe
    from lox.backtest.vol_factors import fetch_universe_closes, realized_vol
    from lox.backtest.earnings_vol import EarningsVolCheck, historical_earnings_moves, implied_move_pct, classify_richness
    from lox.altdata.fmp import fetch_earnings_calendar
    from lox.data.alpaca import make_clients, fetch_option_chain, to_candidates

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

    today = date.today()
    to_date = today + timedelta(days=days_ahead)

    with console.status(f"Fetching earnings calendar ({today} → {to_date})…"):
        cal_rows = fetch_earnings_calendar(settings=settings, tickers=symbols, from_date=str(today), to_date=str(to_date))

    upcoming: dict[str, date] = {}
    for r in cal_rows:
        sym = str(r.get("symbol") or "").strip().upper()
        d = r.get("date")
        if not sym or not d:
            continue
        try:
            edate = pd.Timestamp(str(d)[:10]).date()
        except Exception:
            continue
        if sym not in upcoming or edate < upcoming[sym]:
            upcoming[sym] = edate

    if not upcoming:
        console.print(f"[yellow]No upcoming earnings found for this universe in the next {days_ahead} days.[/yellow]")
        raise typer.Exit(0)

    console.print(Panel(
        Text(f"Universe: {universe}  ·  {len(upcoming)} names reporting within {days_ahead} days", style="dim"),
        title="[bold]LOX VOLTRADES EARNINGS — Don't Overpay for Vol[/bold]",
        border_style="bright_blue",
    ))
    console.print("[yellow]Compares live implied move vs each name's OWN historical earnings-day moves — a decision aid, not a backtested edge (no historical options/IV data exists in lox).[/yellow]")
    console.print()

    lookback_start = today - timedelta(days=365 * 4)
    with console.status(f"Fetching price history for {len(upcoming)} names…"):
        closes = fetch_universe_closes(settings=settings, symbols=list(upcoming.keys()), start=str(lookback_start))

    try:
        _, data_client = make_clients(settings)
    except Exception as exc:
        console.print(f"[red]Alpaca option client unavailable: {exc}[/red]")
        raise typer.Exit(1)

    rv20 = realized_vol(closes, 20)
    checks: list[EarningsVolCheck] = []

    def _one(sym: str, edate: date):
        if sym not in closes.columns:
            return None
        hist_moves = historical_earnings_moves(settings=settings, ticker=sym, closes=closes[sym])
        if len(hist_moves) < min_history:
            return None
        med = sorted(hist_moves)[len(hist_moves) // 2]
        avg = sum(hist_moves) / len(hist_moves)

        try:
            chain = fetch_option_chain(data_client, sym, feed=settings.alpaca_options_feed)
            cands = list(to_candidates(chain, sym))
        except Exception:
            cands = []

        spot = closes[sym].dropna().iloc[-1] if sym in closes.columns and not closes[sym].dropna().empty else None
        trailing_vol = rv20[sym].dropna().iloc[-1] if sym in rv20.columns and not rv20[sym].dropna().empty else None
        raw_move, expiry_used, implied = (None, None, None)
        if cands and spot:
            raw_move, expiry_used, implied = implied_move_pct(
                cands, float(spot), edate, daily_vol_annualized=float(trailing_vol) if trailing_vol else None,
            )

        days_past = (expiry_used - edate).days if expiry_used else None
        ratio = (implied / med) if (implied is not None and med > 0) else None
        return EarningsVolCheck(
            ticker=sym, earnings_date=edate, n_history=len(hist_moves),
            median_historical_move_pct=round(med, 2), avg_historical_move_pct=round(avg, 2),
            raw_straddle_move_pct=round(raw_move, 2) if raw_move is not None else None,
            implied_move_pct=round(implied, 2) if implied is not None else None,
            expiry_used=expiry_used,
            days_past_event=days_past,
            richness_ratio=round(ratio, 2) if ratio is not None else None,
            label=classify_richness(ratio),
        )

    with console.status(f"Pulling live option chains for {len(upcoming)} names…"):
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = [ex.submit(_one, sym, edate) for sym, edate in upcoming.items()]
            for fut in as_completed(futures):
                r = fut.result()
                if r is not None:
                    checks.append(r)

    if not checks:
        console.print("[yellow]No names had enough earnings history + live option data to compare.[/yellow]")
        raise typer.Exit(0)

    # Cheapest (most underpriced vs history) first; names with no implied-move data go last.
    checks.sort(key=lambda c: (c.richness_ratio is None, c.richness_ratio if c.richness_ratio is not None else 0))

    table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold dim", expand=False)
    table.add_column("Ticker",        width=8,  no_wrap=True)
    table.add_column("Earnings",      width=11)
    table.add_column("Hist Med Move", width=13, justify="right")
    table.add_column("Implied (adj)", width=13, justify="right")
    table.add_column("Ratio",         width=7,  justify="right")
    table.add_column("Read",          width=8)
    table.add_column("Expiry +Nd",    width=11, justify="right")
    table.add_column("N hist",        width=7,  justify="right")

    label_style = {"CHEAP": "bold green", "FAIR": "white", "RICH": "bold red", "NO DATA": "dim"}

    for c in checks:
        expiry_str = f"+{c.days_past_event}d" if c.days_past_event is not None else "—"
        expiry_style = "yellow" if (c.days_past_event or 0) > 7 else "dim"
        table.add_row(
            f"[bold]{c.ticker}[/bold]",
            str(c.earnings_date),
            f"{c.median_historical_move_pct:.1f}%",
            f"{c.implied_move_pct:.1f}%" if c.implied_move_pct is not None else "—",
            f"{c.richness_ratio:.2f}×" if c.richness_ratio is not None else "—",
            Text(c.label, style=label_style.get(c.label, "white")),
            Text(expiry_str, style=expiry_style),
            str(c.n_history),
        )

    console.print(table)
    console.print()
    console.print(
        "[dim]Ratio = implied (event-adjusted) move ÷ historical median earnings move. "
        "CHEAP (<0.85×) = market pricing less move than usual; RICH (>1.2×) = pricing more than usual.[/dim]"
    )
    console.print(
        "[dim]Expiry +Nd = days between the earnings date and the nearest available expiry. "
        "When >7d (no weeklies), \"Implied (adj)\" backs the extra weeks' ordinary vol out using trailing realized vol — "
        "trust ratios less when this gap is large, since the adjustment is an estimate, not a market price.[/dim]"
    )
    console.print("[dim]Historical move = 2-day close-to-close span around each past print (bmo/amc timing not always reliable in the data).[/dim]")

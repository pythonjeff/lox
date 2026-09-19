"""
lox backtest — Congress signal backtester.

Evaluates historical win-rates and alpha for congressional trade signals
across multiple filter criteria (cluster size, lag, amount, etc.).
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from statistics import mean
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

app = typer.Typer(add_completion=False, help="Congress signal backtest")


# ── Signal construction ───────────────────────────────────────────────────────

def signals_from_congress(
    congress_df,
    trump_df,
    since: date,
    max_lag_days: int = 90,
    congress_api_key: str | None = None,
):
    """
    Convert congress + trump DataFrames into a list of Signal objects.

    Groups by (ticker, side) — uses the EARLIEST transaction date in each group
    as signal_date. Sets cluster_count, avg lag, total amount, and whether any
    trader in the group is committee-aligned with the ticker.
    """
    from collections import defaultdict
    from lox.backtest.engine import Signal
    from lox.quiver.signal import records_from_congress, records_from_trump
    from lox.quiver.committees import is_committee_aligned, alignment_note, get_member_committees

    all_records = []
    if congress_df is not None and not congress_df.empty:
        all_records.extend(records_from_congress(congress_df))
    if trump_df is not None and not trump_df.empty:
        all_records.extend(records_from_trump(trump_df))

    # Build rep name → bioguide_id lookup from the congress DataFrame
    bioguide_map: dict[str, str] = {}
    if congress_df is not None and not congress_df.empty:
        cols = [c.lower() for c in congress_df.columns]
        rep_col = next((c for c in congress_df.columns if c.lower() == "representative"), None)
        bio_col = next((c for c in congress_df.columns if c.lower() == "bioguideid"), None)
        if rep_col and bio_col:
            for _, row in congress_df.iterrows():
                name = str(row[rep_col]).strip()
                bio = str(row[bio_col]).strip()
                if name and bio and bio.lower() not in ("nan", "none", ""):
                    bioguide_map[name] = bio

    # Pre-warm committee cache for all unique bioguide IDs via Congress.gov API.
    # get_member_committees() updates MEMBER_COMMITTEES in-memory so subsequent
    # is_committee_aligned() calls find the data without needing the API key.
    if congress_api_key:
        for bio_id in set(bioguide_map.values()):
            if bio_id:
                get_member_committees(bio_id, congress_api_key)

    # Filter by lag and since date
    # Filter and enter on FILED (disclosure) date, not the underlying transaction
    # date — a trader can't act on a congressional trade until it's publicly
    # disclosed, which lags the transaction by up to 45 days under the STOCK Act.
    # Using transaction date as signal_date is look-ahead bias: it lets the
    # backtest "buy" at a price the public couldn't have traded at yet.
    filtered = [
        r for r in all_records
        if r.lag_days <= max_lag_days and r.filed is not None and r.filed >= since
    ]
    if not filtered:
        return []

    groups: dict[tuple[str, str], list] = defaultdict(list)
    for r in filtered:
        groups[(r.ticker, r.side)].append(r)

    signals = []
    for (ticker, side), records in groups.items():
        dates = [r.filed for r in records if r.filed is not None]
        if not dates:
            continue
        signal_date = min(dates)
        cluster_count = len({r.official for r in records})
        avg_lag = int(mean(r.lag_days for r in records))
        total_amount = sum(r.amount_mid_usd for r in records)
        sources = sorted({r.source for r in records})

        # Committee alignment: True if ANY official in this group is aligned
        aligned = False
        matched_committee = ""
        for r in records:
            bio = bioguide_map.get(r.official, "")
            is_aligned, committee = is_committee_aligned(bio, ticker)
            if is_aligned:
                aligned = True
                matched_committee = committee
                break

        signals.append(Signal(
            ticker=ticker,
            signal_date=signal_date,
            side=side,
            cluster_count=cluster_count,
            lag_days=avg_lag,
            amount_usd=total_amount,
            source=",".join(sources),
            committee_aligned=aligned,
            committee_note=alignment_note(matched_committee) if matched_committee else "",
        ))

    # Populate sector on each signal via FMP batch profiles
    if signals:
        try:
            from lox.config import load_settings
            from lox.altdata.fmp import fetch_batch_profiles
            _settings = load_settings()
            unique_tickers = list({s.ticker for s in signals})
            profiles = fetch_batch_profiles(settings=_settings, tickers=unique_tickers)
            _FMP_NORM = {
                "Technology": "Information Technology",
                "Healthcare": "Health Care",
                "Financial Services": "Financials",
                "Consumer Cyclical": "Consumer Discretionary",
                "Consumer Defensive": "Consumer Staples",
                "Basic Materials": "Materials",
            }
            for sig in signals:
                raw = profiles.get(sig.ticker)
                if raw and raw.sector:
                    sig.sector = _FMP_NORM.get(raw.sector, raw.sector)
        except Exception:
            pass

    return signals


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _win_rate_style(win_pct: float) -> str:
    if win_pct >= 60:
        return "bold green"
    if win_pct >= 50:
        return "yellow"
    return "red"


def _tstat_style(t: float) -> str:
    if abs(t) >= 2.0:
        return "bold green"
    return "white"


def _fmt_return(r: float) -> str:
    sign = "+" if r >= 0 else ""
    return f"{sign}{r:.1f}%"


# ── ML helpers ────────────────────────────────────────────────────────────────

_GROWTH     = {"Information Technology", "Communication Services", "Consumer Discretionary"}
_HARD_ASSET = {"Energy", "Materials", "Industrials"}
_DEFENSIVE  = {"Health Care", "Consumer Staples", "Utilities"}

_FEATURE_NAMES = [
    "cluster_count", "freshness", "committee_aligned",
    "rsi_normalized", "log_amount", "is_trump",
    "sec_growth", "sec_hard_asset", "sec_defensive",
]

_FEATURE_INTERP = {
    "cluster_count":      "more officials → more conviction",
    "freshness":          "lower lag → stronger signal",
    "committee_aligned":  "oversight committee member bought",
    "rsi_normalized":     ">0 = overbought; prefer negative (buy into weakness)",
    "log_amount":         "larger dollar amount bet",
    "is_trump":           "Trump family trade",
    "sec_growth":         "IT / Comm / Consumer Disc",
    "sec_hard_asset":     "Energy / Materials / Industrials",
    "sec_defensive":      "Health Care / Staples / Utilities",
}


def _extract_features_labels(all_results: list, horizon: int):
    """
    Build feature matrix X (chronologically sorted), binary label y (win=1),
    and sorted date list.  Returns (X, y, sorted_dates) or (None, None, None).
    """
    import numpy as np

    rows, labels, dates = [], [], []

    for r in all_results:
        if not r.tradeable or horizon not in r.returns:
            continue
        s = r.signal
        rsi_norm = ((s.rsi_at_entry - 50.0) / 50.0) if s.rsi_at_entry is not None else 0.0
        row = [
            min(s.cluster_count, 5) / 5.0,                     # cluster_count
            max(0.0, 1.0 - s.lag_days / 90.0),                 # freshness
            float(s.committee_aligned),                         # committee_aligned
            rsi_norm,                                           # rsi_normalized
            math.log1p(s.amount_usd) / math.log1p(1_000_000),  # log_amount
            float("trump" in (s.source or "")),                 # is_trump
            float(s.sector in _GROWTH),                         # sec_growth
            float(s.sector in _HARD_ASSET),                     # sec_hard_asset
            float(s.sector in _DEFENSIVE),                      # sec_defensive
        ]
        rows.append(row)
        labels.append(1 if r.returns[horizon] > 0 else 0)
        dates.append(s.signal_date)

    if not rows:
        return None, None, None

    order = sorted(range(len(dates)), key=lambda i: dates[i])
    X = np.array([rows[i] for i in order], dtype=np.float64)
    y = np.array([labels[i] for i in order])
    sorted_dates = [dates[i] for i in order]
    return X, y, sorted_dates


def _run_ml_model(all_results: list, horizon: int, console: Console) -> None:
    """Train LogisticRegression; display walk-forward AUC + feature coefficients."""
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        console.print("[yellow]scikit-learn not installed — run: pip install scikit-learn[/yellow]")
        return

    X, y, _ = _extract_features_labels(all_results, horizon)
    if X is None or len(X) < 20:
        console.print("[yellow]Not enough tradeable signals for ML model (need ≥20)[/yellow]")
        return

    n = len(y)
    baseline_win = float(y.mean() * 100)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    tscv = TimeSeriesSplit(n_splits=3)
    auc_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring="roc_auc")

    model.fit(X_scaled, y)
    coeffs = model.coef_[0]

    # Quintile win rates: does ranking by predicted prob stratify outcomes?
    probs = model.predict_proba(X_scaled)[:, 1]
    rank_order = np.argsort(probs)[::-1]
    q = max(1, n // 5)
    top_win  = float(y[rank_order[:q]].mean() * 100)
    bot_win  = float(y[rank_order[-q:]].mean() * 100)

    # ── Header ────────────────────────────────────────────────────────────────
    auc_str = "  /  ".join(f"{a:.3f}" for a in auc_scores)
    auc_mean = auc_scores.mean()
    auc_style = "bold green" if auc_mean >= 0.60 else "yellow" if auc_mean >= 0.54 else "red"
    body = (
        f"AUC: [{auc_style}]{auc_mean:.3f}[/{auc_style}]  ({auc_str})  ·  "
        f"n={n}  ·  baseline {baseline_win:.0f}% win rate"
    )
    console.print()
    console.print(Panel(
        body,
        title=f"[bold]ML SIGNAL MODEL[/bold]  LogReg · {horizon}d horizon · walk-forward CV",
        border_style="magenta",
    ))
    console.print()

    # ── Coefficients table ────────────────────────────────────────────────────
    sorted_feat = sorted(range(len(_FEATURE_NAMES)), key=lambda i: -abs(coeffs[i]))

    coeff_table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold dim")
    coeff_table.add_column("Feature",       width=22)
    coeff_table.add_column("Coeff",         width=8,  justify="right")
    coeff_table.add_column("Interpretation", width=45)

    for i in sorted_feat:
        c = coeffs[i]
        sign = "+" if c >= 0 else ""
        style = "green" if c >= 0 else "red"
        coeff_table.add_row(
            _FEATURE_NAMES[i],
            Text(f"{sign}{c:.3f}", style=style),
            _FEATURE_INTERP.get(_FEATURE_NAMES[i], ""),
        )

    console.print(coeff_table)
    console.print()
    console.print(
        f"[dim]Predicted-rank quintiles — top 20%: {top_win:.0f}% wins  ·  "
        f"bottom 20%: {bot_win:.0f}% wins[/dim]"
    )
    console.print("[dim]AUC > 0.60 = meaningful predictive power; coefficients = learned feature weights[/dim]")
    console.print()


# ── Command ───────────────────────────────────────────────────────────────────

@app.callback(invoke_without_command=True)
def backtest(
    ctx: typer.Context,
    horizon: int = typer.Option(10, "--horizon", help="Holding period (business days) to evaluate"),
    since: str = typer.Option("", "--since", help="Start date like 2025-01-01 (default: 90 days ago)"),
    show_trades: bool = typer.Option(False, "--show-trades", help="Show individual trade results table"),
    ml: bool = typer.Option(False, "--ml", help="Train logistic regression on signals; show feature weights"),
) -> None:
    """Congress signal backtest — win rates, alpha, t-stats by filter."""
    if ctx.invoked_subcommand is not None:
        return

    console = Console()

    # ── Resolve since date ────────────────────────────────────────────────────
    today = date.today()
    if since:
        try:
            import datetime
            since_date = datetime.date.fromisoformat(since)
        except ValueError:
            console.print(f"[red]Invalid --since date: {since}. Use YYYY-MM-DD format.[/red]")
            raise typer.Exit(1)
    else:
        since_date = today - timedelta(days=90)

    # ── Fetch quiver data ─────────────────────────────────────────────────────
    from lox.quiver.loader import fetch_congress_live, fetch_trump_live, get_api_key

    api_key = get_api_key()
    if not api_key:
        console.print("[red]QUIVER_API_KEY not set — add it to .env[/red]")
        raise typer.Exit(1)

    congress_df = trump_df = None
    with console.status("Fetching congressional + Trump trades…"):
        try:
            congress_df = fetch_congress_live(api_key)
        except Exception as exc:
            console.print(f"[yellow]Congress fetch failed: {exc}[/yellow]")
        try:
            trump_df = fetch_trump_live(api_key)
        except Exception as exc:
            console.print(f"[yellow]Trump trades fetch failed: {exc}[/yellow]")

    if congress_df is None and trump_df is None:
        console.print("[red]No trade data available.[/red]")
        raise typer.Exit(1)

    # ── Load settings (needed for Congress.gov API key) ───────────────────────
    from lox.config import load_settings
    settings = load_settings()

    # ── Build signals ─────────────────────────────────────────────────────────
    with console.status("Building signals + fetching committee data…"):
        all_signals = signals_from_congress(
            congress_df, trump_df,
            since=since_date,
            max_lag_days=90,
            congress_api_key=settings.congress_gov_api_key,
        )

    # Filter to Buy side for the congress backtest
    buy_signals = [s for s in all_signals if s.side == "Buy"]

    # ── Header panel ──────────────────────────────────────────────────────────
    n = len(buy_signals)
    header = Text()
    header.append(f"Period: {since_date}  →  {today}", style="dim")
    header.append(f"  ·  {n} signals", style="white")
    header.append(f"  ·  Evaluating at {horizon} days", style="dim")
    console.print(Panel(header, title="[bold]CONGRESS SIGNAL BACKTEST[/bold]", border_style="bright_blue"))
    console.print()

    if n < 30:
        console.print(f"[yellow]Warning: only {n} signals in window — results may not be statistically meaningful (need ≥30)[/yellow]")
        console.print()

    if n == 0:
        console.print("[yellow]No qualifying buy signals in the selected period.[/yellow]")
        raise typer.Exit(0)

    # ── Engine ────────────────────────────────────────────────────────────────
    from lox.backtest.engine import BacktestEngine, compare_filters

    engine = BacktestEngine(settings, horizons=[5, 10, 20, 30])

    # ── Pre-compute RSI at entry for all signals (one price fetch pass) ─────────
    # Saves results for ML training; also mutates rsi_at_entry on each Signal.
    with console.status("Pre-computing RSI at entry for all signals…"):
        all_results = engine.run(buy_signals)

    # ── Named filters ─────────────────────────────────────────────────────────
    named_filters = [
        ("All congress buys",           lambda s: True),
        ("Committee-aligned",           lambda s: s.committee_aligned),
        ("Committee + cluster ≥2",      lambda s: s.committee_aligned and s.cluster_count >= 2),
        ("Cluster ≥2 officials",        lambda s: s.cluster_count >= 2),
        ("Lag ≤14 days",                lambda s: s.lag_days <= 14),
        ("Large amount ≥$50K",          lambda s: s.amount_usd >= 50_000),
        # RSI filters — does entry RSI matter?
        ("RSI < 50 at entry",           lambda s: s.rsi_at_entry is not None and s.rsi_at_entry < 50),
        ("Cmt+cluster≥2 + RSI<50",      lambda s: s.committee_aligned and s.cluster_count >= 2
                                                   and s.rsi_at_entry is not None and s.rsi_at_entry < 50),
        # Sector filters — which sectors have the best congressional signal?
        ("Hard-asset sectors",          lambda s: s.sector in _HARD_ASSET),
        ("Growth sectors",              lambda s: s.sector in _GROWTH),
        ("Defensive sectors",           lambda s: s.sector in _DEFENSIVE),
        ("Cmt+cluster≥2 + hard-asset",  lambda s: s.committee_aligned and s.cluster_count >= 2
                                                   and s.sector in _HARD_ASSET),
    ]

    with console.status(f"Running backtest ({n} signals × {len(named_filters)} filters)…"):
        filter_results = compare_filters(buy_signals, named_filters, engine, horizon=horizon)

    # ── Summary table ─────────────────────────────────────────────────────────
    table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold dim", expand=False)
    table.add_column("Filter",          width=30)
    table.add_column("N",               width=5,  justify="right")
    table.add_column("Win%",            width=6,  justify="right")
    table.add_column(f"Avg {horizon}d", width=9,  justify="right")
    table.add_column("Alpha",           width=8,  justify="right")
    table.add_column("t-stat",          width=7,  justify="right")

    for filter_name, summary in filter_results:
        win_pct = summary.win_rates.get(horizon, 0.0)
        avg_ret = summary.avg_returns.get(horizon, 0.0)
        avg_alp = summary.avg_alpha.get(horizon, 0.0)
        t = summary.t_stats.get(horizon, 0.0)

        win_style = _win_rate_style(win_pct)
        t_style = _tstat_style(t)

        table.add_row(
            filter_name,
            str(summary.n_signals),
            Text(f"{win_pct:.0f}%", style=win_style),
            Text(_fmt_return(avg_ret), style="green" if avg_ret >= 0 else "red"),
            Text(_fmt_return(avg_alp), style="green" if avg_alp >= 0 else "red"),
            Text(f"{t:.2f}", style=t_style),
        )

    console.print(table)
    console.print()
    console.print("[dim]t-stat >2.0 indicates statistically meaningful edge[/dim]")
    console.print()

    # ── ML model (optional) ──────────────────────────────────────────────────
    if ml:
        _run_ml_model(all_results, horizon, console)

    # ── Individual trades table (optional) ───────────────────────────────────
    if show_trades:
        tradeable_results = [r for r in all_results if r.tradeable and horizon in r.returns]
        tradeable_results.sort(key=lambda r: -r.returns.get(horizon, 0.0))

        trades_table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold dim")
        trades_table.add_column("Ticker",   width=8)
        trades_table.add_column("Date",     width=12)
        trades_table.add_column("Cluster",  width=8,  justify="right")
        trades_table.add_column("Lag",      width=5,  justify="right")
        trades_table.add_column("Amount",   width=10, justify="right")
        trades_table.add_column(f"{horizon}d Ret", width=9, justify="right")
        trades_table.add_column("Alpha",    width=8,  justify="right")

        for r in tradeable_results:
            ret = r.returns.get(horizon, 0.0)
            alp = r.alpha.get(horizon, 0.0)
            amt = r.signal.amount_usd
            amt_str = (
                f"${amt/1_000:.0f}K" if amt < 1_000_000 else f"${amt/1_000_000:.1f}M"
            ) if amt > 0 else "—"

            ret_style = "green" if ret >= 0 else "red"
            alp_style = "green" if alp >= 0 else "red"

            trades_table.add_row(
                f"[bold]{r.signal.ticker}[/bold]",
                str(r.signal.signal_date),
                str(r.signal.cluster_count),
                f"{r.signal.lag_days}d",
                amt_str,
                Text(_fmt_return(ret), style=ret_style),
                Text(_fmt_return(alp), style=alp_style),
            )

        console.print("[bold]Individual Trades[/bold]")
        console.print(trades_table)


# ── Volatility factor backtest ─────────────────────────────────────────────────

@app.command("vol-factors")
def vol_factors_cmd(
    since: str = typer.Option("", "--since", help="Start date like 2023-01-01 (default: 2 years ago)"),
    universe: str = typer.Option("sp500", "--universe", help="sp500 | dow30 | scan"),
    horizon: int = typer.Option(10, "--horizon", help="Primary holding period (business days) to display"),
    vol_top_pct: float = typer.Option(0.2, "--vol-top-pct", help="Fraction of universe kept as 'most volatile' each rebalance"),
) -> None:
    """
    Volatility-tilted factor backtest — within the most-volatile slice of the
    universe, does momentum / RSI-reversion predict forward moves? Answers
    "prove the factor first" before building a high-volume options picker.

    No historical options/IV data exists in lox, so this scores the
    UNDERLYING's realized move (direction + magnitude), not option P&L.
    """
    import datetime
    from lox.config import load_settings
    from lox.universe.sp500 import fetch_sp500_symbols, fetch_dow30_symbols, build_scan_universe
    from lox.backtest.vol_factors import fetch_universe_closes, run_vol_factor_backtest

    console = Console()
    settings = load_settings()

    today = date.today()
    since_date = datetime.date.fromisoformat(since) if since else today - timedelta(days=730)

    if universe == "dow30":
        symbols = fetch_dow30_symbols(settings)
    elif universe == "scan":
        symbols = build_scan_universe(settings)
    else:
        symbols = fetch_sp500_symbols(settings)

    if not symbols:
        console.print("[red]Universe fetch returned no symbols (check FMP_API_KEY).[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        Text(f"Universe: {universe} ({len(symbols)} tickers)  ·  Since: {since_date}  ·  Top {vol_top_pct:.0%} by trailing 20d realized vol", style="dim"),
        title="[bold]VOLATILITY FACTOR BACKTEST[/bold]",
        border_style="bright_blue",
    ))
    console.print("[yellow]Note: scores the underlying's realized move — no historical options/IV data exists to backtest actual option P&L.[/yellow]")
    console.print()

    with console.status(f"Fetching {len(symbols)} tickers + SPY (cold cache = slow first run)…"):
        spy_df = fetch_universe_closes(settings=settings, symbols=["SPY"], start=str(since_date))
        spy = spy_df["SPY"] if "SPY" in spy_df.columns else None
        closes = fetch_universe_closes(settings=settings, symbols=symbols, start=str(since_date))

    if closes.empty:
        console.print("[red]No price data returned.[/red]")
        raise typer.Exit(1)

    console.print(f"[dim]{closes.shape[1]} tickers with price history, {closes.shape[0]} trading days.[/dim]")
    console.print()

    with console.status("Running walk-forward factor backtest…"):
        results = run_vol_factor_backtest(closes, spy, vol_top_pct=vol_top_pct, horizons=(5, 10, 20))

    if not results:
        console.print("[yellow]No results — try a longer --since window.[/yellow]")
        raise typer.Exit(0)

    table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold dim", expand=False)
    table.add_column("Factor",       width=14, no_wrap=True)
    table.add_column("Quintile",     width=10, justify="center", no_wrap=True)
    table.add_column("N",            width=6,  justify="right")
    table.add_column("Win%",         width=6,  justify="right")
    table.add_column(f"Avg {horizon}d", width=9,  justify="right")
    table.add_column("Alpha",        width=8,  justify="right")
    table.add_column("Avg |Move|",   width=11, justify="right")
    table.add_column("t-stat",       width=8,  justify="right", no_wrap=True)

    q_label = {1: "Q1 (low)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 (high)"}

    for r in [r for r in results if r.horizon == horizon]:
        win_style = _win_rate_style(r.win_rate)
        t_style = _tstat_style(r.t_stat)
        table.add_row(
            r.factor,
            q_label.get(r.quintile, str(r.quintile)),
            str(r.n),
            Text(f"{r.win_rate:.0f}%", style=win_style),
            Text(_fmt_return(r.avg_return), style="green" if r.avg_return >= 0 else "red"),
            Text(_fmt_return(r.avg_alpha) if not math.isnan(r.avg_alpha) else "—",
                 style="green" if (not math.isnan(r.avg_alpha) and r.avg_alpha >= 0) else "red"),
            f"{r.avg_abs_move:.1f}%",
            Text(f"{r.t_stat:.2f}", style=t_style),
        )

    console.print(table)
    console.print()
    console.print("[dim]t-stat >2.0 indicates statistically meaningful edge. Q5 vs Q1 spread is what matters — a factor with no spread has no edge regardless of individual t-stats.[/dim]")
    console.print("[dim]Rebalances are spaced by the longest horizon tested to reduce (not eliminate) overlap; within-date names still share market/sector moves.[/dim]")
    console.print()

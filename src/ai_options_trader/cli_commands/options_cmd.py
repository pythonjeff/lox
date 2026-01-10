from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ai_options_trader.config import load_settings
from ai_options_trader.data.alpaca import fetch_option_chain, make_clients, to_candidates
from ai_options_trader.data.market import fetch_equity_daily_closes
from ai_options_trader.execution.alpaca import submit_option_order
from ai_options_trader.options.historical import fetch_option_bar_volumes
from ai_options_trader.options.budget_scan import affordable_options_for_ticker, pick_best_affordable
from ai_options_trader.options.most_traded import most_traded_options
from ai_options_trader.options.moonshot import rank_moonshots, rank_moonshots_unconditional
from ai_options_trader.llm.moonshot_theory import llm_moonshot_theory
from ai_options_trader.regimes.feature_matrix import build_regime_feature_matrix
from ai_options_trader.universe.sp500 import load_sp500_universe
from ai_options_trader.portfolio.universe import STARTER_UNIVERSE


def register(options_app: typer.Typer) -> None:
    def _fmt_int(x: int | None) -> str:
        return f"{int(x):,}" if isinstance(x, int) else "n/a"

    def _fmt_price(x: float | None) -> str:
        return f"{float(x):.2f}" if isinstance(x, (int, float)) else "n/a"

    def _fmt_pct(x: float | None) -> str:
        return f"{100.0*float(x):.1f}%" if isinstance(x, (int, float)) else "n/a"

    @options_app.command("most-traded")
    def options_most_traded(
        ticker: str = typer.Option("SPY", "--ticker", "-t", help="Underlying ticker, e.g. SPY"),
        min_days: int = typer.Option(0, "--min-days", help="Min days-to-expiry (DTE) to include"),
        max_days: int = typer.Option(90, "--max-days", help="Max days-to-expiry (DTE) to include"),
        top: int = typer.Option(25, "--top", help="How many contracts to print"),
        calls: bool = typer.Option(False, "--calls", help="Only calls"),
        puts: bool = typer.Option(False, "--puts", help="Only puts"),
        sort: str = typer.Option(
            "volume",
            "--sort",
            help="volume|open_interest|delta|abs_delta|gamma|theta|vega|iv",
        ),
        mode: str = typer.Option("snapshot", "--mode", help="snapshot|historical (last-day bars volume)"),
        lookback_days: int = typer.Option(1, "--lookback-days", help="(historical) Lookback window in days"),
        chunk_size: int = typer.Option(200, "--chunk-size", help="(historical) Batch size for option bar requests"),
    ):
        """
        Print the most-traded option contracts for `ticker` expiring within the next N days.

        This ranks contracts by the snapshot `volume` field when present (falls back to OI tiebreaks).
        """
        want = "both"
        if calls and not puts:
            want = "call"
        elif puts and not calls:
            want = "put"

        s = sort.strip().lower()
        if s.startswith("open"):
            sort_key = "open_interest"
        elif s in {"delta", "abs_delta", "gamma", "theta", "vega", "iv"}:
            sort_key = s
        else:
            sort_key = "volume"

        settings = load_settings()
        _trading, data = make_clients(settings)

        chain = fetch_option_chain(data, ticker, feed=settings.alpaca_options_feed)
        candidates = list(to_candidates(chain, ticker))
        mode_norm = mode.strip().lower()
        volume_by_symbol = None
        if mode_norm.startswith("hist"):
            # Aggregate last-day contract volume from historical option bars.
            # (This avoids relying on snapshot volume/OI, which may be missing depending on subscription/feed.)
            today = date.today()
            # Filter to candidate symbols in the DTE window before hitting the bars endpoint.
            symbols_in_window = [
                c.symbol
                for c in most_traded_options(
                    candidates,
                    ticker=ticker,
                    min_dte_days=int(min_days),
                    max_dte_days=int(max_days),
                    want=want,  # type: ignore[arg-type]
                    top=max(1, len(candidates)),
                    sort="volume",
                    today=today,
                )
            ]
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=int(lookback_days))
            try:
                hv = fetch_option_bar_volumes(
                    data,
                    option_symbols=symbols_in_window,
                    start=start,
                    end=end,
                    feed=settings.alpaca_options_feed,
                    chunk_size=int(chunk_size),
                )
                volume_by_symbol = hv.volume_by_symbol
                print(
                    f"[dim]Historical bars window: {hv.start.isoformat()} → {hv.end.isoformat()} | "
                    f"symbols: {hv.symbols_requested} | returned: {hv.symbols_returned} | chunks: {hv.chunks}[/dim]"
                )
            except Exception as e:
                print(
                    "[red]Error:[/red] Failed to fetch historical option bars volume. "
                    "This is usually a subscription limitation (e.g., Basic plan only allows the latest ~15 minutes) "
                    "or a feed entitlement issue. See Alpaca docs: "
                    "`https://docs.alpaca.markets/docs/historical-option-data` and "
                    "`https://docs.alpaca.markets/docs/real-time-option-data`.\n"
                    f"[dim]{type(e).__name__}: {e}[/dim]"
                )
                raise typer.Exit(code=2)

        ranked = most_traded_options(
            candidates,
            ticker=ticker,
            min_dte_days=int(min_days),
            max_dte_days=int(max_days),
            want=want,  # type: ignore[arg-type]
            top=int(top),
            sort=sort_key,  # type: ignore[arg-type]
            volume_by_symbol=volume_by_symbol,
            today=date.today(),
        )

        vol_missing = sum(1 for c in candidates if c.volume is None)
        oi_missing = sum(1 for c in candidates if c.oi is None)
        delta_missing = sum(1 for c in candidates if c.delta is None)
        iv_missing = sum(1 for c in candidates if c.iv is None)
        print(
            "[dim]"
            f"Chain snapshots: {len(candidates)} | volume missing: {vol_missing} | OI missing: {oi_missing} | "
            f"Δ missing: {delta_missing} | IV missing: {iv_missing}"
            "[/dim]"
        )

        if volume_by_symbol is None and candidates and vol_missing == len(candidates) and oi_missing == len(candidates):
            print(
                "[yellow]Warning:[/yellow] Alpaca did not provide volume/open-interest in the option chain snapshot. "
                "This is usually a data-permissions/feed issue (US options typically require OPRA). "
                "Try setting `ALPACA_OPTIONS_FEED=opra` in your `.env` and ensure your Alpaca plan includes options data. "
                "If volume/OI still come back empty, we can switch this command to aggregate volume from historical option bars/trades instead."
            )
        if volume_by_symbol is not None:
            covered = sum(1 for o in ranked if isinstance(o.volume, int))
            print(f"[dim]Historical volume coverage (in ranked set): {covered}/{len(ranked)}[/dim]")

        tbl = Table(title=f"{ticker.upper()} options: most traded ({int(min_days)}..{int(max_days)} DTE)")
        tbl.add_column("Rank", justify="right")
        tbl.add_column("Symbol", style="bold")
        tbl.add_column("Type")
        tbl.add_column("Exp")
        tbl.add_column("DTE", justify="right")
        tbl.add_column("Strike", justify="right")
        tbl.add_column("Vol", justify="right")
        show_oi = any(isinstance(o.open_interest, int) for o in ranked)
        if show_oi:
            tbl.add_column("OI", justify="right")
        tbl.add_column("Bid", justify="right")
        tbl.add_column("Ask", justify="right")
        tbl.add_column("Mid", justify="right")
        show_gamma = sort_key == "gamma" or any(isinstance(o.gamma, float) for o in ranked)
        show_theta = sort_key == "theta" or any(isinstance(o.theta, float) for o in ranked)
        show_vega = sort_key == "vega" or any(isinstance(o.vega, float) for o in ranked)

        tbl.add_column("Δ", justify="right")
        if show_gamma:
            tbl.add_column("Γ", justify="right")
        if show_theta:
            tbl.add_column("Θ", justify="right")
        if show_vega:
            tbl.add_column("V", justify="right")
        tbl.add_column("IV", justify="right")

        if not show_oi:
            print(
                "[dim]OI: n/a for all contracts (not provided by current data source; "
                "requires a feed/entitlement that includes OI, or a dedicated OI endpoint).[/dim]"
            )

        for i, o in enumerate(ranked, start=1):
            mid = o.mid
            row = [
                str(i),
                o.symbol,
                o.opt_type,
                o.expiry.isoformat(),
                str(o.dte_days),
                f"{o.strike:.2f}",
                _fmt_int(o.volume),
            ]
            if show_oi:
                row.append(_fmt_int(o.open_interest))
            row.extend(
                [
                    _fmt_price(o.bid),
                    _fmt_price(o.ask),
                    _fmt_price(mid),
                    _fmt_price(o.delta),
                    _fmt_price(o.gamma) if show_gamma else None,
                    _fmt_price(o.theta) if show_theta else None,
                    _fmt_price(o.vega) if show_vega else None,
                    _fmt_pct(o.iv),
                ]
            )
            # Drop Nones inserted for hidden columns
            row = [x for x in row if x is not None]
            tbl.add_row(*row)

        Console().print(tbl)

    @options_app.command("sp500-under-budget")
    def sp500_under_budget(
        max_premium_usd: float = typer.Option(100.0, "--max-premium", help="Max option premium per contract (USD)"),
        min_days: int = typer.Option(7, "--min-days", help="Min DTE"),
        max_days: int = typer.Option(45, "--max-days", help="Max DTE"),
        calls: bool = typer.Option(False, "--calls", help="Only calls"),
        puts: bool = typer.Option(False, "--puts", help="Only puts"),
        price_basis: str = typer.Option("ask", "--price-basis", help="ask|mid|last (premium basis)"),
        min_price: float = typer.Option(0.05, "--min-price", help="Minimum option price (filters out $0.01 junk)"),
        target_abs_delta: float = typer.Option(0.30, "--target-abs-delta", help="Target |delta| for the picked contract"),
        max_spread_pct: float = typer.Option(0.30, "--max-spread-pct", help="Prefer contracts with spread <= this"),
        limit: int = typer.Option(500, "--limit", help="Limit tickers scanned (set 500 for full S&P 500)"),
        workers: int = typer.Option(8, "--workers", help="Parallel workers (API load)."),
        refresh_universe: bool = typer.Option(False, "--refresh-universe", help="Force refresh S&P 500 universe CSV"),
        refresh_chains: bool = typer.Option(False, "--refresh-chains", help="(not implemented) placeholder; chains are live"),
        max_results: int = typer.Option(200, "--max-results", help="Max rows to print"),
    ):
        """
        Scan S&P 500 tickers and find at least one option contract per ticker with premium <= $max_premium_usd.

        This is a **scanner** only — it does not place orders.
        """
        _ = refresh_chains  # placeholder for future caching

        want = "both"
        if calls and not puts:
            want = "call"
        elif puts and not calls:
            want = "put"

        pb = price_basis.strip().lower()
        if pb not in {"ask", "mid", "last"}:
            pb = "ask"

        settings = load_settings()
        _trading, data = make_clients(settings)

        uni = load_sp500_universe(refresh=bool(refresh_universe), fmp_api_key=settings.fmp_api_key)
        tickers = uni.tickers[: max(0, int(limit))]
        if uni.skipped:
            print(f"[dim]Skipped dotted tickers: {len(uni.skipped)} (e.g., {', '.join(uni.skipped[:3])})[/dim]")
        print(f"[dim]Universe: {len(tickers)} tickers (source={uni.source})[/dim]")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _scan_one(t: str):
            chain = fetch_option_chain(data, t, feed=settings.alpaca_options_feed)
            candidates = list(to_candidates(chain, t))
            opts = affordable_options_for_ticker(
                candidates,
                ticker=t,
                max_premium_usd=float(max_premium_usd),
                min_dte_days=int(min_days),
                max_dte_days=int(max_days),
                want=want,  # type: ignore[arg-type]
                price_basis=pb,  # type: ignore[arg-type]
                min_price=float(min_price),
                require_delta=True,
                today=date.today(),
            )
            best = pick_best_affordable(opts, target_abs_delta=float(target_abs_delta), max_spread_pct=float(max_spread_pct))
            return t, best

        results = []
        errors = 0
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            futs = {ex.submit(_scan_one, t): t for t in tickers}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    _t, best = fut.result()
                    if best is not None:
                        results.append(best)
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"[dim]{t}: {type(e).__name__}: {e}[/dim]")

        # User intent: only show contracts with delta and sort by delta strength.
        # Use |Δ| so calls/puts can be compared consistently.
        results.sort(key=lambda o: (-abs(float(o.delta)), o.premium_usd, o.ticker))
        shown = results[: max(0, int(max_results))]

        tbl = Table(title=f"S&P 500: options under ${float(max_premium_usd):.0f} (picked 1/ticker)")
        tbl.add_column("Ticker", style="bold")
        tbl.add_column("Symbol", style="bold")
        tbl.add_column("Type")
        tbl.add_column("Exp")
        tbl.add_column("DTE", justify="right")
        tbl.add_column("Strike", justify="right")
        tbl.add_column("Price", justify="right")
        tbl.add_column("Premium", justify="right")
        tbl.add_column("Spread%", justify="right")
        tbl.add_column("Δ", justify="right")
        tbl.add_column("IV", justify="right")

        for o in shown:
            sp = (100.0 * float(o.spread_pct)) if isinstance(o.spread_pct, float) else None
            tbl.add_row(
                o.ticker,
                o.symbol,
                o.opt_type,
                o.expiry.isoformat(),
                str(o.dte_days),
                f"{o.strike:.2f}",
                _fmt_price(o.price),
                f"${o.premium_usd:,.0f}",
                (f"{sp:.1f}%" if sp is not None else "n/a"),
                _fmt_price(o.delta),
                _fmt_pct(o.iv),
            )

        Console().print(tbl)
        print(
            f"[dim]Found: {len(results)}/{len(tickers)} tickers with at least one contract <= ${float(max_premium_usd):.0f}. "
            f"Errors: {errors}. Price basis: {pb}. DTE: {min_days}..{max_days}[/dim]"
        )

    @options_app.command("etf-under-budget")
    def etf_under_budget(
        max_premium_usd: float = typer.Option(100.0, "--max-premium", help="Max option premium per contract (USD)"),
        min_days: int = typer.Option(7, "--min-days", help="Min DTE"),
        max_days: int = typer.Option(45, "--max-days", help="Max DTE"),
        calls: bool = typer.Option(False, "--calls", help="Only calls"),
        puts: bool = typer.Option(False, "--puts", help="Only puts"),
        price_basis: str = typer.Option("ask", "--price-basis", help="ask|mid|last (premium basis)"),
        min_price: float = typer.Option(0.05, "--min-price", help="Minimum option price (filters out $0.01 junk)"),
        target_abs_delta: float = typer.Option(0.30, "--target-abs-delta", help="Target |delta| for the picked contract"),
        max_spread_pct: float = typer.Option(0.30, "--max-spread-pct", help="Prefer contracts with spread <= this"),
        workers: int = typer.Option(6, "--workers", help="Parallel workers (API load)."),
        tickers: str = typer.Option(
            "",
            "--tickers",
            help="Optional comma-separated ETF tickers (defaults to STARTER_UNIVERSE basket)",
        ),
        max_results: int = typer.Option(50, "--max-results", help="Max rows to print"),
    ):
        """
        Scan a small ETF universe and pick 1 contract per ticker under a premium budget.

        Defaults are tuned for small accounts:
        - premium <= $100 (ask)
        - delta required
        - DTE 7..45
        """
        want = "both"
        if calls and not puts:
            want = "call"
        elif puts and not calls:
            want = "put"

        pb = price_basis.strip().lower()
        if pb not in {"ask", "mid", "last"}:
            pb = "ask"

        if tickers.strip():
            uni = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        else:
            uni = [t.strip().upper() for t in STARTER_UNIVERSE.basket_equity]

        settings = load_settings()
        _trading, data = make_clients(settings)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _scan_one(t: str):
            chain = fetch_option_chain(data, t, feed=settings.alpaca_options_feed)
            candidates = list(to_candidates(chain, t))
            opts = affordable_options_for_ticker(
                candidates,
                ticker=t,
                max_premium_usd=float(max_premium_usd),
                min_dte_days=int(min_days),
                max_dte_days=int(max_days),
                want=want,  # type: ignore[arg-type]
                price_basis=pb,  # type: ignore[arg-type]
                min_price=float(min_price),
                require_delta=True,
                today=date.today(),
            )
            best = pick_best_affordable(opts, target_abs_delta=float(target_abs_delta), max_spread_pct=float(max_spread_pct))
            return t, best

        results = []
        errors = 0
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            futs = {ex.submit(_scan_one, t): t for t in uni}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    _t, best = fut.result()
                    if best is not None:
                        results.append(best)
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"[dim]{t}: {type(e).__name__}: {e}[/dim]")

        # Sort by |Δ| descending (user intent), then cheapest premium.
        results.sort(key=lambda o: (-abs(float(o.delta)), o.premium_usd, o.ticker))
        shown = results[: max(0, int(max_results))]

        tbl = Table(title=f"ETF options under ${float(max_premium_usd):.0f} (1 pick / ticker)")
        tbl.add_column("Ticker", style="bold")
        tbl.add_column("Symbol", style="bold")
        tbl.add_column("Type")
        tbl.add_column("Exp")
        tbl.add_column("DTE", justify="right")
        tbl.add_column("Strike", justify="right")
        tbl.add_column("Price", justify="right")
        tbl.add_column("Premium", justify="right")
        tbl.add_column("Spread%", justify="right")
        tbl.add_column("Δ", justify="right")
        tbl.add_column("IV", justify="right")

        for o in shown:
            sp = (100.0 * float(o.spread_pct)) if isinstance(o.spread_pct, float) else None
            tbl.add_row(
                o.ticker,
                o.symbol,
                o.opt_type,
                o.expiry.isoformat(),
                str(o.dte_days),
                f"{o.strike:.2f}",
                _fmt_price(o.price),
                f"${o.premium_usd:,.0f}",
                (f"{sp:.1f}%" if sp is not None else "n/a"),
                _fmt_price(o.delta),
                _fmt_pct(o.iv),
            )

        Console().print(tbl)
        print(
            f"[dim]Found: {len(results)}/{len(uni)} tickers with at least one contract <= ${float(max_premium_usd):.0f} "
            f"(delta required). Errors: {errors}. Price basis: {pb}. DTE: {min_days}..{max_days}[/dim]"
        )

    @options_app.command("moonshot")
    def options_moonshot(
        basket: str = typer.Option("starter", "--basket", help="starter|extended (universe)"),
        ticker: str = typer.Option("", "--ticker", "-t", help="Optional: restrict to a single underlying ticker (e.g. SLV)"),
        tickers: str = typer.Option("", "--tickers", help="Optional: restrict to a comma-separated list of tickers (overrides --basket)"),
        start_date: str = typer.Option("2012-01-01", "--start-date", help="History start date (YYYY-MM-DD)"),
        horizon_days: int = typer.Option(63, "--horizon-days", help="Forward return horizon for 'extreme move' search"),
        k_analogs: int = typer.Option(250, "--k-analogs", help="How many closest regime analog days to use"),
        min_abs_extreme: float = typer.Option(0.15, "--min-abs-extreme", help="Require an extreme analog move of at least this (e.g. 0.15 = 15%)"),
        min_samples: int = typer.Option(40, "--min-samples", help="Min analog samples required per ticker"),
        direction: str = typer.Option("both", "--direction", help="bullish|bearish|both"),
        # Volatility focus (so moonshots are drawn from tickers that can actually move hard either way)
        vol_lookback_days: int = typer.Option(14, "--vol-lookback-days", help="Lookback window (days) for realized-vol filter"),
        vol_top_pct: float = typer.Option(0.35, "--vol-top-pct", help="Keep only the top X%% most-volatile tickers (0..1). Set 1.0 to disable."),
        vol_min_ann: float = typer.Option(0.0, "--vol-min-ann", help="Optional minimum annualized realized vol (e.g., 0.40 = 40%%)."),
        # Option picking / affordability
        min_days: int = typer.Option(14, "--min-days", help="Min DTE for the recommended contract"),
        max_days: int = typer.Option(60, "--max-days", help="Max DTE for the recommended contract"),
        price_basis: str = typer.Option("ask", "--price-basis", help="ask|mid|last (premium basis)"),
        min_price: float = typer.Option(0.05, "--min-price", help="Minimum option price (filters out $0.01 junk)"),
        target_abs_delta: float = typer.Option(0.12, "--target-abs-delta", help="Target |delta| for a 'moonshot' (smaller = more OTM)"),
        max_spread_pct: float = typer.Option(0.60, "--max-spread-pct", help="Prefer contracts with spread <= this"),
        top: int = typer.Option(10, "--top", help="How many ranked candidates to print"),
        # Budget / moonshot behavior
        cash_usd: float = typer.Option(0.0, "--cash", help="Override cash budget (USD). If 0, uses Alpaca account cash."),
        all_in_threshold_usd: float = typer.Option(50.0, "--all-in-threshold", help="If cash <= this, spend ~all cash on one contract"),
        max_premium_usd: float = typer.Option(0.0, "--max-premium", help="Optional cap on contract premium. If 0, uses cash (or a fraction when cash is larger)."),
        review: bool = typer.Option(True, "--review/--no-review", help="Step through moonshot candidates one by one and prompt to execute per ticker"),
        review_limit: int = typer.Option(25, "--review-limit", help="Max tickers to step through in review mode"),
        with_theory: bool = typer.Option(True, "--theory/--no-theory", help="Generate a short grounded thesis for each ticker (requires OPENAI_API_KEY)"),
        theory_model: str = typer.Option("", "--theory-model", help="Optional override model for theory generation"),
        theory_temperature: float = typer.Option(0.2, "--theory-temperature", help="Theory temperature (0.0..1.0)"),
        execute: bool = typer.Option(False, "--execute", help="If set, can submit the order after confirmation (paper by default; use --live for live)"),
        live: bool = typer.Option(False, "--live", help="Allow LIVE execution when ALPACA_PAPER is false (guarded with extra confirmations)"),
    ):
        """
        Find "extreme move happened before" analogs across ALL regime features, then recommend an OTM option.

        This is intentionally a high-variance scanner. If cash <= --all-in-threshold, it will try to recommend
        a single contract spending ~all available cash ("moonshot").
        """
        pb = price_basis.strip().lower()
        if pb not in {"ask", "mid", "last"}:
            pb = "ask"

        settings = load_settings()
        trading, data = make_clients(settings)

        live_ok = bool(live) and (not bool(settings.alpaca_paper))
        if execute and (not settings.alpaca_paper) and (not live_ok):
            Console().print(
                Panel(
                    "[red]Refusing to execute[/red] because ALPACA_PAPER is false.\n"
                    "If you intend LIVE trading, re-run with [b]--live --execute[/b].",
                    title="Safety",
                    expand=False,
                )
            )
            raise typer.Exit(code=1)
        if execute and live_ok:
            Console().print(
                Panel(
                    "[yellow]LIVE MODE ENABLED[/yellow]\n"
                    "Orders will be submitted to your LIVE Alpaca account.\n"
                    "You will be asked to confirm again before submission.",
                    title="Safety",
                    expand=False,
                )
            )
            if not typer.confirm("Confirm LIVE mode (ALPACA_PAPER=false) and proceed?", default=False):
                raise typer.Exit(code=0)
            if not typer.confirm("Second confirmation: proceed with LIVE trading now?", default=False):
                raise typer.Exit(code=0)

        # Budget: default to live cash unless overridden.
        cash_live = None
        if float(cash_usd) > 0:
            cash_live = float(cash_usd)
        else:
            try:
                acct = trading.get_account()
                cash_live = float(getattr(acct, "cash", 0.0) or 0.0)
            except Exception:
                cash_live = 0.0

        # Reasonable default cap: all-in when tiny; otherwise limit per-contract to avoid accidental full spend.
        if float(max_premium_usd) > 0:
            prem_cap = float(max_premium_usd)
        else:
            prem_cap = float(cash_live) if float(cash_live) <= float(all_in_threshold_usd) else float(min(100.0, 0.35 * float(cash_live)))
        prem_cap = float(max(0.0, prem_cap))

        # Universe
        try:
            from ai_options_trader.portfolio.universe import get_universe
        except Exception:  # pragma: no cover
            from ai_options_trader.portfolio.universe import STARTER_UNIVERSE, DEFAULT_UNIVERSE

            def get_universe(name: str):
                n = (name or "starter").strip().lower()
                if n.startswith("d"):
                    return DEFAULT_UNIVERSE
                return STARTER_UNIVERSE

        symbols: list[str]
        if tickers.strip():
            symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        elif ticker.strip():
            symbols = [ticker.strip().upper()]
        else:
            uni = get_universe(basket)
            symbols = sorted(set(uni.basket_equity))

        # Regimes + prices
        X = build_regime_feature_matrix(settings=settings, start_date=str(start_date), refresh_fred=False)
        px = fetch_equity_daily_closes(settings=settings, symbols=symbols, start=str(start_date), refresh=False).sort_index().ffill()

        asof = min(pd.to_datetime(X.index.max()), pd.to_datetime(px.index.max()))
        asof_str = str(pd.to_datetime(asof).date())
        feat_row = {}
        try:
            feat_row = X.loc[pd.to_datetime(asof)].to_dict()
        except Exception:
            feat_row = {}

        # Realized volatility (annualized) from daily returns, for "can move either way" filtering.
        rv_ann = None
        try:
            r = px.pct_change().replace([float("inf"), float("-inf")], pd.NA)
            rv = r.rolling(max(5, int(vol_lookback_days))).std() * (252.0 ** 0.5)
            if pd.to_datetime(asof) in rv.index:
                rv_ann = rv.loc[pd.to_datetime(asof)]
            else:
                rv_ann = rv.iloc[-1]
        except Exception:
            rv_ann = None
        ranked = rank_moonshots(
            px=px,
            regimes=X,
            asof=asof,
            horizon_days=int(horizon_days),
            k_analogs=int(k_analogs),
            min_abs_extreme=float(min_abs_extreme),
            min_samples=int(min_samples),
            direction=str(direction),
        )

        if not ranked:
            # Fallback ladder: relax extreme threshold + samples, then fall back to unconditional tails.
            Console().print(
                Panel(
                    "No moonshot candidates found under the current analog constraints.\n"
                    "Falling back by relaxing thresholds, then (if needed) using unconditional tail ranking.",
                    title="Moonshot fallback",
                    expand=False,
                )
            )
            tried = [
                # (min_abs_extreme, min_samples)
                (float(min_abs_extreme) * 0.75, max(25, int(min_samples))),
                (float(min_abs_extreme) * 0.50, max(20, int(min_samples) // 2)),
                (max(0.05, float(min_abs_extreme) * 0.33), max(15, int(min_samples) // 3)),
            ]
            for mae, ms in tried:
                ranked = rank_moonshots(
                    px=px,
                    regimes=X,
                    asof=asof,
                    horizon_days=int(horizon_days),
                    k_analogs=int(k_analogs),
                    min_abs_extreme=float(mae),
                    min_samples=int(ms),
                    direction=str(direction),
                )
                if ranked:
                    Console().print(Panel(f"Recovered {len(ranked)} candidate(s) after relaxing: min_abs_extreme={mae:.3f} min_samples={ms}", title="Fallback", expand=False))
                    break
            if not ranked:
                ranked = rank_moonshots_unconditional(
                    px=px,
                    asof=asof,
                    horizon_days=int(horizon_days),
                    min_samples=max(60, int(min_samples)),
                    direction=str(direction),
                )
                if ranked:
                    Console().print(Panel(f"Using unconditional tail ranking ({len(ranked)} candidate(s)).", title="Fallback", expand=False))
                else:
                    print("[yellow]No moonshot candidates available[/yellow] (insufficient price history).")
                    raise typer.Exit(code=0)

        # Volatility filter: keep only the most volatile tickers so the option has a chance to pay in either direction.
        if rv_ann is not None and len(symbols) > 1:
            try:
                v = pd.to_numeric(rv_ann, errors="coerce").dropna()
                if not v.empty:
                    keep = set(v.index)
                    # Apply min realized vol (optional)
                    if float(vol_min_ann) > 0:
                        keep = {t for t in keep if float(v.get(t, 0.0) or 0.0) >= float(vol_min_ann)}
                    # Apply top-percentile filter (default keeps top 35%)
                    top_pct = float(vol_top_pct)
                    if 0.0 < top_pct < 1.0:
                        thr = float(v.quantile(max(0.0, min(1.0, 1.0 - top_pct))))
                        keep = {t for t in keep if float(v.get(t, 0.0) or 0.0) >= thr}
                    ranked_vol = [r for r in ranked if str(r.ticker).strip().upper() in keep]
                    if ranked_vol:
                        ranked = ranked_vol
                        Console().print(
                            Panel(
                                f"Volatility filter applied: lookback={int(vol_lookback_days)}d, top_pct={float(vol_top_pct):.2f}, "
                                f"min_ann={float(vol_min_ann):.2f} → candidates={len(ranked)}",
                                title="Vol filter",
                                expand=False,
                            )
                        )
                    else:
                        Console().print(Panel("Volatility filter would remove all candidates; keeping unfiltered list.", title="Vol filter", expand=False))
            except Exception:
                pass

        # Print top candidates
        tbl = Table(title=f"Moonshot scan (asof={asof.date()} | horizon={horizon_days}d | analogs={k_analogs} | cash≈${cash_live:,.2f})")
        tbl.add_column("Rank", justify="right")
        tbl.add_column("Ticker", style="bold")
        tbl.add_column("Dir")
        tbl.add_column("rv", justify="right")
        tbl.add_column("Score", justify="right")
        tbl.add_column("q95", justify="right")
        tbl.add_column("q50", justify="right")
        tbl.add_column("q05", justify="right")
        tbl.add_column("best", justify="right")
        tbl.add_column("worst", justify="right")
        tbl.add_column("extreme (date, ret)", justify="right")
        tbl.add_column("n", justify="right")

        shown = ranked[: max(1, int(top))]
        for i, r in enumerate(shown, start=1):
            ex = "—"
            if r.extreme_date is not None and r.extreme_return is not None:
                ex = f"{pd.to_datetime(r.extreme_date).date()} {100.0*float(r.extreme_return):+.1f}%"
            rv_s = "—"
            try:
                if rv_ann is not None:
                    vv = float(pd.to_numeric(rv_ann.get(str(r.ticker).strip().upper()), errors="coerce"))
                    if vv == vv:  # not nan
                        rv_s = f"{100.0*vv:.0f}%"
            except Exception:
                rv_s = "—"
            tbl.add_row(
                str(i),
                str(r.ticker),
                "CALL" if r.direction == "bullish" else "PUT",
                rv_s,
                f"{float(r.score):.3f}",
                f"{100.0*float(r.q95):+.1f}%" if r.q95 is not None else "—",
                f"{100.0*float(r.q50):+.1f}%" if r.q50 is not None else "—",
                f"{100.0*float(r.q05):+.1f}%" if r.q05 is not None else "—",
                f"{100.0*float(r.best):+.1f}%" if r.best is not None else "—",
                f"{100.0*float(r.worst):+.1f}%" if r.worst is not None else "—",
                ex,
                str(int(r.samples)),
            )
        Console().print(tbl)

        def _scan_best_contract_for_candidate(r) -> tuple[object | None, float]:
            """Return (AffordableOption|None, premium_cap_used) for this ticker under current budget."""
            # Update premium cap against remaining cash.
            cap_now = float(prem_cap)
            if float(remaining_cash) > 0:
                cap_now = float(min(cap_now, float(remaining_cash)))
            if cap_now <= 0:
                return None, cap_now

            want = "call" if r.direction == "bullish" else "put"
            chain = fetch_option_chain(data, r.ticker, feed=settings.alpaca_options_feed)
            candidates = list(to_candidates(chain, r.ticker))

            # Fallback ladder: progressively relax constraints until we find *some* in-budget longshot.
            # Goal: always produce a trade when possible (budget-fit contract exists).
            attempts = [
                # (min_dte, max_dte, min_price, target_abs_delta, max_spread_pct)
                (int(min_days), int(max_days), float(min_price), float(target_abs_delta), float(max_spread_pct)),
                (int(min_days), int(max_days), 0.02, max(0.15, float(target_abs_delta)), max(0.80, float(max_spread_pct))),
                (max(7, int(min_days) // 2), max(30, int(max_days)), 0.01, max(0.25, float(target_abs_delta)), 1.20),
                (0, max(90, int(max_days) * 2), 0.01, 0.35, 2.00),
            ]

            for mind, maxd, minpx, targd, spmax in attempts:
                opts = affordable_options_for_ticker(
                    candidates,
                    ticker=r.ticker,
                    max_premium_usd=float(cap_now),
                    min_dte_days=int(mind),
                    max_dte_days=int(maxd),
                    want=want,  # type: ignore[arg-type]
                    price_basis=pb,  # type: ignore[arg-type]
                    min_price=float(minpx),
                    require_delta=True,
                    today=pd.Timestamp(asof).date(),
                )
                best = pick_best_affordable(opts, target_abs_delta=float(targd), max_spread_pct=float(spmax))
                if best is not None:
                    return best, cap_now

            return None, cap_now

        # Interactive review deck: step through candidates and prompt per ticker.
        remaining_cash = float(cash_live or 0.0)
        all_in = float(remaining_cash) <= float(all_in_threshold_usd) and float(remaining_cash) > 0
        if all_in:
            prem_cap = float(remaining_cash)
            Console().print(
                Panel(
                    f"[red]ALL-IN MODE[/red]: cash≈${remaining_cash:,.2f} <= ${float(all_in_threshold_usd):.0f}\n"
                    "You can still skip/choose per ticker; remaining cash will be tracked.",
                    title="Moonshot",
                    expand=False,
                )
            )

        if review:
            label = "LIVE" if live_ok else "PAPER"
            n_review = min(int(review_limit), len(ranked))
            Console().print(Panel(f"Reviewing {n_review} moonshot candidate(s). Remaining cash≈${remaining_cash:,.2f}.", title="Review", expand=False))

            for i, r in enumerate(ranked[:n_review], start=1):
                if float(remaining_cash) <= 0 and float(cash_live or 0.0) > 0:
                    Console().print(Panel("Remaining cash is ~0; stopping review.", title="Budget", expand=False))
                    break

                # Try to pick a contract for this ticker.
                try:
                    best, cap_now = _scan_best_contract_for_candidate(r)
                except Exception as e:
                    Console().print(f"[dim]{i}/{n_review} {r.ticker}: skip (option scan error: {type(e).__name__})[/dim]")
                    continue
                if best is None:
                    Console().print(f"[dim]{i}/{n_review} {r.ticker}: no contract found under cap=${cap_now:,.2f}[/dim]")
                    continue

                ex = "—"
                if r.extreme_date is not None and r.extreme_return is not None:
                    ex = f"{pd.to_datetime(r.extreme_date).date()} {100.0*float(r.extreme_return):+.1f}%"

                hdr = (
                    f"{i}/{n_review}  {r.ticker}  "
                    f"{'CALL' if r.direction == 'bullish' else 'PUT'}  "
                    f"score={float(r.score):.3f}  "
                    f"extreme={ex}  "
                    f"cash≈${remaining_cash:,.2f}"
                )
                Console().print(Panel(hdr, title="Candidate", expand=False))

                if with_theory:
                    try:
                        theory = llm_moonshot_theory(
                            settings=settings,
                            asof=asof_str,
                            ticker=str(r.ticker),
                            direction=str(r.direction),
                            horizon_days=int(horizon_days),
                            regime_features=feat_row,
                            analog_stats={
                                "samples": int(r.samples),
                                "q05": r.q05,
                                "q50": r.q50,
                                "q95": r.q95,
                                "best": r.best,
                                "worst": r.worst,
                                "extreme_date": r.extreme_date,
                                "extreme_return": r.extreme_return,
                            },
                            model=(theory_model.strip() or None),
                            temperature=float(theory_temperature),
                        )
                        Console().print(Panel(theory, title="Theory (grounded)", expand=False))
                    except Exception as e:
                        Console().print(f"[dim]Theory unavailable: {type(e).__name__}: {e}[/dim]")

                # Contract preview
                tbl2 = Table(title="Proposed contract")
                tbl2.add_column("Underlying", style="bold")
                tbl2.add_column("Type", style="bold")
                tbl2.add_column("Contract", style="bold")
                tbl2.add_column("Exp")
                tbl2.add_column("DTE", justify="right")
                tbl2.add_column("Strike", justify="right")
                tbl2.add_column("Price", justify="right")
                tbl2.add_column("Premium", justify="right")
                tbl2.add_column("|Δ|", justify="right")
                tbl2.add_column("Analog extreme", justify="right")
                tbl2.add_row(
                    str(best.ticker),
                    "CALL" if best.opt_type == "call" else "PUT",
                    str(best.symbol),
                    best.expiry.isoformat(),
                    str(int(best.dte_days)),
                    f"{float(best.strike):.2f}",
                    f"{float(best.price):.2f}",
                    f"${float(best.premium_usd):,.0f}",
                    f"{abs(float(best.delta)):.2f}" if best.delta is not None else "n/a",
                    ex,
                )
                Console().print(tbl2)

                # Sizing: in all-in mode, buy as many as fit in remaining cash; otherwise 1.
                qty = 1
                if float(remaining_cash) > 0 and float(best.premium_usd) > 0:
                    qty = max(1, int(float(remaining_cash) // float(best.premium_usd))) if all_in else 1
                est_total = float(qty) * float(best.premium_usd)

                if not typer.confirm(f"Execute: BUY {qty}x {best.symbol} (limit≈{float(best.price):.2f})? [{label}]", default=False):
                    # Allow fast exit if user wants.
                    if typer.confirm("Stop reviewing moonshots?", default=False):
                        break
                    continue

                # Reserve budget immediately (even in dry-run) so the review behaves realistically.
                remaining_cash = max(0.0, float(remaining_cash) - float(est_total))
                Console().print(Panel(f"Reserved ≈${est_total:,.2f}. Remaining cash≈${remaining_cash:,.2f}.", title="Budget", expand=False))

                if not execute:
                    print("[dim]DRY RUN[/dim]: re-run with `--execute` to submit orders.")
                    continue

                try:
                    resp = submit_option_order(
                        trading=trading,
                        symbol=str(best.symbol),
                        qty=int(qty),
                        side="buy",
                        limit_price=float(best.price),
                        tif="day",
                    )
                    print(f"[green]Submitted {label} order[/green]: {resp}")
                except Exception as e:
                    print(f"[red]Order submission failed[/red]: {type(e).__name__}: {e}")
                    raise typer.Exit(code=2)

            raise typer.Exit(code=0)

        # Non-review mode (single best pick from the printed top list)
        recommended = None
        rec_stats = None
        remaining_cash = float(cash_live or 0.0)
        all_in = float(remaining_cash) <= float(all_in_threshold_usd) and float(remaining_cash) > 0
        if all_in:
            prem_cap = float(remaining_cash)

        for r in shown:
            try:
                best, _cap_now = _scan_best_contract_for_candidate(r)
                if best is not None:
                    recommended = best
                    rec_stats = r
                    break
            except Exception:
                continue

        if recommended is None:
            print(
                "[yellow]No affordable option found[/yellow] under the current constraints.\n"
                "Try increasing --max-premium, widening --max-spread-pct, or raising --target-abs-delta."
            )
            raise typer.Exit(code=0)

        r = rec_stats
        assert r is not None
        tbl2 = Table(title="Moonshot recommendation (scanner)")
        tbl2.add_column("Underlying", style="bold")
        tbl2.add_column("Type", style="bold")
        tbl2.add_column("Contract", style="bold")
        tbl2.add_column("Exp")
        tbl2.add_column("DTE", justify="right")
        tbl2.add_column("Strike", justify="right")
        tbl2.add_column("Price", justify="right")
        tbl2.add_column("Premium", justify="right")
        tbl2.add_column("|Δ|", justify="right")
        tbl2.add_column("Analog extreme", justify="right")
        ex = "—"
        if r.extreme_date is not None and r.extreme_return is not None:
            ex = f"{pd.to_datetime(r.extreme_date).date()} {100.0*float(r.extreme_return):+.1f}%"
        tbl2.add_row(
            str(recommended.ticker),
            "CALL" if recommended.opt_type == "call" else "PUT",
            str(recommended.symbol),
            recommended.expiry.isoformat(),
            str(int(recommended.dte_days)),
            f"{float(recommended.strike):.2f}",
            f"{float(recommended.price):.2f}",
            f"${float(recommended.premium_usd):,.0f}",
            f"{abs(float(recommended.delta)):.2f}" if recommended.delta is not None else "n/a",
            ex,
        )
        Console().print(tbl2)
        qty = 1
        try:
            if all_in and float(recommended.premium_usd) > 0:
                qty = max(1, int(float(remaining_cash) // float(recommended.premium_usd)))
        except Exception:
            qty = 1
        label = "LIVE" if live_ok else "PAPER"
        if typer.confirm(f"Execute: BUY {qty}x {recommended.symbol} (limit≈{float(recommended.price):.2f}) now? [{label}]", default=False):
            if not execute:
                print("[dim]DRY RUN[/dim]: re-run with `--execute` to submit this order.")
                raise typer.Exit(code=0)
            try:
                resp = submit_option_order(
                    trading=trading,
                    symbol=str(recommended.symbol),
                    qty=int(qty),
                    side="buy",
                    limit_price=float(recommended.price),
                    tif="day",
                )
                print(f"[green]Submitted {label} order[/green]: {resp}")
            except Exception as e:
                print(f"[red]Order submission failed[/red]: {type(e).__name__}: {e}")
                raise typer.Exit(code=2)



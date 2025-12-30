from __future__ import annotations

import typer
from rich import print

from ai_options_trader.config import StrategyConfig, RiskConfig, load_settings
from ai_options_trader.data.alpaca import make_clients, fetch_option_chain, to_candidates
from ai_options_trader.strategy.selector import choose_best_option


def register(ideas_app: typer.Typer) -> None:
    @ideas_app.command("ai-bubble")
    def ideas_ai_bubble(
        start: str = typer.Option("2011-01-01", "--start"),
        refresh: bool = typer.Option(False, "--refresh"),
        macro_window: int = typer.Option(252, "--macro-window", help="Rolling window (trading days) for sensitivity"),
        macro_benchmark: str = typer.Option("QQQ", "--macro-benchmark"),
        tariff_benchmark: str = typer.Option("XLY", "--tariff-benchmark"),
        tariff_baskets: str = typer.Option("all", "--tariff-baskets", help="Comma-separated or 'all'"),
        tech_tickers: str = typer.Option("", "--tech-tickers", help="Override AI/tech universe (comma-separated)"),
        top: int = typer.Option(12, "--top", help="Number of ideas to print"),
        with_legs: bool = typer.Option(False, "--with-legs", help="Fetch option chains and pick best leg per idea"),
        target_dte: int = typer.Option(30, "--target-dte", help="Target days-to-expiry for option selection"),
        interactive: bool = typer.Option(
            False,
            "--interactive",
            help="Show one idea at a time and (optionally) submit paper orders after confirmation",
        ),
        execute: bool = typer.Option(
            False,
            "--execute",
            help="When used with --interactive, allow submitting orders (still asks for confirmation)",
        ),
    ):
        """
        Generate option-trading ideas for the thesis:
        AI bubble + inflation underpriced in tech + tariffs underpriced in import-exposed stocks.
        """
        from rich.table import Table
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        from ai_options_trader.ideas.ai_bubble import build_ai_bubble_ideas
        from ai_options_trader.tracking.store import TrackerStore, default_tracker_db_path

        settings = load_settings()
        tracker = TrackerStore(default_tracker_db_path())
        run_id = tracker.new_run_id()

        if tariff_baskets.strip().lower() == "all":
            basket_list = None
        else:
            basket_list = [b.strip() for b in tariff_baskets.split(",") if b.strip()]

        tech_list = [t.strip().upper() for t in tech_tickers.split(",") if t.strip()] or None

        try:
            _ctx, ideas = build_ai_bubble_ideas(
                settings=settings,
                start_date=start,
                refresh=refresh,
                macro_window=macro_window,
                macro_benchmark=macro_benchmark,
                tariff_benchmark=tariff_benchmark,
                tariff_baskets=basket_list,
                tech_tickers=tech_list,
            )
        except Exception as e:
            # Keep output readable; point to likely causes.
            Console().print(
                Panel(
                    f"[red]Failed to build ideas.[/red]\n\n{e}\n\n"
                    "Common fixes:\n"
                    "- Re-run (FRED occasionally times out)\n"
                    "- Try `--refresh` if your cache is stale\n"
                    "- Ensure `FRED_API_KEY` is set\n",
                    title="Error",
                )
            )
            raise typer.Exit(code=1)

        ideas = ideas[: max(1, int(top))]

        # Option legs (optional)
        if with_legs:
            trading, data = make_clients(settings)
            equity_usd = float(trading.get_account().equity)
            strat = StrategyConfig(target_dte_days=target_dte)
            risk = RiskConfig()

            for idea in ideas:
                want = "put" if idea.direction == "bearish" else "call"
                chain = fetch_option_chain(data, idea.ticker)
                candidates = list(to_candidates(chain, idea.ticker))
                best = choose_best_option(
                    candidates,
                    idea.ticker,
                    want=want,
                    equity_usd=equity_usd,
                    strat=strat,
                    risk=risk,
                )
                if best:
                    idea.option_leg = {
                        "symbol": best.symbol,
                        "type": best.opt_type,
                        "expiry": str(best.expiry),
                        "strike": best.strike,
                        "dte_days": best.dte_days,
                        "mid": best.mid,
                        "delta": best.delta,
                        "gamma": best.gamma,
                        "theta": best.theta,
                        "vega": best.vega,
                        "iv": best.iv,
                        "contracts": best.size.max_contracts,
                        "budget_usd": best.size.budget_usd,
                    }

        # Always log recommendations (one row per idea) so we can track later.
        reco_ids: dict[str, str] = {}
        for idea in ideas:
            reco_id = tracker.log_recommendation(
                run_id=run_id,
                ticker=idea.ticker,
                direction=idea.direction,
                score=float(idea.score),
                tags=list(idea.tags),
                thesis=idea.thesis,
                rationale=idea.rationale,
                why=dict(idea.why or {}),
                option_leg=dict(idea.option_leg) if idea.option_leg else None,
            )
            reco_ids[idea.ticker] = reco_id

        # Interactive: one idea at a time
        if interactive:
            console = Console()
            if execute:
                from ai_options_trader.execution.alpaca import submit_option_order

            for i, idea in enumerate(ideas, start=1):
                why = idea.why or {}
                macro = why.get("macro", {})
                tech = why.get("tech_sensitivity", {})
                tariff = why.get("tariff", {})
                comps = why.get("score_components", {})

                body = Text()
                body.append(f"Rank: {i}/{len(ideas)}\n", style="bold")
                body.append(f"Ticker: {idea.ticker} | Direction: {idea.direction} | Score: {idea.score:.2f}\n\n")
                body.append("WHY THIS IDEA\n", style="bold")
                if macro:
                    body.append(
                        f"- Macro regime: {macro.get('regime')} (infl={macro.get('infl_trend')}, real={macro.get('real_yield_trend')})\n"
                    )
                    if macro.get("z_infl_mom_minus_be5y") is not None:
                        body.append(f"- z(infl mom - BE5y): {macro.get('z_infl_mom_minus_be5y'):.2f}\n")
                if tech:
                    br = tech.get("beta_d_real")
                    rp = tech.get("rank_pct")
                    body.append(f"- Tech sensitivity: beta_d_real={br:.4f} | rank={rp:.0%}\n" if br is not None else "")
                if tariff and tariff.get("baskets"):
                    bs = tariff.get("baskets", [])
                    body.append("- Tariff exposure:\n")
                    for b in bs:
                        body.append(
                            f"  - {b.get('basket')}: is_regime={b.get('is_tariff_regime')} score={b.get('tariff_regime_score')}\n"
                        )
                if comps:
                    body.append("\nSCORE BREAKDOWN\n", style="bold")
                    for k, v in comps.items():
                        body.append(f"- {k}: {float(v):.2f}\n")

                body.append("\nRECOMMENDED STRUCTURE\n", style="bold")
                body.append(
                    "Put debit spread (starter workflow: select a liquid long put leg first)\n"
                    if idea.direction == "bearish"
                    else "Call (starter)\n"
                )

                leg = idea.option_leg
                body.append("\nOPTION LEG\n", style="bold")
                if leg:
                    body.append(
                        f"- {leg['symbol']} exp={leg['expiry']} strike={leg['strike']} dte={leg['dte_days']} mid=${leg['mid']:.2f}\n"
                    )
                    body.append(
                        f"- Δ={leg.get('delta'):.3f} Γ={leg.get('gamma')} Θ={leg.get('theta')} ν={leg.get('vega')} IV={leg.get('iv')}\n"
                    )
                    body.append(f"- Suggested qty={leg['contracts']} (budget=${leg['budget_usd']:,.2f})\n")
                else:
                    body.append("- (run with --with-legs to select an options leg)\n")

                console.print(Panel(body, title="Idea", expand=False))

                if execute and leg:
                    if not settings.ALPACA_PAPER:
                        console.print("[red]Refusing to submit orders because ALPACA_PAPER is false.[/red]")
                    else:
                        if typer.confirm("Submit this order to Alpaca PAPER now?", default=False):
                            qty = int(leg["contracts"]) if leg.get("contracts") else 1
                            if qty < 1:
                                qty = 1
                            # Use a limit at mid by default
                            resp = submit_option_order(
                                trading=trading,
                                symbol=str(leg["symbol"]),
                                qty=qty,
                                side="buy",
                                limit_price=float(leg["mid"]),
                                tif="day",
                            )
                            console.print(f"[green]Submitted[/green]: {resp}")
                            # Track execution by linking to the recommendation we just logged.
                            reco_id = reco_ids.get(idea.ticker)
                            if reco_id:
                                tracker.log_execution(
                                    recommendation_id=reco_id,
                                    alpaca_order_id=str(resp.id),
                                    symbol=str(resp.symbol),
                                    qty=int(qty),
                                    side=str(resp.side),
                                    order_type=str(resp.order_type),
                                    limit_price=float(getattr(resp, "limit_price", None) or leg.get("mid")),
                                    status=str(resp.status),
                                    filled_qty=int(getattr(resp, "filled_qty", 0) or 0),
                                    filled_avg_price=float(getattr(resp, "filled_avg_price", 0) or 0) or None,
                                    filled_at=str(getattr(resp, "filled_at", None) or "") or None,
                                    raw=resp.model_dump() if hasattr(resp, "model_dump") else str(resp),
                                )

                if i < len(ideas):
                    if not typer.confirm("Next idea?", default=True):
                        break
            return

        # Print table
        tbl = Table(title="AI Bubble / Inflation / Tariff Thesis — Idea List")
        tbl.add_column("Ticker", style="bold")
        tbl.add_column("Direction")
        tbl.add_column("Score", justify="right")
        tbl.add_column("Tags")
        tbl.add_column("Why (short)")
        tbl.add_column("Structure")
        if with_legs:
            tbl.add_column("Leg (greeks)")

        for idea in ideas:
            structure = "Put debit spread" if idea.direction == "bearish" else "Call"
            why = idea.why or {}
            macro = why.get("macro", {})
            tech = why.get("tech_sensitivity", {})
            tariff = why.get("tariff", {})
            short_bits = []
            if macro.get("regime"):
                short_bits.append(f"macro={macro.get('regime')}")
            if tech.get("beta_d_real") is not None:
                short_bits.append(f"β_real={tech.get('beta_d_real'):.2f}")
            if tariff.get("baskets"):
                short_bits.append(f"tariff={len(tariff.get('baskets'))}b")
            why_short = " ".join(short_bits) if short_bits else "—"
            row = [
                idea.ticker,
                idea.direction,
                f"{idea.score:.2f}",
                ",".join(idea.tags),
                why_short,
                structure,
            ]
            if with_legs:
                leg = idea.option_leg
                if leg:
                    row.append(f"{leg['symbol']} Δ={leg.get('delta'):.2f} mid=${leg.get('mid'):.2f}")
                else:
                    row.append("—")
            tbl.add_row(*row)

        Console().print(tbl)
        print("\nNotes:")
        print("- These are research ideas, not advice. Use defined-risk spreads and size conservatively.")
        print("- Use `--with-legs` to pull current option chains and select a liquid leg per idea.")
        print("- Use `--interactive --with-legs --execute` for a one-by-one review + optional paper execution.")
        print(f"- This run logged to tracker DB: {default_tracker_db_path()} (run_id={run_id})")



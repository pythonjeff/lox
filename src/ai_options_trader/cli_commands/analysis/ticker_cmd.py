from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
import os
import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from rich.markdown import Markdown

from ai_options_trader.config import load_settings
from ai_options_trader.overlay.context import extract_underlyings
from ai_options_trader.utils.settings import safe_load_settings


def _fmt_market_cap(val: float | None) -> str:
    """Format market cap for display."""
    if val is None:
        return "—"
    if val >= 1e12:
        return f"${val/1e12:.2f}T"
    if val >= 1e9:
        return f"${val/1e9:.1f}B"
    if val >= 1e6:
        return f"${val/1e6:.0f}M"
    return f"${val:,.0f}"


def register(ticker_app: typer.Typer) -> None:
    @ticker_app.command("snapshot")
    def ticker_snapshot(
        ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol (e.g., AAPL)"),
        benchmark: str = typer.Option("SPY", "--benchmark", help="Benchmark symbol for relative strength (default SPY)"),
        start: str = typer.Option("2011-01-01", "--start", help="Start date YYYY-MM-DD"),
    ):
        """Print a quantitative snapshot for a ticker (returns, vol, drawdown, rel strength)."""
        settings = load_settings()
        from ai_options_trader.ticker.snapshot import build_ticker_snapshot

        snap = build_ticker_snapshot(settings=settings, ticker=ticker, benchmark=benchmark, start=start)
        print(snap)

    @ticker_app.command("outlook")
    def ticker_outlook(
        ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol (e.g., AAPL)"),
        benchmark: str = typer.Option("SPY", "--benchmark", help="Benchmark symbol for relative strength (default SPY)"),
        year: int = typer.Option(2026, "--year", help="Focus year for the outlook (e.g., 2026)"),
        start: str = typer.Option("2011-01-01", "--start", help="Start date YYYY-MM-DD"),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh FRED downloads (for regimes)"),
        llm_model: str = typer.Option("", "--llm-model", help="Override OPENAI_MODEL (optional)"),
        llm_temperature: float = typer.Option(0.2, "--llm-temperature", help="LLM temperature (0..2)"),
    ):
        """
        Ask an LLM for a 3/6/12 month outlook for a ticker, grounded in:
        - ticker quantitative snapshot (Alpaca daily closes)
        - current regimes (macro/liquidity/usd + optional tariff summary)
        """
        settings = load_settings()

        # --- Ticker snapshot ---
        from ai_options_trader.ticker.snapshot import build_ticker_snapshot

        snap = build_ticker_snapshot(settings=settings, ticker=ticker, benchmark=benchmark, start=start)

        # --- Regime context (keep it lightweight / non-leaky) ---
        from ai_options_trader.macro.signals import build_macro_state
        from ai_options_trader.macro.regime import classify_macro_regime_from_state
        from ai_options_trader.funding.signals import build_funding_state
        from ai_options_trader.usd.signals import build_usd_state

        macro_state = build_macro_state(settings=settings, start_date=start, refresh=refresh)
        macro_regime = classify_macro_regime_from_state(
            cpi_yoy=macro_state.inputs.cpi_yoy,
            payrolls_3m_annualized=macro_state.inputs.payrolls_3m_annualized,
            inflation_momentum_minus_be5y=macro_state.inputs.inflation_momentum_minus_be5y,
            real_yield_proxy_10y=macro_state.inputs.real_yield_proxy_10y,
            z_inflation_momentum_minus_be5y=macro_state.inputs.components.get("z_infl_mom_minus_be5y") if macro_state.inputs.components else None,
            z_real_yield_proxy_10y=macro_state.inputs.components.get("z_real_yield_proxy_10y") if macro_state.inputs.components else None,
            use_zscores=True,
            cpi_target=3.0,
            infl_thresh=0.0,
            real_thresh=0.0,
        )
        liq_state = build_funding_state(settings=settings, start_date=start, refresh=refresh)
        usd_state = build_usd_state(settings=settings, start_date=start, refresh=refresh)

        regimes = {
            "macro": {"state": macro_state.model_dump(), "regime": macro_regime.__dict__},
            "liquidity": liq_state.model_dump(),
            "usd": usd_state.model_dump(),
        }

        # --- LLM ---
        from ai_options_trader.llm.outlooks.ticker_outlook import llm_ticker_outlook

        text = llm_ticker_outlook(
            settings=settings,
            ticker_snapshot=snap,
            regimes=regimes,
            year=int(year),
            model=llm_model.strip() or None,
            temperature=float(llm_temperature),
        )
        print(text)

    @ticker_app.command("dossier")
    def ticker_dossier(
        ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol (e.g., AAPL)"),
        days_ahead: int = typer.Option(180, "--days-ahead", help="How far ahead to look for next earnings (days)"),
    ):
        """
        Build a minimal alternative-data dossier for a ticker (start slow):
        - company profile (sector/industry/market cap)
        - next earnings date (if available)
        """
        settings = load_settings()
        from ai_options_trader.altdata.fmp import build_ticker_dossier

        d = build_ticker_dossier(settings=settings, ticker=ticker, days_ahead=int(days_ahead))
        Console().print(Panel(Pretty(d, expand_all=True), title=f"Dossier: {ticker.upper()}", expand=False))

    @ticker_app.command("news")
    def ticker_news(
        ticker: str = typer.Option("", "--ticker", "-t", help="Ticker symbol (e.g., NVDA)"),
        from_options: bool = typer.Option(
            False,
            "--from-options",
            help="If set, pick the top option underlying from current positions (by abs market value).",
        ),
        lookback_days: int = typer.Option(7, "--days", help="Lookback window for news (days)"),
        max_items: int = typer.Option(10, "--max-items", help="Max news items to show"),
        llm: bool = typer.Option(True, "--llm/--no-llm", help="Enable LLM summary"),
        llm_model: str = typer.Option("", "--llm-model", help="Override OPENAI_MODEL (optional)"),
        llm_temperature: float = typer.Option(0.2, "--llm-temperature", help="LLM temperature (0..2)"),
    ):
        """
        Build a basic ticker profile + recent news summary.
        """
        settings = safe_load_settings()
        if not settings:
            raise typer.BadParameter("Settings unavailable (missing env/.env).")
        c = Console()

        if from_options and not ticker:
            try:
                from ai_options_trader.data.alpaca import make_clients

                trading, _data = make_clients(settings)
                positions = trading.get_all_positions()
            except Exception:
                positions = []

            option_syms: list[tuple[float, str]] = []
            for p in positions or []:
                sym = str(getattr(p, "symbol", "") or "").upper()
                if re.search(r"\d{6}[CP]\d{8}$", sym):
                    mv = float(getattr(p, "market_value", 0.0) or 0.0)
                    option_syms.append((abs(mv), sym))
            if option_syms:
                top_sym = sorted(option_syms, key=lambda x: x[0], reverse=True)[0][1]
                underlying = next(iter(extract_underlyings([top_sym])), "")
                ticker = underlying or ticker

        if not ticker:
            raise typer.BadParameter("Provide --ticker or use --from-options.")

        t = ticker.strip().upper()

        # --- Ticker profile (basic) ---
        from ai_options_trader.altdata.fmp import build_ticker_dossier

        dossier = build_ticker_dossier(settings=settings, ticker=t, days_ahead=180)
        prof = dossier.get("profile") or {}
        next_ev = dossier.get("next_earnings") or {}
        prof_lines = [
            f"Ticker: {t}",
            f"Company: {prof.get('company_name') or 'DATA NOT PROVIDED'}",
            f"Sector/Industry: {(prof.get('sector') or '—')} / {(prof.get('industry') or '—')}",
            f"Exchange: {prof.get('exchange') or 'DATA NOT PROVIDED'}",
            f"Market cap: {prof.get('market_cap') or 'DATA NOT PROVIDED'}",
        ]
        if next_ev:
            prof_lines.append(f"Next earnings: {next_ev.get('date')} {next_ev.get('time') or ''}".strip())
        c.print(Panel("\n".join(prof_lines), title=f"Ticker Profile: {t}", expand=False))

        # --- News ---
        from ai_options_trader.llm.outlooks.ticker_news import fetch_fmp_stock_news, llm_recent_news_brief

        now = datetime.now(timezone.utc).date()
        from_date = (now - timedelta(days=int(lookback_days))).isoformat()
        to_date = now.isoformat()

        try:
            items = fetch_fmp_stock_news(
                settings=settings,
                tickers=[t],
                from_date=from_date,
                to_date=to_date,
                max_pages=3,
            )
        except Exception as e:
            items = []
            c.print(Panel(f"News fetch failed: {e}", title="Ticker News", expand=False))

        if items:
            t_news = Table(title=f"Recent News ({lookback_days}d)")
            t_news.add_column("date")
            t_news.add_column("source")
            t_news.add_column("title")
            t_news.add_column("url")
            for it in items[: max(1, int(max_items))]:
                t_news.add_row(
                    str(it.published_at),
                    str(it.source or "—"),
                    str(it.title or ""),
                    str(it.url or "—"),
                )
            c.print(t_news)
        else:
            c.print(Panel("No news items available for this lookback.", title="Ticker News", expand=False))

        # --- LLM summary ---
        if bool(llm):
            try:
                summary = llm_recent_news_brief(
                    settings=settings,
                    ticker=t,
                    items=items,
                    model=llm_model.strip() or None,
                    temperature=float(llm_temperature),
                    lookback_label=f"the last {int(lookback_days)} days",
                )
                c.print(Panel(summary, title="LLM News Brief", expand=False))
            except Exception as e:
                c.print(Panel(f"LLM summary unavailable: {e}", title="LLM News Brief", expand=False))

    @ticker_app.command("filings")
    def ticker_filings(
        ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol (e.g., AAPL)"),
        form_types: str = typer.Option("8-K,10-K,10-Q", "--forms", help="Comma-separated form types"),
        limit: int = typer.Option(15, "--limit", help="Max filings to show"),
    ):
        """
        Show recent SEC filings (8-K, 10-K, 10-Q, etc.) for a ticker.
        
        Uses SEC EDGAR API directly (no API key required).
        """
        settings = safe_load_settings()
        if not settings:
            raise typer.BadParameter("Settings unavailable.")
        c = Console()
        
        from ai_options_trader.altdata.sec import fetch_sec_filings, summarize_filings, categorize_8k_items
        
        t = ticker.strip().upper()
        forms = [f.strip().upper() for f in form_types.split(",") if f.strip()]
        
        filings = fetch_sec_filings(
            settings=settings,
            ticker=t,
            form_types=forms if forms else None,
            limit=int(limit),
        )
        
        if not filings:
            c.print(Panel(f"No SEC filings found for {t}", title="SEC Filings", expand=False))
            return
        
        # Summary
        summary = summarize_filings(filings)
        summary_lines = [
            f"[bold]Total filings:[/bold] {summary['total']}",
            f"[bold]By type:[/bold] {summary['by_type']}",
            f"[bold]Most recent:[/bold] {summary['most_recent_form']} on {summary['most_recent_date']}",
        ]
        
        if summary.get("8k_categories"):
            cats = summary["8k_categories"]
            flags = [k.replace("has_", "").replace("_", " ") for k, v in cats.items() if v]
            if flags:
                summary_lines.append(f"[bold]8-K categories:[/bold] {', '.join(flags)}")
        
        c.print(Panel("\n".join(summary_lines), title=f"SEC Filings Summary: {t}", expand=False))
        
        # Table
        tbl = Table(title=f"Recent Filings ({limit})")
        tbl.add_column("Date", style="cyan")
        tbl.add_column("Form")
        tbl.add_column("Description")
        tbl.add_column("Items")
        tbl.add_column("URL", no_wrap=True)
        
        for f in filings:
            items_str = ", ".join(f.items[:3]) if f.items else "—"
            url_short = f.filing_url[:60] + "..." if len(f.filing_url) > 60 else f.filing_url
            tbl.add_row(
                f.filed_date,
                f.form_type,
                f.description[:40] + "..." if len(f.description) > 40 else f.description or "—",
                items_str,
                url_short or "—",
            )
        
        c.print(tbl)

    @ticker_app.command("earnings")
    def ticker_earnings(
        ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
        history: int = typer.Option(8, "--history", help="Number of historical quarters"),
        transcript: bool = typer.Option(False, "--transcript", help="Show most recent transcript excerpt"),
    ):
        """
        Show earnings history, surprises, and analysis for a ticker.
        """
        settings = safe_load_settings()
        if not settings:
            raise typer.BadParameter("Settings unavailable.")
        c = Console()
        
        t = ticker.strip().upper()
        
        from ai_options_trader.altdata.earnings import (
            fetch_earnings_surprises,
            fetch_recent_transcripts,
            fetch_upcoming_earnings,
            analyze_earnings_history,
            extract_transcript_insights,
        )
        
        # Earnings surprises
        surprises = fetch_earnings_surprises(settings=settings, ticker=t, limit=int(history))
        
        if surprises:
            analysis = analyze_earnings_history(surprises)
            
            c.print(Panel(
                f"[bold]Quarters analyzed:[/bold] {analysis['count']}\n"
                f"[bold]Beat rate:[/bold] {analysis['beat_rate']:.0f}% ({analysis['beats']}/{analysis['beats']+analysis['misses']})\n"
                f"[bold]Avg surprise:[/bold] {analysis['avg_surprise_pct']:+.1f}%\n"
                f"[bold]Current streak:[/bold] {analysis['streak']} {analysis['streak_type'] or 'N/A'}",
                title=f"Earnings Analysis: {t}",
                expand=False,
            ))
            
            tbl = Table(title="Earnings History")
            tbl.add_column("Date", style="cyan")
            tbl.add_column("EPS Actual", justify="right")
            tbl.add_column("EPS Est", justify="right")
            tbl.add_column("Surprise", justify="right")
            tbl.add_column("Surprise %", justify="right")
            
            for s in surprises[:8]:
                surprise_style = "green" if (s.eps_surprise or 0) > 0 else "red" if (s.eps_surprise or 0) < 0 else ""
                tbl.add_row(
                    s.date,
                    f"${s.eps_actual:.2f}" if s.eps_actual is not None else "—",
                    f"${s.eps_estimated:.2f}" if s.eps_estimated is not None else "—",
                    f"[{surprise_style}]${s.eps_surprise:+.2f}[/{surprise_style}]" if s.eps_surprise is not None else "—",
                    f"[{surprise_style}]{s.eps_surprise_pct:+.1f}%[/{surprise_style}]" if s.eps_surprise_pct is not None else "—",
                )
            
            c.print(tbl)
        else:
            c.print(Panel(f"No earnings data found for {t}", title="Earnings", expand=False))
        
        # Upcoming earnings
        upcoming = fetch_upcoming_earnings(settings=settings, tickers=[t], days_ahead=90)
        if upcoming:
            next_event = upcoming[0]
            c.print(Panel(
                f"[bold]Date:[/bold] {next_event.date}\n"
                f"[bold]Time:[/bold] {next_event.time or 'TBD'}\n"
                f"[bold]EPS Est:[/bold] ${next_event.eps_estimated:.2f}" if next_event.eps_estimated else "",
                title=f"Next Earnings: {t}",
                expand=False,
            ))
        
        # Transcript
        if transcript:
            transcripts = fetch_recent_transcripts(settings=settings, ticker=t, num_quarters=1)
            if transcripts:
                tr = transcripts[0]
                insights = extract_transcript_insights(tr)
                
                c.print(Panel(
                    f"[bold]Quarter:[/bold] {insights['quarter']}\n"
                    f"[bold]Date:[/bold] {insights['date']}\n"
                    f"[bold]Length:[/bold] {insights['length']:,} chars\n\n"
                    f"[bold]Key mentions:[/bold]\n"
                    + "\n".join([f"  {k}: {v}" for k, v in insights['mentions'].items() if v > 0])
                    + f"\n\n[dim]{insights['preview']}[/dim]",
                    title=f"Latest Transcript: {t}",
                    expand=False,
                ))
            else:
                c.print(Panel(f"No transcripts available for {t}", title="Transcript", expand=False))

    @ticker_app.command("deep")
    def ticker_deep(
        ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
        llm: bool = typer.Option(True, "--llm/--no-llm", help="Include LLM analysis"),
        llm_model: str = typer.Option("", "--llm-model", help="Override LLM model"),
    ):
        """
        Deep dive on a ticker: profile, filings, earnings, news, sentiment.
        
        This is the comprehensive research command that pulls together all
        available data sources for a single ticker.
        """
        settings = safe_load_settings()
        if not settings:
            raise typer.BadParameter("Settings unavailable.")
        c = Console()
        
        t = ticker.strip().upper()
        c.print(f"\n[bold cyan]Deep dive: {t}[/bold cyan]\n")
        
        # 1. Profile
        from ai_options_trader.altdata.fmp import fetch_profile
        
        profile = fetch_profile(settings=settings, ticker=t)
        if profile:
            c.print(Panel(
                f"[bold]Company:[/bold] {profile.company_name or t}\n"
                f"[bold]Sector:[/bold] {profile.sector or '—'} / {profile.industry or '—'}\n"
                f"[bold]Exchange:[/bold] {profile.exchange or '—'}\n"
                f"[bold]Market Cap:[/bold] {_fmt_market_cap(profile.market_cap)}\n"
                f"[dim]{(profile.description or '')[:300]}...[/dim]" if profile.description and len(profile.description) > 300 else f"[dim]{profile.description or ''}[/dim]",
                title="Company Profile",
                expand=False,
            ))
        
        # 2. Quantitative snapshot
        try:
            from ai_options_trader.ticker.snapshot import build_ticker_snapshot
            snap = build_ticker_snapshot(settings=settings, ticker=t, benchmark="SPY", start="2020-01-01")
            c.print(Panel(str(snap), title="Quantitative Snapshot", expand=False))
        except Exception as e:
            c.print(Panel(f"Snapshot unavailable: {e}", title="Quantitative Snapshot", expand=False))
        
        # 3. SEC Filings summary
        from ai_options_trader.altdata.sec import fetch_8k_filings, fetch_annual_quarterly_reports, summarize_filings
        
        filings_8k = fetch_8k_filings(settings=settings, ticker=t, limit=5)
        filings_periodic = fetch_annual_quarterly_reports(settings=settings, ticker=t, limit=4)
        all_filings = sorted(filings_8k + filings_periodic, key=lambda x: x.filed_date, reverse=True)
        
        if all_filings:
            summary = summarize_filings(all_filings)
            filing_lines = [f"[bold]Recent filings:[/bold] {summary['total']} ({summary['by_type']})"]
            
            for f in all_filings[:5]:
                items_str = f" ({', '.join(f.items[:2])})" if f.items else ""
                filing_lines.append(f"  • {f.filed_date}: {f.form_type}{items_str}")
            
            c.print(Panel("\n".join(filing_lines), title="SEC Filings", expand=False))
        
        # 4. Earnings
        from ai_options_trader.altdata.earnings import fetch_earnings_surprises, fetch_upcoming_earnings, analyze_earnings_history
        
        surprises = fetch_earnings_surprises(settings=settings, ticker=t, limit=4)
        upcoming = fetch_upcoming_earnings(settings=settings, tickers=[t], days_ahead=90)
        
        earnings_lines = []
        if surprises:
            analysis = analyze_earnings_history(surprises)
            earnings_lines.append(f"[bold]Beat rate:[/bold] {analysis['beat_rate']:.0f}% (last {analysis['count']} quarters)")
            earnings_lines.append(f"[bold]Avg surprise:[/bold] {analysis['avg_surprise_pct']:+.1f}%")
            if analysis['streak'] > 1:
                earnings_lines.append(f"[bold]Streak:[/bold] {analysis['streak']} consecutive {analysis['streak_type']}s")
        
        if upcoming:
            next_ev = upcoming[0]
            earnings_lines.append(f"[bold]Next earnings:[/bold] {next_ev.date} {next_ev.time or ''}")
        
        if earnings_lines:
            c.print(Panel("\n".join(earnings_lines), title="Earnings", expand=False))
        
        # 5. News with sentiment
        from ai_options_trader.llm.outlooks.ticker_news import fetch_fmp_stock_news
        from ai_options_trader.llm.core.sentiment import analyze_article_sentiment, aggregate_sentiment
        
        now = datetime.now(timezone.utc).date()
        from_date = (now - timedelta(days=14)).isoformat()
        
        try:
            news_items = fetch_fmp_stock_news(
                settings=settings,
                tickers=[t],
                from_date=from_date,
                to_date=now.isoformat(),
                max_pages=2,
            )
            
            if news_items:
                # Analyze sentiment
                article_sentiments = [
                    analyze_article_sentiment(
                        headline=item.title or "",
                        content=item.snippet or "",
                        source=item.source or "",
                        url=item.url or "",
                        published_at=str(item.published_at or ""),
                    )
                    for item in news_items[:10]
                ]
                
                agg = aggregate_sentiment(article_sentiments)
                
                sentiment_color = "green" if agg.label == "positive" else "red" if agg.label == "negative" else "yellow"
                
                news_lines = [
                    f"[bold]Aggregate sentiment:[/bold] [{sentiment_color}]{agg.label.upper()}[/{sentiment_color}] (score: {agg.score:+.2f})",
                    f"[bold]Articles:[/bold] {agg.positive_count} positive, {agg.negative_count} negative, {agg.neutral_count} neutral",
                    "",
                    "[bold]Recent headlines:[/bold]",
                ]
                
                for item in news_items[:5]:
                    sent = analyze_article_sentiment(item.title or "")
                    sent_icon = "🟢" if sent.sentiment.label == "positive" else "🔴" if sent.sentiment.label == "negative" else "⚪"
                    news_lines.append(f"  {sent_icon} {item.title[:70]}...")
                
                c.print(Panel("\n".join(news_lines), title="News & Sentiment (14d)", expand=False))
        except Exception as e:
            c.print(Panel(f"News unavailable: {e}", title="News", expand=False))
        
        # 6. LLM Analysis
        if llm and settings.openai_api_key:
            c.print("\n[bold cyan]Generating LLM analysis...[/bold cyan]\n")
            
            try:
                # Build context
                context = {
                    "ticker": t,
                    "profile": profile.__dict__ if profile else None,
                    "recent_filings": [
                        {"date": f.filed_date, "form": f.form_type, "items": f.items}
                        for f in all_filings[:5]
                    ] if all_filings else [],
                    "earnings_analysis": analyze_earnings_history(surprises) if surprises else None,
                    "next_earnings": {"date": upcoming[0].date, "time": upcoming[0].time} if upcoming else None,
                    "news_sentiment": {
                        "label": agg.label,
                        "score": agg.score,
                        "positive": agg.positive_count,
                        "negative": agg.negative_count,
                    } if 'agg' in dir() else None,
                }
                
                from openai import OpenAI
                import json
                
                client = OpenAI(api_key=settings.openai_api_key)
                model = llm_model.strip() or settings.openai_model or "gpt-4o-mini"
                
                prompt = f"""You are a senior equity research analyst. Provide a concise investment analysis for {t}.

Context data:
{json.dumps(context, indent=2, default=str)}

Provide:
1. **Bull Case** (2-3 sentences): Key positives and upside catalysts
2. **Bear Case** (2-3 sentences): Key risks and concerns
3. **Near-term View** (1-2 sentences): What to watch in next 30-60 days
4. **Sentiment Summary**: How does recent news/filings affect the thesis?

Be specific and reference the data provided. Keep total response under 300 words."""

                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500,
                )
                
                analysis = resp.choices[0].message.content or ""
                c.print(Panel(Markdown(analysis), title="LLM Analysis", expand=False))
                
            except Exception as e:
                c.print(Panel(f"LLM analysis unavailable: {e}", title="LLM Analysis", expand=False))
        
        c.print(f"\n[dim]Deep dive complete for {t}[/dim]\n")



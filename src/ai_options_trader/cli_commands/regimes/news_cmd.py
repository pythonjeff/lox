"""
News Sentiment Regime CLI Command

Tracks aggregate market news sentiment and classifies it into actionable regimes.

Author: Lox Capital Research
"""
from __future__ import annotations

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def _fmt_score(val: float | None, decimals: int = 2) -> str:
    """Format a score for display."""
    if val is None:
        return "—"
    return f"{val:+.{decimals}f}"


def _fmt_pct(val: float | None) -> str:
    """Format a percentage."""
    if val is None:
        return "—"
    return f"{val:.0f}%"


def _sentiment_color(score: float | None) -> str:
    """Get color based on sentiment score."""
    if score is None:
        return "white"
    if score >= 0.3:
        return "green"
    if score >= 0.1:
        return "bright_green"
    if score <= -0.3:
        return "red"
    if score <= -0.1:
        return "bright_red"
    return "yellow"


def _run_news_snapshot(
    lookback: int = 7,
    refresh: bool = False,
    llm: bool = False,
    features: bool = False,
    json_output: bool = False,
):
    """Shared implementation for news sentiment snapshot."""
    from ai_options_trader.config import load_settings
    from ai_options_trader.news.signals import build_news_sentiment_state
    from ai_options_trader.news.regime import classify_news_regime
    
    settings = load_settings()
    state = build_news_sentiment_state(
        settings=settings,
        lookback_days=lookback,
        refresh=refresh,
    )
    regime = classify_news_regime(state.inputs)
    inp = state.inputs
    
    # JSON output
    if json_output:
        import json
        output = {
            "asof": state.asof,
            "lookback_days": state.lookback_days,
            "regime": {
                "name": regime.name,
                "label": regime.label,
                "description": regime.description,
                "tags": list(regime.tags),
                "market_implications": regime.market_implications,
                "contrarian_signal": regime.contrarian_signal,
            },
            "metrics": {
                "market_sentiment_score": inp.market_sentiment_score,
                "fear_greed_score": inp.fear_greed_score,
                "total_articles": inp.total_articles,
                "positive_articles": inp.positive_articles,
                "negative_articles": inp.negative_articles,
                "neutral_articles": inp.neutral_articles,
                "best_sector": inp.best_sector,
                "worst_sector": inp.worst_sector,
                "dominant_theme": inp.dominant_theme,
            },
        }
        print(json.dumps(output, indent=2, default=str))
        return
    
    # Features output
    if features:
        from ai_options_trader.news.features import news_feature_vector
        vec = news_feature_vector(state, regime)
        
        tbl = Table(title="News Sentiment Features")
        tbl.add_column("Feature", style="cyan")
        tbl.add_column("Value", justify="right")
        
        for k, v in sorted(vec.features.items()):
            tbl.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        
        print(tbl)
        return
    
    c = Console()
    
    # Regime header
    regime_color = _sentiment_color(inp.market_sentiment_score)
    c.print(Panel(
        f"[bold]Regime:[/bold] [{regime_color}]{regime.label}[/{regime_color}]\n"
        f"[bold]As of:[/bold] {state.asof} ({state.lookback_days}d lookback)\n\n"
        f"[dim]{regime.description}[/dim]",
        title="News Sentiment Regime",
        expand=False,
    ))
    
    # Market implications
    if regime.market_implications:
        c.print(Panel(
            f"{regime.market_implications}\n\n"
            f"[bold cyan]Contrarian view:[/bold cyan] {regime.contrarian_signal}",
            title="Market Implications",
            expand=False,
        ))
    
    # Main metrics
    score_color = _sentiment_color(inp.market_sentiment_score)
    fg_color = "green" if (inp.fear_greed_score or 50) > 60 else "red" if (inp.fear_greed_score or 50) < 40 else "yellow"
    
    c.print(Panel(
        f"[bold]Market Sentiment Score:[/bold] [{score_color}]{_fmt_score(inp.market_sentiment_score)}[/{score_color}] (−1 to +1)\n"
        f"[bold]Fear/Greed Score:[/bold] [{fg_color}]{_fmt_pct(inp.fear_greed_score)}[/{fg_color}] (0=Fear, 100=Greed)\n"
        f"[bold]Risk Appetite:[/bold] [{score_color}]{_fmt_score(inp.risk_appetite_score)}[/{score_color}]\n\n"
        f"[bold]Articles Analyzed:[/bold] {inp.total_articles}\n"
        f"  🟢 Positive: {inp.positive_articles} ({inp.high_conviction_positive} high conviction)\n"
        f"  🔴 Negative: {inp.negative_articles} ({inp.high_conviction_negative} high conviction)\n"
        f"  ⚪ Neutral: {inp.neutral_articles}",
        title="Sentiment Metrics",
        expand=False,
    ))
    
    # Sector sentiment table
    sectors = [
        ("Tech", inp.tech_sentiment_score),
        ("Financials", inp.financials_sentiment_score),
        ("Energy", inp.energy_sentiment_score),
        ("Healthcare", inp.healthcare_sentiment_score),
        ("Consumer", inp.consumer_sentiment_score),
        ("Industrials", inp.industrials_sentiment_score),
    ]
    
    tbl = Table(title="Sector Sentiment")
    tbl.add_column("Sector")
    tbl.add_column("Score", justify="right")
    tbl.add_column("Status")
    
    for sector, score in sectors:
        color = _sentiment_color(score)
        status = "Best" if sector.lower() == (inp.best_sector or "").lower() else "Worst" if sector.lower() == (inp.worst_sector or "").lower() else ""
        status_style = "bold green" if status == "Best" else "bold red" if status == "Worst" else ""
        tbl.add_row(
            sector,
            f"[{color}]{_fmt_score(score)}[/{color}]",
            f"[{status_style}]{status}[/{status_style}]",
        )
    
    c.print(tbl)
    
    # Sector dispersion
    if inp.sector_dispersion is not None:
        disp_label = "High (diverging)" if inp.sector_dispersion > 0.4 else "Low (consensus)" if inp.sector_dispersion < 0.2 else "Moderate"
        c.print(f"[bold]Sector Dispersion:[/bold] {inp.sector_dispersion:.2f} ({disp_label})")
    
    # Theme mentions
    themes = [
        ("Earnings", inp.earnings_mentions),
        ("Fed/Monetary", inp.fed_mentions),
        ("Tariffs", inp.tariff_mentions),
        ("Recession", inp.recession_mentions),
        ("AI", inp.ai_mentions),
        ("Layoffs", inp.layoffs_mentions),
    ]
    
    theme_lines = [f"[bold]Dominant theme:[/bold] {inp.dominant_theme or 'None'}"]
    for name, count in themes:
        if count > 0:
            theme_lines.append(f"  {name}: {count} mentions")
    
    c.print(Panel("\n".join(theme_lines), title="Theme Analysis", expand=False))
    
    # LLM analysis
    if llm and settings.openai_api_key:
        c.print("\n[bold cyan]Generating LLM analysis...[/bold cyan]\n")
        
        try:
            from ai_options_trader.llm.core.analyst import llm_analyze_regime
            
            snapshot_data = {
                "market_sentiment_score": inp.market_sentiment_score,
                "fear_greed_score": inp.fear_greed_score,
                "positive_articles": inp.positive_articles,
                "negative_articles": inp.negative_articles,
                "best_sector": inp.best_sector,
                "worst_sector": inp.worst_sector,
                "dominant_theme": inp.dominant_theme,
                "sector_dispersion": inp.sector_dispersion,
                "earnings_mentions": inp.earnings_mentions,
                "fed_mentions": inp.fed_mentions,
                "recession_mentions": inp.recession_mentions,
            }
            
            analysis = llm_analyze_regime(
                settings=settings,
                domain="news_sentiment",
                snapshot=snapshot_data,
                regime_label=regime.label,
                regime_description=regime.description,
            )
            
            from rich.markdown import Markdown
            c.print(Panel(Markdown(analysis), title="LLM Analysis", expand=False))
            
        except Exception as e:
            c.print(Panel(f"LLM analysis unavailable: {e}", title="LLM Analysis", expand=False))


def register(news_app: typer.Typer) -> None:
    """Register news sentiment regime commands."""
    
    @news_app.callback(invoke_without_command=True)
    def news_default(
        ctx: typer.Context,
        lookback: int = typer.Option(7, "--lookback", "-d", help="Days of news to analyze"),
        llm: bool = typer.Option(False, "--llm", help="Get LLM analysis"),
        features: bool = typer.Option(False, "--features", help="Output ML feature vector"),
        json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    ):
        """
        News sentiment regime: aggregate market news sentiment.
        
        Regimes:
        - NEWS_EUPHORIA: Extreme positive (contrarian sell signal)
        - NEWS_BULLISH: Strong positive sentiment
        - NEWS_NEUTRAL: Mixed or balanced
        - NEWS_CAUTIOUS: Moderately negative
        - NEWS_FEARFUL: Extreme negative (contrarian buy signal)
        """
        if ctx.invoked_subcommand is None:
            _run_news_snapshot(
                lookback=lookback,
                llm=llm,
                features=features,
                json_output=json_output,
            )
    
    @news_app.command("snapshot")
    def snapshot(
        lookback: int = typer.Option(7, "--lookback", "-d", help="Days of news to analyze"),
        refresh: bool = typer.Option(False, "--refresh", help="Force refresh cached data"),
        llm: bool = typer.Option(False, "--llm", help="Get LLM analysis"),
        features: bool = typer.Option(False, "--features", help="Output ML feature vector"),
        json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    ):
        """
        News sentiment snapshot.
        
        Analyzes recent market news across sectors and themes to determine
        aggregate sentiment and regime classification.
        """
        _run_news_snapshot(
            lookback=lookback,
            refresh=refresh,
            llm=llm,
            features=features,
            json_output=json_output,
        )
    
    @news_app.command("sectors")
    def sector_sentiment(
        lookback: int = typer.Option(7, "--lookback", "-d", help="Days of news to analyze"),
    ):
        """
        Show sector-by-sector news sentiment breakdown.
        """
        from ai_options_trader.config import load_settings
        from ai_options_trader.news.signals import build_news_sentiment_state, SECTOR_TICKERS
        
        settings = load_settings()
        state = build_news_sentiment_state(settings=settings, lookback_days=lookback)
        inp = state.inputs
        
        c = Console()
        
        tbl = Table(title=f"Sector News Sentiment ({lookback}d)")
        tbl.add_column("Sector")
        tbl.add_column("Tickers")
        tbl.add_column("Sentiment", justify="right")
        tbl.add_column("Rank", justify="center")
        
        sectors = [
            ("Tech", inp.tech_sentiment_score, SECTOR_TICKERS.get("tech", [])),
            ("Financials", inp.financials_sentiment_score, SECTOR_TICKERS.get("financials", [])),
            ("Energy", inp.energy_sentiment_score, SECTOR_TICKERS.get("energy", [])),
            ("Healthcare", inp.healthcare_sentiment_score, SECTOR_TICKERS.get("healthcare", [])),
            ("Consumer", inp.consumer_sentiment_score, SECTOR_TICKERS.get("consumer", [])),
            ("Industrials", inp.industrials_sentiment_score, SECTOR_TICKERS.get("industrials", [])),
        ]
        
        # Sort by sentiment
        sectors_sorted = sorted(
            [(s, score, t) for s, score, t in sectors if score is not None],
            key=lambda x: x[1],
            reverse=True,
        )
        
        for rank, (sector, score, tickers) in enumerate(sectors_sorted, 1):
            color = _sentiment_color(score)
            rank_str = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else str(rank)
            tbl.add_row(
                sector,
                ", ".join(tickers[:4]) + "...",
                f"[{color}]{_fmt_score(score)}[/{color}]",
                rank_str,
            )
        
        c.print(tbl)
        
        # Best/worst summary
        if sectors_sorted:
            best = sectors_sorted[0]
            worst = sectors_sorted[-1]
            c.print(Panel(
                f"[bold green]Best:[/bold green] {best[0]} ({_fmt_score(best[1])})\n"
                f"[bold red]Worst:[/bold red] {worst[0]} ({_fmt_score(worst[1])})\n"
                f"[bold]Dispersion:[/bold] {inp.sector_dispersion:.2f}" if inp.sector_dispersion else "",
                title="Sector Summary",
                expand=False,
            ))
    
    @news_app.command("themes")
    def theme_analysis(
        lookback: int = typer.Option(7, "--lookback", "-d", help="Days of news to analyze"),
    ):
        """
        Show theme/topic mentions in recent news.
        """
        from ai_options_trader.config import load_settings
        from ai_options_trader.news.signals import build_news_sentiment_state, THEME_KEYWORDS
        
        settings = load_settings()
        state = build_news_sentiment_state(settings=settings, lookback_days=lookback)
        inp = state.inputs
        
        c = Console()
        
        themes = [
            ("Earnings", inp.earnings_mentions, THEME_KEYWORDS.get("earnings", [])),
            ("Fed/Monetary Policy", inp.fed_mentions, THEME_KEYWORDS.get("fed", [])),
            ("Tariffs/Trade", inp.tariff_mentions, THEME_KEYWORDS.get("tariff", [])),
            ("Recession", inp.recession_mentions, THEME_KEYWORDS.get("recession", [])),
            ("AI/Technology", inp.ai_mentions, THEME_KEYWORDS.get("ai", [])),
            ("Layoffs", inp.layoffs_mentions, THEME_KEYWORDS.get("layoffs", [])),
        ]
        
        tbl = Table(title=f"Theme Mentions ({lookback}d)")
        tbl.add_column("Theme")
        tbl.add_column("Mentions", justify="right")
        tbl.add_column("Keywords")
        
        # Sort by mentions
        themes_sorted = sorted(themes, key=lambda x: x[1], reverse=True)
        
        for theme, count, keywords in themes_sorted:
            style = "bold" if count > 5 else ""
            tbl.add_row(
                f"[{style}]{theme}[/{style}]",
                str(count),
                ", ".join(keywords[:4]) + "...",
            )
        
        c.print(tbl)
        
        if inp.dominant_theme:
            c.print(f"\n[bold]Dominant theme:[/bold] {inp.dominant_theme}")

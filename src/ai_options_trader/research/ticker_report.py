"""
Ticker Research Report - Comprehensive uniform reports for stocks and ETFs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd

from ai_options_trader.config import Settings
from ai_options_trader.research.momentum import MomentumMetrics, calculate_momentum_metrics
from ai_options_trader.research.hf_metrics import HedgeFundMetrics, calculate_hf_metrics


@dataclass
class CompanyFundamentals:
    """Fundamental data for stocks."""
    name: str
    sector: str
    industry: str
    market_cap_b: float
    enterprise_value_b: float
    
    # Valuation
    pe_ratio: float
    forward_pe: float
    ps_ratio: float
    pb_ratio: float
    ev_ebitda: float
    peg_ratio: float
    
    # Profitability
    gross_margin: float
    operating_margin: float
    net_margin: float
    roe: float
    roa: float
    roic: float
    
    # Growth
    revenue_growth_yoy: float
    earnings_growth_yoy: float
    revenue_growth_3y_cagr: float
    
    # Balance sheet
    debt_to_equity: float
    current_ratio: float
    quick_ratio: float
    cash_b: float
    total_debt_b: float
    
    # Dividends
    dividend_yield: float
    payout_ratio: float
    
    # Earnings
    eps_ttm: float
    next_earnings_date: str
    earnings_surprise_last: float


@dataclass
class ETFFundamentals:
    """Fundamental data for ETFs."""
    name: str
    category: str
    expense_ratio: float
    aum_b: float
    inception_date: str
    holdings_count: int
    
    # Top holdings
    top_holdings: list[dict]  # [{ticker, name, weight_pct}]
    top_10_weight: float
    
    # Sector/country exposure
    sector_weights: dict[str, float]
    country_weights: dict[str, float]


@dataclass
class SECFilingSummary:
    """Summary of SEC filings."""
    filings_30d: int
    filings_90d: int
    recent_8k_items: list[str]
    most_recent_form: str
    most_recent_date: str
    has_insider_selling: bool
    has_insider_buying: bool
    insider_net_30d: float  # Net insider transactions in $M
    notable_filings: list[dict]  # [{date, form, description}]


@dataclass
class AnalystData:
    """Analyst ratings and price targets."""
    consensus_rating: str  # Strong Buy / Buy / Hold / Sell / Strong Sell
    num_analysts: int
    target_price_mean: float
    target_price_high: float
    target_price_low: float
    upside_to_target: float
    rating_changes_30d: list[dict]


@dataclass
class NewsAnalysis:
    """Categorized news analysis."""
    total_articles_30d: int
    sentiment_score: float  # -1 to +1
    sentiment_label: str  # Positive/Neutral/Negative
    key_themes: list[str]
    recent_headlines: list[dict]  # [{date, title, sentiment}]


@dataclass
class TickerResearchReport:
    """Complete research report for a ticker."""
    ticker: str
    asset_type: str  # "stock" or "etf"
    generated_at: str
    
    # Core data
    current_price: float
    momentum: MomentumMetrics
    hf_metrics: HedgeFundMetrics
    
    # Fundamentals (varies by type)
    fundamentals: CompanyFundamentals | ETFFundamentals | None
    
    # Additional research
    sec_filings: SECFilingSummary | None
    analyst_data: AnalystData | None
    news_analysis: NewsAnalysis | None
    
    # Summary scores
    overall_score: int  # 0-100 composite
    key_strengths: list[str]
    key_risks: list[str]
    
    # LLM analysis (if requested)
    llm_summary: str | None = None


def build_ticker_research_report(
    settings: Settings,
    ticker: str,
    include_llm: bool = False,
) -> TickerResearchReport:
    """
    Build a comprehensive research report for any ticker.
    
    Provides uniform output format for both stocks and ETFs.
    """
    import requests
    
    t = ticker.strip().upper()
    is_etf = _is_etf(t)
    
    # Fetch price data
    from ai_options_trader.data.market import fetch_equity_daily_closes
    
    try:
        px_df = fetch_equity_daily_closes(
            settings=settings,
            symbols=[t, "SPY", "QQQ"],
            start="2023-01-01",
            refresh=False,
        )
        prices = px_df[t].dropna() if t in px_df.columns else pd.Series()
        spy_prices = px_df["SPY"].dropna() if "SPY" in px_df.columns else None
        qqq_prices = px_df["QQQ"].dropna() if "QQQ" in px_df.columns else None
    except Exception:
        prices = pd.Series()
        spy_prices = None
        qqq_prices = None
    
    current_price = float(prices.iloc[-1]) if len(prices) > 0 else 0
    
    # Calculate momentum metrics
    momentum = calculate_momentum_metrics(prices, t)
    
    # Calculate HF metrics
    hf_metrics = calculate_hf_metrics(
        prices, t, spy_prices=spy_prices, qqq_prices=qqq_prices
    )
    
    # Fetch fundamentals
    if is_etf:
        fundamentals = _fetch_etf_fundamentals(settings, t)
    else:
        fundamentals = _fetch_stock_fundamentals(settings, t)
    
    # Fetch SEC filings
    sec_filings = _fetch_sec_summary(settings, t)
    
    # Fetch analyst data
    analyst_data = _fetch_analyst_data(settings, t) if not is_etf else None
    
    # Fetch news analysis
    news_analysis = _fetch_news_analysis(settings, t)
    
    # Calculate overall score and identify strengths/risks
    overall_score, strengths, risks = _calculate_overall_score(
        momentum, hf_metrics, fundamentals, sec_filings, analyst_data, is_etf
    )
    
    # LLM summary if requested
    llm_summary = None
    if include_llm:
        llm_summary = _generate_llm_summary(
            settings, t, is_etf, momentum, hf_metrics, fundamentals, 
            sec_filings, analyst_data, news_analysis
        )
    
    return TickerResearchReport(
        ticker=t,
        asset_type="etf" if is_etf else "stock",
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        current_price=current_price,
        momentum=momentum,
        hf_metrics=hf_metrics,
        fundamentals=fundamentals,
        sec_filings=sec_filings,
        analyst_data=analyst_data,
        news_analysis=news_analysis,
        overall_score=overall_score,
        key_strengths=strengths,
        key_risks=risks,
        llm_summary=llm_summary,
    )


def _is_etf(ticker: str) -> bool:
    """Detect if ticker is an ETF."""
    known_etfs = {
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VXX", "VIXY", "UVXY",
        "XLF", "XLE", "XLK", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE",
        "GLD", "SLV", "USO", "UNG", "DBC", "GLDM", "IAU",
        "TLT", "IEF", "SHY", "HYG", "LQD", "JNK", "TBF", "TBT", "BND", "AGG",
        "EEM", "EFA", "VWO", "FXI", "MCHI", "KWEB", "EWJ", "EWZ",
        "SMH", "SOXX", "XRT", "XHB", "ITB", "IYT", "XAR", "XBI", "IBB",
        "ARKK", "ARKG", "ARKW", "ARKF", "ARKQ",
        "SQQQ", "TQQQ", "SPXU", "UPRO", "PSQ", "SH",
        "SLX", "MOO", "DBA", "CORN", "SOYB", "PAVE", "TAN", "ICLN", "QCLN",
        "IBIT", "BITO", "VIXM", "VXZ",
    }
    return ticker.upper() in known_etfs


def _fetch_stock_fundamentals(settings: Settings, ticker: str) -> CompanyFundamentals | None:
    """Fetch fundamental data for a stock."""
    import requests
    
    base_url = "https://financialmodelingprep.com/api/v3"
    
    try:
        # Profile
        resp = requests.get(
            f"{base_url}/profile/{ticker}",
            params={"apikey": settings.fmp_api_key},
            timeout=10,
        )
        profile = resp.json()[0] if resp.ok and resp.json() else {}
        
        # Key metrics
        resp = requests.get(
            f"{base_url}/key-metrics-ttm/{ticker}",
            params={"apikey": settings.fmp_api_key},
            timeout=10,
        )
        metrics = resp.json()[0] if resp.ok and resp.json() else {}
        
        # Ratios
        resp = requests.get(
            f"{base_url}/ratios-ttm/{ticker}",
            params={"apikey": settings.fmp_api_key},
            timeout=10,
        )
        ratios = resp.json()[0] if resp.ok and resp.json() else {}
        
        # Growth
        resp = requests.get(
            f"{base_url}/financial-growth/{ticker}",
            params={"apikey": settings.fmp_api_key, "limit": 1},
            timeout=10,
        )
        growth = resp.json()[0] if resp.ok and resp.json() else {}
        
        def _f(x):
            try:
                return float(x) if x is not None else 0
            except (ValueError, TypeError):
                return 0
        
        return CompanyFundamentals(
            name=str(profile.get("companyName", ticker)),
            sector=str(profile.get("sector", "Unknown")),
            industry=str(profile.get("industry", "Unknown")),
            market_cap_b=_f(profile.get("mktCap", 0)) / 1e9,
            enterprise_value_b=_f(metrics.get("enterpriseValueTTM", 0)) / 1e9,
            pe_ratio=_f(ratios.get("priceEarningsRatioTTM", 0)),
            forward_pe=_f(profile.get("forwardPE", 0)),
            ps_ratio=_f(ratios.get("priceToSalesRatioTTM", 0)),
            pb_ratio=_f(ratios.get("priceBookValueRatioTTM", 0)),
            ev_ebitda=_f(ratios.get("enterpriseValueMultipleTTM", 0)),
            peg_ratio=_f(ratios.get("pegRatioTTM", 0)),
            gross_margin=_f(ratios.get("grossProfitMarginTTM", 0)),
            operating_margin=_f(ratios.get("operatingProfitMarginTTM", 0)),
            net_margin=_f(ratios.get("netProfitMarginTTM", 0)),
            roe=_f(ratios.get("returnOnEquityTTM", 0)),
            roa=_f(ratios.get("returnOnAssetsTTM", 0)),
            roic=_f(ratios.get("returnOnCapitalEmployedTTM", 0)),
            revenue_growth_yoy=_f(growth.get("revenueGrowth", 0)),
            earnings_growth_yoy=_f(growth.get("epsgrowth", 0)),
            revenue_growth_3y_cagr=_f(growth.get("threeYRevenueGrowthPerShare", 0)),
            debt_to_equity=_f(ratios.get("debtEquityRatioTTM", 0)),
            current_ratio=_f(ratios.get("currentRatioTTM", 0)),
            quick_ratio=_f(ratios.get("quickRatioTTM", 0)),
            cash_b=_f(metrics.get("cashPerShareTTM", 0)) * _f(profile.get("shareOutstanding", 0)) / 1e9,
            total_debt_b=_f(metrics.get("debtToAssetsTTM", 0)) * _f(profile.get("mktCap", 0)) / 1e9 / max(_f(ratios.get("debtEquityRatioTTM", 1)), 0.01),
            dividend_yield=_f(ratios.get("dividendYielTTM", 0)),
            payout_ratio=_f(ratios.get("payoutRatioTTM", 0)),
            eps_ttm=_f(metrics.get("earningsYieldTTM", 0)) * _f(profile.get("price", 0)),
            next_earnings_date="TBD",
            earnings_surprise_last=0,
        )
    except Exception:
        return None


def _fetch_etf_fundamentals(settings: Settings, ticker: str) -> ETFFundamentals | None:
    """Fetch fundamental data for an ETF."""
    import requests
    
    base_url = "https://financialmodelingprep.com/api/v3"
    
    try:
        # ETF info
        resp = requests.get(
            f"{base_url}/etf-info",
            params={"apikey": settings.fmp_api_key, "symbol": ticker},
            timeout=10,
        )
        info = resp.json()[0] if resp.ok and resp.json() else {}
        
        # Holdings
        resp = requests.get(
            f"{base_url}/etf-holder/{ticker}",
            params={"apikey": settings.fmp_api_key},
            timeout=10,
        )
        holdings = resp.json()[:20] if resp.ok else []
        
        # Sector weights
        resp = requests.get(
            f"{base_url}/etf-sector-weightings/{ticker}",
            params={"apikey": settings.fmp_api_key},
            timeout=10,
        )
        sectors = resp.json() if resp.ok else []
        
        # Country weights
        resp = requests.get(
            f"{base_url}/etf-country-weightings/{ticker}",
            params={"apikey": settings.fmp_api_key},
            timeout=10,
        )
        countries = resp.json() if resp.ok else []
        
        top_holdings = []
        for h in holdings[:10]:
            top_holdings.append({
                "ticker": h.get("asset", ""),
                "name": h.get("name", ""),
                "weight_pct": h.get("weightPercentage", 0),
            })
        
        top_10_weight = sum(h["weight_pct"] for h in top_holdings)
        
        sector_weights = {}
        for s in sectors:
            sector_weights[s.get("sector", "Other")] = s.get("weightPercentage", 0)
        
        country_weights = {}
        for c in countries:
            country_weights[c.get("country", "Other")] = c.get("weightPercentage", 0)
        
        return ETFFundamentals(
            name=str(info.get("name", ticker)),
            category=str(info.get("assetClass", "Unknown")),
            expense_ratio=float(info.get("expenseRatio", 0) or 0),
            aum_b=float(info.get("totalAssets", 0) or 0) / 1e9,
            inception_date=str(info.get("inceptionDate", "Unknown")),
            holdings_count=len(holdings),
            top_holdings=top_holdings,
            top_10_weight=top_10_weight,
            sector_weights=sector_weights,
            country_weights=country_weights,
        )
    except Exception:
        return None


def _fetch_sec_summary(settings: Settings, ticker: str) -> SECFilingSummary | None:
    """Fetch SEC filing summary."""
    try:
        from ai_options_trader.altdata.sec import (
            fetch_sec_filings,
            fetch_insider_filings,
            summarize_filings,
        )
        
        # All filings (90 days)
        all_filings = fetch_sec_filings(
            settings=settings,
            ticker=ticker,
            form_types=["8-K", "10-K", "10-Q", "4"],
            limit=30,
        )
        
        summary = summarize_filings(all_filings)
        
        # Count by time period
        from datetime import datetime, timedelta
        now = datetime.now()
        filings_30d = sum(1 for f in all_filings if (now - datetime.strptime(f.filed_date, "%Y-%m-%d")).days <= 30)
        filings_90d = len(all_filings)
        
        # Insider activity
        insider_filings = fetch_insider_filings(settings=settings, ticker=ticker, limit=20)
        has_buying = False
        has_selling = False
        net_insider = 0
        
        for f in insider_filings:
            if "P" in str(f.description).upper():
                has_buying = True
            if "S" in str(f.description).upper():
                has_selling = True
        
        # Notable filings
        notable = []
        for f in all_filings[:5]:
            notable.append({
                "date": f.filed_date,
                "form": f.form_type,
                "description": f.description[:100] if f.description else "",
            })
        
        return SECFilingSummary(
            filings_30d=filings_30d,
            filings_90d=filings_90d,
            recent_8k_items=summary.get("recent_8k_items", [])[:5],
            most_recent_form=summary.get("most_recent_form", ""),
            most_recent_date=summary.get("most_recent_date", ""),
            has_insider_selling=has_selling,
            has_insider_buying=has_buying,
            insider_net_30d=net_insider,
            notable_filings=notable,
        )
    except Exception:
        return None


def _fetch_analyst_data(settings: Settings, ticker: str) -> AnalystData | None:
    """Fetch analyst ratings and targets."""
    import requests
    
    base_url = "https://financialmodelingprep.com/api/v4"
    
    try:
        # Price target consensus
        resp = requests.get(
            f"{base_url}/price-target-consensus",
            params={"apikey": settings.fmp_api_key, "symbol": ticker},
            timeout=10,
        )
        consensus = resp.json()[0] if resp.ok and resp.json() else {}
        
        # Rating consensus
        resp = requests.get(
            f"https://financialmodelingprep.com/api/v3/rating/{ticker}",
            params={"apikey": settings.fmp_api_key},
            timeout=10,
        )
        rating = resp.json()[0] if resp.ok and resp.json() else {}
        
        # Get current price for upside calc
        resp = requests.get(
            f"https://financialmodelingprep.com/api/v3/quote/{ticker}",
            params={"apikey": settings.fmp_api_key},
            timeout=10,
        )
        quote = resp.json()[0] if resp.ok and resp.json() else {}
        current_price = float(quote.get("price", 0) or 0)
        
        target_mean = float(consensus.get("targetConsensus", 0) or 0)
        upside = ((target_mean / current_price) - 1) * 100 if current_price > 0 else 0
        
        # Map rating
        rating_score = rating.get("ratingScore", 3)
        if rating_score >= 4:
            consensus_label = "Strong Buy"
        elif rating_score >= 3.5:
            consensus_label = "Buy"
        elif rating_score >= 2.5:
            consensus_label = "Hold"
        elif rating_score >= 2:
            consensus_label = "Sell"
        else:
            consensus_label = "Strong Sell"
        
        return AnalystData(
            consensus_rating=consensus_label,
            num_analysts=int(consensus.get("numberOfAnalysts", 0) or 0),
            target_price_mean=target_mean,
            target_price_high=float(consensus.get("targetHigh", 0) or 0),
            target_price_low=float(consensus.get("targetLow", 0) or 0),
            upside_to_target=round(upside, 1),
            rating_changes_30d=[],
        )
    except Exception:
        return None


def _fetch_news_analysis(settings: Settings, ticker: str) -> NewsAnalysis | None:
    """Fetch and analyze recent news."""
    import requests
    
    try:
        resp = requests.get(
            f"https://financialmodelingprep.com/api/v3/stock_news",
            params={"apikey": settings.fmp_api_key, "tickers": ticker, "limit": 30},
            timeout=10,
        )
        news = resp.json() if resp.ok else []
        
        if not news:
            return None
        
        # Simple sentiment analysis (keyword-based)
        positive_words = {"beat", "surge", "soar", "jump", "rally", "upgrade", "bullish", "growth", "profit", "strong"}
        negative_words = {"miss", "drop", "fall", "decline", "downgrade", "bearish", "loss", "weak", "cut", "warning"}
        
        sentiment_scores = []
        headlines = []
        themes = {}
        
        for n in news[:20]:
            title = (n.get("title", "") or "").lower()
            
            pos_count = sum(1 for w in positive_words if w in title)
            neg_count = sum(1 for w in negative_words if w in title)
            
            if pos_count > neg_count:
                score = 1
                label = "positive"
            elif neg_count > pos_count:
                score = -1
                label = "negative"
            else:
                score = 0
                label = "neutral"
            
            sentiment_scores.append(score)
            
            headlines.append({
                "date": n.get("publishedDate", "")[:10],
                "title": n.get("title", ""),
                "sentiment": label,
            })
            
            # Extract themes
            for theme in ["earnings", "acquisition", "partnership", "product", "regulation", "lawsuit"]:
                if theme in title:
                    themes[theme] = themes.get(theme, 0) + 1
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        if avg_sentiment > 0.3:
            sentiment_label = "Positive"
        elif avg_sentiment < -0.3:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        key_themes = sorted(themes.keys(), key=lambda x: themes[x], reverse=True)[:3]
        
        return NewsAnalysis(
            total_articles_30d=len(news),
            sentiment_score=round(avg_sentiment, 2),
            sentiment_label=sentiment_label,
            key_themes=key_themes,
            recent_headlines=headlines[:5],
        )
    except Exception:
        return None


def _calculate_overall_score(
    momentum: MomentumMetrics,
    hf_metrics: HedgeFundMetrics,
    fundamentals: Any,
    sec_filings: Any,
    analyst_data: Any,
    is_etf: bool,
) -> tuple[int, list[str], list[str]]:
    """Calculate overall score and identify key strengths/risks."""
    score = 50  # Start neutral
    strengths = []
    risks = []
    
    # Momentum contribution (±20 points)
    score += min(momentum.momentum_score // 5, 20)
    score = max(score - 20, score + momentum.momentum_score // 5)
    
    if momentum.momentum_label in ["Strong Bullish", "Bullish"]:
        strengths.append(f"Strong momentum ({momentum.momentum_label})")
    elif momentum.momentum_label in ["Strong Bearish", "Bearish"]:
        risks.append(f"Weak momentum ({momentum.momentum_label})")
    
    # Risk-adjusted quality (±15 points)
    score += (hf_metrics.quality_score - 50) // 3
    
    if hf_metrics.sharpe_ratio_1y > 1.5:
        strengths.append(f"Excellent risk-adjusted returns (Sharpe: {hf_metrics.sharpe_ratio_1y})")
    elif hf_metrics.sharpe_ratio_1y < 0:
        risks.append("Negative risk-adjusted returns")
    
    # Drawdown risk
    if hf_metrics.max_drawdown_1y > 30:
        risks.append(f"High drawdown risk ({hf_metrics.max_drawdown_1y:.1f}%)")
        score -= 10
    elif hf_metrics.max_drawdown_1y < 15:
        strengths.append("Low drawdown history")
        score += 5
    
    # Fundamentals (for stocks)
    if not is_etf and fundamentals:
        # Valuation
        if 0 < fundamentals.pe_ratio < 15:
            strengths.append(f"Attractive valuation (P/E: {fundamentals.pe_ratio:.1f})")
            score += 5
        elif fundamentals.pe_ratio > 50:
            risks.append(f"High valuation (P/E: {fundamentals.pe_ratio:.1f})")
            score -= 5
        
        # Profitability
        if fundamentals.roe > 0.2:
            strengths.append(f"Strong profitability (ROE: {fundamentals.roe*100:.1f}%)")
            score += 5
        
        # Balance sheet
        if fundamentals.debt_to_equity > 2:
            risks.append(f"High leverage (D/E: {fundamentals.debt_to_equity:.1f})")
            score -= 5
        
        # Growth
        if fundamentals.revenue_growth_yoy > 0.2:
            strengths.append(f"Strong growth ({fundamentals.revenue_growth_yoy*100:.1f}% rev growth)")
            score += 5
    
    # Analyst sentiment
    if analyst_data:
        if analyst_data.upside_to_target > 20:
            strengths.append(f"Analyst upside: {analyst_data.upside_to_target:.1f}%")
            score += 5
        elif analyst_data.upside_to_target < -10:
            risks.append(f"Analyst downside: {analyst_data.upside_to_target:.1f}%")
            score -= 5
    
    # SEC filings
    if sec_filings:
        if sec_filings.has_insider_buying and not sec_filings.has_insider_selling:
            strengths.append("Insider buying activity")
            score += 3
        elif sec_filings.has_insider_selling and not sec_filings.has_insider_buying:
            risks.append("Insider selling activity")
            score -= 3
    
    # Clamp score
    score = max(0, min(100, score))
    
    return score, strengths[:5], risks[:5]


def _generate_llm_summary(
    settings: Settings,
    ticker: str,
    is_etf: bool,
    momentum: MomentumMetrics,
    hf_metrics: HedgeFundMetrics,
    fundamentals: Any,
    sec_filings: Any,
    analyst_data: Any,
    news_analysis: Any,
) -> str | None:
    """Generate LLM research summary."""
    if not settings.openai_api_key:
        return None
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=settings.openai_api_key)
        
        # Build context
        context = f"""
TICKER: {ticker}
TYPE: {"ETF" if is_etf else "Stock"}

MOMENTUM:
- 1M Return: {momentum.return_1m:+.1f}%
- 3M Return: {momentum.return_3m:+.1f}%
- RSI(14): {momentum.rsi_14} ({momentum.rsi_interpretation})
- Momentum Score: {momentum.momentum_score}/100 ({momentum.momentum_label})
- Trend: {momentum.trend_direction}
- vs 52W High: {momentum.pct_from_52w_high:.1f}%

RISK METRICS:
- Volatility (1Y): {hf_metrics.volatility_1y}%
- Beta: {hf_metrics.beta_1y}
- Max Drawdown (1Y): {hf_metrics.max_drawdown_1y:.1f}%
- Sharpe Ratio (1Y): {hf_metrics.sharpe_ratio_1y}
- Risk Label: {hf_metrics.risk_label}
"""
        
        if not is_etf and fundamentals:
            context += f"""
FUNDAMENTALS:
- Sector: {fundamentals.sector}
- Market Cap: ${fundamentals.market_cap_b:.1f}B
- P/E: {fundamentals.pe_ratio:.1f}
- Revenue Growth: {fundamentals.revenue_growth_yoy*100:+.1f}%
- Net Margin: {fundamentals.net_margin*100:.1f}%
- ROE: {fundamentals.roe*100:.1f}%
- D/E: {fundamentals.debt_to_equity:.1f}
"""
        
        if analyst_data:
            context += f"""
ANALYST:
- Consensus: {analyst_data.consensus_rating}
- Target: ${analyst_data.target_price_mean:.2f} ({analyst_data.upside_to_target:+.1f}% upside)
- # Analysts: {analyst_data.num_analysts}
"""
        
        if news_analysis:
            context += f"""
NEWS (30D):
- Sentiment: {news_analysis.sentiment_label} ({news_analysis.sentiment_score:+.2f})
- Articles: {news_analysis.total_articles_30d}
- Themes: {', '.join(news_analysis.key_themes) if news_analysis.key_themes else 'None'}
"""
        
        prompt = f"""You are a senior hedge fund analyst. Provide a concise research summary for {ticker}.

{context}

Write a 3-4 paragraph research brief covering:
1. Current momentum and technical setup
2. {"ETF characteristics and exposure" if is_etf else "Fundamental quality and valuation"}
3. Key risks and catalysts
4. Bottom line: Is this attractive at current levels? For what type of investor/trade?

Be direct and specific. Use the data provided. No generic statements."""

        resp = client.chat.completions.create(
            model=settings.openai_model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"LLM analysis unavailable: {e}"

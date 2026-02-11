"""
Universal LLM Analyst Module - Research Grade

Provides PhD-level macro analysis for any regime/domain with:
- Real-time news with source citations
- Market data with timestamps
- Economic calendar integration
- Analyst-grade output with references

Usage:
    from lox.llm.core.analyst import llm_analyze_regime
    
    result = llm_analyze_regime(
        settings=settings,
        domain="volatility",
        snapshot={"vix": 18.5, "term_spread": -2.1, ...},
        regime_label="Elevated Volatility",
    )
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any

from lox.config import Settings


# Domain-specific context for the LLM
DOMAIN_CONTEXT = {
    "volatility": {
        "description": "VIX-based volatility regime (level, momentum, term structure, persistence)",
        "key_metrics": ["VIX level", "VIX term spread", "VIX momentum", "persistence", "vol pressure score"],
        "related_tickers": ["VIX", "VIXY", "VXX", "UVXY", "SVXY", "SPY", "QQQ"],
        "news_keywords": ["volatility", "VIX", "market fear", "options", "hedging", "risk-off", "selloff", "correction", "CBOE", "implied volatility"],
        "macro_keywords": ["Fed", "FOMC", "recession", "inflation", "geopolitical", "market", "stocks"],
        "trading_focus": "vol products, hedging strategies, straddles/strangles, tail risk",
        "data_sources": ["CBOE VIX Index", "FRED VIXCLS", "Alpaca Options Data"],
        "sector_implications": {
            "low_vol_complacent": {
                "benefit": ["QQQ (tech)", "XLY (consumer disc)", "ARKK (innovation)", "IWM (small caps)", "XLF (financials)"],
                "hurt": ["XLU (utilities)", "XLP (staples)", "VIXY (vol long)"],
                "rationale": "Low vol favors risk-on/high-beta; defensive sectors underperform",
            },
            "rising_vol_stress": {
                "benefit": ["XLU (utilities)", "XLP (staples)", "XLV (healthcare)", "GLD (gold)", "TLT (bonds)"],
                "hurt": ["QQQ (tech)", "XLY (consumer disc)", "IWM (small caps)", "XLF (financials)"],
                "rationale": "Vol spike triggers flight to safety; high-beta gets crushed",
            },
            "elevated_vol_persistent": {
                "benefit": ["XLV (healthcare)", "XLP (staples)", "USMV (low vol factor)"],
                "hurt": ["ARKK (innovation)", "XLY (consumer disc)", "XLF (financials)"],
                "rationale": "Persistent high vol = defensive rotation, avoid leverage & beta",
            },
            "vol_crush_mean_reversion": {
                "benefit": ["QQQ (tech)", "XLF (financials)", "SVXY (short vol)", "IWM (small caps)"],
                "hurt": ["VIXY (vol long)", "VXX (vol long)"],
                "rationale": "Vol crush = sell hedges, buy beaten-down cyclicals",
            },
        },
    },
    "commodities": {
        "description": "Commodities regime (oil, gold, copper, broad index momentum and shocks)",
        "key_metrics": ["WTI price", "gold price", "copper", "broad commodity index", "energy shock", "metals impulse"],
        "related_tickers": ["USO", "GLDM", "GLD", "SLV", "CPER", "DBC", "XLE", "XME", "OIH", "CL=F", "GC=F"],
        "news_keywords": ["oil", "crude", "OPEC", "gold", "copper", "commodities", "inflation", "supply chain", "WTI", "Brent", "precious metals"],
        "macro_keywords": ["China", "demand", "inventory", "production", "sanctions", "tariff"],
        "trading_focus": "commodity ETFs, energy/materials equities, inflation hedges",
        "data_sources": ["FRED DCOILWTICO", "FRED GOLDAMGBD228NLBM", "FMP Real-time Quotes"],
        "sector_implications": {
            "oil_rising": {
                "benefit": ["XLE (energy)", "OIH (oil services)", "XOP (E&P)", "VDE (energy)"],
                "hurt": ["XLI (industrials)", "XLY (consumer disc)", "JETS (airlines)", "XTN (transports)"],
                "rationale": "Oil up = energy profits but consumer/transport cost squeeze",
            },
            "oil_falling": {
                "benefit": ["XLY (consumer disc)", "JETS (airlines)", "XTN (transports)", "XLI (industrials)"],
                "hurt": ["XLE (energy)", "OIH (oil services)", "XOP (E&P)"],
                "rationale": "Cheap oil = consumer tailwind, energy earnings collapse",
            },
            "gold_rising": {
                "benefit": ["GDX (gold miners)", "GDXJ (junior miners)", "SIL (silver miners)", "RING (gold miners)"],
                "hurt": ["USD longs", "UUP (dollar bull)"],
                "rationale": "Gold up signals fear/inflation hedge demand; often inverse USD",
            },
            "copper_rising": {
                "benefit": ["COPX (copper miners)", "XLB (materials)", "FCX (Freeport)", "XLI (industrials)"],
                "hurt": [],
                "rationale": "Dr. Copper signals global growth; benefits industrials/materials",
            },
            "broad_commodity_rally": {
                "benefit": ["XLB (materials)", "XLE (energy)", "XME (metals/mining)", "TIP (TIPS)"],
                "hurt": ["XLU (utilities)", "XLP (staples)", "long duration bonds"],
                "rationale": "Commodity rally = inflation regime; hurts bond proxies",
            },
        },
    },
    "rates": {
        "description": "Rates/yield curve regime (UST level, slope, momentum, term premium)",
        "key_metrics": ["10Y yield", "2Y yield", "2s10s spread", "yield momentum", "real rates"],
        "related_tickers": ["TLT", "IEF", "SHY", "TBT", "TMV", "TMF", "GOVT", "BND"],
        "news_keywords": ["treasury", "yields", "Fed", "FOMC", "rate hike", "rate cut", "bonds", "duration", "10-year", "2-year"],
        "macro_keywords": ["inflation", "employment", "GDP", "Powell", "dot plot"],
        "trading_focus": "duration plays, curve trades, rate-sensitive equities",
        "data_sources": ["FRED DGS10", "FRED DGS2", "FRED T10Y2Y", "Treasury Direct"],
        "sector_implications": {
            "rising_rates": {
                "hurt": ["XLU (utilities)", "XLRE (REITs)", "XLP (staples)", "VNQ (REITs)", "XHB (homebuilders)", "ITB (homebuilders)"],
                "benefit": ["XLF (financials)", "KRE (regional banks)", "IAK (insurance)"],
                "rationale": "Higher rates hurt dividend proxies & rate-sensitive sectors; banks benefit from NIM expansion",
            },
            "falling_rates": {
                "benefit": ["XLU (utilities)", "XLRE (REITs)", "XLP (staples)", "XHB (homebuilders)", "ITB (homebuilders)"],
                "hurt": ["XLF (financials)", "KRE (regional banks)"],
                "rationale": "Lower rates boost dividend proxies & housing; bank margins compress",
            },
            "steepening_curve": {
                "benefit": ["XLF (financials)", "KRE (regional banks)", "XLI (industrials)", "XLB (materials)"],
                "hurt": ["TLT (long duration)"],
                "rationale": "Steepening signals growth expectations; banks earn more on term spread",
            },
            "flattening_curve": {
                "benefit": ["XLU (utilities)", "XLP (staples)", "TLT (long duration)"],
                "hurt": ["XLF (financials)", "KRE (regional banks)", "IWM (small caps)"],
                "rationale": "Flattening signals slowdown fears; defensive sectors outperform",
            },
        },
    },
    "funding": {
        "description": "Funding regime (SOFR, TGCR, BGCR corridor, repo market stress)",
        "key_metrics": ["SOFR", "SOFR-Fed Funds spread", "TGCR", "BGCR", "repo stress indicators"],
        "related_tickers": ["BIL", "SGOV", "SHV", "FLOT", "TFLO"],
        "news_keywords": ["SOFR", "repo", "funding", "money market", "Fed", "liquidity", "reserves", "overnight"],
        "macro_keywords": ["QT", "balance sheet", "RRP", "Treasury"],
        "trading_focus": "short-duration, money market sensitivity, bank funding stress",
        "data_sources": ["FRED SOFR", "FRED TGCR", "FRED BGCR", "NY Fed"],
    },
    "monetary": {
        "description": "Monetary regime (Fed balance sheet, reserves, RRP, policy stance)",
        "key_metrics": ["Fed balance sheet", "bank reserves", "RRP usage", "policy rate", "QT pace"],
        "related_tickers": ["SPY", "QQQ", "TLT", "XLF", "KRE", "BTC-USD"],
        "news_keywords": ["Fed", "FOMC", "QT", "QE", "balance sheet", "reserves", "liquidity", "Powell", "monetary policy"],
        "macro_keywords": ["inflation", "employment", "rate decision", "dot plot"],
        "trading_focus": "liquidity-sensitive assets, bank equities, risk assets broadly",
        "data_sources": ["FRED WALCL", "FRED TOTRESNS", "FRED RRPONTSYD", "Federal Reserve H.4.1"],
    },
    "fiscal": {
        "description": "US fiscal regime (deficits, Treasury issuance, TGA, auction demand)",
        "key_metrics": ["deficit/GDP", "TGA balance", "issuance mix", "auction demand", "dealer absorption"],
        "related_tickers": ["TLT", "IEF", "TMV", "TMF", "SPY"],
        "news_keywords": ["deficit", "Treasury", "auction", "debt ceiling", "fiscal", "government spending", "issuance", "TGA"],
        "macro_keywords": ["Congress", "budget", "stimulus", "tax"],
        "trading_focus": "duration risk, term premium, crowding out",
        "data_sources": ["FRED MTSDS133FMS", "Treasury FiscalData MSPD", "Treasury Direct Auctions"],
    },
    "inflation": {
        "description": "Inflation regime (CPI, PCE, breakevens, sticky vs flexible)",
        "key_metrics": ["CPI YoY", "core CPI", "PCE", "5Y breakeven", "10Y breakeven", "sticky CPI"],
        "related_tickers": ["TIP", "SCHP", "VTIP", "RINF"],
        "news_keywords": ["inflation", "CPI", "PCE", "prices", "deflation", "disinflation", "Fed target", "core inflation"],
        "macro_keywords": ["Fed", "rate", "consumer", "wage"],
        "trading_focus": "TIPS, inflation hedges, real assets, rate-sensitive",
        "data_sources": ["FRED CPIAUCSL", "FRED PCEPILFE", "FRED T5YIE", "BLS"],
        "sector_implications": {
            "inflation_rising": {
                "benefit": ["XLE (energy)", "XLB (materials)", "GLD (gold)", "TIP (TIPS)", "XME (mining)", "PDBC (commodities)"],
                "hurt": ["XLU (utilities)", "TLT (long bonds)", "XLRE (REITs)", "XLP (staples)"],
                "rationale": "Inflation benefits real assets, crushes duration & bond proxies",
            },
            "inflation_falling": {
                "benefit": ["TLT (long bonds)", "XLU (utilities)", "XLRE (REITs)", "QQQ (growth)", "XLY (discretionary)"],
                "hurt": ["XLE (energy)", "XLB (materials)", "TIP (TIPS)", "GLD (gold)"],
                "rationale": "Disinflation = duration rally, growth stocks outperform",
            },
            "stagflation": {
                "benefit": ["XLE (energy)", "GLD (gold)", "XLV (healthcare)", "XLP (staples)"],
                "hurt": ["XLY (discretionary)", "XLF (financials)", "IWM (small caps)", "QQQ (tech)"],
                "rationale": "Stagflation = real assets + defensives; avoid cyclicals",
            },
        },
    },
    "growth": {
        "description": "Growth regime (employment, payrolls, claims, PMI, consumer)",
        "key_metrics": ["payrolls", "unemployment rate", "initial claims", "PMI", "consumer confidence"],
        "related_tickers": ["SPY", "IWM", "XLY", "XLI", "XLF"],
        "news_keywords": ["jobs", "employment", "payrolls", "GDP", "recession", "growth", "PMI", "manufacturing", "labor"],
        "macro_keywords": ["Fed", "soft landing", "hard landing", "consumer"],
        "trading_focus": "cyclicals vs defensives, small caps, consumer discretionary",
        "data_sources": ["FRED PAYEMS", "FRED UNRATE", "FRED ICSA", "ISM PMI", "BLS"],
        "sector_implications": {
            "growth_accelerating": {
                "benefit": ["XLY (discretionary)", "XLI (industrials)", "XLF (financials)", "IWM (small caps)", "XLB (materials)"],
                "hurt": ["XLU (utilities)", "XLP (staples)", "TLT (long bonds)"],
                "rationale": "Accelerating growth = cyclicals lead, defensives lag",
            },
            "growth_decelerating": {
                "benefit": ["XLU (utilities)", "XLP (staples)", "XLV (healthcare)", "TLT (long bonds)", "USMV (low vol)"],
                "hurt": ["XLY (discretionary)", "XLI (industrials)", "IWM (small caps)", "XLF (financials)"],
                "rationale": "Slowing growth = defensive rotation, quality over beta",
            },
            "recession_imminent": {
                "benefit": ["TLT (long bonds)", "XLU (utilities)", "XLP (staples)", "GLD (gold)", "SH (short SPY)"],
                "hurt": ["XLF (financials)", "XLY (discretionary)", "XLI (industrials)", "IWM (small caps)", "HYG (high yield)"],
                "rationale": "Recession = aggressive defense, short cyclicals, long duration",
            },
            "recovery_early_cycle": {
                "benefit": ["XLF (financials)", "XLI (industrials)", "XLB (materials)", "IWM (small caps)", "XLY (discretionary)"],
                "hurt": ["XLU (utilities)", "XLP (staples)"],
                "rationale": "Early recovery = buy beaten-down cyclicals, financials lead",
            },
        },
    },
    "liquidity": {
        "description": "Liquidity regime (credit spreads, funding conditions, financial conditions)",
        "key_metrics": ["HY spread", "IG spread", "Fed funds", "financial conditions index", "TED spread"],
        "related_tickers": ["HYG", "LQD", "JNK", "BKLN", "XLF", "KRE"],
        "news_keywords": ["credit", "spreads", "liquidity", "financial conditions", "stress", "tightening", "high yield"],
        "macro_keywords": ["Fed", "QT", "bank", "lending"],
        "trading_focus": "credit risk, bank equities, liquidity-sensitive assets",
        "data_sources": ["FRED BAMLH0A0HYM2", "FRED NFCI", "ICE BofA Indices"],
    },
    "usd": {
        "description": "USD regime (DXY strength/weakness, rate differentials, flows)",
        "key_metrics": ["DXY", "EUR/USD", "USD momentum", "rate differentials"],
        "related_tickers": ["UUP", "USDU", "FXE", "FXY", "EEM", "EFA"],
        "news_keywords": ["dollar", "USD", "DXY", "currency", "FX", "emerging markets", "rate differential", "forex"],
        "macro_keywords": ["Fed", "ECB", "BOJ", "trade", "tariff"],
        "trading_focus": "EM vs DM, commodity currencies, FX-sensitive equities",
        "data_sources": ["FRED DTWEXBGS", "ICE US Dollar Index"],
    },
    "housing": {
        "description": "Housing/MBS regime (mortgage rates, housing proxies, MBS spreads)",
        "key_metrics": ["30Y mortgage rate", "MBS spread", "housing starts", "home prices"],
        "related_tickers": ["XHB", "ITB", "XLRE", "VNQ", "MBB", "VMBS"],
        "news_keywords": ["housing", "mortgage", "MBS", "home prices", "real estate", "rates", "homebuilders"],
        "macro_keywords": ["Fed", "affordability", "inventory", "construction"],
        "trading_focus": "homebuilders, REITs, mortgage REITs, rate sensitivity",
        "data_sources": ["FRED MORTGAGE30US", "FRED HOUST", "Case-Shiller"],
    },
    "crypto": {
        "description": "Crypto perpetual futures regime (BTC/ETH/SOL prices, funding rates, OI, momentum, correlation to risk assets)",
        "key_metrics": ["BTC price", "ETH price", "SOL price", "funding rates", "open interest", "RSI", "MACD", "EMA trend", "BTC momentum", "crypto vs SPY correlation"],
        "related_tickers": ["GBTC", "ETHE", "BITO", "COIN", "MARA", "RIOT"],
        "news_keywords": ["bitcoin", "crypto", "ethereum", "blockchain", "SEC", "regulation", "halving", "ETF", "perpetual", "futures", "funding rate", "liquidation", "DeFi", "stablecoin"],
        "macro_keywords": ["Fed", "liquidity", "risk", "institutional", "stablecoin", "USDT"],
        "trading_focus": "crypto perps (BTC, ETH, SOL), funding rate arb, momentum, mean reversion, miners, crypto-correlated equities",
        "data_sources": ["CCXT (OKX/Bybit/Binance)", "Aster DEX", "Alpaca Crypto Data"],
        "sector_implications": {
            "risk_on_crypto": {
                "benefit": ["COIN", "MARA", "RIOT", "BITO", "GBTC"],
                "hurt": ["UUP (dollar)", "TLT (bonds)"],
                "rationale": "Crypto rallies correlate with risk-on, liquidity expansion",
            },
            "crypto_stress": {
                "benefit": ["TLT (bonds)", "UUP (dollar)"],
                "hurt": ["COIN", "MARA", "RIOT", "GBTC", "BITO"],
                "rationale": "Crypto drawdowns correlate with risk-off, deleveraging",
            },
        },
    },
    "tariff": {
        "description": "Tariff/cost-push regime (trade policy impact on specific sectors)",
        "key_metrics": ["basket performance vs benchmark", "import prices", "tariff exposure"],
        "related_tickers": ["EEM", "FXI", "KWEB"],
        "news_keywords": ["tariff", "trade war", "import", "China", "supply chain", "reshoring", "duties"],
        "macro_keywords": ["Trump", "trade", "manufacturing", "Section 301"],
        "trading_focus": "exposed sectors, domestic alternatives, supply chain plays",
        "data_sources": ["USTR Tariff Schedule", "FRED Import Price Index"],
    },
}

# Canonical scenario framework — must match Monte Carlo regimes
# These are the ONLY scenarios the LLM should use for institutional consistency
SCENARIO_FRAMEWORK = {
    "GOLDILOCKS": {
        "label": "Goldilocks / Disinflationary",
        "description": "Low inflation + solid growth. Risk-on conditions with controlled vol.",
        "equity_drift": "+8% annualized",
        "equity_vol": "15% realized",
        "iv_target": "VIX ~18",
        "jump_risk": "Low (3%)",
        "typical_triggers": ["Soft landing confirmed", "Fed pivot to cuts", "Earnings beat cycle"],
        "spx_target_3m": "+4-6%",
        "tlt_target_3m": "+2-4%",
        "base_probability": 25,
    },
    "STAGFLATION": {
        "label": "Stagflation / Inflationary",
        "description": "High inflation + weak growth. Worst for 60/40, good for real assets.",
        "equity_drift": "-2% annualized",
        "equity_vol": "22% realized",
        "iv_target": "VIX ~25",
        "jump_risk": "Elevated (10%)",
        "typical_triggers": ["CPI re-acceleration", "Wage-price spiral", "Supply shock"],
        "spx_target_3m": "-3-8%",
        "tlt_target_3m": "-5-10%",
        "base_probability": 15,
    },
    "RISK_OFF": {
        "label": "Risk-Off / Crash",
        "description": "Sharp equity drawdown with vol spike. Flight to safety.",
        "equity_drift": "-25% annualized",
        "equity_vol": "35% realized",
        "iv_target": "VIX 35-50+",
        "jump_risk": "High (25%)",
        "typical_triggers": ["Credit event", "Geopolitical shock", "Systemic stress", "Liquidity crisis"],
        "spx_target_3m": "-15-25%",
        "tlt_target_3m": "+5-10%",
        "base_probability": 10,
    },
    "RATES_SHOCK": {
        "label": "Rates Shock / Term Premium",
        "description": "Aggressive Fed hiking or bond market revolt. Duration pain.",
        "equity_drift": "-10% annualized",
        "equity_vol": "24% realized",
        "iv_target": "VIX ~22",
        "jump_risk": "Moderate (8%)",
        "typical_triggers": ["Hot CPI print", "Fed hawkish surprise", "Fiscal blowout fears"],
        "spx_target_3m": "-5-10%",
        "tlt_target_3m": "-8-15%",
        "base_probability": 10,
    },
    "CREDIT_STRESS": {
        "label": "Credit Stress / Credit Event",
        "description": "HY spreads widen, bank stress. Risk-off with credit leading.",
        "equity_drift": "-15% annualized",
        "equity_vol": "28% realized",
        "iv_target": "VIX 28-35",
        "jump_risk": "High (15%)",
        "typical_triggers": ["Bank failure", "HY defaults spike", "EM crisis spillover"],
        "spx_target_3m": "-10-18%",
        "tlt_target_3m": "+3-8%",
        "base_probability": 8,
    },
    "SLOW_BLEED": {
        "label": "Slow Bleed / Grinding Down",
        "description": "Persistent drift lower without vol spike. Death by 1000 cuts.",
        "equity_drift": "-8% annualized",
        "equity_vol": "16% realized",
        "iv_target": "VIX ~19",
        "jump_risk": "Low (2%)",
        "typical_triggers": ["Earnings recession", "Multiple compression", "Persistent outflows"],
        "spx_target_3m": "-3-6%",
        "tlt_target_3m": "Flat to +2%",
        "base_probability": 12,
    },
    "VOL_CRUSH": {
        "label": "Vol Crush / Complacent",
        "description": "Melt-up with collapsing IV. Worst scenario for hedges.",
        "equity_drift": "+10% annualized",
        "equity_vol": "12% realized",
        "iv_target": "VIX 12-14",
        "jump_risk": "Very low (1%)",
        "typical_triggers": ["Earnings beat with guidance raise", "Fed on hold + QT taper", "Low event risk"],
        "spx_target_3m": "+5-8%",
        "tlt_target_3m": "+1-3%",
        "base_probability": 15,
    },
    "BASE_CASE": {
        "label": "Base Case / Neutral",
        "description": "Balanced conditions. Modest drift, normal vol, no extremes.",
        "equity_drift": "+5% annualized",
        "equity_vol": "18% realized",
        "iv_target": "VIX ~20",
        "jump_risk": "Normal (5%)",
        "typical_triggers": ["Current trajectory continues", "No major surprises"],
        "spx_target_3m": "+2-4%",
        "tlt_target_3m": "Flat to +2%",
        "base_probability": 5,  # Residual after other scenarios
    },
}


def _fetch_domain_news(
    settings: Settings,
    domain: str,
    max_items: int = 20,
    lookback_days: int = 5,
) -> list[dict[str, Any]]:
    """
    Fetch real-time news from BOTH FMP and Alpaca APIs.
    
    Uses unified news fetcher to aggregate and deduplicate.
    Alpaca provides full article content for deeper analysis.
    """
    ctx = DOMAIN_CONTEXT.get(domain, {})
    tickers = ctx.get("related_tickers", [])[:8]
    keywords = ctx.get("news_keywords", []) + ctx.get("macro_keywords", [])
    
    try:
        from lox.altdata.news import fetch_unified_news, format_news_for_llm
        
        # Fetch from both FMP and Alpaca
        articles = fetch_unified_news(
            settings=settings,
            symbols=tickers if tickers else None,
            keywords=keywords if keywords else None,
            lookback_days=lookback_days,
            limit=max_items * 2,  # Fetch more, filter down
            include_content=True,  # Get full content from Alpaca
        )
        
        # Format for LLM with content snippets
        return format_news_for_llm(
            articles,
            max_articles=max_items,
            include_snippets=True,
            include_content=True,  # Include full content for research-grade output
        )
    
    except Exception as e:
        # Fallback to FMP direct if unified fetch fails
        try:
            from lox.altdata.fmp import fetch_stock_news
            
            all_items = []
            for ticker in tickers[:5]:
                items = fetch_stock_news(settings=settings, ticker=ticker, limit=10)
                all_items.extend(items)
            
            formatted = []
            for i, it in enumerate(all_items[:max_items]):
                formatted.append({
                    "index": i + 1,
                    "title": it.get("title", ""),
                    "source": it.get("site", "") or "FMP",
                    "provider": "FMP",
                    "published_at": it.get("publishedDate", ""),
                    "url": it.get("url", ""),
                    "snippet": (it.get("text", "") or "")[:300],
                })
            return formatted
        except Exception:
            return []


def _fetch_ticker_news(
    settings: Settings,
    ticker: str,
    max_items: int = 15,
    lookback_days: int = 7,
) -> list[dict[str, Any]]:
    """Fetch news specifically for a given ticker symbol."""
    try:
        from lox.altdata.news import fetch_unified_news, format_news_for_llm
        
        articles = fetch_unified_news(
            settings=settings,
            symbols=[ticker],
            keywords=None,
            lookback_days=lookback_days,
            limit=max_items * 2,
            include_content=True,
        )
        
        return format_news_for_llm(
            articles,
            max_articles=max_items,
            include_snippets=True,
            include_content=True,
        )
    except Exception:
        try:
            from lox.altdata.fmp import fetch_stock_news
            
            items = fetch_stock_news(settings=settings, ticker=ticker, limit=max_items)
            formatted = []
            for i, it in enumerate(items[:max_items]):
                formatted.append({
                    "index": i + 1,
                    "title": it.get("title", ""),
                    "source": it.get("site", "") or "FMP",
                    "provider": "FMP",
                    "published_at": it.get("publishedDate", ""),
                    "url": it.get("url", ""),
                    "snippet": (it.get("text", "") or "")[:300],
                })
            return formatted
        except Exception:
            return []


def _fetch_general_macro_news(
    settings: Settings,
    domain: str,
    max_items: int = 10,
) -> list[dict[str, Any]]:
    """Fetch general macro news filtered by domain keywords."""
    if not settings.fmp_api_key:
        return []
    
    ctx = DOMAIN_CONTEXT.get(domain, {})
    keywords = ctx.get("news_keywords", []) + ctx.get("macro_keywords", [])
    
    try:
        from lox.altdata.fmp import fetch_stock_news
        
        # Use FMP stock news with general tickers as proxy for macro news
        macro_tickers = ["SPY", "TLT", "HYG", "VIX", "DXY"]
        all_items = []
        for ticker in macro_tickers[:3]:
            items = fetch_stock_news(settings=settings, ticker=ticker, limit=10)
            all_items.extend(items)
        
        # Filter by keywords
        if keywords:
            filtered = []
            for item in all_items:
                title = (item.get("title", "") or "").lower()
                text_body = (item.get("text", "") or "").lower()
                combined = f"{title} {text_body}"
                if any(kw.lower() in combined for kw in keywords):
                    filtered.append(item)
            all_items = filtered if filtered else all_items
        
        return [
            {
                "index": i + 1,
                "title": it.get("title", ""),
                "source": it.get("site", "") or "FMP",
                "published_at": it.get("publishedDate", ""),
                "url": it.get("url", ""),
                "topic": "macro",
            }
            for i, it in enumerate(all_items[:max_items])
        ]
    except Exception:
        return []


def _fetch_realtime_prices(
    settings: Settings,
    domain: str,
) -> dict[str, Any]:
    """Fetch real-time prices with change data for domain-related tickers."""
    ctx = DOMAIN_CONTEXT.get(domain, {})
    tickers = ctx.get("related_tickers", [])[:10]
    
    if not tickers or not settings.fmp_api_key:
        return {}
    
    # Filter out crypto-style tickers for FMP
    equity_tickers = [t for t in tickers if "-" not in t and "=" not in t]
    
    try:
        from lox.altdata.fmp import fetch_realtime_quotes
        prices = fetch_realtime_quotes(settings=settings, tickers=equity_tickers)
        
        # Add timestamp
        return {
            "asof": datetime.now(timezone.utc).isoformat(),
            "prices": prices,
            "source": "FMP Real-time Quotes",
        }
    except Exception:
        return {}


def _fetch_calendar_events(
    settings: Settings,
    days_ahead: int = 14,
    max_items: int = 15,
) -> list[dict[str, Any]]:
    """Fetch upcoming economic calendar events with enhanced detail."""
    try:
        from lox.overlay.context import fetch_calendar_events
        events = fetch_calendar_events(settings=settings, days_ahead=days_ahead, max_items=max_items)
        
        # Add index for citation
        for i, ev in enumerate(events):
            ev["index"] = i + 1
        
        return events
    except Exception:
        return []


def _compute_historical_context(
    settings: Settings,
    domain: str,
    snapshot_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute historical context for quantitative calibration.
    
    Provides the LLM with:
    - Current levels vs 52-week ranges
    - Recent realized volatility (for move size calibration)
    - Typical event-day moves (if available)
    """
    ctx = DOMAIN_CONTEXT.get(domain.lower(), {})
    tickers = ctx.get("related_tickers", [])[:5]
    
    if not tickers:
        return {}
    
    historical = {
        "calibration_data": {},
        "typical_moves": {},
    }
    
    try:
        from lox.altdata.fmp import fetch_realtime_quotes
        
        quotes = fetch_realtime_quotes(settings=settings, tickers=tickers)
        
        for ticker, data in quotes.items():
            if not isinstance(data, dict):
                continue
            
            price = data.get("price")
            high_52w = data.get("yearHigh")
            low_52w = data.get("yearLow")
            
            if price and high_52w and low_52w and high_52w > low_52w:
                range_52w = high_52w - low_52w
                pct_of_range = ((price - low_52w) / range_52w * 100) if range_52w > 0 else 50
                
                historical["calibration_data"][ticker] = {
                    "current": round(price, 2),
                    "52w_high": round(high_52w, 2),
                    "52w_low": round(low_52w, 2),
                    "pct_of_52w_range": round(pct_of_range, 1),
                    "distance_to_high_pct": round((high_52w - price) / price * 100, 1) if price > 0 else None,
                    "distance_to_low_pct": round((price - low_52w) / price * 100, 1) if price > 0 else None,
                }
    except Exception:
        pass
    
    # Add typical event-day volatility estimates (hard-coded based on historical analysis)
    # These are approximate 1-day moves for major releases
    historical["typical_event_moves"] = {
        "CPI": {"direction": "rates", "typical_move_bps": "10-25bps on 10Y if surprise > 0.2%"},
        "FOMC": {"direction": "rates + equities", "typical_move_bps": "15-30bps on 2Y; SPX ±1-2%"},
        "NFP": {"direction": "rates + USD", "typical_move_bps": "10-20bps on 10Y if miss > 50k"},
        "GDP": {"direction": "rates + equities", "typical_move_bps": "5-15bps on 10Y"},
        "PMI": {"direction": "equities + commodities", "typical_move": "SPX ±0.5-1%"},
        "Housing": {"direction": "rates", "typical_move_bps": "3-8bps on 10Y"},
    }
    
    # Extract volatility from snapshot if available
    vol_keys = ["vix", "VIX", "rv20", "realized_vol", "20d_std", "volatility"]
    for key in vol_keys:
        if key in snapshot_dict:
            val = snapshot_dict[key]
            if isinstance(val, (int, float)):
                historical["current_vol_regime"] = {
                    "metric": key,
                    "value": round(val, 2),
                    "interpretation": "low" if val < 15 else "elevated" if val < 25 else "high",
                    "typical_daily_move": f"~{val/16:.1f}% daily" if val else None,  # VIX/16 ≈ daily move
                }
                break
    
    return historical


# =========================================================================
# Ticker-specific scenario section builder
# =========================================================================

def _build_ticker_scenario_section(ticker: str, snapshot_dict: dict[str, Any]) -> str:
    """
    Build the SCENARIO ANALYSIS prompt section for a single ticker.

    If ``snapshot_dict`` contains a ``quant_scenarios`` key (pre-computed by
    the regime-conditional bootstrap engine), the table rows are pre-filled
    and the LLM is instructed to supply narrative triggers only.

    Otherwise falls back to the original prompt that asks the LLM to generate
    everything (targets + triggers).
    """
    qs = snapshot_dict.get("quant_scenarios")

    if qs and isinstance(qs, dict) and "scenarios" in qs:
        return _quant_scenario_section(ticker, qs)

    # ---- Fallback: no pre-computed scenarios available --------------------
    return f"""### SCENARIO ANALYSIS ({ticker}-Specific)

Provide **4-5 scenarios** for {ticker} over the next 3 months. Each scenario must include a **specific price target for {ticker}** and the **valuation methodology** used to derive it (e.g. "based on 35x EV/Revenue at $7.7B revenue" or "based on DCF 12% WACC" — not just a number).

**Probability assignments need basis:** Do not use a generic 20/45/25/10 template. Justify with historical base rates or analytical reasoning (e.g. "Bull 20% — historically, stocks at this valuation sustain >30% growth only ~20% of the time").

**Ground targets** in snapshot and 52w range; the Key Trigger must justify the magnitude.

| Scenario | Probability | Key Trigger | {ticker} Target (3M) | Methodology | {ticker} Price Range |
|----------|-------------|-------------|---------------------|-------------|---------------------|
| BULL | X% | [catalyst] | $XX.XX | [e.g. 35x EV/Rev] | $XX - $XX |
| BASE | X% | [trajectory] | $XX.XX | [e.g. current multiple] | $XX - $XX |
| BEAR | X% | [risk catalyst] | $XX.XX | [e.g. 15x EV/Rev] | $XX - $XX |
| TAIL RISK | X% | [extreme] | $XX.XX | [e.g. stress multiple] | $XX - $XX |

**Add one Expected value line:** E[V] = (Bull% × Bull target) + (Base% × Base) + (Bear% × Bear) + (Tail% × Tail) = $X.XX

**Rules:** Probabilities sum to 100%. Targets must state methodology. Probabilities must be justified, not arbitrary. Calibrate to snapshot volatility.
"""


def _quant_scenario_section(ticker: str, qs: dict[str, Any]) -> str:
    """
    Build a scenario section where price targets, ranges, and probabilities
    are already computed.  The LLM is told to use these numbers exactly and
    only supply the narrative Key Trigger for each row.
    """
    sc = qs["scenarios"]
    bull = sc.get("BULL", {})
    base = sc.get("BASE", {})
    bear = sc.get("BEAR", {})
    tail = sc.get("TAIL_RISK", {})

    def _fmt(s: dict) -> tuple[str, str, str]:
        """Return (probability, target, range) formatted strings."""
        prob = f"{s.get('probability', '?')}%"
        tgt = f"${s.get('target', '?')}"
        rng = s.get("range", [])
        rng_str = f"${rng[0]} - ${rng[1]}" if len(rng) == 2 else "?"
        return prob, tgt, rng_str

    bp, bt, br = _fmt(bull)
    bap, bat, bar_ = _fmt(base)
    bep, bet, ber = _fmt(bear)
    tp, tt, tr = _fmt(tail)

    method = qs.get("method", "bootstrap")
    n_sims = qs.get("n_simulations", "N/A")
    horizon = qs.get("horizon", "3M")
    vol = qs.get("realized_vol_annual", "?")
    regime_adj = qs.get("regime_adjustments") or {}
    drift_adj = regime_adj.get("equity_drift_adj", 0)
    vol_adj = regime_adj.get("equity_vol_adj", 1.0)

    adj_desc = "none"
    if regime_adj:
        parts = []
        if drift_adj:
            parts.append(f"drift {drift_adj:+.1%}")
        if vol_adj != 1.0:
            parts.append(f"vol x{vol_adj:.2f}")
        if parts:
            adj_desc = ", ".join(parts)

    return f"""### SCENARIO ANALYSIS ({ticker}-Specific, Quantitative)

**Pre-computed targets** — generated via regime-conditional block-bootstrap ({n_sims:,} simulations, {horizon} horizon, realized vol {vol}%, regime adjustments: {adj_desc}).

**IMPORTANT — Use the price targets and probabilities below exactly as given.** They are derived from the historical return distribution conditioned on the current macro regime. Do NOT change the numbers. Your job is to provide the **Key Trigger** narrative for each row — a specific, plausible catalyst from the news, calendar, or fundamental data that would drive {ticker} to that level.

| Scenario | Probability | Key Trigger (YOU FILL THIS IN) | {ticker} Target (3M) | {ticker} Price Range |
|----------|-------------|-------------------------------|---------------------|---------------------|
| BULL | {bp} | [provide a specific catalyst that would drive {ticker} to {bt}] | {bt} | {br} |
| BASE | {bap} | [current trajectory / status quo narrative] | {bat} | {bar_} |
| BEAR | {bep} | [provide a specific risk catalyst that would push {ticker} to {bet}] | {bet} | {ber} |
| TAIL RISK | {tp} | [provide an extreme but plausible scenario for {ticker} at {tt}] | {tt} | {tr} |

**Rules:**
1. Do NOT modify the price targets, ranges, or probabilities — they are quantitatively derived.
2. **Key Trigger must justify the target:** Explain *why* {ticker} reaches that level. Reference specific upcoming events from the catalyst calendar, earnings, macro releases, or sector dynamics.
3. If a target seems stretched relative to recent history, acknowledge that in the trigger (e.g., "would require [specific condition] — historically rare but regime-consistent").
4. Triggers should be specific to {ticker}, not generic macro platitudes.
5. **Add one Expected value line** (CFA-correct): E[V] = (Bull% × Bull target) + (Base% × Base target) + (Bear% × Bear target) + (Tail% × Tail target) = $X.XX
"""


def llm_analyze_regime(
    *,
    settings: Settings,
    domain: str,
    snapshot: dict[str, Any] | Any,
    regime_label: str | None = None,
    regime_description: str | None = None,
    include_news: bool = True,
    include_prices: bool = True,
    include_calendar: bool = True,
    model: str | None = None,
    temperature: float = 0.25,
    ticker: str | None = None,
) -> str:
    """
    Universal LLM regime analyst - Research Grade.
    
    Provides PhD-level macro analysis for any domain (volatility, commodities, rates, etc.)
    with real-time news (cited), prices, and economic calendar integration.
    
    Output includes:
    - Regime status with confidence
    - Key metrics with data sources
    - News-driven context with citations [1], [2], etc.
    - Outlook with specific catalysts
    - Trade expressions
    - Sources section for research compliance
    
    Args:
        settings: App settings with API keys
        domain: Domain name (volatility, commodities, rates, funding, monetary, fiscal, etc.)
        snapshot: Current regime snapshot data (dict or dataclass)
        regime_label: Optional regime classification label
        regime_description: Optional regime description
        include_news: Fetch real-time news for the domain
        include_prices: Fetch real-time prices for related tickers
        include_calendar: Include upcoming economic events
        model: OpenAI model override
        temperature: LLM temperature
    
    Returns:
        Formatted analysis text with citations for terminal display
    """
    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment / .env")
    
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package is not installed. Try: pip install openai") from e
    
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.OPENAI_BASE_URL)
    chosen_model = model or settings.openai_model
    
    # Convert snapshot to dict if needed
    if hasattr(snapshot, "model_dump"):
        snapshot_dict = snapshot.model_dump()
    elif hasattr(snapshot, "__dataclass_fields__"):
        snapshot_dict = asdict(snapshot)
    elif hasattr(snapshot, "__dict__"):
        snapshot_dict = {k: v for k, v in snapshot.__dict__.items() if not k.startswith("_")}
    else:
        snapshot_dict = dict(snapshot) if snapshot else {}
    
    # Get domain context
    ctx = DOMAIN_CONTEXT.get(domain.lower(), {
        "description": f"{domain} regime analysis",
        "key_metrics": [],
        "related_tickers": [],
        "news_keywords": [],
        "macro_keywords": [],
        "trading_focus": "general macro trading",
        "data_sources": [],
        "sector_implications": {},
    })
    
    # Build payload
    now_utc = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "report_timestamp": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
        "domain": domain.upper(),
        "domain_description": ctx["description"],
        "trading_focus": ctx["trading_focus"],
        "primary_data_sources": ctx.get("data_sources", []),
        "snapshot": snapshot_dict,
        "regime": {
            "label": regime_label or "unclassified",
            "description": regime_description or "",
        },
    }
    
    # Add sector implications if available
    sector_impl = ctx.get("sector_implications", {})
    if sector_impl:
        payload["sector_implications"] = sector_impl
    
    # Add canonical scenario framework (only for regime analysis, not ticker-specific)
    if not ticker:
        payload["scenario_framework"] = SCENARIO_FRAMEWORK
    else:
        payload["ticker"] = ticker
    
    # Fetch real-time data with source tracking
    sources_used = []
    
    if include_prices:
        price_data = _fetch_realtime_prices(settings, domain)
        if price_data:
            payload["realtime_prices"] = price_data
            sources_used.append(f"FMP Real-time Quotes ({price_data.get('asof', 'N/A')})")
    
    if include_news:
        if ticker:
            # Ticker-specific news fetching
            ticker_news = _fetch_ticker_news(settings, ticker, max_items=15, lookback_days=7)
            if ticker_news:
                payload["news_articles"] = ticker_news
                alpaca_count = sum(1 for n in ticker_news if n.get("provider") == "Alpaca")
                fmp_count = sum(1 for n in ticker_news if n.get("provider") == "FMP")
                content_count = sum(1 for n in ticker_news if n.get("content"))
                if alpaca_count > 0:
                    sources_used.append(f"Alpaca News API ({alpaca_count} articles, {content_count} with full content)")
                if fmp_count > 0:
                    sources_used.append(f"FMP News API ({fmp_count} articles)")
        else:
            # Domain-wide news for regime analysis
            unified_news = _fetch_domain_news(settings, domain, max_items=15, lookback_days=5)
            if unified_news:
                payload["news_articles"] = unified_news
                alpaca_count = sum(1 for n in unified_news if n.get("provider") == "Alpaca")
                fmp_count = sum(1 for n in unified_news if n.get("provider") == "FMP")
                content_count = sum(1 for n in unified_news if n.get("content"))
                if alpaca_count > 0:
                    sources_used.append(f"Alpaca News API ({alpaca_count} articles, {content_count} with full content)")
                if fmp_count > 0:
                    sources_used.append(f"FMP News API ({fmp_count} articles)")
        
        # Also fetch general macro news for broader context
        macro_news = _fetch_general_macro_news(settings, domain, max_items=8)
        if macro_news:
            payload["macro_news"] = macro_news
            sources_used.append(f"FMP General News ({len(macro_news)} articles)")
    
    if include_calendar:
        events = _fetch_calendar_events(settings, days_ahead=14, max_items=12)
        if events:
            payload["upcoming_events"] = events
            sources_used.append(f"FMP Economic Calendar ({len(events)} events)")
    
    # Add historical context for quantitative calibration
    historical_context = _compute_historical_context(settings, domain, snapshot_dict)
    if historical_context:
        payload["historical_context"] = historical_context
        sources_used.append("Historical calibration data (52-week ranges, typical event moves)")
    
    payload["sources_consulted"] = sources_used
    
    payload_json = json.dumps(payload, indent=2, default=str)
    
    # Build ticker-specific or regime-generic prompt sections
    if ticker:
        subject = f"{ticker}"
        scenario_section = _build_ticker_scenario_section(ticker, snapshot_dict)
        catalyst_section = f"""### CATALYST CALENDAR ({ticker} Impact)

**Company-specific catalysts first:** Next earnings date, product/contract announcements, analyst days — these matter most for a single name. Then macro events.

**Do NOT fabricate specific dollar impacts** for macro events on {ticker} (e.g. avoid "{ticker} -$2.00 if Retail Sales < 0.2%" unless you have empirical data). Instead use:
- **Qualitative direction**: "Retail sales miss would likely pressure cyclicals; {ticker} could underperform."
- Or **sector sensitivity**: "Industrials (XLI) typically move ±0.5% on retail sales misses; {ticker} beta to sector ~1.8x."
- Or **historical**: "In the last 4 retail sales reports, {ticker} moved an average of ±X% on release day" (only if supported by data).

| Date | Event | Type (Co/Macro) | Expected | Impact if Miss | Impact if Beat |
|------|-------|-----------------|----------|----------------|-----------------|
| [earnings date if known] | [company event] | Company | — | [qualitative] | [qualitative] |
| [date] | [macro event] | Macro | [forecast] | [directional, no fabricated $] | [directional] |
"""
        trade_section = f"""### TRADE EXPRESSIONS ({ticker}-Focused)

Provide **4-5 specific trade ideas for {ticker}**. **Do NOT suggest outright shorting shares** (unlimited risk; inappropriate for a research tool). Use long equity, options, or relative value only.

**Required:**
- 1-2 **Long** directional (shares or calls) with entry/target/stop
- 1-2 **Options** (spreads, straddles, covered calls) — for each options trade include: **max risk in dollars**, **key Greeks (delta, theta)**, **break-even at expiry**
- 1 **Relative value** or pair trade ({ticker} vs related instrument)
- **Position sizing:** For at least one idea, add portfolio context: e.g. "At 5% allocation with $140 stop, max loss = 9.7% × 5% = 0.49% of portfolio."
- **Options:** State "Max risk = net debit. Max gain = [width] - debit." and delta/theta/days to expiry where relevant.

**For each trade:**
| Direction | Instrument | Strategy | Conviction | Entry | Target | Stop | Max Risk / Greeks | Timeframe |
|-----------|-----------|----------|------------|-------|--------|------|-------------------|-----------|
| Long | {ticker} or option | [detail] | HIGH/MED/LOW | $XX | $XX | $XX | [dollar risk or delta/theta] | Xd/Xw |
"""
    else:
        subject = domain.upper()
        scenario_section = """### SCENARIO ANALYSIS (Monte Carlo Aligned)

**REQUIRED**: Use scenarios from the `scenario_framework` in the data payload. These match our Monte Carlo simulations for institutional consistency.

Select **4-5 scenarios** most relevant to current conditions and assign probability weights:

| Scenario | Probability | Key Trigger | SPX Target (3M) | TLT Target (3M) | VIX Target |
|----------|-------------|-------------|-----------------|-----------------|------------|
| [scenario_label] | X% | [specific catalyst from current news/data] | [from framework] | [from framework] | [from framework] |

**Rules for scenario selection:**
1. Always include BASE_CASE with residual probability
2. Include at least one RISK_OFF / CREDIT_STRESS / RATES_SHOCK (tail risk)
3. Include the scenario most consistent with current regime
4. Probabilities must sum to 100%
5. Adjust base probabilities based on current metrics (e.g., if VIX is 8th percentile, increase VOL_CRUSH probability)

**Available scenarios** (from Monte Carlo framework):
- GOLDILOCKS: Soft landing, +8% drift, VIX ~18
- STAGFLATION: Inflation + weak growth, -2% drift, VIX ~25
- RISK_OFF: Sharp crash, -25% drift, VIX 35-50
- RATES_SHOCK: Duration pain, -10% drift, TLT -8-15%
- CREDIT_STRESS: HY spreads blow out, -15% drift
- SLOW_BLEED: Grinding down, -8% drift, VIX ~19
- VOL_CRUSH: Melt-up, IV collapse, +10% drift, VIX 12-14
- BASE_CASE: Neutral, +5% drift, VIX ~20
"""
        catalyst_section = """### CATALYST CALENDAR (Event-Driven)

For each upcoming calendar event, estimate **market impact**:

| Date | Event | Expected | Consensus Miss Impact | Consensus Beat Impact |
|------|-------|----------|----------------------|----------------------|
| [date] | [event] | [forecast if available] | [direction + magnitude] | [direction + magnitude] |

*Be specific: "10Y +15-20bps if CPI > 3.5%" not just "rates could rise"*
"""
        trade_section = """### TRADE EXPRESSIONS (Cross-Asset)

Provide **7 specific trade ideas** spanning multiple asset classes. Use the sector_implications data when available.

**Required mix:**
- 2-3 **Sector ETF plays** (XLF, XLU, XLE, XLI, etc.) based on regime implications
- 1-2 **Fixed income/duration** plays if rates matter
- 1-2 **Equity index** plays (SPY, QQQ, IWM) or factor plays (MTUM, USMV, QUAL)
- 1 **Hedge/defensive** idea (inverse ETF, gold, puts, or defensive sector)

**For each trade:**
| Direction | Ticker | Sizing | Conviction | Entry Trigger | Target | Stop |
|-----------|--------|--------|------------|---------------|--------|------|
| Long/Short | [specific ticker] | X% | HIGH/MED/LOW | [condition] | [price or %] | [price or %] |

*Reference sector_implications from the data payload to justify sector picks.*
"""

    prompt = f"""You are a PhD-level macro strategist at a top-tier hedge fund (Citadel/Renaissance caliber) writing a **research brief** on {subject}.

You are given comprehensive data including:
- Quantitative snapshot with key metrics
- Real-time prices with timestamps  
- News articles from multiple providers (FMP + Alpaca) with sources
- Some articles include FULL CONTENT for deeper analysis
- Economic calendar events with dates
- **Historical context for calibration:**
  - 52-week ranges for key instruments (current position within range)
  - Typical event-day moves for major releases (CPI, FOMC, NFP, etc.)
  - Current volatility regime (for sizing move estimates)
- Data source provenance

Your task: Produce a **research-grade brief** suitable for portfolio managers.{' Focus ALL analysis on ' + ticker + ' — not generic market indices.' if ticker else ''}

## OUTPUT FORMAT

### REGIME STATUS
- Current regime: [label] — [1-line interpretation]. **Source signal:** [e.g. "based on: ISM expanding, earnings beats broadening, VIX below 20th percentile"] so the reader knows how the regime was derived.
- Confidence: [HIGH/MEDIUM/LOW] — [rationale based on signal consistency]

### THESIS SUMMARY
Replace a repeat of the structured data with **2-3 sentences of synthesis**: WHY this stock is interesting right now, what the market is pricing in, and what could change the view. Do NOT restate price, market cap, or 52-week range — that is already in the tables above. Synthesize and add insight.

### NEWS & MARKET CONTEXT
Synthesize the provided news into 5-8 bullets. **Cite sources using [N] format** (N = article index).
- What's driving current prices? Reference specific articles.
- Key themes emerging across multiple sources
- Any analyst calls, price targets, or research notes mentioned in content?
- Notable quotes from articles (if full content available)
- Divergences between news narrative and quant signals?

{scenario_section}
{catalyst_section}
### HISTORICAL CONTEXT

Reference 1-2 **historical analogs** if applicable:
- When was the last time we saw similar conditions{'for ' + ticker if ticker else ''}?
- What happened next? (with timeframes and magnitudes)
- What's different this time?

{trade_section}
### RISKS & INVALIDATION
4-5 bullets. **Be specific; use qualitative language — do NOT fabricate specific dollar impacts** (e.g. avoid "CPI would push [ticker] -$2.00" unless you have empirical data). Frame as **"What would make me wrong?"** (strongest bear case, intellectual honesty).

- What would **invalidate the base view**? A specific level or release is fine; for macro events use **directional** language (e.g. "Higher-than-expected inflation would pressure growth valuations broadly; [ticker] is exposed given negative earnings and high multiple").
- Tie risks to **named calendar events** and **idiosyncratic** risks (sector, supply, positioning, concentration). Include **concentration risk** if relevant (e.g. % of revenue from top customers, key contracts).
- Do NOT use generic bullets without a concrete trigger. Do NOT invent precise $ move amounts for macro events on single names.

### SOURCES
List the primary data sources and APIs consulted for this analysis.

## RULES
1. **Cite news articles as [N]** where N is the index in news_articles array
2. When articles have FULL CONTENT, extract key insights and quotes
3. Use ONLY provided data — do not hallucinate facts
4. Be direct and opinionated — this is for trading decisions
5. If data is missing, explicitly note the gap and adjust confidence
6. **QUANTITATIVE RIGOR**: Every outlook statement needs a probability, target/range, and timeframe. Calibrate moves using snapshot volatility and historical percentiles.
7. Total length: 700-1000 words (tables count toward limit)
{('8. ALL scenarios, catalysts, and trade ideas MUST reference ' + ticker + ' directly. Do NOT substitute generic SPX/TLT/VIX targets.') if ticker else ''}

## IMPORTANT RULES FOR ANALYSIS (CFA-aligned)
1. **Never restate** data already in the structured tables (price, market cap, 52-week range). Your job is to SYNTHESIZE and provide INSIGHT, not summarize.
2. **Price targets** must include the valuation methodology (e.g. "based on 25x forward EV/EBITDA" or "based on DCF with 12% WACC"). If targets are pre-computed (quant scenarios), say so and add E[V] = sum(probability × target) in one line.
3. **Scenario probabilities** must be justified with historical base rates or analytical reasoning, not arbitrary. If same probabilities for every ticker, they are useless.
4. **Never fabricate** specific dollar price impacts for macro events on individual stocks unless you have empirical data. Use qualitative directional language.
5. Identify the **1-2 metrics that matter MOST** for this company right now (e.g. pre-profit: revenue growth and path to profitability; mature: earnings growth and capital returns).
6. Frame risks as **"what would make the bull thesis wrong"** and include concentration risk where relevant.
7. For **options** suggestions: always include max risk in dollars, key Greeks (delta, theta), and break-even price at expiry.

## DATA PAYLOAD
{payload_json}
"""
    
    resp = client.chat.completions.create(
        model=chosen_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=float(temperature),
    )
    
    return (resp.choices[0].message.content or "").strip()


def quick_llm_summary(
    *,
    settings: Settings,
    title: str,
    data: dict[str, Any] | Any,
    question: str = "Summarize this data and its implications for trading.",
    model: str | None = None,
    temperature: float = 0.25,
) -> str:
    """
    Quick LLM summary for any data payload.
    
    Simpler than llm_analyze_regime - just pass data and a question.
    Useful for ad-hoc analysis.
    """
    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment / .env")
    
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package is not installed.") from e
    
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.OPENAI_BASE_URL)
    chosen_model = model or settings.openai_model
    
    # Convert to dict
    if hasattr(data, "model_dump"):
        data_dict = data.model_dump()
    elif hasattr(data, "__dataclass_fields__"):
        data_dict = asdict(data)
    else:
        data_dict = dict(data) if data else {}
    
    payload_json = json.dumps({"title": title, "data": data_dict}, indent=2, default=str)
    
    prompt = f"""You are a senior macro analyst. {question}

Be concise (under 200 words), actionable, and cite specific data points.

JSON:
{payload_json}
"""
    
    resp = client.chat.completions.create(
        model=chosen_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=float(temperature),
    )
    
    return (resp.choices[0].message.content or "").strip()

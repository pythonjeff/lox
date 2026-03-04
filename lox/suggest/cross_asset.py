"""
Cross-asset suggest: quant-level trade generation engine.

Orchestrates:
1. Rule-based candidate sourcing (scenarios + macro quadrant)
2. Playbook k-NN analog scoring (regime-conditioned forward returns)
3. Correlation vs benchmark
4. Optional Monte Carlo forward scoring (--deep)
5. Composite scoring (weighted combination of all signals)
6. Portfolio-aware adjustments (sector concentration, delta direction)

Output: ScoredCandidate list with conviction scores, expected returns, VaR.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Categorized candidate universe — each category can be filtered via --category / -c
UNIVERSE_BY_CATEGORY: dict[str, list[str]] = {
    "sectors": [
        "XLE", "XLF", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLK",
        "XOP", "OIH", "VDE",       # energy sub-sectors
        "KRE", "IAK",              # financials sub-sectors
        "XBI", "IBB",              # biotech
        "XME",                     # metals & mining
        "XLRE",                    # real estate sector
        "XLC",                     # communication services
    ],
    "commodities": [
        "GLD", "SLV", "GDX", "GDXJ",  # precious metals
        "USO", "BNO", "DBC", "PDBC",   # oil & broad commodities
        "COPX", "FCX",                  # copper / miners
        "UNG",                          # natural gas
        "WEAT", "CORN", "SOYB",        # agriculture
        "SIL", "RING",                  # silver miners, gold miners
    ],
    "rates": [
        "TLT", "IEF", "SHY", "BIL",   # duration spectrum
        "TIP", "STIP",                  # inflation-linked
        "TMF", "TBT",                   # leveraged rates
        "GOVT", "VGSH",                 # short-term govies
    ],
    "credit": [
        "HYG", "JNK",                  # high yield
        "LQD", "VCIT",                 # investment grade
        "BKLN", "SRLN",               # leveraged loans
        "EMB", "PCY",                  # EM debt
        "ANGL",                        # fallen angels
    ],
    "equity_indices": [
        "QQQ", "IWM", "DIA",          # US large/small/mid
        "EEM", "FXI", "EWZ", "EFA",   # international
        "VTI", "VOO", "RSP",          # broad US
        "ARKK", "ARKG",               # innovation / genomics
        "MDY",                         # mid-cap
    ],
    "dollar": [
        "UUP", "UDN",                 # dollar bull / bear
        "FXE", "FXY", "FXB",          # euro, yen, pound
        "FXA", "FXC",                  # AUD, CAD
        "EEM",                         # EM proxy
        "DXJ",                         # hedged Japan
    ],
    "real_estate": [
        "VNQ", "XLRE",                # broad REITs
        "XHB", "ITB",                 # homebuilders
        "IYR", "REM",                 # REIT variants
        "MORT",                        # mortgage REITs
    ],
    "volatility": [
        "VIXY", "UVXY", "SVXY",       # VIX ETPs
        "VXX",                         # VIX short-term
        "TAIL",                        # tail risk
        "USMV",                        # min vol equity
    ],
    "crypto": [
        "COIN", "MARA", "RIOT",       # crypto equities
        "BITO", "GBTC",               # BTC ETFs
        "ETHE",                        # ETH ETF
    ],
}

# Flat universe (all categories combined, deduplicated)
CANDIDATE_UNIVERSE = list(dict.fromkeys(
    ticker for tickers in UNIVERSE_BY_CATEGORY.values() for ticker in tickers
))

# Valid category names for CLI autocomplete
VALID_CATEGORIES = sorted(UNIVERSE_BY_CATEGORY.keys())

QUADRANT_TICKERS = {
    "STAGFLATION": [("LONG", "GLD"), ("LONG", "XLE"), ("SHORT", "TLT"), ("SHORT", "XLY")],
    "GOLDILOCKS": [("LONG", "QQQ"), ("LONG", "XLF"), ("LONG", "IWM")],
    "REFLATION": [("LONG", "XLE"), ("LONG", "DBC"), ("SHORT", "TLT"), ("LONG", "XLI")],
    "DISINFLATION": [("LONG", "TLT"), ("LONG", "QQQ"), ("LONG", "XLF")],
    "RISK_OFF": [("LONG", "TLT"), ("LONG", "GLD"), ("SHORT", "HYG"), ("LONG", "XLP")],
    "DEFAULT": [("LONG", "GLD"), ("LONG", "XLE"), ("LONG", "TLT"), ("LONG", "XLF")],
}

# Sector ETF mapping for portfolio concentration checks
SECTOR_MAP: dict[str, str] = {
    # Energy
    "XLE": "energy", "USO": "energy", "BNO": "energy", "XOP": "energy",
    "OIH": "energy", "VDE": "energy", "UNG": "energy",
    # Financials
    "XLF": "financials", "KRE": "financials", "IAK": "financials",
    # Healthcare / Biotech
    "XLV": "healthcare", "XBI": "healthcare", "IBB": "healthcare",
    # Industrials
    "XLI": "industrials",
    # Consumer
    "XLY": "consumer_disc", "XLP": "consumer_staples",
    # Utilities
    "XLU": "utilities",
    # Materials / Mining
    "XLB": "materials", "GDX": "materials", "GDXJ": "materials",
    "SLV": "materials", "SIL": "materials", "RING": "materials",
    "COPX": "materials", "FCX": "materials", "XME": "materials",
    # Tech
    "XLK": "tech", "QQQ": "tech", "ARKK": "tech", "ARKG": "tech", "XLC": "tech",
    # Commodities (broad)
    "GLD": "safe_haven", "DBC": "commodities", "PDBC": "commodities",
    "WEAT": "commodities", "CORN": "commodities", "SOYB": "commodities",
    # Duration / Rates
    "TLT": "duration", "IEF": "duration", "TIP": "duration", "STIP": "duration",
    "SHY": "duration", "BIL": "duration", "TMF": "duration", "TBT": "duration",
    "GOVT": "duration", "VGSH": "duration",
    # Credit
    "HYG": "credit", "LQD": "credit", "JNK": "credit", "VCIT": "credit",
    "BKLN": "credit", "SRLN": "credit", "EMB": "credit", "PCY": "credit",
    "ANGL": "credit",
    # Equity indices
    "IWM": "small_cap", "DIA": "large_cap", "VTI": "broad_equity",
    "VOO": "broad_equity", "RSP": "broad_equity", "MDY": "mid_cap",
    # International
    "EEM": "emerging", "FXI": "china", "EWZ": "emerging",
    "EFA": "developed_intl", "DXJ": "developed_intl",
    # Real estate / Housing
    "VNQ": "real_estate", "XLRE": "real_estate", "XHB": "housing",
    "ITB": "housing", "IYR": "real_estate", "REM": "real_estate", "MORT": "real_estate",
    # Dollar / FX
    "UUP": "usd", "UDN": "usd", "FXE": "fx", "FXY": "fx", "FXB": "fx",
    "FXA": "fx", "FXC": "fx",
    # Volatility
    "VIXY": "volatility", "UVXY": "volatility", "SVXY": "volatility",
    "VXX": "volatility", "TAIL": "volatility", "USMV": "volatility",
    # Crypto
    "COIN": "crypto", "MARA": "crypto", "RIOT": "crypto",
    "BITO": "crypto", "GBTC": "crypto", "ETHE": "crypto",
}


# Ticker descriptions for thesis generation
TICKER_DESC: dict[str, str] = {
    # Credit
    "VCIT": "intermediate IG corporates",
    "LQD": "IG corporate duration",
    "HYG": "high-yield credit",
    "JNK": "junk bonds",
    "EMB": "EM sovereign USD debt",
    "PCY": "EM sovereign local-currency debt",
    "ANGL": "fallen angels",
    "BKLN": "senior bank loans",
    "SRLN": "senior loans (floating rate)",
    # Rates
    "TLT": "long-duration Treasuries",
    "IEF": "intermediate Treasuries",
    "SHY": "short-duration Treasuries",
    "TIP": "TIPS",
    "STIP": "short TIPS",
    "TMF": "3x long bonds",
    "TBT": "2x short bonds",
    "BIL": "T-bills",
    "GOVT": "Treasury aggregate",
    "VGSH": "short-term govies",
    # Commodities
    "GLD": "gold",
    "SLV": "silver",
    "GDX": "gold miners",
    "GDXJ": "junior gold miners",
    "USO": "crude oil",
    "BNO": "Brent crude",
    "DBC": "broad commodities",
    "PDBC": "broad commodities",
    "COPX": "copper miners",
    "FCX": "copper/gold miner",
    "UNG": "natural gas",
    "WEAT": "wheat",
    "CORN": "corn",
    "SOYB": "soybeans",
    "SIL": "silver miners",
    "RING": "gold miners (global)",
    # Sectors
    "XLE": "energy",
    "XLF": "financials",
    "XLV": "healthcare",
    "XLI": "industrials",
    "XLY": "consumer discretionary",
    "XLP": "consumer staples",
    "XLU": "utilities",
    "XLB": "materials",
    "XLK": "tech",
    "XOP": "oil & gas E&P",
    "OIH": "oil services",
    "VDE": "energy (broad)",
    "KRE": "regional banks",
    "IAK": "insurance",
    "XBI": "biotech",
    "IBB": "biotech (large cap)",
    "XME": "metals & mining",
    "XLRE": "real estate",
    "XLC": "communication services",
    # Equity indices
    "QQQ": "Nasdaq-100",
    "IWM": "small-cap",
    "DIA": "Dow 30",
    "EEM": "emerging markets",
    "FXI": "China large-cap",
    "EWZ": "Brazil",
    "EFA": "developed intl ex-US",
    "VTI": "total US market",
    "VOO": "S&P 500",
    "RSP": "equal-weight S&P",
    "ARKK": "disruptive innovation",
    "ARKG": "genomics",
    "MDY": "mid-cap",
    # Dollar / FX
    "UUP": "dollar bull",
    "UDN": "dollar bear",
    "FXE": "euro",
    "FXY": "yen",
    "FXB": "pound sterling",
    "FXA": "Aussie dollar",
    "FXC": "Canadian dollar",
    "DXJ": "Japan hedged equity",
    # Real estate
    "VNQ": "REITs",
    "XHB": "homebuilders",
    "ITB": "homebuilders",
    "IYR": "real estate",
    "REM": "mortgage REITs",
    "MORT": "mortgage REITs",
    # Volatility
    "VIXY": "VIX short-term futures",
    "UVXY": "2x VIX",
    "SVXY": "short VIX",
    "VXX": "VIX short-term",
    "TAIL": "tail-risk hedge",
    "USMV": "min-vol equity",
    # Crypto
    "COIN": "Coinbase",
    "MARA": "Bitcoin miner",
    "RIOT": "Bitcoin miner",
    "BITO": "Bitcoin futures ETF",
    "GBTC": "Bitcoin trust",
    "ETHE": "Ethereum trust",
}


def _regime_snippet(regime_state, ticker: str) -> str:
    """Extract the most relevant regime read for a ticker's asset class."""
    sector = SECTOR_MAP.get(ticker, "")
    if sector in ("credit",) and regime_state.credit:
        lbl = (regime_state.credit.label or "").lower().replace("credit", "").strip()
        return f"credit {lbl}" if lbl else "credit regime"
    if sector in ("duration",) and regime_state.rates:
        lbl = (regime_state.rates.label or "").lower().replace("rates", "").strip()
        return f"rates {lbl}" if lbl else "rates regime"
    if sector in ("energy", "commodities") and regime_state.commodities:
        lbl = (regime_state.commodities.label or "").lower().replace("commodities", "").strip()
        return lbl if lbl else "commodities regime"
    if sector in ("volatility",) and regime_state.volatility:
        lbl = (regime_state.volatility.label or "").lower().replace("volatility", "").strip()
        return f"vol {lbl}" if lbl else "vol regime"
    if sector in ("safe_haven",):
        parts = []
        if regime_state.growth:
            lbl = (regime_state.growth.label or "").lower()
            if lbl:
                parts.append(f"growth {lbl}")
        if regime_state.volatility and regime_state.volatility.score >= 55:
            parts.append("elevated vol")
        return ", ".join(parts) if parts else ""
    if sector in ("emerging", "china"):
        if regime_state.usd:
            lbl = (regime_state.usd.label or "").lower()
            if lbl:
                return f"USD {lbl}"
        return ""
    # Fallback: macro quadrant
    quad = (regime_state.macro_quadrant or "").strip("— ").lower()
    return quad if quad else ""


def _build_thesis(
    ticker: str,
    direction: str,
    pb,
    regime_state,
    corr: float | None = None,
) -> str:
    """Build a concise, regime-aware thesis for a candidate.

    Rules:
    - Don't repeat numbers already in the table (E[R], Hit%, VaR, Sharpe)
    - Skip neutral/baseline regime context (it's in the banner)
    - Focus on qualitative differentiators
    """
    desc = TICKER_DESC.get(ticker, ticker)
    parts = [desc]

    # Regime context — only if directional (skip neutral/normal/baseline)
    regime_ctx = _regime_snippet(regime_state, ticker) if regime_state else ""
    if regime_ctx:
        ctx_lower = regime_ctx.lower()
        if not any(skip in ctx_lower for skip in ("neutral", "normal", "baseline", "mixed")):
            parts.append(regime_ctx)

    # Playbook signal quality (qualitative, not numbers)
    if pb and pb.n_analogs > 0:
        if pb.hit_rate >= 0.70:
            parts.append("strong analog signal")
        elif pb.hit_rate >= 0.53:
            parts.append("positive playbook")

    # Correlation qualifier
    if corr is not None:
        if direction == "LONG" and corr < 0.2:
            parts.append("strong diversifier")
        elif direction == "LONG" and corr < 0.4:
            parts.append("low benchmark corr")
        elif direction == "SHORT" and corr > 0.7:
            parts.append("effective hedge")

    return "; ".join(parts)


def _underlying_from_symbol(sym: str) -> str:
    """Extract underlying ticker from position symbol (handles options)."""
    from lox.utils.occ import parse_occ_option_full
    if "/" in sym:
        return sym.split("/")[0].strip().upper()
    if len(sym) > 12 and any(c.isdigit() for c in sym):
        parsed = parse_occ_option_full(sym)
        if parsed:
            return parsed.get("underlying", sym).upper()
        for i, c in enumerate(sym):
            if c.isdigit():
                return sym[:i].upper() if i > 0 else sym.upper()
    return sym.strip().upper()


def _portfolio_underlyings(positions: list) -> set[str]:
    """Extract set of underlying tickers from positions."""
    out = set()
    for p in positions:
        sym = str(getattr(p, "symbol", ""))
        if not sym:
            continue
        u = _underlying_from_symbol(sym)
        if u:
            out.add(u)
    return out


def _portfolio_sector_weights(positions: list) -> dict[str, float]:
    """Compute sector weights from portfolio positions."""
    import numpy as np
    sector_mv: dict[str, float] = {}
    total_mv = 0.0
    for p in positions:
        mv = abs(float(getattr(p, "market_value", 0) or 0))
        sym = str(getattr(p, "symbol", ""))
        underlying = _underlying_from_symbol(sym)
        sector = SECTOR_MAP.get(underlying, "other")
        sector_mv[sector] = sector_mv.get(sector, 0) + mv
        total_mv += mv
    if total_mv <= 0:
        return {}
    return {s: v / total_mv for s, v in sector_mv.items()}


def _compute_correlation_scores(
    px_df,
    benchmark: str,
    candidates: list[str],
    window: int = 60,
) -> dict[str, float]:
    """Compute rolling correlation of each candidate vs benchmark."""
    import numpy as np
    if px_df is None or px_df.empty or benchmark not in px_df.columns:
        return {}
    ret = px_df.pct_change().dropna()
    if benchmark not in ret.columns or len(ret) < window:
        return {}
    bench_ret = ret[benchmark]
    out = {}
    for sym in candidates:
        if sym not in ret.columns:
            continue
        try:
            corr = bench_ret.rolling(window).corr(ret[sym]).iloc[-1]
            if np.isfinite(corr):
                out[sym] = float(corr)
        except Exception:
            pass
    return out


def _apply_portfolio_adjustments(
    scored: list,
    positions: list,
    portfolio_greeks=None,
) -> list:
    """
    Apply portfolio-aware score adjustments:
    - Penalise candidates that increase sector concentration above 25%
    - Prefer direction that offsets portfolio delta skew
    - Bonus for candidates filling factor gaps
    """
    if not positions or not scored:
        return scored

    sector_weights = _portfolio_sector_weights(positions)
    max_sector = max(sector_weights.values()) if sector_weights else 0

    # Portfolio delta direction from Greeks (if available)
    net_delta = None
    if portfolio_greeks is not None:
        net_delta = getattr(portfolio_greeks, "net_delta", None)

    # Sectors already in portfolio
    covered_sectors = {s for s, w in sector_weights.items() if w > 0.05}
    all_sectors = set(SECTOR_MAP.values())
    gap_sectors = all_sectors - covered_sectors

    for cand in scored:
        ticker_sector = SECTOR_MAP.get(cand.ticker, "other")
        adjustment = 0.0

        # Penalise concentration: if this sector is already >25%, penalise
        if sector_weights.get(ticker_sector, 0) > 0.25:
            adjustment -= 10.0

        # Bonus for filling a gap sector
        if ticker_sector in gap_sectors:
            adjustment += 5.0

        # Delta direction preference
        if net_delta is not None:
            if net_delta > 200 and cand.direction == "SHORT":
                adjustment += 5.0
            elif net_delta < -100 and cand.direction == "LONG":
                adjustment += 5.0

        cand.composite_score = max(0.0, min(100.0, cand.composite_score + adjustment))

    scored.sort(key=lambda x: x.composite_score, reverse=True)
    return scored


def suggest_cross_asset(
    *,
    regime_state,
    positions: list,
    benchmark: str = "SPY",
    count: int = 3,
    settings=None,
    use_correlation: bool = True,
    deep: bool = False,
    category: str = "",
):
    """
    Quant-level trade suggestion engine.

    Returns list of ScoredCandidate (from lox.suggest.scoring) with
    conviction scores, expected returns, VaR, and thesis.

    Parameters
    ----------
    deep : bool
        If True, run per-candidate Monte Carlo simulation (slower, ~15-20s).
        Default False uses playbook analog returns only (~3-5s).
    category : str
        If set, restrict universe to one category (e.g. "sectors", "commodities",
        "dollar", "rates", "credit"). Empty string = all categories.
    """
    from lox.regimes.scenarios import evaluate_scenarios, SCENARIOS
    from lox.suggest.scoring import compute_composite_scores, SuggestResult

    excluded = _portfolio_underlyings(positions)

    # --- 1. Evaluate active scenarios ---
    try:
        active_scenarios = evaluate_scenarios(regime_state, SCENARIOS)
    except Exception as e:
        logger.warning("Scenario evaluation failed: %s", e)
        active_scenarios = []

    # --- 2. Resolve category filter ---
    cat_key = category.lower().strip() if category else ""
    if cat_key and cat_key in UNIVERSE_BY_CATEGORY:
        universe_pool = UNIVERSE_BY_CATEGORY[cat_key]
        cat_filter = set(universe_pool)
        cat_label = cat_key
    else:
        universe_pool = CANDIDATE_UNIVERSE
        cat_filter = None  # no filtering
        cat_label = "all"

    # --- 3. Collect raw candidates from scenarios + quadrant ---
    ideas: list[dict] = []
    seen: set[str] = set()

    for s in active_scenarios:
        for t in s.trades:
            ticker = t.ticker.upper()
            if ticker in excluded or ticker in seen:
                continue
            if cat_filter and ticker not in cat_filter:
                continue
            seen.add(ticker)
            direction = t.direction.upper()
            if direction in ("LONG", "SHORT"):
                ideas.append({
                    "ticker": ticker,
                    "direction": direction,
                    "thesis": f"{s.name}: {t.rationale}",
                    "source": "scenario",
                    "conviction": s.conviction,
                })

    macro_quad = (regime_state.macro_quadrant or "").upper()
    if not ideas and macro_quad:
        quad_key = "DEFAULT"
        for k in QUADRANT_TICKERS:
            if k in macro_quad or (k == "STAGFLATION" and "STAG" in macro_quad):
                quad_key = k
                break
        for direction, ticker in QUADRANT_TICKERS.get(quad_key, QUADRANT_TICKERS["DEFAULT"]):
            if ticker in excluded or ticker in seen:
                continue
            if cat_filter and ticker not in cat_filter:
                continue
            seen.add(ticker)
            ideas.append({
                "ticker": ticker,
                "direction": direction,
                "thesis": f"Macro {macro_quad}: regime-aligned exposure",
                "source": "quadrant",
                "conviction": "MEDIUM",
            })

    if not ideas:
        for direction, ticker in QUADRANT_TICKERS["DEFAULT"]:
            if ticker in excluded or ticker in seen:
                continue
            if cat_filter and ticker not in cat_filter:
                continue
            seen.add(ticker)
            ideas.append({
                "ticker": ticker,
                "direction": direction,
                "thesis": "Diversifier vs benchmark",
                "source": "default",
                "conviction": "LOW",
            })

    # Expand pool: add universe tickers not already in ideas (direction TBD by playbook)
    for ticker in universe_pool:
        if ticker in excluded or ticker in seen:
            continue
        seen.add(ticker)
        ideas.append({
            "ticker": ticker,
            "direction": "LONG",  # default; playbook may flip
            "thesis": f"Universe ({cat_label}) — scored by playbook",
            "source": "universe",
            "conviction": "LOW",
        })

    candidate_tickers = [i["ticker"] for i in ideas]

    # --- 3. Playbook scoring (k-NN analog returns) ---
    # Score scenario/quadrant tickers first (fast), then optionally score universe
    playbook_ideas: dict = {}
    _shared_price_panel = None
    try:
        from lox.regimes.feature_matrix import build_regime_feature_matrix
        from lox.data.market import fetch_equity_daily_closes
        from lox.ideas.macro_playbook import rank_macro_playbook

        feature_matrix = build_regime_feature_matrix(
            settings=settings, start_date="2015-01-01", refresh_fred=False,
        )

        # Priority tickers: scenario/quadrant first, then universe
        # When a category is selected, score all tickers in that category
        priority_tickers = [i["ticker"] for i in ideas if i["source"] in ("scenario", "quadrant")]
        if cat_key and cat_key in UNIVERSE_BY_CATEGORY:
            cat_tickers = [t for t in UNIVERSE_BY_CATEGORY[cat_key] if t not in excluded]
            priority_tickers = list(dict.fromkeys(priority_tickers + cat_tickers))
        elif len(priority_tickers) < count * 2:
            extra = [t for t in universe_pool if t not in set(priority_tickers) and t not in excluded]
            priority_tickers.extend(extra[:10])

        all_price_syms = list(set(priority_tickers + [benchmark]))
        start = (datetime.now() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")
        price_panel = fetch_equity_daily_closes(
            settings=settings,
            symbols=all_price_syms,
            start=start,
            refresh=False,
        )
        _shared_price_panel = price_panel
        price_aligned = price_panel.reindex(feature_matrix.index).ffill()

        pb_results = rank_macro_playbook(
            features=feature_matrix,
            prices=price_aligned,
            tickers=priority_tickers,
            horizon_days=20,
            k=120,
            min_matches=50,
        )
        playbook_ideas = {r.ticker: r for r in pb_results}
        logger.info("Playbook scored %d/%d tickers", len(playbook_ideas), len(priority_tickers))

        # Flip direction for universe candidates if playbook says bearish
        for idea in ideas:
            pb = playbook_ideas.get(idea["ticker"])
            if pb and idea["source"] == "universe":
                if pb.direction == "bearish":
                    idea["direction"] = "SHORT"
                    idea["thesis"] = "Universe candidate — bearish analog signal"

    except Exception as e:
        logger.warning("Playbook scoring failed (using scenario order): %s", e)
        import traceback
        logger.debug(traceback.format_exc())

    # --- 4. Correlation scoring ---
    # Reuse the playbook price panel when available to avoid duplicate FMP calls
    correlation_scores: dict = {}
    if use_correlation and settings:
        try:
            corr_tickers = [i["ticker"] for i in ideas if i["source"] in ("scenario", "quadrant")] or candidate_tickers[:15]
            if _shared_price_panel is not None and not _shared_price_panel.empty:
                correlation_scores = _compute_correlation_scores(_shared_price_panel, benchmark, corr_tickers, window=60)
            else:
                from lox.data.market import fetch_equity_daily_closes
                corr_start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
                px = fetch_equity_daily_closes(
                    settings=settings,
                    symbols=list(set([benchmark] + corr_tickers)),
                    start=corr_start,
                    refresh=False,
                )
                correlation_scores = _compute_correlation_scores(px, benchmark, corr_tickers, window=60)
        except Exception as e:
            logger.debug("Correlation fetch failed: %s", e)

    # --- 4b. Build smart theses for non-scenario candidates ---
    for idea in ideas:
        if idea["source"] in ("universe", "quadrant", "default"):
            pb = playbook_ideas.get(idea["ticker"])
            corr = correlation_scores.get(idea["ticker"])
            idea["thesis"] = _build_thesis(
                ticker=idea["ticker"],
                direction=idea["direction"],
                pb=pb,
                regime_state=regime_state,
                corr=corr,
            )

    # --- 5. Optional MC scoring (--deep) ---
    mc_scores: dict = {}
    if deep and settings:
        try:
            from lox.suggest.mc_scoring import score_candidates_mc
            # Only MC the top candidates by playbook to limit runtime
            mc_tickers = candidate_tickers[:15] if len(candidate_tickers) > 15 else candidate_tickers
            mc_scores = score_candidates_mc(
                tickers=mc_tickers,
                regime_state=regime_state,
                settings=settings,
                benchmark=benchmark,
            )
        except Exception as e:
            logger.warning("MC scoring failed: %s", e)

    # --- 6. Composite scoring ---
    scored = compute_composite_scores(
        candidates=ideas,
        playbook_ideas=playbook_ideas,
        mc_scores=mc_scores if deep else None,
        correlation_scores=correlation_scores,
        active_scenarios=active_scenarios,
        macro_quadrant=macro_quad,
        deep=deep and bool(mc_scores),
    )

    # --- 7. Portfolio-aware adjustments ---
    portfolio_greeks = None
    try:
        from lox.risk.greeks import compute_portfolio_greeks
        portfolio_greeks = compute_portfolio_greeks(settings)
    except Exception:
        pass

    scored = _apply_portfolio_adjustments(scored, positions, portfolio_greeks)
    scored = scored[:count]

    # --- 8. Position sizing, signal flags, conviction override ---
    from lox.suggest.sizing import (
        compute_realized_vols,
        size_positions,
        detect_signal_flags,
        apply_smart_conviction,
        compute_portfolio_impact,
    )

    realized_vols = {}
    if _shared_price_panel is not None:
        realized_vols = compute_realized_vols(
            _shared_price_panel,
            [c.ticker for c in scored],
        )

    acct_equity = 0.0
    current_delta = 0.0
    if portfolio_greeks is not None:
        acct_equity = getattr(portfolio_greeks, "account_equity", 0.0) or 0.0
        current_delta = getattr(portfolio_greeks, "net_delta", 0.0) or 0.0

    # Size each position
    size_positions(scored, realized_vols, acct_equity)

    # Detect signal flags and override conviction
    for cand in scored:
        pb = playbook_ideas.get(cand.ticker)
        pb_dir = pb.direction if pb else None
        raw_corr = correlation_scores.get(cand.ticker)
        cand.signal_flags = detect_signal_flags(cand, pb_direction=pb_dir, raw_corr=raw_corr)
        cand.conviction = apply_smart_conviction(cand, cand.signal_flags)

    # Re-size after conviction override (conviction affects size)
    size_positions(scored, realized_vols, acct_equity)

    # Portfolio impact
    impact = compute_portfolio_impact(scored, current_delta, acct_equity)

    return SuggestResult(
        scored=scored,
        portfolio_greeks=portfolio_greeks,
        portfolio_impact=impact,
    )

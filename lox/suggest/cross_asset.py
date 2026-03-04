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

CANDIDATE_UNIVERSE = [
    "XLE", "XLF", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLK",
    "GLD", "SLV", "GDX", "USO", "DBC",
    "TLT", "IEF", "HYG", "LQD", "TIP",
    "QQQ", "IWM", "EEM", "FXI",
    "UUP", "VNQ", "KRE", "XHB", "ITB",
    "VIXY", "UVXY", "SVXY",
]

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
    "XLE": "energy", "USO": "energy", "DBC": "commodities", "XOP": "energy",
    "XLF": "financials", "KRE": "financials",
    "XLV": "healthcare",
    "XLI": "industrials",
    "XLY": "consumer_disc",
    "XLP": "consumer_staples",
    "XLU": "utilities",
    "XLB": "materials", "GDX": "materials", "SLV": "materials",
    "XLK": "tech", "QQQ": "tech",
    "GLD": "safe_haven", "TLT": "duration", "IEF": "duration", "TIP": "duration",
    "HYG": "credit", "LQD": "credit",
    "IWM": "small_cap", "EEM": "emerging",
    "VNQ": "real_estate", "XHB": "housing", "ITB": "housing",
    "UUP": "usd",
    "VIXY": "volatility", "UVXY": "volatility", "SVXY": "volatility",
    "FXI": "china",
}


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
    """
    from lox.regimes.scenarios import evaluate_scenarios, SCENARIOS
    from lox.suggest.scoring import compute_composite_scores

    excluded = _portfolio_underlyings(positions)

    # --- 1. Evaluate active scenarios ---
    try:
        active_scenarios = evaluate_scenarios(regime_state, SCENARIOS)
    except Exception as e:
        logger.warning("Scenario evaluation failed: %s", e)
        active_scenarios = []

    # --- 2. Collect raw candidates from scenarios + quadrant ---
    ideas: list[dict] = []
    seen: set[str] = set()

    for s in active_scenarios:
        for t in s.trades:
            ticker = t.ticker.upper()
            if ticker in excluded or ticker in seen:
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
            seen.add(ticker)
            ideas.append({
                "ticker": ticker,
                "direction": direction,
                "thesis": "Diversifier vs benchmark",
                "source": "default",
                "conviction": "LOW",
            })

    # Expand pool: add universe tickers not already in ideas (direction TBD by playbook)
    for ticker in CANDIDATE_UNIVERSE:
        if ticker in excluded or ticker in seen:
            continue
        seen.add(ticker)
        ideas.append({
            "ticker": ticker,
            "direction": "LONG",  # default; playbook may flip
            "thesis": "Universe candidate — scored by playbook",
            "source": "universe",
            "conviction": "LOW",
        })

    candidate_tickers = [i["ticker"] for i in ideas]

    # --- 3. Playbook scoring (k-NN analog returns) ---
    # Score scenario/quadrant tickers first (fast), then optionally score universe
    playbook_ideas: dict = {}
    try:
        from lox.regimes.feature_matrix import build_regime_feature_matrix
        from lox.data.market import fetch_equity_daily_closes
        from lox.ideas.macro_playbook import rank_macro_playbook

        feature_matrix = build_regime_feature_matrix(
            settings=settings, start_date="2015-01-01", refresh_fred=False,
        )

        # Priority tickers: scenario/quadrant first, then universe if count > ideas
        priority_tickers = [i["ticker"] for i in ideas if i["source"] in ("scenario", "quadrant")]
        if len(priority_tickers) < count * 2:
            extra = [t for t in CANDIDATE_UNIVERSE if t not in set(priority_tickers) and t not in excluded]
            priority_tickers.extend(extra[:10])  # cap universe expansion

        all_price_syms = list(set(priority_tickers + [benchmark]))
        start = (datetime.now() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")
        price_panel = fetch_equity_daily_closes(
            settings=settings,
            symbols=all_price_syms,
            start=start,
            refresh=False,
        )
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
    correlation_scores: dict = {}
    if use_correlation and settings:
        try:
            from lox.data.market import fetch_equity_daily_closes
            corr_start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
            px = fetch_equity_daily_closes(
                settings=settings,
                symbols=list(set([benchmark] + candidate_tickers)),
                start=corr_start,
                refresh=False,
            )
            correlation_scores = _compute_correlation_scores(px, benchmark, candidate_tickers, window=60)
        except Exception as e:
            logger.debug("Correlation fetch failed: %s", e)

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

    return scored[:count]

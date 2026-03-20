"""
Two-pass opportunity scanner — the main orchestrator.

Pass 1: Quick scan ~550 tickers via batch FMP quotes.
         Filter to ~75 "interesting" tickers (big movers + volume surges).
Pass 2: Deep signal scoring (4 independent pillars).
         Composite scoring with regime-conditional weights + anti-staleness.

Usage:
    from lox.suggest.scanner import run_opportunity_scan
    result = run_opportunity_scan(settings=settings, count=10)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from lox.config import Settings
from lox.suggest.opportunity import ScoredOpportunity

logger = logging.getLogger(__name__)


@dataclass
class ScannerResult:
    """Full output from the opportunity scanner."""
    candidates: list[ScoredOpportunity] = field(default_factory=list)
    regime_headline: str = ""
    regime_confidence: float = 0.0
    composite_regime: str = ""
    universe_size: int = 0
    pass1_survivors: int = 0
    scan_timestamp: str = ""
    rotation_applied: bool = False


def _pass1_filter(
    quotes: list[dict[str, Any]],
    top_n: int = 75,
) -> list[dict[str, Any]]:
    """Filter batch quotes to the most interesting tickers.

    Scores each ticker on:
    - |changesPercentage| (big movers)
    - volume / avgVolume (flow surge)
    Takes union of top 25 by each dimension, capped at top_n.
    Ensures minimum diversity (at least 5 with volume data).
    """
    # Pre-compute scores
    for q in quotes:
        try:
            q["_abs_change"] = abs(float(q.get("changesPercentage", 0) or 0))
        except (ValueError, TypeError):
            q["_abs_change"] = 0.0

        try:
            vol = float(q.get("volume", 0) or 0)
            avg_vol = float(q.get("avgVolume", 0) or 0)
            q["_vol_surge"] = vol / avg_vol if avg_vol > 0 else 1.0
        except (ValueError, TypeError, ZeroDivisionError):
            q["_vol_surge"] = 1.0

    # ── Multi-dimensional selection ──
    # Take top from each dimension, plus any with combined signal strength.
    # More lenient than before to catch quality setups with moderate moves.
    seen: set[str] = set()
    survivors: list[dict[str, Any]] = []

    # Top movers (biggest |change|)
    by_change = sorted(quotes, key=lambda q: q["_abs_change"], reverse=True)
    for q in by_change[:35]:
        sym = str(q.get("symbol", "")).upper()
        if sym and sym not in seen:
            seen.add(sym)
            survivors.append(q)

    # Top volume surges (flow signal)
    by_volume = sorted(quotes, key=lambda q: q["_vol_surge"], reverse=True)
    for q in by_volume[:30]:
        sym = str(q.get("symbol", "")).upper()
        if sym and sym not in seen:
            seen.add(sym)
            survivors.append(q)

    # Combined: moderate change + elevated volume (the best setups)
    for q in quotes:
        sym = str(q.get("symbol", "")).upper()
        if sym not in seen and q["_abs_change"] > 0.8 and q["_vol_surge"] > 1.2:
            seen.add(sym)
            survivors.append(q)

    # Also include any ETFs from the macro basket (always worth scoring)
    from lox.suggest.reversion import CORE_UNIVERSE
    for q in quotes:
        sym = str(q.get("symbol", "")).upper()
        if sym not in seen and sym in set(CORE_UNIVERSE) and q["_abs_change"] > 0.3:
            seen.add(sym)
            survivors.append(q)

    # Sort by combined activity score
    survivors.sort(
        key=lambda q: q["_abs_change"] * max(1.0, q["_vol_surge"]),
        reverse=True,
    )

    return survivors[:top_n]


def run_opportunity_scan(
    *,
    settings: Settings,
    count: int = 10,
    refresh: bool = False,
    deep: bool = False,
    ticker: str = "",
    signal_filter: str = "",
    etf_only: bool = False,
) -> ScannerResult:
    """Main entry point for the opportunity scanner.

    Args:
        count: number of top candidates to return.
        deep: run Monte Carlo + extended flow analysis.
        ticker: single-ticker mode (bypass universe scan).
        signal_filter: filter by signal type (tailwind, flow, reversion, catalyst).
        etf_only: exclude individual stocks.
    """
    from lox.suggest.cross_asset import CANDIDATE_UNIVERSE, SECTOR_MAP
    from lox.suggest.opportunity import compute_opportunity_scores
    from lox.suggest.staleness import (
        compute_rotation_penalties,
        enforce_diversity,
        load_rotation_history,
        save_rotation_history,
    )
    from lox.suggest.track_record import (
        backfill_returns,
        get_weight_adjustments,
        log_recommendations,
    )

    # ── 0. Backfill track record returns from past recommendations ──
    try:
        updated = backfill_returns(settings)
        if updated:
            logger.info("Backfilled %d track record returns", updated)
    except Exception as e:
        logger.debug("Track record backfill failed: %s", e)

    # ── 1. Build universe ──
    if ticker:
        universe = [ticker.strip().upper()]
    else:
        from lox.universe.sp500 import build_scan_universe
        universe = build_scan_universe(settings)

    if etf_only:
        etf_set = set(CANDIDATE_UNIVERSE)
        universe = [t for t in universe if t in etf_set]

    # ── 2. Build regime state ──
    regime_state = None
    composite_regime = ""
    composite_confidence = 0.0
    try:
        from lox.regimes.features import build_unified_regime_state
        regime_state = build_unified_regime_state(settings=settings, refresh=refresh)
        if regime_state and getattr(regime_state, "composite", None):
            composite_regime = regime_state.composite.regime
            composite_confidence = regime_state.composite.confidence
    except Exception as e:
        logger.warning("Regime state build failed: %s", e)

    # ── 3. Pass 1: Batch quote scan ──
    from lox.altdata.fmp import fetch_batch_quotes_full
    all_quotes = fetch_batch_quotes_full(settings=settings, tickers=universe)

    if not all_quotes:
        return ScannerResult(
            universe_size=len(universe),
            scan_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # Build quote lookup
    quote_lookup: dict[str, dict[str, Any]] = {}
    for q in all_quotes:
        sym = str(q.get("symbol", "")).upper()
        if sym:
            quote_lookup[sym] = q

    # Filter pass 1 survivors
    if ticker:
        survivors = [quote_lookup.get(ticker.upper(), {})] if ticker.upper() in quote_lookup else []
        survivor_tickers = [ticker.upper()] if survivors else []
    else:
        survivors = _pass1_filter(all_quotes, top_n=75)
        survivor_tickers = [
            str(q.get("symbol", "")).upper()
            for q in survivors
            if q.get("symbol")
        ]

    if not survivor_tickers:
        return ScannerResult(
            universe_size=len(universe),
            pass1_survivors=0,
            scan_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ── 4. Fetch profiles for sector data ──
    etf_set = set(CANDIDATE_UNIVERSE)
    stock_tickers = [t for t in survivor_tickers if t not in etf_set]
    ticker_sectors: dict[str, str] = {}
    ticker_names: dict[str, str] = {}

    # ETFs get sector from SECTOR_MAP
    for t in survivor_tickers:
        if t in SECTOR_MAP:
            ticker_sectors[t] = SECTOR_MAP[t]
        if t in etf_set:
            from lox.suggest.cross_asset import TICKER_DESC
            ticker_names[t] = TICKER_DESC.get(t, t)

    # Stocks get sector from FMP profiles (batch, cached 7d)
    if stock_tickers:
        try:
            from lox.altdata.fmp import fetch_batch_profiles
            profiles = fetch_batch_profiles(settings=settings, tickers=stock_tickers)
            for sym, prof in profiles.items():
                if prof.sector:
                    ticker_sectors[sym] = prof.sector.lower()
                if prof.company_name:
                    ticker_names[sym] = prof.company_name
        except Exception as e:
            logger.debug("Batch profile fetch failed: %s", e)

    # ── 5. Pass 2: Score across 4 signal pillars ──

    # Pillar 1: Momentum
    from lox.suggest.signals.momentum import score_momentum
    momentum_signals, price_panel = score_momentum(
        settings=settings,
        tickers=survivor_tickers,
        refresh=refresh,
    )

    # Pillar 2: Flow (now uses price panel for money flow computation)
    from lox.suggest.signals.flow import score_flow
    flow_signals = score_flow(
        settings=settings,
        tickers=survivor_tickers,
        quote_data=quote_lookup,
        price_panel=price_panel,
        fetch_si=True,
    )

    # Pillar 3: Regime Alignment
    regime_signals: dict = {}
    if regime_state:
        from lox.suggest.signals.regime_alignment import score_regime_alignment
        regime_signals = score_regime_alignment(
            regime_state=regime_state,
            tickers=survivor_tickers,
            ticker_sectors=ticker_sectors,
        )

    # Pillar 4: Catalyst (uses quote data for gap detection + broad earnings calendar)
    from lox.suggest.signals.catalyst import score_catalyst
    catalyst_signals = score_catalyst(
        settings=settings,
        tickers=survivor_tickers,
        quote_data=quote_lookup,
        fetch_news=True,
        max_news_tickers=30,
    )

    # ── 6. Composite scoring ──
    rotation_history = load_rotation_history()
    rotation_penalties = compute_rotation_penalties(survivor_tickers, rotation_history)

    # Track record weight adjustments
    weight_adjustments = {}
    try:
        weight_adjustments = get_weight_adjustments()
    except Exception:
        pass

    scored = compute_opportunity_scores(
        tickers=survivor_tickers,
        momentum_signals=momentum_signals,
        flow_signals=flow_signals,
        regime_signals=regime_signals,
        catalyst_signals=catalyst_signals,
        quote_data=quote_lookup,
        ticker_names=ticker_names,
        ticker_sectors=ticker_sectors,
        etf_tickers=etf_set,
        composite_regime=composite_regime,
        rotation_penalties=rotation_penalties,
        weight_adjustments=weight_adjustments if weight_adjustments else None,
        regime_state=regime_state,
    )

    # ── 7. Anti-staleness: diversity enforcement ──
    scored = enforce_diversity(scored, max_per_category=3)

    # ── 8. Signal filter ──
    _FILTER_MAP = {
        "tailwind": "REGIME_TAILWIND",
        "flow": "FLOW_ACCELERATION",
        "reversion": "REVERSION_SETUP",
        "catalyst": "CATALYST_DRIVEN",
    }
    if signal_filter:
        target = _FILTER_MAP.get(signal_filter.lower(), signal_filter.upper())
        scored = [c for c in scored if c.signal_type == target]

    # Take top N
    top = scored[:count]

    # ── 9. Log to track record + rotation history ──
    if top:
        try:
            log_recommendations(top)
            save_rotation_history([c.ticker for c in top])
        except Exception as e:
            logger.debug("Track record logging failed: %s", e)

    # ── 10. Build headline ──
    headline = composite_regime or "UNKNOWN"
    if composite_confidence:
        headline = f"{headline} ({composite_confidence:.0%} conf)"

    return ScannerResult(
        candidates=top,
        regime_headline=headline,
        regime_confidence=composite_confidence,
        composite_regime=composite_regime,
        universe_size=len(universe),
        pass1_survivors=len(survivor_tickers),
        scan_timestamp=datetime.now(timezone.utc).isoformat(),
        rotation_applied=bool(rotation_penalties),
    )

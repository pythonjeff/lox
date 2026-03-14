"""
Mean-reversion screener — find overextended ETFs, attribute moves to macro
factors, assess reversion probability, and recommend equity vs options.

Pure-function module: no display code, no side effects beyond data fetching.

Usage:
    from lox.suggest.reversion import run_reversion_screen

    result = run_reversion_screen(settings=settings, universe="core")
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from lox.config import Settings

logger = logging.getLogger(__name__)

# ── Core macro ETF universe (~30 most liquid) ─────────────────────────────────
CORE_UNIVERSE: list[str] = [
    # Equity indices
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    # Sectors
    "XLE", "XLF", "XLI", "XLY", "XLP", "XLU", "XLK", "XLB", "XLV",
    # Rates
    "TLT", "IEF", "TIP",
    # Credit
    "HYG", "LQD",
    # Commodities
    "GLD", "SLV", "USO", "DBC",
    # Dollar
    "UUP",
    # Real estate
    "VNQ",
    # Volatility
    "VIXY",
]

# Factor → regime domain mapping (which regime pillar is most associated)
_FACTOR_TO_DOMAIN: dict[str, str] = {
    "equity_beta": "growth",
    "duration": "rates",
    "credit": "credit",
    "commodities": "commodities",
    "gold": "commodities",
    "vol": "volatility",
    "inflation": "inflation",
}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class MomentumScan:
    """Raw momentum scan result for one ticker."""
    ticker: str
    description: str
    current_price: float
    ret_5d: float
    ret_20d: float
    ret_60d: float
    zscore_20d: float
    rsi_14: float
    dist_200d_pct: float  # distance from 200d MA as %
    signal: str  # "EXTENDED_UP", "EXTENDED_DOWN", or "NEUTRAL"


@dataclass
class FactorAttribution:
    """Which macro factors drove a ticker's recent move."""
    ticker: str
    primary_factor: str
    primary_loading: float
    regime_drivers: list[tuple[str, str, float]]  # (domain, trend_dir, velocity_7d)
    attribution_text: str


@dataclass
class ReversionAssessment:
    """Probability and supporting evidence for mean reversion."""
    ticker: str
    reversion_score: float  # 0-100
    playbook_reversion: bool  # k-NN analogs show reversion?
    playbook_hit_rate: float
    playbook_exp_return: float
    factor_decelerating: bool
    thesis: str


@dataclass
class InstrumentReco:
    """Whether to use equity or options, and which."""
    ticker: str
    direction: str  # "LONG" or "SHORT"
    instrument: str  # "equity", "call", "put"
    dte_target: str
    delta_target: str
    iv_rank: float | None  # 0-100
    rv_20d: float | None
    iv_rv_ratio: float | None
    rationale: str
    notional: float
    pct_nav: float
    conviction: str  # HIGH / MEDIUM / LOW


@dataclass
class ReversionCandidate:
    """Complete candidate combining all analysis layers."""
    scan: MomentumScan
    attribution: FactorAttribution
    assessment: ReversionAssessment
    recommendation: InstrumentReco


@dataclass
class ReversionScreenResult:
    """Full output from the screener."""
    candidates: list[ReversionCandidate] = field(default_factory=list)
    all_scans: list[MomentumScan] = field(default_factory=list)
    composite_regime: str = ""
    composite_confidence: float = 0.0
    universe_size: int = 0
    extended_count: int = 0
    scan_timestamp: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rsi(closes: np.ndarray, period: int = 14) -> float:
    """Standard Wilder RSI."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss < 1e-12:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _zscore_rolling_return(closes: np.ndarray, lookback: int) -> float:
    """Z-score the most recent `lookback`-day return vs trailing 252d of overlapping returns."""
    if len(closes) < lookback + 1:
        return 0.0
    # Build rolling lookback-day return series
    rets = closes[lookback:] / closes[:-lookback] - 1.0
    if len(rets) < 10:
        return 0.0
    current = float(rets[-1])
    mu = float(np.mean(rets))
    std = float(np.std(rets, ddof=1))
    if std < 1e-9:
        return 0.0
    return (current - mu) / std


def _pct_return(closes: np.ndarray, days: int) -> float:
    """Simple N-day return."""
    if len(closes) < days + 1:
        return 0.0
    p0 = float(closes[-(days + 1)])
    p1 = float(closes[-1])
    return (p1 - p0) / p0 if p0 > 0 else 0.0


def _dist_from_ma(closes: np.ndarray, ma_period: int = 200) -> float:
    """% distance of current price from N-day SMA."""
    if len(closes) < ma_period:
        return 0.0
    ma = float(np.mean(closes[-ma_period:]))
    if ma < 1e-6:
        return 0.0
    return (float(closes[-1]) - ma) / ma


# ── Core functions ────────────────────────────────────────────────────────────

def scan_momentum(
    *,
    settings: Settings,
    universe: list[str],
    lookback: int = 20,
    threshold: float = 1.5,
    refresh: bool = False,
) -> tuple[list[MomentumScan], pd.DataFrame]:
    """
    Scan universe for momentum extremes.

    Returns (scans sorted by |z-score|, price_panel for reuse downstream).
    """
    from lox.data.market import fetch_equity_daily_closes
    from lox.suggest.cross_asset import TICKER_DESC

    start = (pd.Timestamp.now() - pd.DateOffset(days=400)).strftime("%Y-%m-%d")
    px = fetch_equity_daily_closes(settings=settings, symbols=universe, start=start, refresh=refresh)
    if px.empty:
        return [], px

    scans: list[MomentumScan] = []
    for ticker in universe:
        if ticker not in px.columns:
            continue
        col = px[ticker].dropna()
        if len(col) < 60:
            continue
        closes = col.values.astype(np.float64)

        ret_5d = _pct_return(closes, 5)
        ret_20d = _pct_return(closes, lookback)
        ret_60d = _pct_return(closes, 60)
        z = _zscore_rolling_return(closes, lookback)
        rsi = _rsi(closes)
        dist_200d = _dist_from_ma(closes, 200)
        current_price = float(closes[-1])

        if z > threshold or ret_20d > 0.15:
            signal = "EXTENDED_UP"
        elif z < -threshold or ret_20d < -0.15:
            signal = "EXTENDED_DOWN"
        else:
            signal = "NEUTRAL"

        desc = TICKER_DESC.get(ticker, ticker)
        scans.append(MomentumScan(
            ticker=ticker,
            description=desc,
            current_price=current_price,
            ret_5d=ret_5d,
            ret_20d=ret_20d,
            ret_60d=ret_60d,
            zscore_20d=z,
            rsi_14=rsi,
            dist_200d_pct=dist_200d,
            signal=signal,
        ))

    scans.sort(key=lambda s: abs(s.zscore_20d), reverse=True)
    return scans, px


def attribute_factors(
    *,
    scans: list[MomentumScan],
    regime_state,
) -> dict[str, FactorAttribution]:
    """Attribute each extended ticker's move to macro factors using static factor map + regime trends."""
    from lox.cli_commands.shared.book_impact import FACTOR_EXPOSURES

    out: dict[str, FactorAttribution] = {}
    for scan in scans:
        if scan.signal == "NEUTRAL":
            continue

        exposures = FACTOR_EXPOSURES.get(scan.ticker)
        if not exposures:
            out[scan.ticker] = FactorAttribution(
                ticker=scan.ticker,
                primary_factor="unknown",
                primary_loading=0.0,
                regime_drivers=[],
                attribution_text="No factor mapping available",
            )
            continue

        # Find dominant factor (highest absolute loading)
        primary_factor = max(exposures, key=lambda k: abs(exposures[k]))
        primary_loading = exposures[primary_factor]

        # Cross-reference with regime trend data
        regime_drivers: list[tuple[str, str, float]] = []
        trends = getattr(regime_state, "trends", {}) or {}
        for factor, domain in _FACTOR_TO_DOMAIN.items():
            if factor not in exposures or abs(exposures[factor]) < 0.2:
                continue
            trend = trends.get(domain)
            if trend is None:
                continue
            vel = getattr(trend, "velocity_7d", None) or 0.0
            trend_dir = getattr(trend, "trend_direction", "stable") or "stable"
            if abs(vel) > 0.1:
                regime_drivers.append((domain, trend_dir, vel))

        # Sort drivers by velocity magnitude
        regime_drivers.sort(key=lambda x: abs(x[2]), reverse=True)

        # Build attribution text
        parts = []
        primary_domain = _FACTOR_TO_DOMAIN.get(primary_factor, primary_factor)
        parts.append(f"{primary_domain} ({primary_factor}={primary_loading:+.1f})")
        for domain, trend_dir, vel in regime_drivers[:2]:
            parts.append(f"{domain} {trend_dir} (v={vel:+.1f})")

        out[scan.ticker] = FactorAttribution(
            ticker=scan.ticker,
            primary_factor=primary_factor,
            primary_loading=primary_loading,
            regime_drivers=regime_drivers,
            attribution_text=", ".join(parts),
        )
    return out


def assess_reversion(
    *,
    scans: list[MomentumScan],
    attributions: dict[str, FactorAttribution],
    regime_state,
    price_panel: pd.DataFrame,
    settings: Settings,
) -> dict[str, ReversionAssessment]:
    """Assess reversion probability using k-NN playbook analogs + factor velocity."""
    from lox.ideas.macro_playbook import rank_macro_playbook
    from lox.regimes.feature_matrix import build_regime_feature_matrix

    extended = [s for s in scans if s.signal != "NEUTRAL"]
    if not extended:
        return {}

    tickers = [s.ticker for s in extended]

    # Build feature matrix for k-NN (cached FRED calls)
    try:
        features = build_regime_feature_matrix(settings=settings, start_date="2013-01-01")
    except Exception as e:
        logger.warning("Could not build feature matrix: %s", e)
        features = pd.DataFrame()

    # Run playbook k-NN for reversion signal
    playbook_results: dict[str, object] = {}
    if not features.empty and not price_panel.empty:
        try:
            ideas = rank_macro_playbook(
                features=features,
                prices=price_panel,
                tickers=tickers,
                horizon_days=20,
                k=120,
                min_matches=40,
            )
            for idea in ideas:
                playbook_results[idea.ticker] = idea
        except Exception as e:
            logger.warning("Playbook scoring failed: %s", e)

    trends = getattr(regime_state, "trends", {}) or {}
    out: dict[str, ReversionAssessment] = {}

    for scan in extended:
        ticker = scan.ticker
        attr = attributions.get(ticker)

        # Playbook reversion: did analogs show the ticker reversing?
        pb = playbook_results.get(ticker)
        pb_reversion = False
        pb_hit_rate = 0.5
        pb_exp_return = 0.0
        pb_score = 50.0  # neutral

        if pb is not None:
            pb_exp_return = pb.exp_return
            pb_hit_rate = pb.hit_rate
            if scan.signal == "EXTENDED_UP" and pb.exp_return < -0.005:
                pb_reversion = True
                pb_score = min(100, 50 + abs(pb.exp_return) * 1000)
            elif scan.signal == "EXTENDED_DOWN" and pb.exp_return > 0.005:
                pb_reversion = True
                pb_score = min(100, 50 + abs(pb.exp_return) * 1000)
            elif scan.signal == "EXTENDED_UP" and pb.exp_return > 0.01:
                pb_score = max(0, 50 - pb.exp_return * 500)
            elif scan.signal == "EXTENDED_DOWN" and pb.exp_return < -0.01:
                pb_score = max(0, 50 - abs(pb.exp_return) * 500)

        # Factor deceleration: is the primary factor losing momentum?
        factor_decel = False
        decel_score = 50.0
        if attr and attr.primary_factor:
            domain = _FACTOR_TO_DOMAIN.get(attr.primary_factor, "")
            trend = trends.get(domain)
            if trend:
                vel = getattr(trend, "velocity_7d", None) or 0.0
                chg_7d = getattr(trend, "score_chg_7d", None) or 0.0
                # Extended up: factor decel if velocity is turning negative (stress easing)
                if scan.signal == "EXTENDED_UP" and vel < -0.3:
                    factor_decel = True
                    decel_score = min(100, 50 + abs(vel) * 20)
                elif scan.signal == "EXTENDED_DOWN" and vel > 0.3:
                    factor_decel = True
                    decel_score = min(100, 50 + abs(vel) * 20)

        # Momentum extreme score (how far past threshold)
        momentum_score = min(100, abs(scan.zscore_20d) * 25)

        # Composite reversion score: playbook 40%, factor decel 30%, momentum 30%
        reversion_score = pb_score * 0.4 + decel_score * 0.3 + momentum_score * 0.3
        reversion_score = max(0.0, min(100.0, reversion_score))

        # Build thesis
        direction = "down" if scan.signal == "EXTENDED_UP" else "up"
        thesis_parts = [f"{ticker} extended {scan.ret_20d:+.1%} ({scan.zscore_20d:+.1f}σ)"]
        if pb_reversion:
            thesis_parts.append(f"analogs show {direction} reversion ({pb_hit_rate:.0%} hit)")
        if factor_decel:
            thesis_parts.append("driving factor decelerating")
        thesis = "; ".join(thesis_parts)

        out[ticker] = ReversionAssessment(
            ticker=ticker,
            reversion_score=reversion_score,
            playbook_reversion=pb_reversion,
            playbook_hit_rate=pb_hit_rate,
            playbook_exp_return=pb_exp_return,
            factor_decelerating=factor_decel,
            thesis=thesis,
        )

    return out


def select_instrument(
    *,
    scan: MomentumScan,
    assessment: ReversionAssessment,
    price_panel: pd.DataFrame,
    settings: Settings,
    account_equity: float,
) -> InstrumentReco:
    """Choose equity vs options and size the position."""
    from lox.suggest.sizing import compute_realized_vols, CONVICTION_SCALE, DEFAULT_VOL

    direction = "SHORT" if scan.signal == "EXTENDED_UP" else "LONG"

    # Compute realized vol from price panel
    rvols = compute_realized_vols(price_panel, [scan.ticker], window=60)
    rv_20d_dict = compute_realized_vols(price_panel, [scan.ticker], window=20)
    rv = rvols.get(scan.ticker, DEFAULT_VOL)
    rv_20d = rv_20d_dict.get(scan.ticker, rv)

    # Try to get IV from option chain
    iv_rank: float | None = None
    iv_rv_ratio: float | None = None
    atm_iv: float | None = None

    try:
        from lox.data.alpaca import make_clients, fetch_option_chain, to_candidates
        _, data_client = make_clients(settings)
        chain = fetch_option_chain(data_client, scan.ticker, feed=settings.alpaca_options_feed)
        candidates = list(to_candidates(chain, scan.ticker))

        # Find near-ATM, 30-60 DTE options for IV estimate
        atm_ivs = []
        for c in candidates:
            if c.dte_days and 20 <= c.dte_days <= 70 and c.iv and c.iv > 0:
                if c.delta and 0.3 <= abs(c.delta) <= 0.7:
                    atm_ivs.append(c.iv)

        if atm_ivs:
            atm_iv = float(np.median(atm_ivs))
            iv_rv_ratio = atm_iv / rv if rv > 0.01 else None
            # Approximate IV rank from IV vs RV spread
            # Simple heuristic: if IV << RV, rank is low; if IV >> RV, rank is high
            if iv_rv_ratio is not None:
                iv_rank = max(0, min(100, (iv_rv_ratio - 0.5) * 100))

    except Exception as e:
        logger.debug("Option chain fetch failed for %s: %s", scan.ticker, e)

    # Instrument decision based on IV/RV
    instrument = "equity"
    dte_target = "—"
    delta_target = "—"
    rationale = ""

    if iv_rank is not None and atm_iv is not None:
        if iv_rank < 30:
            # IV cheap → buy options for leverage
            instrument = "put" if direction == "SHORT" else "call"
            dte_target = "30-60d"
            delta_target = "0.30-0.40"
            rationale = f"IV cheap (rank {iv_rank:.0f}), buy {instrument}s for leverage"
        elif iv_rank > 70:
            # IV expensive → prefer equity
            instrument = "equity"
            rationale = f"IV rich (rank {iv_rank:.0f}), use equity to avoid premium decay"
        else:
            # Middle ground — options if high conviction, else equity
            if assessment.reversion_score >= 65:
                instrument = "put" if direction == "SHORT" else "call"
                dte_target = "45-90d"
                delta_target = "0.35-0.50"
                rationale = f"IV moderate (rank {iv_rank:.0f}), high conviction → options"
            else:
                instrument = "equity"
                rationale = f"IV moderate (rank {iv_rank:.0f}), moderate conviction → equity"
    else:
        instrument = "equity"
        rationale = "No options data, default to equity"

    # Conviction from reversion score
    if assessment.reversion_score >= 70:
        conviction = "HIGH"
    elif assessment.reversion_score >= 45:
        conviction = "MEDIUM"
    else:
        conviction = "LOW"

    # Vol-target sizing
    conv_scale = CONVICTION_SCALE.get(conviction, 0.5)
    vol_budget = 0.01  # 1% annualized vol per position
    if account_equity > 0 and rv > 0:
        base_notional = (vol_budget * account_equity) / rv
        notional = min(base_notional * conv_scale, 0.05 * account_equity)
    else:
        notional = 0.0

    pct_nav = notional / account_equity if account_equity > 0 else 0.0

    return InstrumentReco(
        ticker=scan.ticker,
        direction=direction,
        instrument=instrument,
        dte_target=dte_target,
        delta_target=delta_target,
        iv_rank=iv_rank,
        rv_20d=rv_20d,
        iv_rv_ratio=iv_rv_ratio,
        rationale=rationale,
        notional=round(notional, 0),
        pct_nav=pct_nav,
        conviction=conviction,
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_reversion_screen(
    *,
    settings: Settings,
    universe: str = "core",
    threshold: float = 1.5,
    lookback: int = 20,
    count: int = 5,
    ticker: str = "",
    refresh: bool = False,
) -> ReversionScreenResult:
    """
    Main entry point. Screens universe for overextended tickers,
    attributes moves to macro factors, assesses reversion probability,
    and recommends equity vs options.
    """
    from lox.regimes.features import build_unified_regime_state

    # Resolve universe
    if ticker:
        tickers = [ticker.upper()]
    elif universe == "full":
        from lox.suggest.cross_asset import UNIVERSE_BY_CATEGORY
        tickers = []
        for cat_tickers in UNIVERSE_BY_CATEGORY.values():
            tickers.extend(cat_tickers)
        tickers = list(dict.fromkeys(tickers))  # dedupe preserving order
    else:
        tickers = list(CORE_UNIVERSE)

    # Build regime state (parallel FRED calls, cached)
    try:
        regime_state = build_unified_regime_state(settings=settings, refresh=refresh)
    except Exception as e:
        logger.warning("Regime state build failed: %s", e)
        regime_state = None

    composite_regime = ""
    composite_confidence = 0.0
    if regime_state and getattr(regime_state, "composite", None):
        composite_regime = regime_state.composite.regime
        composite_confidence = regime_state.composite.confidence

    # Step 1: Momentum scan
    scans, price_panel = scan_momentum(
        settings=settings,
        universe=tickers,
        lookback=lookback,
        threshold=threshold,
        refresh=refresh,
    )

    extended = [s for s in scans if s.signal != "NEUTRAL"]

    if not extended:
        return ReversionScreenResult(
            all_scans=scans,
            universe_size=len(tickers),
            extended_count=0,
            composite_regime=composite_regime,
            composite_confidence=composite_confidence,
            scan_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # Step 2: Factor attribution
    attributions = attribute_factors(scans=extended, regime_state=regime_state) if regime_state else {}

    # Step 3: Reversion assessment
    assessments = assess_reversion(
        scans=extended,
        attributions=attributions,
        regime_state=regime_state,
        price_panel=price_panel,
        settings=settings,
    ) if regime_state else {}

    # Step 4: Sort by reversion score, take top N
    extended.sort(key=lambda s: assessments.get(s.ticker, ReversionAssessment(
        ticker=s.ticker, reversion_score=0, playbook_reversion=False,
        playbook_hit_rate=0.5, playbook_exp_return=0, factor_decelerating=False, thesis="",
    )).reversion_score, reverse=True)
    top = extended[:count]

    # Step 5: Instrument selection (option chain fetch only for top N)
    account_equity = 0.0
    try:
        from lox.data.alpaca import make_clients
        trading, _ = make_clients(settings)
        acct = trading.get_account()
        account_equity = float(getattr(acct, "equity", 0) or 0)
    except Exception:
        pass

    candidates: list[ReversionCandidate] = []
    for scan in top:
        attr = attributions.get(scan.ticker, FactorAttribution(
            ticker=scan.ticker, primary_factor="unknown", primary_loading=0,
            regime_drivers=[], attribution_text="N/A",
        ))
        assess = assessments.get(scan.ticker, ReversionAssessment(
            ticker=scan.ticker, reversion_score=0, playbook_reversion=False,
            playbook_hit_rate=0.5, playbook_exp_return=0, factor_decelerating=False, thesis="N/A",
        ))
        reco = select_instrument(
            scan=scan,
            assessment=assess,
            price_panel=price_panel,
            settings=settings,
            account_equity=account_equity,
        )
        candidates.append(ReversionCandidate(
            scan=scan,
            attribution=attr,
            assessment=assess,
            recommendation=reco,
        ))

    return ReversionScreenResult(
        candidates=candidates,
        all_scans=scans,
        composite_regime=composite_regime,
        composite_confidence=composite_confidence,
        universe_size=len(tickers),
        extended_count=len(extended),
        scan_timestamp=datetime.now(timezone.utc).isoformat(),
    )

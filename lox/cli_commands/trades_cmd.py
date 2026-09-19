"""
lox trades — S&P 500 multi-signal scanner.

Universe: ~500 S&P 500 stocks
Score formula (additive, weights sum to 1.0):
    momentum  30% — 12-1 month cross-sectional rank (Jegadeesh-Titman)
    regime    25% — sector × macro regime fit
    analyst   25% — Wall Street net-buy ratio
    congress  20% — committee-aligned gov insider buy; 0 if no qualifying activity

Congress weight reduced from 30% to 20%: standalone backtests showed weak,
inconsistent edge (mean/median divergence, sub-50% hit rate at most horizons).
Momentum raised to 30%: stronger empirical signal across all market regimes.
Congress only scores if a committee-aligned official bought within 30 days.
A stock with no congressional activity can still score 0.70 on the other three.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import typer
from rich.console import Console
from rich.text import Text

app = typer.Typer(add_completion=False, help="S&P 500 multi-signal trade scanner")


# ── Weights ───────────────────────────────────────────────────────────────────
_W_MOMENTUM = 0.25
_W_REGIME   = 0.22
_W_ANALYST  = 0.22
_W_CONGRESS = 0.16
_W_FLOW     = 0.15


# ── Sector × Regime multipliers ───────────────────────────────────────────────
# Each (regime, GICS_sector) → float in [0.70, 1.40].
# Higher = better sector/regime alignment for a Buy idea.
_SECTOR_REGIME_MULT: dict[tuple[str, str], float] = {
    # RISK_ON — growth + tech leadership
    ("RISK_ON", "Information Technology"):    1.30,
    ("RISK_ON", "Communication Services"):    1.25,
    ("RISK_ON", "Consumer Discretionary"):    1.25,
    ("RISK_ON", "Financials"):                1.25,
    ("RISK_ON", "Industrials"):               1.25,
    ("RISK_ON", "Materials"):                 1.20,
    ("RISK_ON", "Energy"):                    1.15,
    ("RISK_ON", "Health Care"):               1.10,
    ("RISK_ON", "Real Estate"):               0.95,
    ("RISK_ON", "Consumer Staples"):          0.90,
    ("RISK_ON", "Utilities"):                 0.85,
    # REFLATION — commodity cycle, capex boom
    ("REFLATION", "Energy"):                  1.40,
    ("REFLATION", "Materials"):               1.30,
    ("REFLATION", "Industrials"):             1.25,
    ("REFLATION", "Financials"):              1.15,
    ("REFLATION", "Consumer Discretionary"):  1.05,
    ("REFLATION", "Information Technology"):  1.05,
    ("REFLATION", "Communication Services"):  1.00,
    ("REFLATION", "Health Care"):             0.95,
    ("REFLATION", "Consumer Staples"):        0.85,
    ("REFLATION", "Utilities"):               0.80,
    ("REFLATION", "Real Estate"):             0.80,
    # STAGFLATION — inflation sticky, growth fading
    ("STAGFLATION", "Energy"):                1.25,
    ("STAGFLATION", "Materials"):             1.15,
    ("STAGFLATION", "Utilities"):             1.10,
    ("STAGFLATION", "Health Care"):           1.10,
    ("STAGFLATION", "Consumer Staples"):      1.05,
    ("STAGFLATION", "Communication Services"):0.85,
    ("STAGFLATION", "Financials"):            0.85,
    ("STAGFLATION", "Real Estate"):           0.85,
    ("STAGFLATION", "Information Technology"):0.80,
    ("STAGFLATION", "Industrials"):           0.80,
    ("STAGFLATION", "Consumer Discretionary"):0.70,
    # RISK_OFF — flight to safety
    ("RISK_OFF", "Utilities"):                1.30,
    ("RISK_OFF", "Consumer Staples"):         1.25,
    ("RISK_OFF", "Health Care"):              1.20,
    ("RISK_OFF", "Real Estate"):              1.05,
    ("RISK_OFF", "Communication Services"):   0.85,
    ("RISK_OFF", "Information Technology"):   0.80,
    ("RISK_OFF", "Consumer Discretionary"):   0.75,
    ("RISK_OFF", "Industrials"):              0.75,
    ("RISK_OFF", "Materials"):                0.75,
    ("RISK_OFF", "Energy"):                   0.75,
    ("RISK_OFF", "Financials"):               0.70,
    # TRANSITION — compressed signals, no clear regime
    ("TRANSITION", "Energy"):                 0.90,
    ("TRANSITION", "Materials"):              0.90,
    ("TRANSITION", "Industrials"):            0.90,
    ("TRANSITION", "Financials"):             0.90,
    ("TRANSITION", "Information Technology"): 0.90,
    ("TRANSITION", "Communication Services"): 0.90,
    ("TRANSITION", "Consumer Discretionary"): 0.90,
    ("TRANSITION", "Health Care"):            0.90,
    ("TRANSITION", "Consumer Staples"):       0.90,
    ("TRANSITION", "Utilities"):              0.90,
    ("TRANSITION", "Real Estate"):            0.90,
}

_REGIME_DEFAULT_MULT: dict[str, float] = {
    "RISK_ON": 1.10, "REFLATION": 1.00,
    "STAGFLATION": 0.85, "RISK_OFF": 0.90, "TRANSITION": 0.90,
}

_FMP_SECTOR_NORM: dict[str, str] = {
    "Technology":             "Information Technology",
    "Healthcare":             "Health Care",
    "Financial Services":     "Financials",
    "Consumer Cyclical":      "Consumer Discretionary",
    "Consumer Defensive":     "Consumer Staples",
    "Basic Materials":        "Materials",
    "Communication Services": "Communication Services",
    "Real Estate":            "Real Estate",
}

_SECTOR_SHORT: dict[str, str] = {
    "Information Technology": "Tech",
    "Communication Services": "Comm",
    "Consumer Discretionary": "Disc",
    "Consumer Staples":       "Stpl",
    "Health Care":            "Hlth",
    "Financials":             "Fin",
    "Energy":                 "Engy",
    "Industrials":            "Ind",
    "Materials":              "Matl",
    "Utilities":              "Util",
    "Real Estate":            "RE",
}

# ── News sentiment keywords ───────────────────────────────────────────────────
_POS_WORDS = {
    "beat", "beats", "raises", "upgrade", "upgraded", "strong", "growth",
    "record", "surges", "surge", "gains", "gain", "bullish", "outperform",
    "exceeds", "increases", "approval", "approved", "launch", "launches",
    "award", "awarded", "win", "wins", "expands", "expansion", "buyback",
    "dividend", "raised", "positive", "accelerat",
}
_NEG_WORDS = {
    "miss", "misses", "downgrade", "downgraded", "weak",
    "falls", "fall", "drops", "drop", "concern", "risk", "loss", "losses",
    "warns", "warning", "decline", "declines", "investigation", "lawsuit",
    "recall", "reduces", "sell", "sells", "negative", "layoffs",
    "probe", "fine", "fined", "breach", "hack", "delay", "delays",
    "writedown", "write-off", "default", "bankruptcy",
}


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class TradeIdea:
    rank: int
    ticker: str
    company: str
    side: str                       # "Buy" (always for now)
    sector: str
    # Component scores — each 0.0 to 1.0
    congress_score: float           # 0 = no activity, 1 = strong cluster+committee buy
    regime_score: float             # sector × regime fit, normalized to 0-1
    analyst_score: float            # net buy ratio from analyst consensus
    momentum_score: float           # cross-sectional 12-1 month rank, 0-1
    flow_score: float               # cross-sectional 10-day money flow rank, 0-1
    insider_boost: float            # 1.0 (none) / 1.15-1.30 (corporate insider buying)
    final_score: float              # weighted sum × insider_boost
    # Display helpers
    regime_fit: str
    regime_multiplier: float        # raw mult for label
    analyst_consensus: str
    flow_direction: str             # ACCUM | NEUTRAL | DIST
    # Congress details (empty if no activity)
    has_congress: bool
    officials: list[str] = field(default_factory=list)
    cluster_count: int = 0
    committee_aligned: bool = False
    committee_note: str = ""
    total_usd: float = 0.0
    avg_lag_days: float = 0.0
    alpha_vs_spy: Optional[float] = None  # gov trade α gap: negative = stock hasn't moved yet
    # Populated in stage 2/3
    momentum_12_1: Optional[float] = None   # raw 12-1 month return, e.g. 0.23 = +23%
    short_interest_pct: Optional[float] = None
    insider_note: str = ""
    news_sentiment: str = "neutral"
    news_headline: str = ""
    # Trade plan — populated in stage 2 (price) + stage 3 (analyst target)
    current_price: Optional[float] = None
    vol_20d: Optional[float] = None          # 20-day realized daily return stdev
    analyst_target: Optional[float] = None   # FMP street consensus target
    composite_target: Optional[float] = None # model-derived target from score
    primary_target: Optional[float] = None   # analyst if available, else composite
    target_source: str = ""                  # "street" | "model"
    expected_upside_pct: Optional[float] = None
    stop_price: Optional[float] = None
    horizon: str = "2-3mo"
    conviction: str = "LOW"
    target_gap_note: str = ""                # e.g. "model sees +12%" when street/model diverge


# ── Score converters ──────────────────────────────────────────────────────────

def _normalize_sector(raw: str) -> str:
    return _FMP_SECTOR_NORM.get(raw, raw)


def _regime_mult(regime_key: str, sector: str) -> float:
    gics = _normalize_sector(sector)
    return _SECTOR_REGIME_MULT.get(
        (regime_key, gics),
        _REGIME_DEFAULT_MULT.get(regime_key, 0.90),
    )


def _regime_to_score(mult: float) -> float:
    """Normalize multiplier 0.70–1.40 → score 0.0–1.0."""
    return max(0.0, min(1.0, (mult - 0.70) / 0.70))


def _analyst_to_score(row: dict) -> float:
    """(strongBuy + buy) / total analysts — simple buy ratio."""
    try:
        sb = float(row.get("strongBuy", 0) or 0)
        b  = float(row.get("buy",       0) or 0)
        h  = float(row.get("hold",      0) or 0)
        s  = float(row.get("sell",      0) or 0)
        ss = float(row.get("strongSell",0) or 0)
        total = sb + b + h + s + ss
        if total == 0:
            return 0.50
        return (sb + b) / total
    except Exception:
        return 0.50


def _analyst_note(row: dict) -> str:
    try:
        consensus = str(row.get("consensus", "")).strip()
        sb = float(row.get("strongBuy", 0) or 0)
        b  = float(row.get("buy",       0) or 0)
        h  = float(row.get("hold",      0) or 0)
        s  = float(row.get("sell",      0) or 0)
        ss = float(row.get("strongSell",0) or 0)
        total = sb + b + h + s + ss
        if total == 0:
            return ""
        pct = int((sb + b) / total * 100)
        return f"{consensus} ({pct}% bull)"
    except Exception:
        return ""


def _compute_momentum_12_1(closes: "np.ndarray") -> Optional[float]:
    """12-1 month momentum: return from t-252 to t-21, skipping recent month."""
    if len(closes) < 253:
        return None
    p_start = float(closes[-252])
    p_end   = float(closes[-21])
    if p_start <= 0:
        return None
    return (p_end - p_start) / p_start


def _rank_normalize(values: dict[str, float]) -> dict[str, float]:
    """Cross-sectional rank → [0, 1]. Highest value = 1.0."""
    if not values:
        return {}
    tickers = sorted(values, key=lambda t: values[t])
    n = len(tickers)
    return {t: i / (n - 1) if n > 1 else 0.5 for i, t in enumerate(tickers)}


def _congress_to_score(sig) -> float:
    """QuiverSignal → 0-1 score. Congress trades only count if committee-aligned."""
    has_trump = "trump" in sig.sources
    has_aligned_congress = "congress" in sig.sources and sig.committee_aligned
    if not has_trump and not has_aligned_congress:
        return 0.0
    base = sig.score
    if sig.committee_aligned and sig.cluster_count >= 2:
        base *= 1.50
    elif sig.committee_aligned:
        base *= 1.15
    return min(1.0, base)


def _final_score(c: float, r: float, a: float, mom: float, flow: float = 0.50, ins: float = 1.0) -> float:
    return round(
        (_W_CONGRESS * c + _W_REGIME * r + _W_ANALYST * a + _W_MOMENTUM * mom + _W_FLOW * flow) * ins,
        3,
    )


def _fetch_short_interest(ticker: str, settings) -> Optional[float]:
    """Return short interest as % of float, or None. Cached 24h."""
    from datetime import timedelta as _td
    try:
        from lox.altdata.cache import cache_path, read_cache, write_cache
        import requests as _req
        key = f"si_{ticker}"
        p = cache_path(key)
        cached = read_cache(p, max_age=_td(hours=24))
        if cached is not None:
            return float(cached) if cached != "None" else None
        resp = _req.get(
            "https://financialmodelingprep.com/api/v4/short-interest",
            params={"symbol": ticker, "apikey": settings.fmp_api_key},
            timeout=10,
        )
        if resp.status_code != 200:
            write_cache(p, "None")
            return None
        data = resp.json()
        if isinstance(data, list) and data:
            for field in ("shortInterestPercentOfFloat", "shortPercentOfFloat", "shortPercentFloat"):
                val = data[0].get(field)
                if val is not None:
                    try:
                        si = float(val)
                        write_cache(p, si)
                        return si
                    except (ValueError, TypeError):
                        continue
        write_cache(p, "None")
        return None
    except Exception:
        return None


def _regime_fit_label(mult: float) -> str:
    if mult >= 1.25: return "strong tailwind"
    if mult >= 1.10: return "tailwind"
    if mult >= 0.88: return "neutral"
    if mult >= 0.70: return "headwind"
    return "strong headwind"


def _fit_color(label: str) -> str:
    return {
        "strong tailwind": "bold green",
        "tailwind":        "green",
        "neutral":         "white",
        "headwind":        "yellow",
        "strong headwind": "red",
    }.get(label, "white")


def _momentum_color(score: float) -> str:
    if score >= 0.70: return "green"
    if score >= 0.40: return "white"
    return "dim"


def _sentiment_tag(sentiment: str) -> tuple[str, str]:
    return {
        "positive": ("news+", "green"),
        "neutral":  ("news~", "dim"),
        "negative": ("news-", "red"),
    }.get(sentiment, ("news~", "dim"))


# ── Enrichment helpers ────────────────────────────────────────────────────────

def _score_sentiment(headlines: list[str]) -> tuple[str, str]:
    pos = neg = 0
    for h in headlines:
        words = set(re.sub(r"[^a-z ]", " ", h.lower()).split())
        pos += len(words & _POS_WORDS)
        neg += len(words & _NEG_WORDS)
    sentiment = "positive" if pos > neg else "negative" if neg > pos else "neutral"
    return sentiment, (headlines[0] if headlines else "")


def _fetch_news_sentiment(ticker: str, settings) -> tuple[str, str]:
    try:
        from lox.altdata.fmp import fetch_stock_news
        articles = fetch_stock_news(settings=settings, ticker=ticker, limit=7)
        ticker_lower = ticker.lower()
        articles = [a for a in articles if ticker_lower in (a.get("title") or "").lower()]
        headlines = [a.get("title", "") for a in articles if a.get("title")]
        return _score_sentiment(headlines)
    except Exception:
        return "neutral", ""


def _fetch_analyst_target(ticker: str, settings) -> Optional[float]:
    """Return FMP price-target-consensus (mean of Street targets), or None."""
    try:
        from datetime import timedelta as _td
        from lox.altdata.cache import cache_path, read_cache, write_cache
        import requests as _req
        key = f"pt_{ticker}"
        p = cache_path(key)
        cached = read_cache(p, max_age=_td(hours=12))
        if cached is not None:
            return float(cached) if cached != "None" else None
        r = _req.get(
            "https://financialmodelingprep.com/api/v4/price-target-consensus",
            params={"symbol": ticker, "apikey": settings.fmp_api_key},
            timeout=10,
        )
        if r.status_code != 200:
            write_cache(p, "None")
            return None
        data = r.json()
        if isinstance(data, list) and data:
            for k in ("targetConsensus", "targetMedian"):
                v = data[0].get(k)
                if v is not None:
                    try:
                        tgt = float(v)
                        if tgt > 0:
                            write_cache(p, tgt)
                            return tgt
                    except (ValueError, TypeError):
                        continue
        write_cache(p, "None")
        return None
    except Exception:
        return None


# ── Trade plan builders ───────────────────────────────────────────────────────

def _conviction_from_score(score: float) -> str:
    if score >= 0.75: return "HIGH"
    if score >= 0.60: return "MED"
    return "LOW"


def _composite_upside_pct(score: float) -> float:
    """Model-derived expected 3-month upside, tied to composite score."""
    base = 5.0 + (score - 0.30) * 30.0
    return max(3.0, min(22.0, base))


def _horizon_from_signals(momentum_score: float, flow_direction: str, regime_key: str) -> str:
    if regime_key in ("STAGFLATION", "RISK_OFF"):
        return "3-6mo"
    if flow_direction == "ACCUM" and momentum_score >= 0.70:
        return "1-2mo"
    return "2-3mo"


def _stop_pct_from_vol(vol_20d: Optional[float], conviction: str) -> float:
    """Stop distance below entry, as decimal. Vol-based when available, floor/cap by conviction."""
    if vol_20d is not None and vol_20d > 0:
        # ~1.5x monthly vol (20-day stdev of daily returns scaled to 20 days)
        import math
        stop = 1.5 * vol_20d * math.sqrt(20)
    else:
        stop = 0.08
    floor, cap = {
        "HIGH": (0.06, 0.12),
        "MED":  (0.05, 0.10),
        "LOW":  (0.04, 0.08),
    }.get(conviction, (0.05, 0.10))
    return max(floor, min(cap, stop))


def _build_trade_plan(idea: TradeIdea, regime_key: str) -> None:
    """Fill in composite_target, primary_target, upside, stop, conviction, horizon, gap_note."""
    idea.conviction = _conviction_from_score(idea.final_score)
    idea.horizon    = _horizon_from_signals(idea.momentum_score, idea.flow_direction, regime_key)

    if idea.current_price is None or idea.current_price <= 0:
        return

    # Composite (model) target from score
    comp_up = _composite_upside_pct(idea.final_score) / 100.0
    idea.composite_target = round(idea.current_price * (1 + comp_up), 2)

    # Primary target: analyst if it points up meaningfully, else composite
    street_up = None
    if idea.analyst_target and idea.analyst_target > 0:
        street_up = (idea.analyst_target - idea.current_price) / idea.current_price

    if street_up is not None and street_up > 0.02:
        idea.primary_target      = idea.analyst_target
        idea.target_source       = "street"
        idea.expected_upside_pct = round(street_up * 100, 1)
        # gap flag: material street/model divergence
        if abs(street_up * 100 - comp_up * 100) >= 8.0:
            idea.target_gap_note = f"model +{comp_up*100:.0f}%"
    else:
        idea.primary_target      = idea.composite_target
        idea.target_source       = "model"
        idea.expected_upside_pct = round(comp_up * 100, 1)

    # Stop
    stop_pct = _stop_pct_from_vol(idea.vol_20d, idea.conviction)
    idea.stop_price = round(idea.current_price * (1 - stop_pct), 2)


def _fetch_insider(ticker: str, settings) -> tuple[float, str]:
    try:
        import requests
        cutoff = (date.today() - timedelta(days=30)).isoformat()
        r = requests.get(
            "https://financialmodelingprep.com/api/v4/insider-trading",
            params={"symbol": ticker, "page": 0, "apikey": settings.fmp_api_key},
            timeout=10,
        )
        if r.status_code != 200:
            return 1.0, ""
        data = r.json()
        qualifying = [
            d for d in data
            if d.get("acquistionOrDisposition") == "A"
            and d.get("typeOfOwner", "").lower() in ("officer", "director")
            and float(d.get("price") or 0) > 0
            and float(d.get("securitiesTransacted") or 0) > 0
            and str(d.get("transactionDate", ""))[:10] >= cutoff
        ]
        if not qualifying:
            return 1.0, ""
        best = max(
            qualifying,
            key=lambda d: float(d.get("price", 0)) * float(d.get("securitiesTransacted", 0)),
        )
        value = float(best.get("price", 0)) * float(best.get("securitiesTransacted", 0))
        name  = best.get("reportingName", "insider")
        role  = best.get("typeOfOwner", "")
        shares = int(float(best.get("securitiesTransacted", 0)))
        price  = float(best.get("price", 0))
        note = f"{name} ({role}) bought {shares:,} @ ${price:.2f}"
        boost = 1.30 if value >= 100_000 else 1.20
        return boost, note
    except Exception:
        return 1.0, ""




# ── Scoring pipeline ──────────────────────────────────────────────────────────

def _stage1(
    universe: list[str],
    names: dict[str, str],
    buy_signals: dict[str, object],
    sectors: dict[str, str],
    analyst_lookup: dict[str, dict],
    regime_key: str,
) -> list[TradeIdea]:
    """
    Score every stock in the universe using regime + analyst + congress.
    No price fetches here — RSI uses placeholder 0.50 until stage 2.
    """
    ideas: list[TradeIdea] = []
    for ticker in universe:
        sector = _normalize_sector(sectors.get(ticker, ""))
        mult   = _regime_mult(regime_key, sector)
        regime_score  = _regime_to_score(mult)

        analyst_row   = analyst_lookup.get(ticker)
        analyst_score = _analyst_to_score(analyst_row) if analyst_row else 0.50
        analyst_str   = _analyst_note(analyst_row) if analyst_row else ""

        sig = buy_signals.get(ticker)
        if sig:
            congress_score    = _congress_to_score(sig)
            officials         = sig.officials
            cluster_count     = sig.cluster_count
            committee_aligned = sig.committee_aligned
            committee_note    = sig.committee_note
            total_usd         = sig.total_usd
            avg_lag_days      = sig.avg_lag_days
            alpha_vs_spy      = sig.alpha_vs_spy
            company           = names.get(ticker) or sig.company or ticker
        else:
            congress_score    = 0.0
            officials         = []
            cluster_count     = 0
            committee_aligned = False
            committee_note    = ""
            total_usd         = 0.0
            avg_lag_days      = 0.0
            alpha_vs_spy      = None
            company           = names.get(ticker) or ticker

        score = _final_score(congress_score, regime_score, analyst_score, 0.50, 0.50)

        ideas.append(TradeIdea(
            rank=0,
            ticker=ticker,
            company=company,
            side="Buy",
            sector=sector,
            congress_score=round(congress_score, 3),
            regime_score=round(regime_score, 3),
            analyst_score=round(analyst_score, 3),
            momentum_score=0.50,
            flow_score=0.50,
            insider_boost=1.0,
            final_score=score,
            regime_fit=_regime_fit_label(mult),
            regime_multiplier=mult,
            analyst_consensus=analyst_str,
            flow_direction="NEUTRAL",
            has_congress=sig is not None and (sig.committee_aligned or "trump" in sig.sources),
            officials=officials,
            cluster_count=cluster_count,
            committee_aligned=committee_aligned,
            committee_note=committee_note,
            total_usd=total_usd,
            avg_lag_days=avg_lag_days,
            alpha_vs_spy=alpha_vs_spy,
        ))

    ideas.sort(key=lambda x: -x.final_score)
    return ideas


def _stage2_prices(ideas: list[TradeIdea], settings, top_n: int = 50) -> list[TradeIdea]:
    """Fetch 13-month price history for top N; compute 12-1 momentum and 10-day money flow."""
    import numpy as np
    from lox.data.market import fetch_equity_daily_closes_fmp

    pool = ideas[:top_n]
    tickers = [idea.ticker for idea in pool]
    start = (date.today() - timedelta(days=390)).isoformat()

    try:
        df = fetch_equity_daily_closes_fmp(settings=settings, symbols=tickers, start=start)
    except Exception:
        return pool

    raw_mom:  dict[str, float] = {}
    raw_flow: dict[str, float] = {}

    for idea in pool:
        if idea.ticker not in df.columns:
            continue
        closes = df[idea.ticker].dropna().values.astype(np.float64)
        if len(closes) == 0:
            continue

        idea.current_price = float(closes[-1])

        # 20-day realized daily-return stdev (for stop sizing)
        if len(closes) >= 21:
            recent = closes[-21:]
            prev = np.where(recent[:-1] > 0, recent[:-1], 1)
            daily_rets = np.diff(recent) / prev
            idea.vol_20d = float(np.std(daily_rets, ddof=1)) if len(daily_rets) > 1 else None

        m = _compute_momentum_12_1(closes)
        if m is not None:
            raw_mom[idea.ticker] = m

        # 10-day directional money flow: (up_magnitude - down_magnitude) / total
        # Positive = buying pressure, negative = selling pressure
        if len(closes) >= 11:
            returns = np.diff(closes[-11:]) / np.where(closes[-11:-1] > 0, closes[-11:-1], 1)
            up   = float(np.sum(np.abs(returns[returns > 0])))
            down = float(np.sum(np.abs(returns[returns < 0])))
            total = up + down
            raw_flow[idea.ticker] = (up - down) / total if total > 0 else 0.0

    mom_scores  = _rank_normalize(raw_mom)
    flow_scores = _rank_normalize(raw_flow)

    for idea in pool:
        idea.momentum_12_1  = raw_mom.get(idea.ticker)
        idea.momentum_score = mom_scores.get(idea.ticker, 0.50)

        flow_raw = raw_flow.get(idea.ticker)
        idea.flow_score = flow_scores.get(idea.ticker, 0.50)
        if flow_raw is not None:
            if flow_raw > 0.15:
                idea.flow_direction = "ACCUM"
            elif flow_raw < -0.15:
                idea.flow_direction = "DIST"
            else:
                idea.flow_direction = "NEUTRAL"

        idea.final_score = _final_score(
            idea.congress_score, idea.regime_score, idea.analyst_score,
            idea.momentum_score, idea.flow_score, idea.insider_boost,
        )

    pool.sort(key=lambda x: -x.final_score)
    return pool


def _stage3_enrich(ideas: list[TradeIdea], settings, regime_key: str, top_n: int = 10) -> list[TradeIdea]:
    """Fetch insider + news + short interest + analyst target for top N; then build trade plan."""
    pool = ideas[:top_n]

    def _enrich_one(idea: TradeIdea) -> TradeIdea:
        ins_boost, insider_note = _fetch_insider(idea.ticker, settings)
        sentiment, headline     = _fetch_news_sentiment(idea.ticker, settings)
        si_pct                  = _fetch_short_interest(idea.ticker, settings)
        analyst_target          = _fetch_analyst_target(idea.ticker, settings)
        idea.insider_boost      = ins_boost
        idea.insider_note       = insider_note
        idea.news_sentiment     = sentiment
        idea.news_headline      = headline
        idea.short_interest_pct = si_pct
        idea.analyst_target     = analyst_target
        idea.final_score = _final_score(
            idea.congress_score, idea.regime_score, idea.analyst_score,
            idea.momentum_score, idea.flow_score, ins_boost,
        )
        return idea

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_enrich_one, idea): idea for idea in pool}
        enriched = [f.result() for f in as_completed(futures)]

    enriched.sort(key=lambda x: -x.final_score)
    for i, idea in enumerate(enriched, 1):
        idea.rank = i
        _build_trade_plan(idea, regime_key)
    return enriched


# ── Rendering ─────────────────────────────────────────────────────────────────

_CONVICTION_STYLE = {
    "HIGH": "bold green",
    "MED":  "cyan",
    "LOW":  "yellow",
}


def _top_pick_reason(idea: TradeIdea) -> str:
    """1-2 driver labels for the TOP PICK line, ordered by narrative weight."""
    reasons: list[str] = []
    if idea.committee_aligned and idea.congress_score >= 0.40:
        reasons.append("committee-aligned gov buy")
    if idea.regime_multiplier >= 1.25:
        reasons.append("sector regime tailwind")
    if idea.momentum_12_1 is not None and idea.momentum_12_1 >= 0.30:
        reasons.append(f"+{idea.momentum_12_1*100:.0f}% 12-1 momentum")
    elif idea.momentum_score >= 0.80:
        reasons.append("momentum leader")
    if idea.flow_direction == "ACCUM":
        reasons.append("accumulation flow")
    if idea.insider_boost > 1.0:
        reasons.append("insider buy")
    if idea.analyst_score >= 0.80:
        reasons.append("street bull consensus")
    if not reasons:
        return "composite score leader"
    return " + ".join(reasons[:2])


def _top_pick_line(idea: TradeIdea) -> str:
    reason = _top_pick_reason(idea)
    if idea.current_price is None or idea.primary_target is None:
        return f"[bold]{idea.ticker}[/bold] — {reason} [dim](plan pending)[/dim]"
    up_sign = "+" if (idea.expected_upside_pct or 0) >= 0 else ""
    return (
        f"[bold]{idea.ticker}[/bold] @ ${idea.current_price:.2f} → "
        f"${idea.primary_target:.2f} ([bold]{up_sign}{idea.expected_upside_pct:.0f}%[/bold], "
        f"{idea.horizon}) — {reason}"
    )


def _theme_summary(ideas: list[TradeIdea]) -> str:
    from collections import Counter
    n = len(ideas)
    parts: list[str] = []

    sector_counts = Counter(i.sector for i in ideas if i.sector)
    top = sector_counts.most_common(2)
    if top:
        s1, c1 = top[0]
        if len(top) > 1 and top[1][1] == c1:
            s2, c2 = top[1]
            names = f"{_SECTOR_SHORT.get(s1, s1[:4])}/{_SECTOR_SHORT.get(s2, s2[:4])}"
            parts.append(f"{names} lead ({c1 + c2}/{n})")
        elif c1 >= 3:
            parts.append(f"{_SECTOR_SHORT.get(s1, s1[:4])} leads ({c1}/{n})")

    gov_count = sum(1 for i in ideas if i.has_congress)
    if gov_count >= 3:
        parts.append(f"{gov_count} gov signals in top-{n}")

    accum_count = sum(1 for i in ideas if i.flow_direction == "ACCUM")
    if accum_count * 2 >= n and accum_count >= 3:
        parts.append(f"{accum_count}/{n} on accumulation")

    if not parts:
        return "no dominant theme — dispersion across sectors"
    return " · ".join(parts)


def _action_line(idea: TradeIdea) -> tuple[str, str]:
    """Return (line1, line2) markup strings for the ACTION ticket, or ('', '') if no plan."""
    if idea.current_price is None or idea.primary_target is None:
        return ("[dim]    plan unavailable — no price data[/dim]", "")
    style = _CONVICTION_STYLE.get(idea.conviction, "white")
    up_sign = "+" if (idea.expected_upside_pct or 0) >= 0 else ""
    src_tag = "" if idea.target_source == "street" else " [dim](model)[/dim]"
    line1 = (
        f"    [{style}]BUY {idea.ticker}[/{style}] @ "
        f"[bold]${idea.current_price:.2f}[/bold] "
        f"→ [bold]${idea.primary_target:.2f}[/bold] "
        f"([{style}]{up_sign}{idea.expected_upside_pct:.0f}%[/{style}], {idea.horizon}){src_tag}"
    )
    gap = f"  [dim yellow]({idea.target_gap_note})[/dim yellow]" if idea.target_gap_note else ""
    line2 = (
        f"         [dim]stop[/dim] [red]${idea.stop_price:.2f}[/red]"
        f"  ·  [dim]conviction[/dim] [{style}]{idea.conviction}[/{style}]{gap}"
    )
    return (line1, line2)


def _render(
    console: Console,
    ideas: list[TradeIdea],
    regime_label: str,
    confidence: float,
    equity_stance: str,
    n_universe: int,
    n_congress: int,
) -> None:
    _REGIME_COLORS = {
        "RISK-ON": "bold green", "REFLATION": "yellow",
        "STAGFLATION": "bold red", "RISK-OFF": "red", "TRANSITION": "dim white",
    }
    label_upper = regime_label.upper()
    rcolor = next((v for k, v in _REGIME_COLORS.items() if k in label_upper), "white")
    stance_color = {"OVERWEIGHT": "green", "NEUTRAL": "white", "UNDERWEIGHT": "red"}.get(equity_stance, "white")

    hdr = Text()
    hdr.append("PICKS  ·  ", style="bold")
    hdr.append(regime_label, style=rcolor)
    hdr.append(f"  ({confidence*100:.0f}%)", style="dim")
    hdr.append("  ·  equity ", style="dim")
    hdr.append(equity_stance, style=stance_color)
    hdr.append(f"  ·  {n_universe} stocks", style="dim")
    if n_congress:
        hdr.append(f"  ·  {n_congress} gov signals", style="cyan")
    console.print(hdr)
    console.print()

    if not ideas:
        console.print("[yellow]No qualifying signals.[/yellow]")
        return

    # ── Column header (manual formatting so we can interleave action lines) ──
    console.print(
        f"[bold dim]{'#':>2}  {'TICK':<5}  {'SCORE':>5}  {'SECT':<5}  "
        f"{'REGIME':<9}  {'CONG':>6}  {'ANL':>4}  {'MOM':>6}  {'FLOW':<6}[/bold dim]"
    )

    gov_notes:  list[tuple[str, str]] = []
    ins_notes:  list[tuple[str, str]] = []
    news_notes: list[tuple[str, str]] = []

    for idea in ideas:
        sect_short = _SECTOR_SHORT.get(idea.sector, idea.sector[:4] if idea.sector else "—")
        sect_style = _fit_color(idea.regime_fit)

        mult = idea.regime_multiplier
        if mult >= 1.25:
            rgm_str, rgm_style = f"{mult:.2f}× ↑↑", "bold green"
        elif mult >= 1.10:
            rgm_str, rgm_style = f"{mult:.2f}×  ↑", "green"
        elif mult >= 0.88:
            rgm_str, rgm_style = f"{mult:.2f}×  ~", "dim"
        else:
            rgm_str, rgm_style = f"{mult:.2f}×  ↓", "yellow"

        if idea.has_congress:
            pfx = "★" if idea.committee_aligned else ""
            cong_raw = f"{pfx}{idea.congress_score:.2f}"
            cong_style = "bold cyan" if idea.committee_aligned else "cyan"
        else:
            cong_raw, cong_style = "—", "dim"

        anl_pct = f"{idea.analyst_score*100:.0f}%"

        if idea.momentum_12_1 is not None:
            sign = "+" if idea.momentum_12_1 >= 0 else ""
            mom_raw = f"{sign}{idea.momentum_12_1*100:.0f}%"
            mom_style = _momentum_color(idea.momentum_score)
        else:
            mom_raw, mom_style = "—", "dim"

        flow_map = {"ACCUM": ("↑↑accum", "green"), "DIST": ("↓ dist", "red"), "NEUTRAL": ("~ flat", "dim")}
        flow_raw, flow_style = flow_map.get(idea.flow_direction, ("—", "dim"))

        # Column widths must match header exactly. Format raw text first, wrap styled after.
        # (Rich markup tags are non-printing so we pad the plain text then embed.)
        console.print(
            f"{idea.rank:>2}  "
            f"[bold]{idea.ticker:<5}[/bold]  "
            f"{idea.final_score:>5.3f}  "
            f"[{sect_style}]{sect_short:<5}[/{sect_style}]  "
            f"[{rgm_style}]{rgm_str:<9}[/{rgm_style}]  "
            f"[{cong_style}]{cong_raw:>6}[/{cong_style}]  "
            f"{anl_pct:>4}  "
            f"[{mom_style}]{mom_raw:>6}[/{mom_style}]  "
            f"[{flow_style}]{flow_raw:<6}[/{flow_style}]"
        )

        # ACTION ticket, indented under the row
        line1, line2 = _action_line(idea)
        if line1:
            console.print(line1)
        if line2:
            console.print(line2)
        console.print()  # blank line between picks

        if idea.has_congress:
            parts: list[str] = []
            if idea.committee_note:
                parts.append(idea.committee_note)
            officials_str = ", ".join(idea.officials[:3])
            if len(idea.officials) > 3:
                officials_str += f" +{len(idea.officials)-3}"
            parts.append(officials_str)
            if idea.total_usd > 0:
                usd = f"${idea.total_usd/1e6:.1f}M" if idea.total_usd >= 1e6 else f"${idea.total_usd/1e3:.0f}K"
                parts.append(usd)
            parts.append(f"{idea.avg_lag_days:.0f}d ago")
            if idea.alpha_vs_spy is not None:
                gap_sign = "+" if idea.alpha_vs_spy >= 0 else ""
                gap_note = f"α{gap_sign}{idea.alpha_vs_spy:.0f}% vs SPY"
                if idea.alpha_vs_spy < 0:
                    gap_note += " (gap open)"
                parts.append(gap_note)
            gov_notes.append((idea.ticker, "  ·  ".join(parts)))

        if idea.insider_note:
            ins_notes.append((idea.ticker, idea.insider_note))
        if idea.short_interest_pct is not None and idea.short_interest_pct >= 10:
            si_note = f"SI {idea.short_interest_pct:.1f}% of float"
            if idea.short_interest_pct >= 20:
                si_note += " (squeeze setup)"
            ins_notes.append((idea.ticker, si_note))

        if idea.news_sentiment == "positive" and idea.news_headline:
            news_notes.append((idea.ticker, idea.news_headline[:85]))

    for ticker, note in gov_notes:
        console.print(f"[bold cyan]GOV[/bold cyan]  [cyan]{ticker}[/cyan]  [dim]{note}[/dim]")
    for ticker, note in ins_notes:
        console.print(f"[bold green]INS[/bold green]  [green]{ticker}[/green]  [dim]{note}[/dim]")
    for ticker, note in news_notes:
        console.print(f"[dim]NEWS  {ticker}  {note}[/dim]")

    console.print()
    console.print(
        f"[dim]momentum({_W_MOMENTUM*100:.0f}%) · flow({_W_FLOW*100:.0f}%) · "
        f"regime({_W_REGIME*100:.0f}%) · analyst({_W_ANALYST*100:.0f}%) · "
        f"congress({_W_CONGRESS*100:.0f}%) · insider boost top 10  ·  "
        f"target: street consensus (fallback: model), stop: 1.5× 20d vol[/dim]"
    )


# ── Command ───────────────────────────────────────────────────────────────────

@app.callback(invoke_without_command=True)
def trades(ctx: typer.Context) -> None:
    """S&P 500 stock picker — momentum · regime · analyst · congress · insider."""
    if ctx.invoked_subcommand is not None:
        return

    console = Console()
    from lox.config import load_settings
    settings = load_settings()

    # ── 1. Universe ───────────────────────────────────────────────────────────
    from lox.universe.sp500 import fetch_sp500_symbols
    with console.status("Loading S&P 500 universe…"):
        universe = fetch_sp500_symbols(settings)

    if not universe:
        console.print("[red]Could not load S&P 500 universe — check FMP_API_KEY[/red]")
        raise typer.Exit(1)

    # ── 2. Congress + Trump signals (optional — if key missing, run without) ──
    buy_signals: dict[str, object] = {}
    api_key = None
    try:
        from lox.quiver.loader import fetch_congress_live, fetch_trump_live, get_api_key
        api_key = get_api_key()
    except Exception:
        pass

    if api_key:
        congress_df = trump_df = None
        with console.status("Fetching congressional + Trump trades…"):
            try:
                congress_df = fetch_congress_live(api_key)
            except Exception:
                pass
            try:
                trump_df = fetch_trump_live(api_key)
            except Exception:
                pass

        if congress_df is not None or trump_df is not None:
            with console.status("Building congress signals…"):
                try:
                    from lox.quiver.signal import build_signals
                    all_sigs = build_signals(
                        congress_df, trump_df,
                        max_lag_days=30,
                        congress_api_key=settings.congress_gov_api_key,
                    )
                    buy_signals = {s.ticker: s for s in all_sigs if s.side == "Buy"}
                except Exception:
                    pass

    # ── 3. Sectors + company names ────────────────────────────────────────────
    sectors: dict[str, str] = {}
    names: dict[str, str] = {}
    with console.status(f"Fetching profiles for {len(universe)} stocks (7d cached)…"):
        try:
            from lox.altdata.fmp import fetch_batch_profiles
            profiles = fetch_batch_profiles(settings=settings, tickers=universe)
            for ticker, profile in profiles.items():
                if profile.sector:
                    sectors[ticker] = profile.sector
                if profile.company_name:
                    names[ticker] = profile.company_name
        except Exception:
            pass

    # ── 4. Analyst consensus (bulk, 12h cached) ───────────────────────────────
    analyst_lookup: dict[str, dict] = {}
    with console.status("Fetching analyst consensus (bulk)…"):
        try:
            from lox.altdata.earnings_market import fetch_upgrades_downgrades_bulk
            rows = fetch_upgrades_downgrades_bulk(settings=settings)
            analyst_lookup = {
                str(row.get("symbol", "")).upper(): row
                for row in rows if row.get("symbol")
            }
        except Exception:
            pass

    # ── 5. Macro regime ───────────────────────────────────────────────────────
    regime_key    = "TRANSITION"
    regime_label  = "TRANSITION"
    confidence    = 0.0
    equity_stance = "NEUTRAL"
    with console.status("Running macro regime analysis…"):
        try:
            from lox.regimes.features import build_unified_regime_state
            state     = build_unified_regime_state(settings=settings)
            composite = state.composite
            if composite is not None:
                regime_key    = composite.regime if isinstance(composite.regime, str) else composite.regime.name
                regime_label  = getattr(composite, "label", regime_key)
                confidence    = getattr(composite, "confidence", 0.0)
                equity_stance = composite.playbook.equity_stance if composite.playbook else "NEUTRAL"
        except Exception as exc:
            console.print(f"[yellow]Regime failed, defaulting to TRANSITION: {exc}[/yellow]")

    # ── 6. Stage 1: score full universe ──────────────────────────────────────
    with console.status(f"Scoring {len(universe)}-stock universe…"):
        ideas = _stage1(universe, names, buy_signals, sectors, analyst_lookup, regime_key)

    # ── 7. Stage 2: price history → momentum + money flow for top 50 ─────────
    with console.status("Fetching 13-month price history for top 50 (momentum + flow)…"):
        ideas = _stage2_prices(ideas, settings, top_n=50)

    ideas.sort(key=lambda x: -x.final_score)

    # ── 8. Stage 3: insider + news + analyst target for top 10 ───────────────
    with console.status("Enriching top 10 with insider + news + analyst targets…"):
        enriched = _stage3_enrich(ideas, settings, regime_key=regime_key, top_n=10)

    _render(
        console, enriched,
        regime_label, confidence, equity_stance,
        n_universe=len(universe),
        n_congress=len(buy_signals),
    )

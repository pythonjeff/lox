"""
Market-wide earnings data for the Earnings regime.

Fetches bulk S&P 500 earnings signals using FMP endpoints that return
aggregate data in 1 API call each (no per-ticker loops):

  1. S&P 500 constituent list            — /v3/sp500_constituent
  2. Earnings surprises bulk (per year)   — /v4/earnings-surprises-bulk
  3. Upgrades/downgrades consensus bulk   — /v4/upgrades-downgrades-consensus-bulk
  4. Earnings calendar (already in earnings.py, reused here)

Total API cost: 4-5 calls per refresh cycle.

v2: Added sector-level decomposition, sparkline time-series, and
    sector dispersion signals — all derived from existing bulk data
    with zero additional API calls.

Author: Lox Capital Research
"""
from __future__ import annotations

import csv
import io
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from lox.altdata.cache import cache_path, read_cache, write_cache
from lox.config import Settings


FMP_BASE_URL = "https://financialmodelingprep.com/api"


def _parse_response(resp: requests.Response) -> list[dict[str, Any]]:
    """Parse FMP response — handles both JSON and CSV (bulk endpoints return CSV)."""
    content_type = resp.headers.get("content-type", "")
    if "csv" in content_type or "text/csv" in content_type:
        reader = csv.DictReader(io.StringIO(resp.text))
        return [row for row in reader]
    # Fall back to JSON
    data = resp.json()
    return data if isinstance(data, list) else []


# ─────────────────────────────────────────────────────────────────────
# S&P 500 Constituents
# ─────────────────────────────────────────────────────────────────────

def fetch_sp500_constituents(
    *,
    settings: Settings,
    cache_max_age: timedelta = timedelta(days=7),
) -> list[dict[str, Any]]:
    """Fetch current S&P 500 constituent list.

    Returns list of dicts with keys: symbol, name, sector, subSector, etc.
    Cached for 7 days (index changes are rare).
    """
    if not settings.fmp_api_key:
        return []

    p = cache_path("fmp_sp500_constituents")
    cached = read_cache(p, max_age=cache_max_age)
    if isinstance(cached, list) and cached:
        return cached

    try:
        resp = requests.get(
            f"{FMP_BASE_URL}/v3/sp500_constituent",
            params={"apikey": settings.fmp_api_key},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            write_cache(p, data)
            return data
    except Exception:
        pass
    return []


# ─────────────────────────────────────────────────────────────────────
# Bulk Earnings Surprises (one call per year)
# ─────────────────────────────────────────────────────────────────────

def fetch_earnings_surprises_bulk(
    *,
    settings: Settings,
    year: int,
    cache_max_age: timedelta = timedelta(hours=12),
) -> list[dict[str, Any]]:
    """Fetch ALL earnings surprises for a given year (bulk endpoint).

    Returns list of dicts with keys like:
      symbol, date, actualEarningResult, estimatedEarning
    """
    if not settings.fmp_api_key:
        return []

    p = cache_path(f"fmp_earnings_surprises_bulk_{year}")
    cached = read_cache(p, max_age=cache_max_age)
    if isinstance(cached, list):
        return cached

    # Try the v4 bulk endpoint (returns CSV)
    try:
        resp = requests.get(
            f"{FMP_BASE_URL}/v4/earnings-surprises-bulk",
            params={"year": year, "apikey": settings.fmp_api_key},
            timeout=60,
        )
        resp.raise_for_status()
        data = _parse_response(resp)
        if data:
            write_cache(p, data)
            return data
    except Exception:
        pass
    return []


# ─────────────────────────────────────────────────────────────────────
# Bulk Upgrades / Downgrades Consensus
# ─────────────────────────────────────────────────────────────────────

def fetch_upgrades_downgrades_bulk(
    *,
    settings: Settings,
    cache_max_age: timedelta = timedelta(hours=12),
) -> list[dict[str, Any]]:
    """Fetch analyst upgrades/downgrades consensus for all stocks (bulk).

    Returns list of dicts with keys like:
      symbol, strongBuy, buy, hold, sell, strongSell, consensus
    """
    if not settings.fmp_api_key:
        return []

    p = cache_path("fmp_upgrades_downgrades_consensus_bulk")
    cached = read_cache(p, max_age=cache_max_age)
    if isinstance(cached, list):
        return cached

    # Try the v4 bulk endpoint (returns CSV)
    try:
        resp = requests.get(
            f"{FMP_BASE_URL}/v4/upgrades-downgrades-consensus-bulk",
            params={"apikey": settings.fmp_api_key},
            timeout=60,
        )
        resp.raise_for_status()
        data = _parse_response(resp)
        if data:
            write_cache(p, data)
            return data
    except Exception:
        pass
    return []


# ─────────────────────────────────────────────────────────────────────
# Orchestrator: compute all regime inputs in one shot
# ─────────────────────────────────────────────────────────────────────

def compute_earnings_regime_inputs(
    *,
    settings: Settings,
    refresh: bool = False,
) -> dict[str, Any]:
    """Compute all inputs needed by ``classify_earnings_regime()``.

    v2 returns sector-level decomposition, sparkline time-series,
    and sector dispersion — all from the same bulk data, zero extra API calls.
    """
    const_age = timedelta(seconds=0) if refresh else timedelta(days=7)
    bulk_age = timedelta(seconds=0) if refresh else timedelta(hours=12)

    # 1. S&P 500 constituent set + sector lookup
    constituents = fetch_sp500_constituents(
        settings=settings, cache_max_age=const_age,
    )
    sp500_set: set[str] = set()
    symbol_to_sector: dict[str, str] = {}
    for c in constituents:
        sym = str(c.get("symbol", "")).upper()
        if sym:
            sp500_set.add(sym)
            sector = str(c.get("sector", "")).strip()
            if sector:
                symbol_to_sector[sym] = sector

    if not sp500_set:
        return _empty_result("No S&P 500 constituents fetched")

    # 2. Bulk earnings surprises (current year + prior year for trailing window)
    now = datetime.now(timezone.utc)
    current_year = now.year
    surprises_current = fetch_earnings_surprises_bulk(
        settings=settings, year=current_year, cache_max_age=bulk_age,
    )
    surprises_prior = fetch_earnings_surprises_bulk(
        settings=settings, year=current_year - 1, cache_max_age=bulk_age,
    )
    all_surprises = surprises_current + surprises_prior

    # Filter to S&P 500 + trailing 90 days
    cutoff_90d = (now - timedelta(days=90)).strftime("%Y-%m-%d")
    sp500_surprises = [
        s for s in all_surprises
        if (
            isinstance(s, dict)
            and str(s.get("symbol", "")).upper() in sp500_set
            and str(s.get("date", "")) >= cutoff_90d
        )
    ]

    # ── Aggregate metrics ──────────────────────────────────────────────
    beat_rate, avg_surprise_pct = _compute_beat_metrics(sp500_surprises)
    total_reporting = len(sp500_surprises)

    # 3. Upgrades / downgrades consensus
    consensus = fetch_upgrades_downgrades_bulk(
        settings=settings, cache_max_age=bulk_age,
    )
    net_revision_ratio = _compute_revision_ratio(consensus, sp500_set)

    # 4. Reporting density (trailing 30d)
    reporting_density = _count_trailing_reporters(sp500_surprises, now)

    # ── v2: Sector-level decomposition ─────────────────────────────────
    sector_stats = _compute_sector_stats(
        sp500_surprises, consensus, symbol_to_sector, sp500_set,
    )

    # Sector dispersion = best sector beat rate - worst sector beat rate
    sector_beat_rates = [
        s["beat_rate"]
        for s in sector_stats.values()
        if s.get("beat_rate") is not None and s.get("count", 0) >= 5
    ]
    sector_dispersion = (
        (max(sector_beat_rates) - min(sector_beat_rates))
        if len(sector_beat_rates) >= 2
        else None
    )

    # Top / worst sectors (by beat rate, min 5 reporters)
    ranked = sorted(
        [
            (name, s)
            for name, s in sector_stats.items()
            if s.get("beat_rate") is not None and s.get("count", 0) >= 5
        ],
        key=lambda x: x[1]["beat_rate"],
        reverse=True,
    )
    top_sectors = [(name, s) for name, s in ranked[:3]]
    worst_sectors = [(name, s) for name, s in ranked[-3:]]

    # Count sectors with beat rate > 65% (broad strength indicator)
    sectors_beating = sum(
        1 for _, s in ranked if s["beat_rate"] > 65
    )

    # ── v2: Sparkline time-series ──────────────────────────────────────
    beat_rate_series = _compute_rolling_series(
        all_surprises, sp500_set, now, metric="beat_rate",
    )
    surprise_series = _compute_rolling_series(
        all_surprises, sp500_set, now, metric="avg_surprise",
    )

    return {
        # v1 signals
        "beat_rate": beat_rate,
        "avg_surprise_pct": avg_surprise_pct,
        "net_revision_ratio": net_revision_ratio,
        "reporting_density": reporting_density,
        "total_sp500_surprises_90d": total_reporting,
        "sp500_count": len(sp500_set),
        "asof": now.strftime("%Y-%m-%d"),
        "error": None,
        # v2 sector decomposition
        "sector_stats": sector_stats,
        "sector_dispersion": sector_dispersion,
        "top_sectors": top_sectors,
        "worst_sectors": worst_sectors,
        "sectors_beating": sectors_beating,
        "total_sectors_rated": len(ranked),
        # v2 sparkline series
        "beat_rate_series": beat_rate_series,
        "surprise_series": surprise_series,
    }


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────

def _empty_result(error: str) -> dict[str, Any]:
    return {
        "beat_rate": None,
        "avg_surprise_pct": None,
        "net_revision_ratio": None,
        "reporting_density": None,
        "total_sp500_surprises_90d": 0,
        "sp500_count": 0,
        "asof": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "error": error,
        "sector_stats": {},
        "sector_dispersion": None,
        "top_sectors": [],
        "worst_sectors": [],
        "sectors_beating": 0,
        "total_sectors_rated": 0,
        "beat_rate_series": [],
        "surprise_series": [],
    }


def _compute_beat_metrics(
    surprises: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    """Compute beat rate and average surprise % from bulk surprise rows."""
    beats = 0
    total = 0
    surprise_pcts: list[float] = []

    for row in surprises:
        actual = row.get("actualEarningResult") or row.get("actualEps")
        estimated = row.get("estimatedEarning") or row.get("estimatedEps")

        if actual is None or estimated is None:
            continue
        try:
            actual_f = float(actual)
            estimated_f = float(estimated)
        except (ValueError, TypeError):
            continue

        total += 1
        if actual_f > estimated_f:
            beats += 1

        if estimated_f != 0:
            surprise_pcts.append(
                (actual_f - estimated_f) / abs(estimated_f) * 100
            )

    beat_rate = (beats / total * 100) if total > 0 else None
    avg_surprise = (
        sum(surprise_pcts) / len(surprise_pcts)
        if surprise_pcts
        else None
    )
    return beat_rate, avg_surprise


def _compute_revision_ratio(
    consensus: list[dict[str, Any]],
    sp500_set: set[str],
) -> float | None:
    """Compute net revision ratio from upgrades/downgrades consensus bulk.

    Proxy: (buy_side - sell_side) / total_rated.
    """
    total_buy_side = 0
    total_sell_side = 0
    total_rated = 0

    for row in consensus:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol", "")).upper()
        if sym not in sp500_set:
            continue

        try:
            strong_buy = int(row.get("strongBuy", 0) or 0)
            buy = int(row.get("buy", 0) or 0)
            hold = int(row.get("hold", 0) or 0)
            sell = int(row.get("sell", 0) or 0)
            strong_sell = int(row.get("strongSell", 0) or 0)
        except (ValueError, TypeError):
            continue

        name_total = strong_buy + buy + hold + sell + strong_sell
        if name_total == 0:
            continue

        total_buy_side += strong_buy + buy
        total_sell_side += sell + strong_sell
        total_rated += name_total

    if total_rated == 0:
        return None

    return (total_buy_side - total_sell_side) / total_rated


def _count_trailing_reporters(
    surprises: list[dict[str, Any]],
    now: datetime,
) -> int:
    """Count unique S&P 500 tickers that reported in the trailing 30 days."""
    cutoff = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    tickers = {
        str(s.get("symbol", "")).upper()
        for s in surprises
        if isinstance(s, dict) and str(s.get("date", "")) >= cutoff
    }
    return len(tickers)


# ─────────────────────────────────────────────────────────────────────
# v2: Sector-level decomposition
# ─────────────────────────────────────────────────────────────────────

def _compute_sector_stats(
    surprises: list[dict[str, Any]],
    consensus: list[dict[str, Any]],
    symbol_to_sector: dict[str, str],
    sp500_set: set[str],
) -> dict[str, dict[str, Any]]:
    """Compute per-sector beat rate, surprise %, and revision ratio.

    Returns dict keyed by sector name → {beat_rate, avg_surprise, revision_ratio, count}.
    """
    # ── Surprise stats by sector ───────────────────────────────────────
    sector_beats: dict[str, int] = defaultdict(int)
    sector_total: dict[str, int] = defaultdict(int)
    sector_surprises: dict[str, list[float]] = defaultdict(list)

    for row in surprises:
        sym = str(row.get("symbol", "")).upper()
        sector = symbol_to_sector.get(sym)
        if not sector:
            continue

        actual = row.get("actualEarningResult") or row.get("actualEps")
        estimated = row.get("estimatedEarning") or row.get("estimatedEps")
        if actual is None or estimated is None:
            continue
        try:
            actual_f = float(actual)
            estimated_f = float(estimated)
        except (ValueError, TypeError):
            continue

        sector_total[sector] += 1
        if actual_f > estimated_f:
            sector_beats[sector] += 1
        if estimated_f != 0:
            sector_surprises[sector].append(
                (actual_f - estimated_f) / abs(estimated_f) * 100
            )

    # ── Revision stats by sector ───────────────────────────────────────
    sector_buy: dict[str, int] = defaultdict(int)
    sector_sell: dict[str, int] = defaultdict(int)
    sector_rated: dict[str, int] = defaultdict(int)

    for row in consensus:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol", "")).upper()
        if sym not in sp500_set:
            continue
        sector = symbol_to_sector.get(sym)
        if not sector:
            continue

        try:
            sb = int(row.get("strongBuy", 0) or 0)
            b = int(row.get("buy", 0) or 0)
            h = int(row.get("hold", 0) or 0)
            s = int(row.get("sell", 0) or 0)
            ss = int(row.get("strongSell", 0) or 0)
        except (ValueError, TypeError):
            continue

        t = sb + b + h + s + ss
        if t == 0:
            continue
        sector_buy[sector] += sb + b
        sector_sell[sector] += s + ss
        sector_rated[sector] += t

    # ── Assemble per-sector output ─────────────────────────────────────
    all_sectors = set(sector_total.keys()) | set(sector_rated.keys())
    result: dict[str, dict[str, Any]] = {}

    for sector in sorted(all_sectors):
        total = sector_total.get(sector, 0)
        beats = sector_beats.get(sector, 0)
        slist = sector_surprises.get(sector, [])
        rated = sector_rated.get(sector, 0)

        result[sector] = {
            "beat_rate": (beats / total * 100) if total > 0 else None,
            "avg_surprise": (sum(slist) / len(slist)) if slist else None,
            "revision_ratio": (
                (sector_buy.get(sector, 0) - sector_sell.get(sector, 0)) / rated
                if rated > 0
                else None
            ),
            "count": total,
        }

    return result


# ─────────────────────────────────────────────────────────────────────
# v2: Rolling time-series for sparklines
# ─────────────────────────────────────────────────────────────────────

def _compute_rolling_series(
    all_surprises: list[dict[str, Any]],
    sp500_set: set[str],
    now: datetime,
    metric: str = "beat_rate",
    window_days: int = 90,
    num_points: int = 12,
) -> list[float]:
    """Compute a rolling time-series of beat_rate or avg_surprise.

    Walks backward in weekly steps, computing the metric over a trailing
    window at each point.  Returns ~12 data points for sparkline rendering.

    This is cheap — no API calls, just re-filtering the same surprise list.
    """
    step_days = 7  # weekly steps
    series: list[float] = []

    for i in range(num_points - 1, -1, -1):
        # "as of" date for this point
        asof = now - timedelta(days=i * step_days)
        cutoff = (asof - timedelta(days=window_days)).strftime("%Y-%m-%d")
        asof_str = asof.strftime("%Y-%m-%d")

        # Filter surprises to [cutoff, asof]
        window = [
            s for s in all_surprises
            if (
                isinstance(s, dict)
                and str(s.get("symbol", "")).upper() in sp500_set
                and cutoff <= str(s.get("date", "")) <= asof_str
            )
        ]

        if not window:
            if series:
                series.append(series[-1])  # carry forward
            continue

        if metric == "beat_rate":
            beats = 0
            total = 0
            for row in window:
                actual = row.get("actualEarningResult") or row.get("actualEps")
                estimated = row.get("estimatedEarning") or row.get("estimatedEps")
                if actual is None or estimated is None:
                    continue
                try:
                    if float(actual) > float(estimated):
                        beats += 1
                    total += 1
                except (ValueError, TypeError):
                    continue
            series.append((beats / total * 100) if total > 0 else 0)

        elif metric == "avg_surprise":
            pcts: list[float] = []
            for row in window:
                actual = row.get("actualEarningResult") or row.get("actualEps")
                estimated = row.get("estimatedEarning") or row.get("estimatedEps")
                if actual is None or estimated is None:
                    continue
                try:
                    a, e = float(actual), float(estimated)
                    if e != 0:
                        pcts.append((a - e) / abs(e) * 100)
                except (ValueError, TypeError):
                    continue
            series.append((sum(pcts) / len(pcts)) if pcts else 0)

    return series


# ─────────────────────────────────────────────────────────────────────
# Sector drill-down: per-stock basket
# ─────────────────────────────────────────────────────────────────────

def get_sector_stocks(
    sector_name: str,
    *,
    settings: Settings,
    refresh: bool = False,
) -> dict[str, Any]:
    """Return per-stock earnings data for a single GICS sector.

    Reuses the same 3 cached bulk endpoints — zero extra API calls.

    Returns dict with:
        sector:          matched sector name
        stocks:          list of per-stock dicts sorted by surprise_pct desc
        sector_summary:  {beat_rate, avg_surprise, revision_ratio, count}
        sp500_beat_rate: aggregate S&P 500 beat rate for comparison
        error:           error string or None
    """
    bulk_age = timedelta(seconds=0) if refresh else timedelta(hours=12)
    const_age = timedelta(seconds=0) if refresh else timedelta(days=7)

    # 1. Constituents — build sector lookup + name lookup
    constituents = fetch_sp500_constituents(
        settings=settings, cache_max_age=const_age,
    )
    sp500_set: set[str] = set()
    symbol_to_sector: dict[str, str] = {}
    symbol_to_name: dict[str, str] = {}
    symbol_to_subsector: dict[str, str] = {}

    for c in constituents:
        sym = str(c.get("symbol", "")).upper()
        if not sym:
            continue
        sp500_set.add(sym)
        sector = str(c.get("sector", "")).strip()
        if sector:
            symbol_to_sector[sym] = sector
        name = str(c.get("name", "")).strip()
        if name:
            symbol_to_name[sym] = name
        sub = str(c.get("subSector", "")).strip()
        if sub:
            symbol_to_subsector[sym] = sub

    if not sp500_set:
        return {"sector": sector_name, "stocks": [], "sector_summary": {},
                "sp500_beat_rate": None, "error": "No S&P 500 constituents"}

    # Fuzzy-match sector name (case-insensitive, hyphens → spaces)
    # Allows: "real-estate", "consumer-cyclical", "Real Estate", "real"
    all_sectors = sorted(set(symbol_to_sector.values()))
    normalized = sector_name.lower().replace("-", " ").replace("_", " ")
    matched = None

    # 1. Exact match (case-insensitive, hyphen-tolerant)
    for s in all_sectors:
        if s.lower() == normalized:
            matched = s
            break

    # 2. Substring match
    if matched is None:
        for s in all_sectors:
            if normalized in s.lower():
                matched = s
                break

    # 3. Word-start match (e.g., "fin" → "Financial Services")
    if matched is None:
        for s in all_sectors:
            if s.lower().startswith(normalized):
                matched = s
                break

    if matched is None:
        short_names = [s.lower().replace(" ", "-") for s in all_sectors]
        return {
            "sector": sector_name,
            "stocks": [],
            "sector_summary": {},
            "sp500_beat_rate": None,
            "error": (
                f"Sector '{sector_name}' not found. Available:\n"
                + "  " + "\n  ".join(
                    f"{s}  (--sector {s.lower().replace(' ', '-')})"
                    for s in all_sectors
                )
            ),
        }

    sector_symbols = {sym for sym, sec in symbol_to_sector.items() if sec == matched}

    # 2. Bulk surprises — filter to sector + trailing 90d
    now = datetime.now(timezone.utc)
    current_year = now.year
    surprises_current = fetch_earnings_surprises_bulk(
        settings=settings, year=current_year, cache_max_age=bulk_age,
    )
    surprises_prior = fetch_earnings_surprises_bulk(
        settings=settings, year=current_year - 1, cache_max_age=bulk_age,
    )
    all_surprises = surprises_current + surprises_prior

    cutoff_90d = (now - timedelta(days=90)).strftime("%Y-%m-%d")
    sp500_surprises_90d = [
        s for s in all_surprises
        if (isinstance(s, dict)
            and str(s.get("symbol", "")).upper() in sp500_set
            and str(s.get("date", "")) >= cutoff_90d)
    ]

    # S&P 500 aggregate beat rate (for comparison)
    sp500_br, _ = _compute_beat_metrics(sp500_surprises_90d)

    # Per-stock: keep most recent report per ticker in sector
    latest_by_sym: dict[str, dict[str, Any]] = {}
    for row in all_surprises:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol", "")).upper()
        if sym not in sector_symbols:
            continue
        date_str = str(row.get("date", ""))
        if date_str < cutoff_90d:
            continue

        actual = row.get("actualEarningResult") or row.get("actualEps")
        estimated = row.get("estimatedEarning") or row.get("estimatedEps")
        if actual is None or estimated is None:
            continue
        try:
            actual_f = float(actual)
            estimated_f = float(estimated)
        except (ValueError, TypeError):
            continue

        # Keep the most recent report per symbol
        if sym in latest_by_sym and latest_by_sym[sym]["date"] >= date_str:
            continue

        surprise_pct = (
            (actual_f - estimated_f) / abs(estimated_f) * 100
            if estimated_f != 0 else 0.0
        )

        latest_by_sym[sym] = {
            "symbol": sym,
            "name": symbol_to_name.get(sym, sym),
            "sub_sector": symbol_to_subsector.get(sym, ""),
            "date": date_str,
            "actual_eps": actual_f,
            "estimated_eps": estimated_f,
            "surprise_pct": round(surprise_pct, 2),
            "beat": actual_f > estimated_f,
        }

    # 3. Consensus — add analyst ratings per stock
    consensus = fetch_upgrades_downgrades_bulk(
        settings=settings, cache_max_age=bulk_age,
    )
    consensus_by_sym: dict[str, dict[str, Any]] = {}
    for row in consensus:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol", "")).upper()
        if sym not in sector_symbols:
            continue
        try:
            sb = int(row.get("strongBuy", 0) or 0)
            b = int(row.get("buy", 0) or 0)
            h = int(row.get("hold", 0) or 0)
            s = int(row.get("sell", 0) or 0)
            ss = int(row.get("strongSell", 0) or 0)
        except (ValueError, TypeError):
            continue
        total = sb + b + h + s + ss
        if total == 0:
            continue

        # Derive consensus label
        buy_side = sb + b
        sell_side = s + ss
        if sb > b and sb > h:
            label = "Strong Buy"
        elif buy_side > sell_side and buy_side > h:
            label = "Buy"
        elif sell_side > buy_side:
            label = "Sell"
        else:
            label = "Hold"

        consensus_by_sym[sym] = {
            "consensus": label,
            "strong_buy": sb,
            "buy": b,
            "hold": h,
            "sell": s,
            "strong_sell": ss,
        }

    # Merge consensus into stock records
    for sym, stock in latest_by_sym.items():
        c = consensus_by_sym.get(sym, {})
        stock["consensus"] = c.get("consensus", "—")

    # Sort by surprise_pct descending
    stocks = sorted(latest_by_sym.values(), key=lambda x: x["surprise_pct"], reverse=True)

    # Sector summary
    sector_rows = [s for s in sp500_surprises_90d
                   if str(s.get("symbol", "")).upper() in sector_symbols]
    sect_br, sect_surp = _compute_beat_metrics(sector_rows)
    sect_rev = _compute_revision_ratio(consensus, sector_symbols)

    return {
        "sector": matched,
        "stocks": stocks,
        "sector_summary": {
            "beat_rate": sect_br,
            "avg_surprise": sect_surp,
            "revision_ratio": sect_rev,
            "count": len(stocks),
        },
        "sp500_beat_rate": sp500_br,
        "error": None,
    }

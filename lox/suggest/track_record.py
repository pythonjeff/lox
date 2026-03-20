"""
Performance tracker & self-evaluation for the suggestion engine.

Logs every recommendation, backfills actual returns on subsequent runs,
and computes hit rates by signal type to dynamically adjust pillar weights.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from lox.config import Settings

logger = logging.getLogger(__name__)

_HISTORY_PATH = Path("data/cache/suggest_track_record.json")

# Schema: list of dicts with these keys
_COLUMNS = [
    "date", "ticker", "direction", "composite_score", "signal_type",
    "price_at_rec", "ret_5d", "ret_20d", "hit_5d", "hit_20d",
]


@dataclass
class TrackRecord:
    """Aggregate performance stats."""
    total_recs: int
    hit_rate_5d: float | None  # % of resolved recs where direction was correct
    hit_rate_20d: float | None
    avg_return_5d: float | None
    avg_return_20d: float | None
    by_signal_type: dict[str, dict[str, Any]]  # signal_type -> stats
    by_conviction: dict[str, dict[str, Any]]
    best_recent: list[dict[str, Any]]  # top 3 best calls (last 30d)
    worst_recent: list[dict[str, Any]]  # top 3 worst calls


def _load_history() -> list[dict[str, Any]]:
    """Load recommendation history from JSON file."""
    try:
        if _HISTORY_PATH.exists():
            data = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def _save_history(records: list[dict[str, Any]]) -> None:
    """Save recommendation history to JSON file."""
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _HISTORY_PATH.write_text(
        json.dumps(records, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def log_recommendations(candidates: list[Any]) -> None:
    """Append new recommendations to the track record.

    Each candidate should have: ticker, direction, composite_score,
    signal_type, price_at_rec (or price).
    """
    records = _load_history()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Don't double-log same day
    existing_today = {
        r["ticker"] for r in records
        if r.get("date") == today
    }

    for c in candidates:
        ticker = getattr(c, "ticker", "")
        if ticker in existing_today:
            continue

        price = getattr(c, "price_at_rec", 0) or getattr(c, "price", 0)
        records.append({
            "date": today,
            "ticker": ticker,
            "direction": getattr(c, "direction", "LONG"),
            "composite_score": round(getattr(c, "composite_score", 0), 1),
            "signal_type": getattr(c, "signal_type", ""),
            "price_at_rec": round(float(price), 2) if price else 0,
            "ret_5d": None,
            "ret_20d": None,
            "hit_5d": None,
            "hit_20d": None,
        })

    _save_history(records)


def backfill_returns(settings: Settings) -> int:
    """Fill in actual 5d and 20d returns for past recommendations.

    Returns number of records updated.
    """
    records = _load_history()
    if not records:
        return 0

    today = datetime.now(timezone.utc).date()
    needs_update: list[dict[str, Any]] = []
    tickers_needed: set[str] = set()

    for r in records:
        try:
            rec_date = datetime.strptime(r["date"], "%Y-%m-%d").date()
        except (ValueError, TypeError, KeyError):
            continue

        days_elapsed = (today - rec_date).days
        price = r.get("price_at_rec", 0)
        if not price or price <= 0:
            continue

        if r.get("ret_5d") is None and days_elapsed >= 7:
            needs_update.append(r)
            tickers_needed.add(r["ticker"])
        elif r.get("ret_20d") is None and days_elapsed >= 28:
            needs_update.append(r)
            tickers_needed.add(r["ticker"])

    if not tickers_needed:
        return 0

    # Batch fetch current prices
    from lox.altdata.fmp import fetch_batch_quotes_full
    quotes = fetch_batch_quotes_full(
        settings=settings,
        tickers=list(tickers_needed),
    )
    price_map: dict[str, float] = {}
    for q in quotes:
        sym = str(q.get("symbol", "")).upper()
        p = q.get("price")
        if sym and p:
            try:
                price_map[sym] = float(p)
            except (ValueError, TypeError):
                pass

    updated = 0
    for r in records:
        try:
            rec_date = datetime.strptime(r["date"], "%Y-%m-%d").date()
        except (ValueError, TypeError, KeyError):
            continue

        days_elapsed = (today - rec_date).days
        price = r.get("price_at_rec", 0)
        current = price_map.get(r["ticker"])

        if not price or not current or price <= 0:
            continue

        raw_ret = (current - price) / price
        direction = r.get("direction", "LONG")
        directed_ret = raw_ret if direction == "LONG" else -raw_ret

        # Fill 5d return (after 7 calendar days to ensure 5 trading days)
        if r.get("ret_5d") is None and days_elapsed >= 7:
            r["ret_5d"] = round(directed_ret, 4)
            r["hit_5d"] = directed_ret > 0
            updated += 1

        # Fill 20d return (after 28 calendar days)
        if r.get("ret_20d") is None and days_elapsed >= 28:
            r["ret_20d"] = round(directed_ret, 4)
            r["hit_20d"] = directed_ret > 0
            updated += 1

    if updated:
        _save_history(records)

    return updated


def compute_track_record(lookback_days: int = 60) -> TrackRecord:
    """Compute aggregate performance statistics.

    Args:
        lookback_days: only consider recs from the last N days.
    """
    records = _load_history()
    if not records:
        return TrackRecord(
            total_recs=0, hit_rate_5d=None, hit_rate_20d=None,
            avg_return_5d=None, avg_return_20d=None,
            by_signal_type={}, by_conviction={},
            best_recent=[], worst_recent=[],
        )

    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    recent = [r for r in records if r.get("date", "") >= cutoff]

    if not recent:
        return TrackRecord(
            total_recs=len(records), hit_rate_5d=None, hit_rate_20d=None,
            avg_return_5d=None, avg_return_20d=None,
            by_signal_type={}, by_conviction={},
            best_recent=[], worst_recent=[],
        )

    # Overall stats
    hits_5d = [r["hit_5d"] for r in recent if r.get("hit_5d") is not None]
    hits_20d = [r["hit_20d"] for r in recent if r.get("hit_20d") is not None]
    rets_5d = [r["ret_5d"] for r in recent if r.get("ret_5d") is not None]
    rets_20d = [r["ret_20d"] for r in recent if r.get("ret_20d") is not None]

    hit_rate_5d = sum(hits_5d) / len(hits_5d) if hits_5d else None
    hit_rate_20d = sum(hits_20d) / len(hits_20d) if hits_20d else None
    avg_ret_5d = sum(rets_5d) / len(rets_5d) if rets_5d else None
    avg_ret_20d = sum(rets_20d) / len(rets_20d) if rets_20d else None

    # By signal type
    by_signal: dict[str, dict[str, Any]] = {}
    for st in set(r.get("signal_type", "") for r in recent):
        if not st:
            continue
        st_recs = [r for r in recent if r.get("signal_type") == st]
        st_hits_5d = [r["hit_5d"] for r in st_recs if r.get("hit_5d") is not None]
        st_hits_20d = [r["hit_20d"] for r in st_recs if r.get("hit_20d") is not None]
        st_rets_20d = [r["ret_20d"] for r in st_recs if r.get("ret_20d") is not None]
        by_signal[st] = {
            "count": len(st_recs),
            "hit_rate_5d": sum(st_hits_5d) / len(st_hits_5d) if st_hits_5d else None,
            "hit_rate_20d": sum(st_hits_20d) / len(st_hits_20d) if st_hits_20d else None,
            "avg_return": sum(st_rets_20d) / len(st_rets_20d) if st_rets_20d else None,
        }

    # By conviction (from composite_score buckets)
    by_conviction: dict[str, dict[str, Any]] = {}
    for label, lo, hi in [("HIGH", 70, 101), ("MEDIUM", 45, 70), ("LOW", 0, 45)]:
        conv_recs = [r for r in recent if lo <= (r.get("composite_score", 0) or 0) < hi]
        conv_hits = [r["hit_20d"] for r in conv_recs if r.get("hit_20d") is not None]
        by_conviction[label] = {
            "count": len(conv_recs),
            "hit_rate": sum(conv_hits) / len(conv_hits) if conv_hits else None,
        }

    # Best / worst recent calls
    resolved = [r for r in recent if r.get("ret_20d") is not None]
    resolved.sort(key=lambda r: r.get("ret_20d", 0), reverse=True)
    best = resolved[:3]
    worst = resolved[-3:] if len(resolved) >= 3 else resolved[::-1][:3]

    return TrackRecord(
        total_recs=len(records),
        hit_rate_5d=hit_rate_5d,
        hit_rate_20d=hit_rate_20d,
        avg_return_5d=avg_ret_5d,
        avg_return_20d=avg_ret_20d,
        by_signal_type=by_signal,
        by_conviction=by_conviction,
        best_recent=best,
        worst_recent=worst,
    )


def get_weight_adjustments(lookback_recs: int = 30) -> dict[str, float]:
    """Compute pillar weight adjustments from track record performance.

    Returns {pillar_name: multiplier} where multiplier is:
    - 0.8 if hit_rate < 40% (reduce weight 20%)
    - 1.1 if hit_rate > 65% (boost weight 10%)
    - 1.0 otherwise (no change)

    Only applies when we have sufficient data (>= 10 resolved recs per signal type).
    """
    records = _load_history()
    if len(records) < 15:
        return {}

    # Map signal types to pillar names
    signal_to_pillar = {
        "REGIME_TAILWIND": "regime",
        "FLOW_ACCELERATION": "flow",
        "REVERSION_SETUP": "momentum",
        "CATALYST_DRIVEN": "catalyst",
    }

    adjustments: dict[str, float] = {}
    for signal_type, pillar in signal_to_pillar.items():
        typed_recs = [
            r for r in records
            if r.get("signal_type") == signal_type and r.get("hit_5d") is not None
        ]
        if len(typed_recs) < 10:
            continue

        # Use last N records
        typed_recs = typed_recs[-lookback_recs:]
        hits = [r["hit_5d"] for r in typed_recs if r.get("hit_5d") is not None]
        if not hits:
            continue

        hit_rate = sum(hits) / len(hits)
        if hit_rate < 0.40:
            adjustments[pillar] = 0.8
        elif hit_rate > 0.65:
            adjustments[pillar] = 1.1
        # else: 1.0 (default, don't include)

    return adjustments

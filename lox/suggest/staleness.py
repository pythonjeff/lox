"""
Anti-staleness engine for the opportunity scanner.

Tracks recommendation history and enforces diversity to prevent
the same names from dominating output day after day.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROTATION_PATH = Path("data/cache/suggest_rotation.json")


def load_rotation_history() -> dict[str, list[str]]:
    """Load {date_str: [tickers]} from rotation file."""
    try:
        if _ROTATION_PATH.exists():
            data = json.loads(_ROTATION_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_rotation_history(tickers: list[str]) -> None:
    """Append today's recommended tickers to rotation file.

    Keeps last 30 days of history.
    """
    history = load_rotation_history()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    history[today] = tickers

    # Prune entries older than 30 days
    cutoff = (datetime.now(timezone.utc).date().toordinal()) - 30
    pruned = {}
    for date_str, tks in history.items():
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            if d.toordinal() >= cutoff:
                pruned[date_str] = tks
        except (ValueError, TypeError):
            continue

    _ROTATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ROTATION_PATH.write_text(
        json.dumps(pruned, ensure_ascii=False), encoding="utf-8",
    )


def compute_rotation_penalties(
    tickers: list[str],
    history: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    """Compute per-ticker rotation penalties based on recent recommendation history.

    Returns ticker -> penalty (0-25 points).
    """
    if history is None:
        history = load_rotation_history()

    today = datetime.now(timezone.utc).date()
    penalties: dict[str, float] = {}

    for ticker in tickers:
        # Find most recent recommendation date
        days_since: int | None = None
        for date_str, rec_tickers in history.items():
            if ticker in rec_tickers:
                try:
                    d = datetime.strptime(date_str, "%Y-%m-%d").date()
                    delta = (today - d).days
                    if days_since is None or delta < days_since:
                        days_since = delta
                except (ValueError, TypeError):
                    continue

        if days_since is None:
            continue  # never recommended → no penalty

        if days_since <= 2:
            penalties[ticker] = 25.0
        elif days_since <= 4:
            penalties[ticker] = 15.0
        elif days_since <= 6:
            penalties[ticker] = 8.0
        # 7+ days: no penalty

    return penalties


def enforce_diversity(
    scored: list[Any],
    max_per_category: int = 3,
) -> list[Any]:
    """Cap the number of candidates from any single sector/category.

    Expects each item to have a .sector attribute.
    Returns a filtered list maintaining rank order.
    """
    category_counts: dict[str, int] = {}
    result = []
    for candidate in scored:
        cat = getattr(candidate, "sector", "") or "other"
        # Normalize sector for grouping
        cat = cat.lower().strip()
        if not cat:
            cat = "other"
        if category_counts.get(cat, 0) < max_per_category:
            result.append(candidate)
            category_counts[cat] = category_counts.get(cat, 0) + 1
    return result

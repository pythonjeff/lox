"""
Regime history tracking — persists regime state for change detection and trend analysis.

Stores at ~/.lox/regime_history.json so it persists across CLI runs.

Structure:
    {
        "domains": { "<domain>": { "label", "score", "date" } },
        "changes": [ { "domain", "from_label", "to_label", "date", "from_score", "to_score" } ],
        "score_series": { "<domain>": [ { "date", "score", "label", "name" }, ... ] },
        "last_updated": "YYYY-MM-DD"
    }

score_series keeps the last MAX_SERIES_LEN snapshots per domain (one per unique date).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HISTORY_FILE = Path.home() / ".lox" / "regime_history.json"
MAX_SERIES_LEN = 90  # Keep ~90 daily snapshots per domain


def load_history() -> dict:
    """Load full regime history from disk."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load regime history: {e}")
    return {}


def save_regime_snapshot(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Save current regime state, detect changes, and append to score series.

    Args:
        results: dict mapping domain name -> RegimeResult-like object
                 (must have .label, .score, .name attributes)

    Returns:
        List of change dicts with keys: domain, from_label, to_label, date,
        from_score, to_score
    """
    history = load_history()
    today = datetime.now().strftime("%Y-%m-%d")
    changes: list[dict[str, Any]] = []

    # Ensure nested structure
    if "domains" not in history:
        history["domains"] = {}
    if "changes" not in history:
        history["changes"] = []
    if "score_series" not in history:
        history["score_series"] = {}

    for domain, result in results.items():
        label = getattr(result, "label", str(result))
        score = getattr(result, "score", 50)
        name = getattr(result, "name", label)

        prev = history["domains"].get(domain, {})
        prev_label = prev.get("label")

        if prev_label is not None and prev_label != label:
            change = {
                "domain": domain,
                "from_label": prev_label,
                "to_label": label,
                "date": today,
                "from_score": prev.get("score"),
                "to_score": score,
            }
            changes.append(change)
            history["changes"].append(change)

        history["domains"][domain] = {
            "label": label,
            "score": score,
            "name": name,
            "date": today,
        }

        # ── Append to score_series (one entry per unique date) ────────────
        series = history["score_series"].get(domain, [])
        entry = {"date": today, "score": round(score, 1), "label": label, "name": name}

        # Replace if same date already exists, otherwise append
        if series and series[-1].get("date") == today:
            series[-1] = entry
        else:
            series.append(entry)

        # Trim to max length
        if len(series) > MAX_SERIES_LEN:
            series = series[-MAX_SERIES_LEN:]

        history["score_series"][domain] = series

    history["last_updated"] = today

    # Persist
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save regime history: {e}")

    return changes


def get_score_series(domain: str) -> list[dict[str, Any]]:
    """Return the score time series for a domain.

    Each entry: {"date": "YYYY-MM-DD", "score": float, "label": str, "name": str}
    """
    history = load_history()
    return history.get("score_series", {}).get(domain, [])


def get_all_score_series() -> dict[str, list[dict[str, Any]]]:
    """Return score_series for all domains."""
    history = load_history()
    return history.get("score_series", {})


def get_recent_changes(days: int = 30) -> list[dict[str, Any]]:
    """Return regime changes from the last N days."""
    history = load_history()
    all_changes = history.get("changes", [])

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    return [c for c in all_changes if c.get("date", "") >= cutoff]

"""
Regime history tracking — persists regime state for change detection.

Stores at ~/.lox/regime_history.json so it persists across CLI runs.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HISTORY_FILE = Path.home() / ".lox" / "regime_history.json"


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
    """Save current regime state and detect changes.

    Args:
        results: dict mapping domain name -> RegimeResult-like object
                 (must have .label, .score attributes)

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

    for domain, result in results.items():
        label = getattr(result, "label", str(result))
        score = getattr(result, "score", 50)

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
            "date": today,
        }

    history["last_updated"] = today

    # Persist
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save regime history: {e}")

    return changes


def get_recent_changes(days: int = 30) -> list[dict[str, Any]]:
    """Return regime changes from the last N days."""
    history = load_history()
    all_changes = history.get("changes", [])

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    return [c for c in all_changes if c.get("date", "") >= cutoff]

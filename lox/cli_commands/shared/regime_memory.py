"""
Persistent regime memory — remembers the last --llm session per domain.

Stores at ~/.lox/regime_memory/<domain>.json so each regime command knows
what the data looked like the last time the user ran --llm, enabling the
LLM to diff the snapshots and discuss catalysts that drove the changes.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MEMORY_DIR = Path.home() / ".lox" / "regime_memory"


def _memory_path(domain: str) -> Path:
    return MEMORY_DIR / f"{domain.lower()}.json"


def save_session(
    domain: str,
    snapshot: dict[str, Any],
    regime_label: str | None,
    regime_description: str | None = None,
) -> None:
    """Persist the current regime snapshot as the 'last viewed' session."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "domain": domain,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "regime_label": regime_label,
        "regime_description": regime_description,
        "snapshot": _safe_serialize(snapshot),
    }
    try:
        with open(_memory_path(domain), "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as exc:
        logger.warning("Failed to save regime memory for %s: %s", domain, exc)


def load_previous_session(domain: str) -> dict[str, Any] | None:
    """Load the last-viewed session for *domain*.  Returns None if no history."""
    path = _memory_path(domain)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load regime memory for %s: %s", domain, exc)
        return None


def build_diff_context(
    current_snapshot: dict[str, Any],
    current_label: str | None,
    previous_session: dict[str, Any],
) -> str:
    """
    Build a human-readable diff between the current and previous snapshot
    suitable for inclusion in the LLM system prompt.
    """
    prev_snap = previous_session.get("snapshot", {})
    prev_label = previous_session.get("regime_label")
    prev_ts = previous_session.get("timestamp", "unknown")

    try:
        dt = datetime.fromisoformat(prev_ts.replace("Z", "+00:00"))
        prev_date_str = dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        prev_date_str = prev_ts

    lines: list[str] = [
        f"## Changes Since Last Review ({prev_date_str})",
    ]

    if prev_label and current_label and prev_label != current_label:
        lines.append(f"**Regime shifted: {prev_label}  →  {current_label}**")
    elif prev_label:
        lines.append(f"Regime unchanged: {current_label}")

    changes: list[str] = []
    all_keys = sorted(set(list(current_snapshot.keys()) + list(prev_snap.keys())))

    for key in all_keys:
        cur = current_snapshot.get(key)
        prev = prev_snap.get(key)

        if cur is None and prev is None:
            continue
        if cur == prev:
            continue

        if isinstance(cur, (int, float)) and isinstance(prev, (int, float)):
            delta = cur - prev
            if abs(prev) > 1e-9:
                pct = (delta / abs(prev)) * 100.0
                changes.append(f"- **{key}**: {prev:.4g} → {cur:.4g}  (Δ {delta:+.4g}, {pct:+.1f}%)")
            else:
                changes.append(f"- **{key}**: {prev} → {cur}  (Δ {delta:+.4g})")
        elif cur is None:
            changes.append(f"- **{key}**: {prev} → n/a  (dropped)")
        elif prev is None:
            changes.append(f"- **{key}**: n/a → {cur}  (new)")
        else:
            changes.append(f"- **{key}**: {prev} → {cur}")

    if changes:
        lines.append("\nMetric changes:")
        lines.extend(changes)
    else:
        lines.append("\nNo numeric metric changes detected.")

    return "\n".join(lines)


def _safe_serialize(obj: Any) -> Any:
    """Recursively make a snapshot JSON-safe."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)

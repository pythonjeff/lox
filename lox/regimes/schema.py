from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional


def _is_finite_number(x: Any) -> bool:
    try:
        # Avoid importing numpy; keep it simple.
        if x is None:
            return False
        v = float(x)
        return v == v and v not in (float("inf"), float("-inf"))
    except Exception:
        return False


def add_feature(features: Dict[str, float], key: str, value: Any) -> None:
    """Add a scalar feature if value is a finite number."""
    if not key or not key.strip():
        return
    if _is_finite_number(value):
        features[key] = float(value)


def add_bool_feature(features: Dict[str, float], key: str, value: Any) -> None:
    """Encode boolean-ish values as 1.0 / 0.0."""
    if value is None:
        return
    features[key] = 1.0 if bool(value) else 0.0


def add_one_hot(
    features: Dict[str, float],
    prefix: str,
    active: Optional[str],
    choices: Iterable[str],
) -> None:
    """Add one-hot features as scalars: {prefix}.{choice} ∈ {0.0, 1.0}."""
    pref = prefix.strip().rstrip(".")
    for c in choices:
        k = f"{pref}.{c}"
        features[k] = 1.0 if (active is not None and active == c) else 0.0


@dataclass(frozen=True)
class RegimeVector:
    """A flat ML-friendly feature vector with optional metadata."""

    asof: str
    features: Dict[str, float]
    notes: str = ""

    def to_flat_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"asof": self.asof, **self.features}
        if self.notes:
            out["notes"] = self.notes
        return out


def merge_feature_dicts(*parts: Mapping[str, float]) -> Dict[str, float]:
    """Merge feature dicts; later dicts override earlier on key collisions."""
    merged: Dict[str, float] = {}
    for p in parts:
        for k, v in p.items():
            merged[k] = float(v)
    return merged



"""
Quarterly Refunding Announcement (QRA) — TGA end-of-quarter targets.

Each QRA (1st Wed of Feb/May/Aug/Nov) Treasury publishes its assumed
end-of-quarter cash balance for the upcoming and following quarter ends.
These targets shape Treasury's bill issuance pace, which directly drives
TGA refilling/draining cycles. Distance-to-target is therefore a leading
signal for upcoming reserve drain (refill phase) or release (drain phase).

Treasury does NOT expose a structured API — values come from the PDF
refunding statement. We hand-maintain a small table here. After each
QRA day, update with the new (announcement_date, eoq_date, target_b)
tuple.

Source documents:
    https://home.treasury.gov/policy-issues/financing-the-government/quarterly-refunding

Public API:
    latest_active_target(today=None) -> dict | None
        Returns the active eoq target whose eoq_date >= today, taken from the
        most recent announcement on or before today. None if no target known.

    compute_qra_target_metrics(refresh=False, today=None) -> dict
        Combines target with current daily TGA level for panel display.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class QRATarget:
    announcement_date: date
    eoq_date: date
    target_b: float


# ─────────────────────────────────────────────────────────────────────────────
# Hand-maintained QRA target table.
# After each QRA, append a row with (announcement, eoq quarter end, target $B).
# Each QRA typically announces a target for the *upcoming* quarter end and a
# preliminary one for the quarter after — store both.
# ─────────────────────────────────────────────────────────────────────────────
QRA_TARGETS: list[QRATarget] = [
    # 2024 cycle (historically confirmed)
    QRATarget(date(2024, 1, 31),  date(2024, 3, 31),  750.0),
    QRATarget(date(2024, 1, 31),  date(2024, 6, 30),  750.0),
    QRATarget(date(2024, 4, 29),  date(2024, 6, 30),  850.0),
    QRATarget(date(2024, 4, 29),  date(2024, 9, 30),  850.0),
    QRATarget(date(2024, 7, 29),  date(2024, 9, 30),  850.0),
    QRATarget(date(2024, 7, 29),  date(2024, 12, 31), 700.0),
    QRATarget(date(2024, 10, 28), date(2024, 12, 31), 700.0),
    QRATarget(date(2024, 10, 28), date(2025, 3, 31),  850.0),
    # 2025 cycle — sourced from Treasury "Marketable Borrowing Estimates"
    # press releases (sb0007, Apr 28 2025, sb0209, sb0300).
    QRATarget(date(2025, 2, 3),   date(2025, 3, 31),  850.0),
    QRATarget(date(2025, 2, 3),   date(2025, 6, 30),  850.0),
    QRATarget(date(2025, 4, 28),  date(2025, 6, 30),  850.0),
    QRATarget(date(2025, 4, 28),  date(2025, 9, 30),  850.0),
    QRATarget(date(2025, 8, 4),   date(2025, 9, 30),  850.0),
    QRATarget(date(2025, 8, 4),   date(2025, 12, 31), 850.0),
    QRATarget(date(2025, 11, 3),  date(2025, 12, 31), 850.0),
    QRATarget(date(2025, 11, 3),  date(2026, 3, 31),  850.0),
    # 2026 cycle (sb0377, Feb 2 2026): cash policy bumped to $900B for Q2.
    QRATarget(date(2026, 2, 2),   date(2026, 3, 31),  850.0),
    QRATarget(date(2026, 2, 2),   date(2026, 6, 30),  900.0),
]


def latest_active_target(today: Optional[date] = None) -> Optional[QRATarget]:
    """
    Most recent target whose eoq_date is in the future and whose announcement
    is on or before today. Returns None if no such target exists.
    """
    if today is None:
        today = date.today()
    candidates = [
        t for t in QRA_TARGETS
        if t.announcement_date <= today and t.eoq_date >= today
    ]
    if not candidates:
        return None
    # Prefer the most recent announcement, then the nearest eoq.
    candidates.sort(key=lambda t: (t.announcement_date, -(t.eoq_date.toordinal())))
    return candidates[-1]


def _business_days_until(target: date, today: date) -> int:
    import pandas as pd
    if target <= today:
        return 0
    return int(len(pd.bdate_range(today, target)) - 1)


def compute_qra_target_metrics(*, refresh: bool = False, today: Optional[date] = None) -> dict:
    """
    Combine the latest QRA target with the current daily TGA level.

    Returns dict with:
        target_b           — announced EOQ target ($B)
        eoq_date           — quarter-end date the target applies to
        days_remaining     — business days until eoq_date
        announcement_date  — QRA date the target was announced on
        current_tga_b      — current daily TGA level (DTS)
        distance_b         — current minus target (positive = above target)
        pct_to_target      — current / target as 0..1+
    Returns dict with None values if no target or no TGA available.
    """
    from lox.gov.dts import compute_tga_daily_metrics

    if today is None:
        today = date.today()

    empty = {
        "target_b": None,
        "eoq_date": None,
        "days_remaining": None,
        "announcement_date": None,
        "current_tga_b": None,
        "distance_b": None,
        "pct_to_target": None,
    }
    target = latest_active_target(today)
    if target is None:
        return empty

    dts = compute_tga_daily_metrics(refresh=refresh)
    current = dts.get("level_b")

    distance = None
    pct = None
    if isinstance(current, (int, float)) and target.target_b > 0:
        distance = float(current) - target.target_b
        pct = float(current) / target.target_b

    return {
        "target_b": target.target_b,
        "eoq_date": str(target.eoq_date),
        "days_remaining": _business_days_until(target.eoq_date, today),
        "announcement_date": str(target.announcement_date),
        "current_tga_b": float(current) if isinstance(current, (int, float)) else None,
        "distance_b": distance,
        "pct_to_target": pct,
    }

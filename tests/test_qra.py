"""Tests for QRA target table and metrics."""
from __future__ import annotations

from datetime import date

import lox.gov.qra as qra
from lox.gov.qra import QRATarget


def _patch_targets(monkeypatch, targets):
    monkeypatch.setattr(qra, "QRA_TARGETS", targets)


def test_latest_active_target_picks_most_recent_announcement(monkeypatch):
    _patch_targets(monkeypatch, [
        QRATarget(date(2026, 2, 4), date(2026, 6, 30), 850.0),
        QRATarget(date(2026, 2, 4), date(2026, 9, 30), 900.0),
        QRATarget(date(2026, 5, 6), date(2026, 9, 30), 925.0),
    ])
    # On May 7, the May 6 announcement is fresh; nearest future eoq is Sep 30.
    t = qra.latest_active_target(today=date(2026, 5, 7))
    assert t is not None
    assert t.announcement_date == date(2026, 5, 6)
    assert t.target_b == 925.0


def test_latest_active_target_skips_past_eoq(monkeypatch):
    _patch_targets(monkeypatch, [
        QRATarget(date(2025, 11, 5), date(2026, 3, 31), 850.0),
    ])
    # Mar 31 already passed; no active target.
    t = qra.latest_active_target(today=date(2026, 5, 1))
    assert t is None


def test_latest_active_target_ignores_future_announcements(monkeypatch):
    _patch_targets(monkeypatch, [
        QRATarget(date(2026, 8, 5), date(2026, 12, 31), 950.0),
    ])
    # Aug 5 announcement is in the future relative to today; skip it.
    t = qra.latest_active_target(today=date(2026, 5, 1))
    assert t is None


def test_latest_active_target_empty_table(monkeypatch):
    _patch_targets(monkeypatch, [])
    assert qra.latest_active_target(today=date(2026, 5, 1)) is None


def test_compute_qra_target_metrics_combines_with_dts(monkeypatch):
    _patch_targets(monkeypatch, [
        QRATarget(date(2026, 2, 4), date(2026, 6, 30), 850.0),
    ])
    monkeypatch.setattr(
        "lox.gov.dts.compute_tga_daily_metrics",
        lambda **kw: {"level_b": 969.0},
    )
    m = qra.compute_qra_target_metrics(today=date(2026, 5, 1))
    assert m["target_b"] == 850.0
    assert m["eoq_date"] == "2026-06-30"
    assert m["current_tga_b"] == 969.0
    assert abs(m["distance_b"] - 119.0) < 1e-6
    assert m["days_remaining"] > 0


def test_compute_qra_target_metrics_no_target(monkeypatch):
    _patch_targets(monkeypatch, [])
    m = qra.compute_qra_target_metrics(today=date(2026, 5, 1))
    assert m["target_b"] is None
    assert m["distance_b"] is None

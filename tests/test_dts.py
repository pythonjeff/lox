"""Tests for daily TGA / DTS helpers."""
from __future__ import annotations

import pandas as pd

import lox.gov.dts as dts


def _series(values):
    return pd.DataFrame({
        "date": pd.bdate_range("2026-01-02", periods=len(values), freq="B").date,
        "tga_close_b": list(values),
    })


def test_avg_daily_change_b_basic():
    # 11 obs, end=200 start=100, window=10 → 10/day
    df = _series(range(100, 211, 10))  # 12 values 100..210
    assert dts._avg_daily_change_b(df, 10) == 10.0


def test_avg_daily_change_b_too_short():
    df = _series([100, 110, 120])
    assert dts._avg_daily_change_b(df, 10) is None


def test_days_to_floor_draining():
    # level $500B, floor $50B, burn -$5B/day → 90 days
    assert dts._days_to_floor(500.0, 50.0, -5.0) == 90.0


def test_days_to_floor_refilling_returns_none():
    assert dts._days_to_floor(500.0, 50.0, +1.0) is None


def test_days_to_floor_already_below_returns_zero():
    assert dts._days_to_floor(40.0, 50.0, -1.0) == 0.0


def test_days_to_floor_no_burn_data():
    assert dts._days_to_floor(500.0, 50.0, None) is None


def test_compute_metrics_includes_burn_and_d2f(monkeypatch):
    # Monotonically draining series → burn negative, d2f finite
    sample = _series([1000.0 - i * 10 for i in range(35)])
    monkeypatch.setattr(dts, "fetch_tga_daily", lambda **kw: sample)

    m = dts.compute_tga_daily_metrics()
    assert m["level_b"] == sample.iloc[-1]["tga_close_b"]
    assert m["burn_10d_b"] == -10.0
    assert m["burn_30d_b"] == -10.0
    # d2f = (level - 50) / 10
    expected = (sample.iloc[-1]["tga_close_b"] - 50.0) / 10.0
    assert abs(m["days_to_floor_10d"] - expected) < 1e-6
    assert abs(m["days_to_floor_30d"] - expected) < 1e-6


def test_compute_metrics_refilling_d2f_none(monkeypatch):
    sample = _series([100.0 + i * 5 for i in range(35)])
    monkeypatch.setattr(dts, "fetch_tga_daily", lambda **kw: sample)
    m = dts.compute_tga_daily_metrics()
    assert m["burn_10d_b"] == 5.0
    assert m["days_to_floor_10d"] is None
    assert m["days_to_floor_30d"] is None

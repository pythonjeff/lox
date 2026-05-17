"""Smoke tests for the funding cross-correlation pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lox.funding.correlations import (
    _classify_strength,
    classify_funding_regime,
    compute_lead_lag,
    compute_pair_correlations,
    EQUITY_LAG_PAIRS,
    EQUITY_PAIR_META,
    PAIR_META,
)


def _synthetic_dataset(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic daily dataset where:
      - reserves_b drifts down ~$300B over period
      - sofr_iorb_bps inversely tracks reserves (the "expected" relationship)
      - rrp_b drifts down + has correlation with reserves
      - tga_b drifts up (drains reserves)
      - walcl_b grows late in period (Fed expansion)
      - bills_b grows late in period
    Then computes Wed-only diffs for change cols.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-04", periods=n, freq="B")  # Wednesday start

    # Reserves: starts $3.5T, drifts down to ~$3.2T with weekly noise
    reserves = 3_500 - 0.4 * np.arange(n) + rng.normal(0, 5, n)
    # SOFR-IORB inversely tracks reserves with lag-shift
    sofr_iorb = -0.05 * (reserves - reserves.mean()) + rng.normal(0, 2, n)
    sofr_effr = sofr_iorb * 0.5 + rng.normal(0, 1, n)
    # RRP: starts $2T, drifts to $50B (the recent regime)
    rrp = np.maximum(50, 2_000 - 2.5 * np.arange(n) + rng.normal(0, 30, n))
    # TGA: bounces around $700B
    tga = 700 + 100 * np.sin(np.arange(n) / 30) + rng.normal(0, 20, n)
    # WALCL: declining QT
    walcl = 7_700 - 1.2 * np.arange(n) + rng.normal(0, 5, n)
    # Bills: late-period growth
    bills = 200 + np.maximum(0, np.arange(n) - 400) * 0.5 + rng.normal(0, 3, n)

    df = pd.DataFrame({
        "date": dates,
        "sofr_iorb_bps": sofr_iorb,
        "sofr_effr_bps": sofr_effr,
        "reserves_b": reserves,
        "rrp_b": rrp,
        "tga_b": tga,
        "walcl_b": walcl,
        "bills_b": bills,
    })
    df["net_liq_t"] = (df["reserves_b"] - df["tga_b"] - df["rrp_b"]) / 1000.0

    # Wednesday-cadence diffs (same logic as build_correlation_dataset)
    wed = df[df["date"].dt.dayofweek == 2].copy()
    for col in ("tga_b", "rrp_b", "reserves_b", "walcl_b", "bills_b"):
        wed[f"d_{col}"] = wed[col].diff()
    diff_cols = [c for c in wed.columns if c.startswith("d_")]
    df = df.merge(wed[["date"] + diff_cols], on="date", how="left")
    for col in diff_cols:
        df[col] = df[col].ffill()

    return df


def test_classify_strength_directions():
    # Strong correlation in expected direction
    label, status = _classify_strength(-0.65, -0.30, "negative")
    assert status == "strong"
    # Sign-flipped current
    label, status = _classify_strength(+0.40, -0.30, "negative")
    assert status == "flipped"
    # Decoupled (small magnitude)
    label, status = _classify_strength(+0.05, -0.30, "negative")
    assert status == "broken"
    # Weaker than baseline
    label, status = _classify_strength(-0.20, -0.50, "negative")
    assert status == "weak"
    # Normal
    label, status = _classify_strength(-0.30, -0.28, "negative")
    assert status == "normal"
    # Positive expected sign
    label, status = _classify_strength(+0.60, +0.30, "positive")
    assert status == "strong"


def test_classify_strength_newly_active():
    # Baseline was sign-flipped weak (+0.10 when expected neg);
    # current is now in expected direction at -0.30 — that's a meaningful swing.
    label, status = _classify_strength(-0.30, +0.10, "negative")
    # cur_norm = +0.30, base_norm = -0.10, delta_norm = +0.40 → "strong"
    assert status == "strong"


def test_compute_pair_correlations_returns_all_pairs():
    df = _synthetic_dataset()
    pairs = compute_pair_correlations(df, window=60)
    # Should get one result per PAIR_META entry (all cols are present in synthetic df)
    names = {p["name"] for p in pairs}
    expected = {m["name"] for m in PAIR_META}
    assert names == expected


def test_compute_pair_correlations_detects_expected_signs():
    """SOFR-IORB ↔ Reserves was constructed with negative correlation — verify."""
    df = _synthetic_dataset()
    pairs = compute_pair_correlations(df, window=60)
    by_name = {p["name"]: p for p in pairs}
    sofr_res = by_name["SOFR-IORB ↔ Bank Reserves"]
    assert sofr_res["current"] is not None
    assert sofr_res["current"] < 0  # negative as constructed


def test_compute_lead_lag_returns_results():
    df = _synthetic_dataset()
    lags = compute_lead_lag(df, window=60, max_lag_days=10)
    # Returns one entry per LAG_PAIRS (some may be filtered if corr too weak)
    assert isinstance(lags, list)
    for l in lags:
        assert "best_lag" in l
        assert "best_corr" in l
        assert 0 <= l["best_lag"] <= 10


def test_regime_classification_reserve_constrained():
    """Hand-crafted pair_results where reserves are dominant and RRP is broken."""
    pair_results = [
        {"name": "SOFR-IORB ↔ Bank Reserves", "current": -0.65, "baseline": -0.30,
         "status": "strong", "expected_sign": "negative", "interpretation": "active"},
        {"name": "SOFR-IORB ↔ ON RRP", "current": -0.05, "baseline": -0.45,
         "status": "broken", "expected_sign": "negative", "interpretation": "broken"},
        {"name": "SOFR-IORB ↔ Net Liquidity", "current": -0.30, "baseline": -0.30,
         "status": "normal", "expected_sign": "negative", "interpretation": None},
    ]
    out = classify_funding_regime(pair_results, [])
    assert out["regime"] == "RESERVE-CONSTRAINED"
    assert "reserves" in out["rationale"].lower()
    # The broken RRP and strong reserves pair both flagged
    assert len(out["divergences"]) >= 2


def test_regime_balanced_when_nothing_dominant():
    pair_results = [
        {"name": "SOFR-IORB ↔ Bank Reserves", "current": -0.25, "baseline": -0.22,
         "status": "normal", "expected_sign": "negative", "interpretation": None},
        {"name": "SOFR-IORB ↔ ON RRP", "current": -0.20, "baseline": -0.25,
         "status": "normal", "expected_sign": "negative", "interpretation": None},
    ]
    out = classify_funding_regime(pair_results, [])
    assert out["regime"] == "BALANCED"


def test_lead_lag_tradeable_filter():
    """Constructed lag result with strong matching-sign correlation → tradeable."""
    df = _synthetic_dataset()
    lags = compute_lead_lag(df, window=60, max_lag_days=10)
    # `tradeable` requires |corr| > 0.30, sign matches, and lag >= 3
    for l in lags:
        if l["tradeable"]:
            assert abs(l["best_corr"]) > 0.30
            assert l["sign_matches"]
            assert l["best_lag"] >= 3


# ── Fed ↔ equity transmission tests ──────────────────────────────────────────

def _synthetic_equity_dataset(n: int = 800, seed: int = 7) -> pd.DataFrame:
    """
    Extend the plumbing synthetic data with sp500/vix that have the *expected*
    couplings: SPX returns track ΔNet Liq; VIX inversely tracks Net Liq and
    co-moves with corridor stress.
    """
    df = _synthetic_dataset(n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    # Wed-cadence diff of net_liq_t — production builder computes this; the
    # base synthetic helper doesn't, so we add it here.
    wed = df[df["date"].dt.dayofweek == 2].copy()
    wed["d_net_liq_t"] = wed["net_liq_t"].diff()
    df = df.merge(wed[["date", "d_net_liq_t"]], on="date", how="left")
    df["d_net_liq_t"] = df["d_net_liq_t"].ffill()
    # VIX: baseline 17, kicks up with sofr_iorb stress, calmer with high net liq
    df["vix"] = (
        17.0
        + 0.4 * df["sofr_iorb_bps"]
        - 6.0 * (df["net_liq_t"] - df["net_liq_t"].mean())
        + rng.normal(0, 1.5, len(df))
    ).clip(lower=9.0)
    # SP500: random walk with drift tied to d_net_liq_t
    rets = 0.0003 + 0.0008 * df["d_net_liq_t"].fillna(0) + rng.normal(0, 0.01, len(df))
    df["sp500"] = 4000.0 * np.exp(np.cumsum(rets))
    # d_sp500_pct on Wed cadence (same construction as build_correlation_dataset)
    wed2 = df[df["date"].dt.dayofweek == 2].copy()
    wed2["d_sp500_pct"] = wed2["sp500"].pct_change() * 100.0
    df = df.merge(wed2[["date", "d_sp500_pct"]], on="date", how="left")
    df["d_sp500_pct"] = df["d_sp500_pct"].ffill()
    return df


def test_equity_pair_meta_columns_resolvable():
    """Every EQUITY_PAIR_META entry references columns the synthetic builder produces."""
    df = _synthetic_equity_dataset()
    for meta in EQUITY_PAIR_META:
        assert meta["x"] in df.columns, f"missing x={meta['x']} for {meta['name']}"
        assert meta["y"] in df.columns, f"missing y={meta['y']} for {meta['name']}"


def test_compute_equity_pair_correlations():
    """Pass EQUITY_PAIR_META through meta_list — should produce one row per pair."""
    df = _synthetic_equity_dataset()
    eq = compute_pair_correlations(df, window=60, meta_list=EQUITY_PAIR_META)
    names = {p["name"] for p in eq}
    expected = {m["name"] for m in EQUITY_PAIR_META}
    assert names == expected


def test_equity_pairs_detect_constructed_signs():
    """ΔNet Liq ↔ ΔSPX was constructed positive; SOFR-IORB ↔ VIX positive too."""
    df = _synthetic_equity_dataset()
    eq = compute_pair_correlations(df, window=60, meta_list=EQUITY_PAIR_META)
    by_name = {p["name"]: p for p in eq}
    nl_spx = by_name["ΔNet Liquidity ↔ ΔSPX"]
    assert nl_spx["current"] is not None
    assert nl_spx["current"] > 0  # constructed positive coupling
    sofr_vix = by_name["SOFR-IORB ↔ VIX"]
    assert sofr_vix["current"] is not None
    assert sofr_vix["current"] > 0  # constructed positive coupling


def test_equity_lead_lag_runs():
    """compute_lead_lag accepts a custom lag_pairs arg and returns shaped results."""
    df = _synthetic_equity_dataset()
    lags = compute_lead_lag(df, window=60, max_lag_days=20, lag_pairs=EQUITY_LAG_PAIRS)
    assert isinstance(lags, list)
    for l in lags:
        assert "best_lag" in l and 0 <= l["best_lag"] <= 20
        assert "best_corr" in l


def test_plumbing_meta_default_unaffected_by_equity():
    """Default meta path (PAIR_META) still produces the original plumbing pair set."""
    df = _synthetic_equity_dataset()
    plumbing = compute_pair_correlations(df, window=60)  # default = PAIR_META
    names = {p["name"] for p in plumbing}
    # Equity pair names must NOT appear in the default plumbing output
    assert not any(
        n in names for n in {m["name"] for m in EQUITY_PAIR_META}
    )

"""Smoke tests for DTS fiscal-behavior breakdown."""
from __future__ import annotations

import pandas as pd

from lox.gov import dts_flows
from lox.gov.dts_flows import (
    REVENUE_BUCKETS,
    SPENDING_BUCKETS,
    _bucket_breakdown,
    _bucket_for,
    _yoy_pace,
)


def test_bucket_for_revenue_keys():
    assert _bucket_for("Taxes - Withheld Individual/FICA", REVENUE_BUCKETS)[0] == "income_withheld"
    assert _bucket_for("Taxes - Non Withheld Ind/SECA Electronic", REVENUE_BUCKETS)[0] == "income_unwithheld"
    assert _bucket_for("Taxes - Corporate Income", REVENUE_BUCKETS)[0] == "corporate_tax"
    assert _bucket_for("DHS - Customs Duties, Taxes, and Fees", REVENUE_BUCKETS)[0] == "customs"
    assert _bucket_for("Federal Reserve Earnings", REVENUE_BUCKETS)[0] == "fed_reserve"
    assert _bucket_for("Random Agency Misc", REVENUE_BUCKETS) is None


def test_bucket_for_spending_keys():
    assert _bucket_for("Interest on Treasury Securities", SPENDING_BUCKETS)[0] == "interest"
    assert _bucket_for("Social Security Admin (SSA) - misc", SPENDING_BUCKETS)[0] == "social_security"
    assert _bucket_for("SSA - Supplemental Security Income", SPENDING_BUCKETS)[0] == "social_security"
    assert _bucket_for("HHS - Grants to States for Medicaid", SPENDING_BUCKETS)[0] == "medicaid"
    assert _bucket_for("HHS - Medicare Prescription Drugs", SPENDING_BUCKETS)[0] == "medicare"
    assert _bucket_for("DoD - Military Active Duty Pay", SPENDING_BUCKETS)[0] == "defense"
    assert _bucket_for("Dept of Veterans Affairs", SPENDING_BUCKETS)[0] == "veterans"


def test_bucket_breakdown_excludes_public_debt():
    """Public Debt Cash Issues/Redemp should NOT show up in operating breakdown."""
    df = pd.DataFrame([
        {"transaction_type": "Deposits", "transaction_catg": "Taxes - Withheld Individual/FICA",
         "amount_m": 50_000, "mtd_m": 0, "fytd_m": 0, "date": "2026-05-08"},
        {"transaction_type": "Deposits", "transaction_catg": "Public Debt Cash Issues (Table IIIB)",
         "amount_m": 500_000, "mtd_m": 0, "fytd_m": 0, "date": "2026-05-08"},
        {"transaction_type": "Deposits", "transaction_catg": "Taxes - Corporate Income",
         "amount_m": 2_000, "mtd_m": 0, "fytd_m": 0, "date": "2026-05-08"},
    ])
    rows = _bucket_breakdown(df, "Deposits", REVENUE_BUCKETS)
    keys = {r["key"] for r in rows}
    assert "income_withheld" in keys
    assert "corporate_tax" in keys
    # The public-debt $500B should NOT inflate any bucket (especially "other")
    other = next((r for r in rows if r["key"] == "other"), None)
    if other:
        assert other["amount_b"] < 1.0  # < $1B, not $500B


def test_yoy_pace_handles_missing_prior():
    current = {"interest": 387.0, "medicaid": 428.0}
    prior = {"interest": 328.0}  # medicaid missing
    out = _yoy_pace(current, prior)
    assert abs(out["interest"] - ((387 - 328) / 328 * 100)) < 0.1
    assert out["medicaid"] is None


def test_yoy_pace_handles_zero_prior():
    current = {"customs": 208.0}
    prior = {"customs": 0.0}
    out = _yoy_pace(current, prior)
    assert out["customs"] is None  # avoid divide-by-zero


def test_compute_breakdown_empty_safe(monkeypatch):
    """When FiscalDataClient returns empty, the breakdown returns empty shape."""
    def _empty_fetch(*args, **kwargs):
        return pd.DataFrame()
    monkeypatch.setattr(dts_flows, "fetch_dts_flows", _empty_fetch)
    out = dts_flows.compute_dts_flow_breakdown()
    assert out["asof"] is None
    assert out["revenue_5d"] == []
    assert out["spending_5d"] == []
    assert out["net_operating_5d_b"] is None


def test_compute_breakdown_with_synthetic_data(monkeypatch):
    """Feed a hand-crafted 5d DTS window and verify the structure."""
    rows = []
    dates = ["2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08"]
    for d in dates:
        rows.extend([
            {"date": pd.to_datetime(d).date(),
             "transaction_type": "Deposits",
             "transaction_catg": "Taxes - Withheld Individual/FICA",
             "amount_m": 10_000, "mtd_m": 50_000, "fytd_m": 2_200_000},
            {"date": pd.to_datetime(d).date(),
             "transaction_type": "Deposits",
             "transaction_catg": "Taxes - Corporate Income",
             "amount_m": 500, "mtd_m": 2_500, "fytd_m": 220_000},
            {"date": pd.to_datetime(d).date(),
             "transaction_type": "Withdrawals",
             "transaction_catg": "Interest on Treasury Securities",
             "amount_m": 200, "mtd_m": 1_000, "fytd_m": 386_000},
            {"date": pd.to_datetime(d).date(),
             "transaction_type": "Withdrawals",
             "transaction_catg": "HHS - Grants to States for Medicaid",
             "amount_m": 4_000, "mtd_m": 20_000, "fytd_m": 427_000},
            {"date": pd.to_datetime(d).date(),
             "transaction_type": "Deposits",
             "transaction_catg": "Public Debt Cash Issues (Table IIIB)",
             "amount_m": 100_000, "mtd_m": 500_000, "fytd_m": 10_000_000},
            {"date": pd.to_datetime(d).date(),
             "transaction_type": "Withdrawals",
             "transaction_catg": "Public Debt Cash Redemp. (Table IIIB)",
             "amount_m": 100_000, "mtd_m": 500_000, "fytd_m": 10_000_000},
        ])
    df = pd.DataFrame(rows)

    def _fake_fetch(*args, **kwargs):
        return df
    monkeypatch.setattr(dts_flows, "fetch_dts_flows", _fake_fetch)

    # Patch the FiscalDataClient.fetch for the prior-year call to return empty
    class _FakeClient:
        def fetch(self, *args, **kwargs):
            return pd.DataFrame()
    monkeypatch.setattr(dts_flows, "FiscalDataClient", lambda: _FakeClient())

    out = dts_flows.compute_dts_flow_breakdown()
    assert out["asof"] == "2026-05-08"
    assert out["window_days"] == 5

    # Revenue breakdown should have withheld and corporate
    rev_keys = {r["key"] for r in out["revenue_5d"]}
    assert "income_withheld" in rev_keys
    assert "corporate_tax" in rev_keys
    withheld = next(r for r in out["revenue_5d"] if r["key"] == "income_withheld")
    # 5 days × $10B = $50B
    assert abs(withheld["amount_b"] - 50.0) < 0.01

    # Spending breakdown should have interest and medicaid
    spd_keys = {r["key"] for r in out["spending_5d"]}
    assert "interest" in spd_keys
    assert "medicaid" in spd_keys

    # Net operating = ($10B + $0.5B - $0.2B - $4B) × 5 days = $31.5B
    expected_op = ((10_000 + 500 - 200 - 4_000) * 5) / 1000.0
    assert abs(out["net_operating_5d_b"] - expected_op) < 0.01

    # Debt flow = issued - redeemed = 0 (equal in synthetic data)
    assert abs(out["debt_flow_5d_b"]) < 0.01

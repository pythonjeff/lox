"""
DTS deposits/withdrawals — fiscal behavior breakdown.

Decomposes daily TGA flow into operating revenue (deposits) and spending
(withdrawals), excluding mechanical public-debt issuance/redemption flows.
Buckets 180+ raw DTS categories into ~12 high-signal lines so you can see
WHY TGA is moving:

    - Customs duties surging?   (tariff regime impact)
    - Interest on debt growing? (debt-service crisis pace)
    - Corporate tax weakening?  (earnings cycle proxy)
    - Medicaid accelerating?    (state-cost passthrough)

This is the "behavior" view that sits underneath the daily TGA balance.

Public API:
    fetch_dts_flows(refresh, lookback_days) -> pd.DataFrame
    compute_dts_flow_breakdown(refresh) -> dict
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd

from lox.data.fiscaldata import FiscalDataClient, FiscalDataEndpoint


_DTS_DEPOSITS_WITHDRAWALS = FiscalDataEndpoint(
    path="/v1/accounting/dts/deposits_withdrawals_operating_cash",
)

# DTS reports amounts in $M; we convert to $B for display.
_M_PER_B = 1000.0

# ── Bucket mapping ───────────────────────────────────────────────────────────
# Each tuple: (bucket_key, display_label, list of substrings to match against
# raw transaction_catg). First match wins. Anything unmatched falls into
# "other_revenue" / "other_spending".
#
# IMPORTANT: "Public Debt Cash Issues" and "Public Debt Cash Redemp." are
# excluded entirely from operating flows — those are mechanical refunding.

REVENUE_BUCKETS: list[tuple[str, str, list[str]]] = [
    ("income_withheld",   "Income tax (withheld)",   ["Taxes - Withheld Individual"]),
    ("income_unwithheld", "Income tax (estimated)",  ["Taxes - Non Withheld Ind"]),
    ("corporate_tax",     "Corporate income tax",    ["Taxes - Corporate Income"]),
    ("customs",           "Customs duties",          ["Customs Duties"]),
    ("fed_reserve",       "Fed Reserve remittances", ["Federal Reserve Earnings"]),
    ("excise",            "Excise taxes",            ["Taxes - Miscellaneous Excise"]),
    ("estate_gift",       "Estate / gift / misc",    ["Estate and Gift", "Estate, Gift"]),
    ("medicare_premium",  "Medicare premiums",       ["Medicare Premiums"]),
    ("unemployment_in",   "Unemployment receipts",   ["State Unemployment", "Taxes - Federal Unemployment"]),
]

SPENDING_BUCKETS: list[tuple[str, str, list[str]]] = [
    ("social_security",   "Social Security",          ["Social Security Admin", "SSA - "]),
    ("medicare",          "Medicare",                 ["Medicare Prescription", "Othr Cent Medicare"]),
    ("medicaid",          "Medicaid",                 ["Medicaid"]),
    ("defense",           "Defense",                  ["DoD ", "Dept of Defense"]),
    ("interest",          "Interest on debt",         ["Interest on Treasury"]),
    ("veterans",          "Veterans Affairs",         ["Dept of Veterans Affairs", "DVA"]),
    ("federal_salaries",  "Federal salaries",         ["Federal Salaries"]),
    ("unemployment_out",  "Unemployment benefits",    ["Unemployment Benefits"]),
    ("education",         "Education",                ["Dept of Education"]),
    ("tax_refunds",       "Tax refunds",              ["IRS Tax Refunds", "Tax Refunds"]),
]

# Public-debt flows — tracked separately, NOT included in operating totals
DEBT_DEPOSIT_KEY = "Public Debt Cash Issues"
DEBT_WITHDRAWAL_KEY = "Public Debt Cash Redemp"


def _bucket_for(category: str, table: list[tuple[str, str, list[str]]]) -> Optional[tuple[str, str]]:
    """Return (key, label) of the matching bucket, or None."""
    for key, label, patterns in table:
        for p in patterns:
            if p in category:
                return (key, label)
    return None


def fetch_dts_flows(
    *,
    refresh: bool = False,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """
    Fetch DTS deposits/withdrawals over the lookback window.

    Returns DataFrame with columns:
        date, transaction_type, transaction_catg, amount_m, mtd_m, fytd_m

    Empty DataFrame if fetch fails.
    """
    client = FiscalDataClient()
    today = date.today()
    start = (pd.Timestamp(today) - pd.Timedelta(days=lookback_days * 2)).date().isoformat()

    df = client.fetch(
        endpoint=_DTS_DEPOSITS_WITHDRAWALS,
        params={
            "filter": f"record_date:gte:{start},account_type:eq:Treasury General Account (TGA)",
            "sort": "-record_date",
            "fields": "record_date,transaction_type,transaction_catg,transaction_today_amt,transaction_mtd_amt,transaction_fytd_amt",
        },
        cache_key=f"dts_flows_{start}",
        refresh=refresh,
    )
    if df.empty:
        return pd.DataFrame(columns=["date", "transaction_type", "transaction_catg", "amount_m", "mtd_m", "fytd_m"])

    df["date"] = pd.to_datetime(df["record_date"]).dt.date
    df["amount_m"] = pd.to_numeric(df["transaction_today_amt"], errors="coerce")
    df["mtd_m"] = pd.to_numeric(df["transaction_mtd_amt"], errors="coerce")
    df["fytd_m"] = pd.to_numeric(df["transaction_fytd_amt"], errors="coerce")
    df = df.dropna(subset=["amount_m"])
    return df[["date", "transaction_type", "transaction_catg", "amount_m", "mtd_m", "fytd_m"]].reset_index(drop=True)


def _bucket_breakdown(
    df_window: pd.DataFrame,
    transaction_type: str,
    table: list[tuple[str, str, list[str]]],
    amount_col: str = "amount_m",
) -> list[dict]:
    """
    Aggregate `amount_col` by bucket for one transaction_type (Deposits or Withdrawals).

    Returns rows sorted by absolute amount (largest first) including "Other" residual.
    """
    sub = df_window[df_window["transaction_type"] == transaction_type].copy()
    if sub.empty:
        return []

    # Skip the mechanical debt flows
    debt_key = DEBT_DEPOSIT_KEY if transaction_type == "Deposits" else DEBT_WITHDRAWAL_KEY
    sub = sub[~sub["transaction_catg"].str.contains(debt_key, na=False)]

    # Aggregate by raw category first
    agg = sub.groupby("transaction_catg", as_index=False)[amount_col].sum()

    # Then map raw to bucket
    bucket_totals: dict[str, dict] = {}
    other_total = 0.0
    for _, row in agg.iterrows():
        cat = row["transaction_catg"]
        amt = float(row[amount_col])
        match = _bucket_for(cat, table)
        if match is None:
            other_total += amt
            continue
        key, label = match
        if key not in bucket_totals:
            bucket_totals[key] = {"key": key, "label": label, "amount_b": 0.0}
        bucket_totals[key]["amount_b"] += amt

    # Convert M → B
    rows = []
    for entry in bucket_totals.values():
        entry["amount_b"] = entry["amount_b"] / _M_PER_B
        rows.append(entry)

    rows.sort(key=lambda r: r["amount_b"], reverse=True)
    rows.append({"key": "other", "label": "Other agencies", "amount_b": other_total / _M_PER_B})
    return rows


def _yoy_pace(
    current_fytd_by_bucket: dict[str, float],
    prior_fytd_by_bucket: dict[str, float],
) -> dict[str, Optional[float]]:
    """% change in FYTD per bucket (current vs prior fiscal year same period)."""
    out: dict[str, Optional[float]] = {}
    for k, cur in current_fytd_by_bucket.items():
        prior = prior_fytd_by_bucket.get(k)
        if prior is None or abs(prior) < 1e-6:
            out[k] = None
        else:
            out[k] = (cur - prior) / abs(prior) * 100.0
    return out


def _fytd_by_bucket(
    df_one_day: pd.DataFrame,
    transaction_type: str,
    table: list[tuple[str, str, list[str]]],
) -> dict[str, float]:
    """Single-day rows have FYTD totals — aggregate those by bucket."""
    return {
        r["key"]: r["amount_b"]
        for r in _bucket_breakdown(df_one_day, transaction_type, table, amount_col="fytd_m")
    }


def compute_dts_flow_breakdown(*, refresh: bool = False) -> dict:
    """
    Top-level: 5d flow per bucket + FYTD vs prior-FY YoY pace + net operating
    summary.

    Returns:
        {
            "asof": "YYYY-MM-DD" | None,
            "window_days": int,
            "revenue_5d": list[dict],      — bucket, label, 5d $B, yoy_pct
            "spending_5d": list[dict],     — bucket, label, 5d $B, yoy_pct
            "net_operating_5d_b": float,
            "debt_flow_5d_b": float,
            "largest_movers": list[dict],  — biggest week-on-week category swings
        }
    """
    empty = {
        "asof": None, "window_days": 0,
        "revenue_5d": [], "spending_5d": [],
        "net_operating_5d_b": None, "debt_flow_5d_b": None,
        "largest_movers": [],
    }

    try:
        df = fetch_dts_flows(refresh=refresh, lookback_days=14)
    except Exception:
        return empty
    if df.empty:
        return empty

    df = df.sort_values("date").reset_index(drop=True)
    asof = max(df["date"])

    # 5-business-day window (use last 5 unique dates, not last 5 calendar days)
    unique_dates = sorted(df["date"].unique())
    window_dates = unique_dates[-5:] if len(unique_dates) >= 5 else unique_dates
    window_df = df[df["date"].isin(window_dates)].copy()

    revenue_5d = _bucket_breakdown(window_df, "Deposits", REVENUE_BUCKETS)
    spending_5d = _bucket_breakdown(window_df, "Withdrawals", SPENDING_BUCKETS)

    # ── FYTD context: pull current asof + ~1y-ago FYTD for YoY ──────────
    asof_df = df[df["date"] == asof]
    current_rev_fytd = _fytd_by_bucket(asof_df, "Deposits", REVENUE_BUCKETS)
    current_spd_fytd = _fytd_by_bucket(asof_df, "Withdrawals", SPENDING_BUCKETS)

    # Prior-year FYTD: fetch single day ~365 days ago. Cache forever (it never changes).
    prior_rev_fytd: dict[str, float] = {}
    prior_spd_fytd: dict[str, float] = {}
    try:
        prior_date = asof - timedelta(days=365)
        prior_client = FiscalDataClient()
        prior_df = prior_client.fetch(
            endpoint=_DTS_DEPOSITS_WITHDRAWALS,
            params={
                "filter": f"record_date:gte:{prior_date - timedelta(days=10)},record_date:lte:{prior_date + timedelta(days=10)},account_type:eq:Treasury General Account (TGA)",
                "sort": "record_date",
                "fields": "record_date,transaction_type,transaction_catg,transaction_today_amt,transaction_mtd_amt,transaction_fytd_amt",
            },
            cache_key=f"dts_flows_prior_yoy_{prior_date}",
            refresh=False,
        )
        if not prior_df.empty:
            prior_df["date"] = pd.to_datetime(prior_df["record_date"]).dt.date
            prior_df["fytd_m"] = pd.to_numeric(prior_df["transaction_fytd_amt"], errors="coerce")
            prior_df["amount_m"] = pd.to_numeric(prior_df["transaction_today_amt"], errors="coerce")
            prior_df = prior_df.dropna(subset=["fytd_m"])
            # Pick the day closest to (asof - 365) that has data
            target = asof - timedelta(days=365)
            prior_df["distance"] = prior_df["date"].apply(lambda d: abs((d - target).days))
            best_day = prior_df.sort_values("distance").iloc[0]["date"]
            prior_one = prior_df[prior_df["date"] == best_day]
            prior_rev_fytd = _fytd_by_bucket(prior_one, "Deposits", REVENUE_BUCKETS)
            prior_spd_fytd = _fytd_by_bucket(prior_one, "Withdrawals", SPENDING_BUCKETS)
    except Exception:
        pass  # YoY is best-effort

    rev_yoy = _yoy_pace(current_rev_fytd, prior_rev_fytd)
    spd_yoy = _yoy_pace(current_spd_fytd, prior_spd_fytd)

    for r in revenue_5d:
        r["yoy_pct"] = rev_yoy.get(r["key"])
        r["fytd_b"] = current_rev_fytd.get(r["key"])
    for r in spending_5d:
        r["yoy_pct"] = spd_yoy.get(r["key"])
        r["fytd_b"] = current_spd_fytd.get(r["key"])

    # ── Net operating + debt-flow 5d ─────────────────────────────────────
    def _sum_amt(sub, label, key_match):
        s = sub[sub["transaction_catg"].str.contains(key_match, na=False)]
        return float(s["amount_m"].sum()) / _M_PER_B if not s.empty else 0.0

    dep = window_df[window_df["transaction_type"] == "Deposits"]
    wdr = window_df[window_df["transaction_type"] == "Withdrawals"]

    dep_op = dep[~dep["transaction_catg"].str.contains(DEBT_DEPOSIT_KEY, na=False)]
    wdr_op = wdr[~wdr["transaction_catg"].str.contains(DEBT_WITHDRAWAL_KEY, na=False)]
    net_op_b = (dep_op["amount_m"].sum() - wdr_op["amount_m"].sum()) / _M_PER_B

    debt_issues = _sum_amt(dep, "issues", DEBT_DEPOSIT_KEY)
    debt_redemp = _sum_amt(wdr, "redemp", DEBT_WITHDRAWAL_KEY)
    debt_flow_b = debt_issues - debt_redemp

    # ── Largest week-on-week movers (current 5d vs prior 5d, by bucket) ──
    movers: list[dict] = []
    if len(unique_dates) >= 10:
        prior_window = unique_dates[-10:-5]
        prior_window_df = df[df["date"].isin(prior_window)]
        for trans_type, table, label_prefix in [
            ("Deposits", REVENUE_BUCKETS, "Revenue: "),
            ("Withdrawals", SPENDING_BUCKETS, "Spend: "),
        ]:
            cur = {r["key"]: r for r in _bucket_breakdown(window_df, trans_type, table)}
            pri = {r["key"]: r for r in _bucket_breakdown(prior_window_df, trans_type, table)}
            for key, c in cur.items():
                p = pri.get(key)
                if not p:
                    continue
                delta = c["amount_b"] - p["amount_b"]
                if abs(delta) < 1.0:  # skip <$1B noise
                    continue
                movers.append({
                    "label": label_prefix + c["label"],
                    "delta_b": delta,
                    "current_b": c["amount_b"],
                    "prior_b": p["amount_b"],
                })
        movers.sort(key=lambda r: abs(r["delta_b"]), reverse=True)
        movers = movers[:5]

    return {
        "asof": str(asof),
        "window_days": len(window_dates),
        "revenue_5d": revenue_5d,
        "spending_5d": spending_5d,
        "net_operating_5d_b": float(net_op_b),
        "debt_flow_5d_b": float(debt_flow_b),
        "debt_issues_5d_b": float(debt_issues),
        "debt_redemp_5d_b": float(debt_redemp),
        "largest_movers": movers,
    }

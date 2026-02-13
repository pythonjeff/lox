"""
Lived Inflation Index — Computation engine.

Computes LII, CPI-weighted equivalent, spread, cumulative price levels,
and per-category breakdowns from BLS time-series data.
"""
from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd

from dashboard.lii_categories import (
    CATEGORIES,
    DEBT_CATEGORIES,
    ESSENTIALITY_MULTIPLIER,
    SCENARIO_PROFILES,
    calculate_lii_weights,
    get_all_series_ids,
)

logger = logging.getLogger(__name__)


def _normalize_cpi_weights(categories: list[dict] | None = None) -> list[dict]:
    """
    Return categories with 'cpi_norm' — the raw CPI weight normalized to sum to 1.0.
    This represents the official CPI weighting (no frequency adjustment).
    """
    from copy import deepcopy
    cats = deepcopy(categories or CATEGORIES)
    total = sum(c["cpi_weight"] for c in cats)
    for c in cats:
        c["cpi_norm"] = c["cpi_weight"] / total if total > 0 else 0.0
    return cats


def compute_lii_timeseries(
    bls_data: dict[str, pd.DataFrame],
    freq_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Compute monthly LII and CPI-weighted YoY series from BLS data.

    Returns DataFrame with columns: date, lii, cpi, spread
    (all in percentage terms, e.g. 3.2 = 3.2%).

    Uses NSA data — YoY naturally removes seasonality.
    """
    cats_lii = calculate_lii_weights(freq_overrides=freq_overrides)
    cats_cpi = _normalize_cpi_weights()  # raw CPI weights (no frequency adjustment)

    # Build a unified monthly index: merge all series by date
    # First, find the common date range
    all_dates: set[pd.Timestamp] = set()
    series_map: dict[str, pd.Series] = {}

    for cat in CATEGORIES:
        sid = cat["series_id"]
        df = bls_data.get(sid)
        if df is None or df.empty:
            logger.warning("Missing BLS data for %s (%s)", cat["name"], sid)
            continue
        s = df.set_index("date")["value"]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        series_map[sid] = s
        all_dates.update(s.index)

    if not all_dates:
        return pd.DataFrame(columns=["date", "lii", "cpi", "spread"])

    # Sort dates and filter to only months where we have enough data
    sorted_dates = sorted(all_dates)

    rows = []
    for dt in sorted_dates:
        # Need at least 12 months prior for YoY
        dt_year_ago = dt - pd.DateOffset(months=12)

        lii_val = 0.0
        cpi_val = 0.0
        lii_valid = 0
        cpi_valid = 0

        for cat_lii, cat_cpi in zip(cats_lii, cats_cpi):
            sid = cat_lii["series_id"]
            s = series_map.get(sid)
            if s is None:
                continue

            # Get current and year-ago values
            if dt not in s.index or dt_year_ago not in s.index:
                continue

            current = s[dt]
            year_ago = s[dt_year_ago]

            if year_ago == 0:
                continue

            yoy_pct = (current - year_ago) / year_ago

            lii_val += cat_lii["lii_weight"] * yoy_pct
            cpi_val += cat_cpi["cpi_norm"] * yoy_pct  # CPI uses raw CPI weights (no freq adjustment)
            lii_valid += 1
            cpi_valid += 1

        # Only include months where we have most categories
        if lii_valid >= 15:  # at least 15 of 23 categories
            rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "lii": round(lii_val * 100, 4),
                "cpi": round(cpi_val * 100, 4),
                "spread": round((lii_val - cpi_val) * 100, 4),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_current(
    bls_data: dict[str, pd.DataFrame],
    freq_overrides: dict[str, float] | None = None,
) -> dict:
    """
    Get the latest month's LII, CPI, spread, and MoM deltas.

    Returns dict with: lii, cpi, spread, lii_mom, cpi_mom, spread_mom, data_month
    """
    ts = compute_lii_timeseries(bls_data, freq_overrides=freq_overrides)
    if ts.empty:
        return {"lii": None, "cpi": None, "spread": None}

    latest = ts.iloc[-1]
    prev = ts.iloc[-2] if len(ts) >= 2 else None

    result = {
        "lii": round(float(latest["lii"]), 2),
        "cpi": round(float(latest["cpi"]), 2),
        "spread": round(float(latest["spread"]), 2),
        "data_month": latest["date"].strftime("%B %Y"),
        "lii_mom": None,
        "cpi_mom": None,
        "spread_mom": None,
    }

    if prev is not None:
        result["lii_mom"] = round(float(latest["lii"] - prev["lii"]), 2)
        result["cpi_mom"] = round(float(latest["cpi"] - prev["cpi"]), 2)
        result["spread_mom"] = round(float(latest["spread"] - prev["spread"]), 2)

    return result


def compute_category_breakdown(
    bls_data: dict[str, pd.DataFrame],
    freq_overrides: dict[str, float] | None = None,
) -> list[dict]:
    """
    Per-category breakdown for the latest month:
    name, freq_label, cpi_weight, lii_weight, weight_delta, yoy_pct,
    cpi_contribution, lii_contribution.
    """
    cats_lii = calculate_lii_weights(freq_overrides=freq_overrides)
    cats_cpi = _normalize_cpi_weights()

    rows = []
    for cat_lii, cat_cpi in zip(cats_lii, cats_cpi):
        sid = cat_lii["series_id"]
        df = bls_data.get(sid)
        if df is None or df.empty:
            continue

        s = df.set_index("date")["value"].sort_index()
        s = s[~s.index.duplicated(keep="last")]

        if len(s) < 13:
            continue

        latest_date = s.index[-1]
        year_ago_date = latest_date - pd.DateOffset(months=12)

        # Find closest date to year_ago
        if year_ago_date not in s.index:
            # Try to find a nearby date
            close_dates = s.index[s.index <= year_ago_date]
            if close_dates.empty:
                continue
            year_ago_date = close_dates[-1]

        current = s.iloc[-1]
        year_ago = s[year_ago_date]

        if year_ago == 0:
            continue

        yoy_pct = (current - year_ago) / year_ago

        cpi_w = cat_cpi["cpi_norm"]  # raw CPI weight (no frequency adjustment)
        lii_w = cat_lii["lii_weight"]

        rows.append({
            "name": cat_lii["name"],
            "series_id": sid,
            "freq_label": cat_lii["freq_label"],
            "freq_score": cat_lii["freq_score"],
            "cpi_weight": round(cat_cpi["cpi_weight"] * 100, 2),
            "lii_weight": round(lii_w * 100, 2),
            "weight_delta": round((lii_w - cpi_w) * 100, 2),
            "yoy_pct": round(yoy_pct * 100, 2),
            "cpi_contribution": round(cpi_w * yoy_pct * 100, 4),
            "lii_contribution": round(lii_w * yoy_pct * 100, 4),
        })

    # Sort by LII contribution descending
    rows.sort(key=lambda r: r["lii_contribution"], reverse=True)
    return rows


def compute_cumulative(
    bls_data: dict[str, pd.DataFrame],
    base_date: str = "2020-01-01",
    freq_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Compute cumulative price level indexed to base_date = 100.

    Returns DataFrame with columns: date, lii_level, cpi_level
    """
    ts = compute_lii_timeseries(bls_data, freq_overrides=freq_overrides)
    if ts.empty:
        return pd.DataFrame(columns=["date", "lii_level", "cpi_level"])

    base = pd.Timestamp(base_date)

    # Build cumulative from monthly YoY rates
    # Approximate: compound monthly changes from the base date
    # For simplicity, compute from base month forward using the MoM implied changes
    rows = []
    lii_level = 100.0
    cpi_level = 100.0
    prev_lii = None
    prev_cpi = None

    for _, row in ts.iterrows():
        dt = row["date"]
        if dt < base:
            prev_lii = row["lii"]
            prev_cpi = row["cpi"]
            continue

        if dt == base:
            lii_level = 100.0
            cpi_level = 100.0
            prev_lii = row["lii"]
            prev_cpi = row["cpi"]
            rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "lii_level": 100.0,
                "cpi_level": 100.0,
            })
            continue

        # Use the YoY rate to approximate monthly price change:
        # monthly_change ≈ (1 + yoy_rate)^(1/12) - 1
        if prev_lii is not None:
            lii_monthly = (1 + row["lii"] / 100) ** (1 / 12) - 1
            cpi_monthly = (1 + row["cpi"] / 100) ** (1 / 12) - 1
            lii_level *= (1 + lii_monthly)
            cpi_level *= (1 + cpi_monthly)

        prev_lii = row["lii"]
        prev_cpi = row["cpi"]

        rows.append({
            "date": dt.strftime("%Y-%m-%d"),
            "lii_level": round(lii_level, 2),
            "cpi_level": round(cpi_level, 2),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# DEBT OVERLAY
# ═══════════════════════════════════════════════════════════════════════

def _interpolate_quarterly_to_monthly(s: pd.Series) -> pd.Series:
    """Interpolate a quarterly FRED series to monthly frequency (linear)."""
    if s.empty:
        return s
    s = s.sort_index()
    idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    s_reindexed = s.reindex(idx)
    return s_reindexed.interpolate(method="linear")


def _load_fred_series(fred_data: dict, series_id: str) -> pd.Series:
    """Load a FRED series from the fetched data dict into a pd.Series."""
    df = fred_data.get(series_id)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    s = df.set_index("date")["value"]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _calc_cc_monthly_cost(balance: float, apr_pct: float) -> float:
    """Monthly interest cost = balance * (APR / 12). APR as percent, e.g. 22.0."""
    return balance * (apr_pct / 100.0 / 12.0)


def _calc_auto_payment(balance: float, annual_rate_pct: float, term_months: int = 72) -> float:
    """Standard amortization: M = P[r(1+r)^n] / [(1+r)^n - 1]."""
    r = annual_rate_pct / 100.0 / 12.0
    if r <= 0:
        return balance / term_months if term_months > 0 else 0
    return balance * (r * (1 + r) ** term_months) / ((1 + r) ** term_months - 1)


def _calc_student_loan_proxy(total_outstanding: float,
                              num_borrowers: float = 45_000_000,
                              avg_term_months: int = 120) -> float:
    """Estimated avg monthly payment proxy from aggregate data."""
    return total_outstanding / num_borrowers / avg_term_months


def compute_debt_cost_series(fred_data: dict) -> dict[str, pd.Series]:
    """
    Build monthly debt-cost-proxy series for each debt category.

    Returns {debt_key: pd.Series} where each series has monthly dates
    and values representing estimated monthly cost (in $).
    """
    result: dict[str, pd.Series] = {}

    # ── Credit card: balance × APR / 12 ──────────────────────────
    revolsl = _load_fred_series(fred_data, "REVOLSL")   # monthly, in billions
    cc_rate = _load_fred_series(fred_data, "TERMCBCCALLNS")  # quarterly, percent

    if not revolsl.empty and not cc_rate.empty:
        cc_rate_m = _interpolate_quarterly_to_monthly(cc_rate)
        # Align on common dates
        common = revolsl.index.intersection(cc_rate_m.index)
        if len(common) > 0:
            costs = pd.Series(
                [_calc_cc_monthly_cost(revolsl[d] * 1e6, cc_rate_m[d]) for d in common],
                index=common,
            )
            result["credit"] = costs

    # ── Student loans: total outstanding / borrowers / term ───────
    sloas = _load_fred_series(fred_data, "SLOAS")  # quarterly, in millions of dollars
    if not sloas.empty:
        sloas_m = _interpolate_quarterly_to_monthly(sloas)
        costs = sloas_m.apply(lambda v: _calc_student_loan_proxy(v * 1e6))
        result["student"] = costs

    # ── Auto loans: amortization(balance, rate, 72mo) ─────────────
    mvloasm = _load_fred_series(fred_data, "MVLOASM")  # quarterly (Mar/Jun/Sep/Dec)
    auto_rate = _load_fred_series(fred_data, "RIFLPBCIANM60NM")  # monthly, percent
    if not mvloasm.empty and not auto_rate.empty:
        # MVLOASM is quarterly — interpolate to monthly to align with rate data
        mvloasm_m = _interpolate_quarterly_to_monthly(mvloasm)
        common = mvloasm_m.index.intersection(auto_rate.index)
        if len(common) > 0:
            # Per-borrower: ~85M auto loans outstanding
            costs = pd.Series(
                [_calc_auto_payment(mvloasm_m[d] * 1e6 / 85_000_000, auto_rate[d])
                 for d in common],
                index=common,
            )
            result["auto"] = costs

    return result


def compute_debt_yoy_series(fred_data: dict) -> dict[str, pd.Series]:
    """
    Compute YoY % change for each debt cost proxy.
    Returns {debt_key: pd.Series} with monthly YoY % values.
    """
    cost_series = compute_debt_cost_series(fred_data)
    yoy_series: dict[str, pd.Series] = {}

    for key, s in cost_series.items():
        if len(s) < 13:
            continue
        yoy = s.pct_change(periods=12).dropna()
        yoy_series[key] = yoy

    return yoy_series


def compute_lii_with_debt(
    bls_data: dict[str, pd.DataFrame],
    fred_data: dict[str, pd.DataFrame],
    freq_overrides: dict[str, float] | None = None,
    enabled_debt: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute LII + Debt timeseries.

    Like compute_lii_timeseries but adds enabled debt categories into the
    weight pool and renormalizes. Returns DataFrame with:
    date, lii, lii_debt, cpi, spread, spread_debt
    """
    if enabled_debt is None:
        enabled_debt = ["student", "credit", "auto"]

    # Get base LII timeseries first (for CPI baseline)
    base_ts = compute_lii_timeseries(bls_data, freq_overrides=freq_overrides)
    if base_ts.empty:
        return base_ts

    # Get debt YoY series
    debt_yoy = compute_debt_yoy_series(fred_data)

    # Filter to enabled categories
    active_debt = [c for c in DEBT_CATEGORIES if c["key"] in enabled_debt]
    if not active_debt or not debt_yoy:
        # No debt data — just return base with lii_debt = lii
        base_ts["lii_debt"] = base_ts["lii"]
        base_ts["spread_debt"] = base_ts["spread"]
        return base_ts

    # Additive model: LII stays unchanged, debt costs are layered on top.
    # Consumers pay rent AND credit card interest — one doesn't replace
    # the other. Debt YoY is floored at 0 (declining debt costs = neutral,
    # not deflationary — you still owe the money).

    # Calculate debt category weights (among themselves, for the additive portion)
    debt_weight_total = sum(dc["weight"] for dc in active_debt)

    # Iterate over the base timeseries dates and compute LII+Debt
    rows = []
    for _, row in base_ts.iterrows():
        dt = row["date"]
        lii_val = float(row["lii"])  # base LII is unchanged

        # Compute additive debt burden
        debt_addon = 0.0
        for dc in active_debt:
            yoy_s = debt_yoy.get(dc["key"])
            if yoy_s is None or yoy_s.empty:
                continue
            # Find closest date in the debt YoY series
            # Use 180-day window — FRED quarterly series (SLOAS) can lag 3-6 months
            if dt in yoy_s.index:
                yoy_val = yoy_s[dt]
            else:
                close = yoy_s.index[yoy_s.index <= dt]
                if close.empty:
                    continue
                nearest = close[-1]
                if (dt - nearest).days > 180:
                    continue
                yoy_val = yoy_s[nearest]

            if pd.isna(yoy_val):
                continue

            # Floor at 0: debt costs declining = neutral, not deflationary
            yoy_val = max(0.0, yoy_val)

            # Weight within debt pool, scaled by total debt share of budget
            w = (dc["weight"] / debt_weight_total) * debt_weight_total
            debt_addon += w * yoy_val

        lii_debt_pct = round(lii_val + debt_addon * 100, 4)
        cpi_pct = float(row["cpi"])
        rows.append({
            "date": dt,
            "lii": lii_val,
            "lii_debt": lii_debt_pct,
            "cpi": cpi_pct,
            "spread": float(row["spread"]),
            "spread_debt": round(lii_debt_pct - cpi_pct, 4),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        base_ts["lii_debt"] = base_ts["lii"]
        base_ts["spread_debt"] = base_ts["spread"]
        return base_ts
    return result


def compute_debt_current(fred_data: dict) -> dict:
    """Get latest debt balances (in trillions) and rates for the callout card."""
    info = {}
    # Balances from FRED are in millions of dollars — convert to trillions
    for sid, label in [("REVOLSL", "revolving_credit_T"),
                       ("SLOAS", "student_loans_T"),
                       ("MVLOASM", "auto_loans_T")]:
        s = _load_fred_series(fred_data, sid)
        if not s.empty:
            info[label] = round(float(s.iloc[-1]) / 1_000_000, 2)  # millions → trillions

    for sid, label in [("TERMCBCCALLNS", "cc_apr"),
                       ("RIFLPBCIANM60NM", "auto_rate")]:
        s = _load_fred_series(fred_data, sid)
        if not s.empty:
            info[label] = round(float(s.iloc[-1]), 2)

    return info

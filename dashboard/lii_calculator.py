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
    OER_SERIES_ID,
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


def _build_series_map(bls_data: dict[str, pd.DataFrame]) -> tuple[dict, set]:
    """Build {series_id: pd.Series} and all_dates from BLS data."""
    all_dates: set[pd.Timestamp] = set()
    series_map: dict[str, pd.Series] = {}
    for cat in CATEGORIES:
        sid = cat["series_id"]
        df = bls_data.get(sid)
        if df is None or df.empty:
            continue
        s = df.set_index("date")["value"]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        series_map[sid] = s
        all_dates.update(s.index)
    return series_map, all_dates


def compute_lii_timeseries(
    bls_data: dict[str, pd.DataFrame],
    freq_overrides: dict[str, float] | None = None,
    cpi_bls_data: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Compute monthly LII and CPI-weighted YoY series from BLS data.

    Returns DataFrame with columns: date, lii, cpi, spread
    (all in percentage terms, e.g. 3.2 = 3.2%).

    Args:
        bls_data: Data used for LII calculation (may have shelter swap).
        cpi_bls_data: If provided, used for CPI calculation (original BLS data).
            This keeps CPI stable when shelter mode changes LII data.
    """
    cats_lii = calculate_lii_weights(freq_overrides=freq_overrides)
    cats_cpi = _normalize_cpi_weights()

    # LII uses bls_data (potentially modified for shelter mode)
    lii_map, lii_dates = _build_series_map(bls_data)
    # CPI always uses original data
    if cpi_bls_data is not None:
        cpi_map, cpi_dates = _build_series_map(cpi_bls_data)
        all_dates = lii_dates & cpi_dates  # only dates in both
    else:
        cpi_map = lii_map
        all_dates = lii_dates

    if not all_dates:
        return pd.DataFrame(columns=["date", "lii", "cpi", "spread"])

    sorted_dates = sorted(all_dates)

    rows = []
    for dt in sorted_dates:
        dt_year_ago = dt - pd.DateOffset(months=12)

        lii_val = 0.0
        cpi_val = 0.0
        lii_valid = 0
        cpi_valid = 0

        for cat_lii, cat_cpi in zip(cats_lii, cats_cpi):
            sid = cat_lii["series_id"]

            # LII: use potentially modified data
            s_lii = lii_map.get(sid)
            if s_lii is not None and dt in s_lii.index and dt_year_ago in s_lii.index:
                cur = s_lii[dt]
                ago = s_lii[dt_year_ago]
                if ago != 0:
                    lii_val += cat_lii["lii_weight"] * ((cur - ago) / ago)
                    lii_valid += 1

            # CPI: always use original data
            s_cpi = cpi_map.get(sid)
            if s_cpi is not None and dt in s_cpi.index and dt_year_ago in s_cpi.index:
                cur = s_cpi[dt]
                ago = s_cpi[dt_year_ago]
                if ago != 0:
                    cpi_val += cat_cpi["cpi_norm"] * ((cur - ago) / ago)
                    cpi_valid += 1

        if lii_valid >= 15:
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
    cpi_bls_data: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """
    Get the latest month's LII, CPI, spread, and MoM deltas.

    Returns dict with: lii, cpi, spread, lii_mom, cpi_mom, spread_mom, data_month
    """
    ts = compute_lii_timeseries(bls_data, freq_overrides=freq_overrides, cpi_bls_data=cpi_bls_data)
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
    cpi_bls_data: dict[str, pd.DataFrame] | None = None,
    shelter_mode: str = "oer",
) -> list[dict]:
    """
    Per-category breakdown for the latest month.
    Uses bls_data for LII YoY and cpi_bls_data (if given) for CPI YoY.
    """
    cats_lii = calculate_lii_weights(freq_overrides=freq_overrides)
    cats_cpi = _normalize_cpi_weights()

    rows = []
    for cat_lii, cat_cpi in zip(cats_lii, cats_cpi):
        sid = cat_lii["series_id"]

        # LII uses potentially modified data (shelter swap)
        df = bls_data.get(sid)
        if df is None or df.empty:
            continue

        s = df.set_index("date")["value"].sort_index()
        s = s[~s.index.duplicated(keep="last")]

        if len(s) < 13:
            continue

        latest_date = s.index[-1]
        year_ago_date = latest_date - pd.DateOffset(months=12)

        if year_ago_date not in s.index:
            close_dates = s.index[s.index <= year_ago_date]
            if close_dates.empty:
                continue
            year_ago_date = close_dates[-1]

        current = s.iloc[-1]
        year_ago = s[year_ago_date]

        if year_ago == 0:
            continue

        yoy_pct = (current - year_ago) / year_ago

        # CPI contribution uses original data
        cpi_yoy = yoy_pct
        if cpi_bls_data is not None:
            df_cpi = cpi_bls_data.get(sid)
            if df_cpi is not None and not df_cpi.empty:
                s_cpi = df_cpi.set_index("date")["value"].sort_index()
                s_cpi = s_cpi[~s_cpi.index.duplicated(keep="last")]
                if len(s_cpi) >= 13:
                    ld = s_cpi.index[-1]
                    ya = ld - pd.DateOffset(months=12)
                    if ya not in s_cpi.index:
                        cd = s_cpi.index[s_cpi.index <= ya]
                        ya = cd[-1] if not cd.empty else ya
                    if ya in s_cpi.index and s_cpi[ya] != 0:
                        cpi_yoy = (s_cpi.iloc[-1] - s_cpi[ya]) / s_cpi[ya]

        cpi_w = cat_cpi["cpi_norm"]
        lii_w = cat_lii["lii_weight"]

        # Rename OER when mortgage mode is active
        name = cat_lii["name"]
        if sid == OER_SERIES_ID and shelter_mode == "mdsp":
            name = "Mortgage burden (MDSP)"
        elif sid == OER_SERIES_ID and shelter_mode == "mortgage":
            name = "New-purchase mortgage"

        rows.append({
            "name": name,
            "series_id": sid,
            "freq_label": cat_lii["freq_label"],
            "freq_score": cat_lii["freq_score"],
            "cpi_weight": round(cat_cpi["cpi_weight"] * 100, 2),
            "lii_weight": round(lii_w * 100, 2),
            "weight_delta": round((lii_w - cpi_w) * 100, 2),
            "yoy_pct": round(yoy_pct * 100, 2),
            "cpi_contribution": round(cpi_w * cpi_yoy * 100, 4),
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
# SHELTER ALTERNATIVES
# ═══════════════════════════════════════════════════════════════════════

def _resample_weekly_to_monthly(s: pd.Series) -> pd.Series:
    """Resample a weekly FRED series to monthly (last value of month)."""
    if s.empty:
        return s
    return s.resample("MS").last().dropna()


def _calc_mortgage_payment(home_price: float, rate_pct: float,
                           down_pct: float = 0.20, term_years: int = 30) -> float:
    """Standard amortization: M = P[r(1+r)^n] / [(1+r)^n - 1]."""
    principal = home_price * (1 - down_pct)
    r = rate_pct / 100.0 / 12.0
    n = term_years * 12
    if r <= 0:
        return principal / n if n > 0 else 0
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def compute_mortgage_proxy(fred_data: dict) -> pd.DataFrame:
    """
    Build a monthly mortgage payment proxy from FRED data.

    Uses MORTGAGE30US (weekly 30yr rate) and MSPUS (quarterly median price)
    to compute monthly P&I assuming 20% down, 30yr fixed.

    Returns DataFrame with columns [date, value] matching BLS data format,
    so it can be swapped directly into the BLS data dict for OER.
    """
    rate_s = _load_fred_series(fred_data, "MORTGAGE30US")
    price_s = _load_fred_series(fred_data, "MSPUS")

    if rate_s.empty or price_s.empty:
        logger.warning("Missing FRED data for mortgage proxy (MORTGAGE30US or MSPUS)")
        return pd.DataFrame(columns=["date", "value"])

    # Resample: weekly rate → monthly, quarterly price → monthly
    rate_m = _resample_weekly_to_monthly(rate_s)
    price_m = _interpolate_quarterly_to_monthly(price_s)

    # Align on common dates
    common = rate_m.index.intersection(price_m.index)
    if len(common) == 0:
        return pd.DataFrame(columns=["date", "value"])

    rows = []
    for dt in sorted(common):
        payment = _calc_mortgage_payment(price_m[dt], rate_m[dt])
        rows.append({"date": dt, "value": payment})

    return pd.DataFrame(rows)


def compute_mdsp_proxy(fred_data: dict) -> pd.DataFrame:
    """
    Build a monthly proxy from the FRED MDSP series (Mortgage Debt Service
    Payments as % of Disposable Personal Income).

    MDSP is quarterly — we interpolate to monthly so it slots into the
    same YoY machinery as a BLS series.

    Returns DataFrame with columns: date, value  (value = ratio like 5.89)
    """
    mdsp_df = fred_data.get("MDSP")
    if mdsp_df is None or mdsp_df.empty:
        logger.warning("MDSP series missing from FRED data")
        return pd.DataFrame(columns=["date", "value"])

    # Build a date→value Series, then interpolate quarterly→monthly
    s = mdsp_df.set_index("date")["value"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s_monthly = _interpolate_quarterly_to_monthly(s)

    if s_monthly.empty:
        return pd.DataFrame(columns=["date", "value"])

    return pd.DataFrame({"date": s_monthly.index, "value": s_monthly.values})


def _forward_fill_proxy(proxy_df: pd.DataFrame, bls_data: dict) -> pd.DataFrame:
    """
    Extend a shelter proxy DataFrame forward to cover the latest BLS date.

    FRED proxies often lag BLS data by months (MDSP quarterly, mortgage proxy
    limited by MSPUS). Without forward-fill, recent months have no shelter
    contribution — producing identical results for all modes.

    Strategy: flat-line the last known proxy value forward (conservative).
    """
    if proxy_df.empty:
        return proxy_df

    # Find the latest date across all BLS series
    latest_bls = pd.Timestamp.min
    for df in bls_data.values():
        if df is not None and not df.empty and "date" in df.columns:
            dt = df["date"].max()
            if dt > latest_bls:
                latest_bls = dt

    proxy_latest = proxy_df["date"].max()
    if proxy_latest >= latest_bls:
        return proxy_df  # already covers the range

    # Generate monthly dates from proxy_latest+1M to latest_bls
    fill_dates = pd.date_range(
        proxy_latest + pd.DateOffset(months=1),
        latest_bls,
        freq="MS",
    )
    if fill_dates.empty:
        return proxy_df

    last_val = proxy_df.iloc[-1]["value"]
    fill_rows = pd.DataFrame({"date": fill_dates, "value": last_val})
    extended = pd.concat([proxy_df, fill_rows], ignore_index=True)
    logger.info(
        "Forward-filled shelter proxy: %s → %s (%d months added, value=%.2f)",
        proxy_latest.strftime("%Y-%m"),
        latest_bls.strftime("%Y-%m"),
        len(fill_dates),
        last_val,
    )
    return extended


def apply_shelter_mode(bls_data: dict, fred_data: dict, shelter_mode: str) -> dict:
    """
    Return a (possibly modified) copy of bls_data with the OER series
    swapped for the selected shelter alternative.

    shelter_mode:
        "oer"       → no change (BLS standard)
        "mdsp"      → replace OER with FRED MDSP ratio (actual mortgage burden)
        "mortgage"  → replace OER with new-purchase mortgage payment proxy

    The calculator doesn't change — it just sees different numbers in
    the OER slot and computes YoY normally.
    """
    if shelter_mode == "oer" or not shelter_mode:
        return bls_data

    if shelter_mode == "mdsp":
        mdsp_df = compute_mdsp_proxy(fred_data)
        mdsp_df = _forward_fill_proxy(mdsp_df, bls_data)
        if mdsp_df.empty:
            logger.warning("MDSP proxy returned empty — falling back to OER")
            return bls_data
        modified = dict(bls_data)
        modified[OER_SERIES_ID] = mdsp_df
        return modified

    if shelter_mode == "mortgage":
        mortgage_df = compute_mortgage_proxy(fred_data)
        mortgage_df = _forward_fill_proxy(mortgage_df, bls_data)
        if mortgage_df.empty:
            logger.warning("Mortgage proxy returned empty — falling back to OER")
            return bls_data
        modified = dict(bls_data)
        modified[OER_SERIES_ID] = mortgage_df
        return modified

    return bls_data


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
    cpi_bls_data: dict[str, pd.DataFrame] | None = None,
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
    base_ts = compute_lii_timeseries(bls_data, freq_overrides=freq_overrides, cpi_bls_data=cpi_bls_data)
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

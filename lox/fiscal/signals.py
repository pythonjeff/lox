from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from lox.config import Settings
from lox.data.fiscaldata import FiscalDataClient, FiscalDataEndpoint
from lox.data.fred import FredClient
from lox.fiscal.models import FiscalInputs, FiscalState
from lox.macro.transforms import merge_series_daily, zscore


# ---------------------------------------------------------------------------
# Data sources (MVP uses FRED only, with best-effort optional series)
# ---------------------------------------------------------------------------
FISCAL_FRED_SERIES: Dict[str, str] = {
    # Monthly Treasury Statement (monthly, flow):
    # "Federal Surplus or Deficit [-]" (typically *millions* of dollars on FRED). Negative => deficit.
    # IMPORTANT: use a *monthly* series so rolling 12m calculations work.
    "SURPLUS_DEFICIT": "MTSDS133FMS",
    # Treasury General Account (weekly, level). Already used in liquidity module.
    "TGA": "WTREGEN",
    # Interest payments proxy (quarterly BEA series, flow). Optional, can be swapped later.
    "INTEREST_EXPENSE": "A091RC1Q027SBEA",
    # Quant upgrade: federal receipts (deficit sustainability), demand structure, bond stress
    "TAX_RECEIPTS": "MTSR133FMS",  # Monthly Treasury Statement receipts (monthly, millions)
    "FOREIGN_HOLDINGS": "FDHBFIN",  # Foreign holdings of US Treasuries (monthly, millions)
    "CUSTODY": "WSHOSHO",  # Fed custody holdings (weekly, millions) — proxy for foreign CB demand
}

_OPTIONAL_SERIES = {"INTEREST_EXPENSE", "TAX_RECEIPTS", "FOREIGN_HOLDINGS", "CUSTODY"}

# ---------------------------------------------------------------------------
# Treasury net issuance (true) via MSPD: Δ outstanding (monthly)
# ---------------------------------------------------------------------------

# MSPD marketable detail endpoint (FiscalData)
_MSPD_MARKET_ENDPOINT = FiscalDataEndpoint(path="/v1/debt/mspd/mspd_table_3_market")

# Treasury auction results endpoint (FiscalData)
# Note: FiscalData’s “Treasury Securities Auctions Data” uses this endpoint.
_AUCTION_ENDPOINT = FiscalDataEndpoint(path="/v1/accounting/od/auctions_query")

# Candidate column names (FiscalData schemas can drift; be defensive).
_AUCT_DATE_COLS = ("auction_date", "record_date", "issue_date", "data_date", "as_of_date")
_AUCT_TYPE_COLS = ("security_type", "security_type_desc", "security_type_name", "security_desc")
_AUCT_TERM_COLS = ("security_term", "term", "original_security_term")
_AUCT_HIGH_YIELD_COLS = ("high_yield", "high_rate", "high_discount_rate", "high_investment_rate")
_AUCT_MEDIAN_YIELD_COLS = (
    "median_yield",
    "median_rate",
    "median_discount_rate",
    "median_investment_rate",
    # auctions_query schema uses this naming (average/median field)
    "avg_med_yield",
)
_AUCT_AVG_YIELD_COLS = ("avg_yield", "average_yield", "avg_rate", "average_rate")
_AUCT_TOTAL_ACCEPTED_COLS = ("total_accepted", "total_accepted_amt", "accepted_amount", "total_amount_accepted")
_AUCT_PD_ACCEPTED_PCT_COLS = (
    "primary_dealer_takedown_pct",
    "primary_dealer_pct",
    "primary_dealer_takedown_percent",
    "pd_takedown_pct",
)
_AUCT_PD_ACCEPTED_AMT_COLS = (
    "primary_dealer_takedown",
    "primary_dealer_takedown_amt",
    "primary_dealer_amount",
    "primary_dealer_accepted",
)
_AUCT_BTC_COLS = (
    "bid_to_cover_ratio",
    "bid_to_cover",
    "btc_ratio",
)

# Candidate column names vary across FiscalData surfaces; handle defensively.
_MSPD_DATE_COLS = ("record_date", "as_of_date", "report_date", "data_date")
_MSPD_MATURITY_COLS = ("maturity_date", "maturity_dt")
_MSPD_TYPE_COLS = ("security_type_desc", "security_type", "security_desc", "security_type_name")
_MSPD_AMT_COLS = (
    "outstanding_amt",
    "outstanding_amount",
    "amount",
    "par_value",
    "current_outstanding_amt",
)


def _first_existing_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _looks_like_coupon_auction(security_type: str | None, security_term: str | None) -> bool:
    """
    Filter auctions to the "coupon" family (notes/bonds) for tail/dealer-take style metrics.
    Bills/CMBs behave differently; TIPS/FRNs have different microstructure.
    """
    s = (security_type or "").lower()
    t = (security_term or "").lower()
    # Exclude obvious bills/cmbs
    if "bill" in s or "bill" in t or "cmb" in s or "cash management" in s:
        return False
    # Exclude TIPS / inflation-linked (different yield semantics)
    if "tips" in s or "inflation" in s:
        return False
    # Exclude FRNs (rate/yield fields can differ; wire later if desired)
    if "frn" in s or "floating" in s:
        return False
    # Prefer notes/bonds; if missing, accept anything not excluded above.
    if "note" in s or "bond" in s:
        return True
    return True


def _compute_auction_tail_and_dealer_take(auctions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly auction-quality metrics from TreasuryDirect auction results.

    Outputs a monthly DataFrame:
      - date (month end)
      - AUCTION_TAIL_BPS
      - DEALER_TAKE_PCT

    Definitions (best-effort, explainable):
    - tail (bps): (high_yield - median_yield) * 100
      (This is a proxy; the true "tail vs WI" needs when-issued yields.)
    - dealer take (%): primary dealer takedown as % of total accepted
    """
    if auctions is None or auctions.empty:
        return pd.DataFrame(columns=["date", "AUCTION_TAIL_BPS", "DEALER_TAKE_PCT"])

    df = auctions.copy()
    date_col = _first_existing_col(df, _AUCT_DATE_COLS)
    typ_col = _first_existing_col(df, _AUCT_TYPE_COLS)
    term_col = _first_existing_col(df, _AUCT_TERM_COLS)
    if date_col is None:
        return pd.DataFrame(columns=["date", "AUCTION_TAIL_BPS", "DEALER_TAKE_PCT"])

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty:
        return pd.DataFrame(columns=["date", "AUCTION_TAIL_BPS", "DEALER_TAKE_PCT"])

    if typ_col is not None:
        df[typ_col] = df[typ_col].astype(str)
    if term_col is not None:
        df[term_col] = df[term_col].astype(str)

    # Coupon-only filter for these metrics
    def _is_coupon_row(r: pd.Series) -> bool:
        return _looks_like_coupon_auction(
            security_type=str(r.get(typ_col)) if typ_col is not None else None,
            security_term=str(r.get(term_col)) if term_col is not None else None,
        )

    df = df[df.apply(_is_coupon_row, axis=1)]
    if df.empty:
        return pd.DataFrame(columns=["date", "AUCTION_TAIL_BPS", "DEALER_TAKE_PCT"])

    # Tail proxy (high - median) in bps
    high_col = _first_existing_col(df, _AUCT_HIGH_YIELD_COLS)
    med_col = _first_existing_col(df, _AUCT_MEDIAN_YIELD_COLS)
    avg_col = _first_existing_col(df, _AUCT_AVG_YIELD_COLS)

    def _to_num(col: str) -> None:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if high_col is not None:
        _to_num(high_col)
    if med_col is not None:
        _to_num(med_col)
    if avg_col is not None:
        _to_num(avg_col)

    tail_bps = None
    if high_col is not None and med_col is not None:
        tail_bps = (df[high_col] - df[med_col]) * 100.0
    elif high_col is not None and avg_col is not None:
        tail_bps = (df[high_col] - df[avg_col]) * 100.0
    if tail_bps is not None:
        df["AUCTION_TAIL_BPS"] = tail_bps
    else:
        df["AUCTION_TAIL_BPS"] = pd.NA

    # Dealer take (%)
    pd_pct_col = _first_existing_col(df, _AUCT_PD_ACCEPTED_PCT_COLS)
    if pd_pct_col is not None:
        _to_num(pd_pct_col)
        # Normalize if some schemas store 0..1.
        v = df[pd_pct_col]
        df["DEALER_TAKE_PCT"] = v.where(v.abs() > 1.0, v * 100.0)
    else:
        pd_amt_col = _first_existing_col(df, _AUCT_PD_ACCEPTED_AMT_COLS)
        tot_col = _first_existing_col(df, _AUCT_TOTAL_ACCEPTED_COLS)
        if pd_amt_col is None or tot_col is None:
            df["DEALER_TAKE_PCT"] = pd.NA
        else:
            _to_num(pd_amt_col)
            _to_num(tot_col)
            df["DEALER_TAKE_PCT"] = 100.0 * df[pd_amt_col] / df[tot_col].replace({0.0: pd.NA})

    # Weights for monthly aggregation: total accepted if available
    tot_col = _first_existing_col(df, _AUCT_TOTAL_ACCEPTED_COLS)
    w = None
    if tot_col is not None:
        _to_num(tot_col)
        w = df[tot_col]

    # Month-end bucket
    df["month_end"] = df[date_col].dt.to_period("M").dt.to_timestamp("M")

    def _wavg(g: pd.DataFrame, col: str) -> float | None:
        s = pd.to_numeric(g[col], errors="coerce")
        s = s.dropna()
        if s.empty:
            return None
        if w is None or tot_col is None:
            return float(s.mean())
        ww = pd.to_numeric(g[tot_col], errors="coerce")
        ww = ww.where(ww.notna() & (ww > 0.0))
        m = g[[col, tot_col]].copy()
        m[col] = pd.to_numeric(m[col], errors="coerce")
        m[tot_col] = pd.to_numeric(m[tot_col], errors="coerce")
        m = m.dropna(subset=[col, tot_col])
        m = m[m[tot_col] > 0.0]
        if m.empty:
            return float(s.mean())
        return float((m[col] * m[tot_col]).sum() / m[tot_col].sum())

    out = (
        df.groupby("month_end", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "AUCTION_TAIL_BPS": _wavg(g, "AUCTION_TAIL_BPS"),
                    "DEALER_TAKE_PCT": _wavg(g, "DEALER_TAKE_PCT"),
                }
            ),
            include_groups=False,
        )
        .rename(columns={"month_end": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    return out[["date", "AUCTION_TAIL_BPS", "DEALER_TAKE_PCT"]]


def _classify_auction_tenor(term_str: str | None) -> str:
    """Classify auction security_term into 'front' (2Y-5Y) or 'back' (7Y-30Y)."""
    t = (term_str or "").lower().replace("-", "").replace(" ", "")
    for prefix in ("2y", "3y", "5y", "2year", "3year", "5year"):
        if prefix in t:
            return "front"
    for prefix in ("7y", "10y", "20y", "30y", "7year", "10year", "20year", "30year"):
        if prefix in t:
            return "back"
    try:
        import re
        m = re.search(r"(\d+)", t)
        if m:
            yrs = int(m.group(1))
            return "front" if yrs <= 5 else "back"
    except Exception:
        pass
    return "unknown"


def _compute_auction_detail_by_tenor(
    auctions: pd.DataFrame,
) -> dict[str, dict[str, float | None]]:
    """
    Split the most recent month's coupon auctions into front-end (2Y-5Y)
    and back-end (7Y-30Y) buckets, returning tail, dealer take, and
    bid-to-cover for each.

    Returns {"front": {...}, "back": {...}, "worst": {...}} where "worst"
    picks whichever bucket shows more stress.
    """
    empty: dict[str, dict[str, float | None]] = {
        "front": {"tail_bps": None, "dealer_take_pct": None, "btc": None, "n": 0},
        "back": {"tail_bps": None, "dealer_take_pct": None, "btc": None, "n": 0},
        "worst": {"tail_bps": None, "dealer_take_pct": None, "btc": None, "bucket": None},
    }
    if auctions is None or auctions.empty:
        return empty

    df = auctions.copy()
    date_col = _first_existing_col(df, _AUCT_DATE_COLS)
    typ_col = _first_existing_col(df, _AUCT_TYPE_COLS)
    term_col = _first_existing_col(df, _AUCT_TERM_COLS)
    if date_col is None:
        return empty

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty:
        return empty

    if typ_col is not None:
        df[typ_col] = df[typ_col].astype(str)
    if term_col is not None:
        df[term_col] = df[term_col].astype(str)

    # Coupon-only filter
    def _is_coupon(r: pd.Series) -> bool:
        return _looks_like_coupon_auction(
            security_type=str(r.get(typ_col)) if typ_col is not None else None,
            security_term=str(r.get(term_col)) if term_col is not None else None,
        )

    df = df[df.apply(_is_coupon, axis=1)]
    if df.empty:
        return empty

    # Classify tenor bucket
    df["_tenor"] = df[term_col].apply(_classify_auction_tenor) if term_col else "unknown"
    df = df[df["_tenor"].isin({"front", "back"})]
    if df.empty:
        return empty

    # Keep last 3 months for robustness
    df = df.sort_values(date_col, ascending=False)
    cutoff = df[date_col].max() - pd.DateOffset(months=3)
    df = df[df[date_col] >= cutoff]

    # Compute per-row metrics
    high_col = _first_existing_col(df, _AUCT_HIGH_YIELD_COLS)
    med_col = _first_existing_col(df, _AUCT_MEDIAN_YIELD_COLS)
    avg_col = _first_existing_col(df, _AUCT_AVG_YIELD_COLS)
    btc_col = _first_existing_col(df, _AUCT_BTC_COLS)

    for c in [high_col, med_col, avg_col, btc_col]:
        if c is not None:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if high_col and med_col:
        df["_tail"] = (df[high_col] - df[med_col]) * 100.0
    elif high_col and avg_col:
        df["_tail"] = (df[high_col] - df[avg_col]) * 100.0
    else:
        df["_tail"] = pd.NA

    pd_pct_col = _first_existing_col(df, _AUCT_PD_ACCEPTED_PCT_COLS)
    if pd_pct_col is not None:
        df[pd_pct_col] = pd.to_numeric(df[pd_pct_col], errors="coerce")
        v = df[pd_pct_col]
        df["_dealer"] = v.where(v.abs() > 1.0, v * 100.0)
    else:
        pd_amt_col = _first_existing_col(df, _AUCT_PD_ACCEPTED_AMT_COLS)
        tot_col = _first_existing_col(df, _AUCT_TOTAL_ACCEPTED_COLS)
        if pd_amt_col and tot_col:
            df[pd_amt_col] = pd.to_numeric(df[pd_amt_col], errors="coerce")
            df[tot_col] = pd.to_numeric(df[tot_col], errors="coerce")
            df["_dealer"] = 100.0 * df[pd_amt_col] / df[tot_col].replace({0.0: pd.NA})
        else:
            df["_dealer"] = pd.NA

    df["_btc"] = df[btc_col] if btc_col else pd.NA

    result: dict[str, dict[str, float | None]] = {}
    for bucket in ("front", "back"):
        sub = df[df["_tenor"] == bucket]
        n = len(sub)
        tail_vals = sub["_tail"].dropna()
        dealer_vals = sub["_dealer"].dropna()
        btc_vals = pd.to_numeric(sub["_btc"], errors="coerce").dropna() if "_btc" in sub else pd.Series(dtype=float)
        result[bucket] = {
            "tail_bps": float(tail_vals.mean()) if not tail_vals.empty else None,
            "dealer_take_pct": float(dealer_vals.mean()) if not dealer_vals.empty else None,
            "btc": float(btc_vals.mean()) if not btc_vals.empty else None,
            "n": n,
        }

    # "worst" = whichever bucket shows more stress (higher tail or higher dealer take)
    def _stress_score(b: dict) -> float:
        s = 0.0
        if isinstance(b.get("tail_bps"), (int, float)):
            s += float(b["tail_bps"])
        if isinstance(b.get("dealer_take_pct"), (int, float)):
            s += float(b["dealer_take_pct"]) * 0.2  # scale dealer take to comparable range
        return s

    worst_bucket = "back" if _stress_score(result.get("back", {})) >= _stress_score(result.get("front", {})) else "front"
    result["worst"] = {**result.get(worst_bucket, {}), "bucket": worst_bucket}

    return result


def _compute_recent_auctions_list(
    auctions: pd.DataFrame, n: int = 8,
) -> list[dict[str, object]]:
    """
    Return the N most recent individual coupon auctions with per-auction
    metrics for display (security term, date, tail, dealer take, BTC).
    """
    if auctions is None or auctions.empty:
        return []

    df = auctions.copy()
    date_col = _first_existing_col(df, _AUCT_DATE_COLS)
    typ_col = _first_existing_col(df, _AUCT_TYPE_COLS)
    term_col = _first_existing_col(df, _AUCT_TERM_COLS)
    if date_col is None:
        return []

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty:
        return []

    if typ_col is not None:
        df[typ_col] = df[typ_col].astype(str)
    if term_col is not None:
        df[term_col] = df[term_col].astype(str)

    def _is_coupon(r: pd.Series) -> bool:
        return _looks_like_coupon_auction(
            security_type=str(r.get(typ_col)) if typ_col is not None else None,
            security_term=str(r.get(term_col)) if term_col is not None else None,
        )

    df = df[df.apply(_is_coupon, axis=1)]
    if df.empty:
        return []

    # Compute tail, dealer, BTC per row
    high_col = _first_existing_col(df, _AUCT_HIGH_YIELD_COLS)
    med_col = _first_existing_col(df, _AUCT_MEDIAN_YIELD_COLS)
    avg_col = _first_existing_col(df, _AUCT_AVG_YIELD_COLS)
    btc_col = _first_existing_col(df, _AUCT_BTC_COLS)

    for c in [high_col, med_col, avg_col, btc_col]:
        if c is not None:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if high_col and med_col:
        df["_tail"] = (df[high_col] - df[med_col]) * 100.0
    elif high_col and avg_col:
        df["_tail"] = (df[high_col] - df[avg_col]) * 100.0
    else:
        df["_tail"] = pd.NA

    pd_pct_col = _first_existing_col(df, _AUCT_PD_ACCEPTED_PCT_COLS)
    if pd_pct_col is not None:
        df[pd_pct_col] = pd.to_numeric(df[pd_pct_col], errors="coerce")
        v = df[pd_pct_col]
        df["_dealer"] = v.where(v.abs() > 1.0, v * 100.0)
    else:
        pd_amt_col = _first_existing_col(df, _AUCT_PD_ACCEPTED_AMT_COLS)
        tot_col = _first_existing_col(df, _AUCT_TOTAL_ACCEPTED_COLS)
        if pd_amt_col and tot_col:
            df[pd_amt_col] = pd.to_numeric(df[pd_amt_col], errors="coerce")
            df[tot_col] = pd.to_numeric(df[tot_col], errors="coerce")
            df["_dealer"] = 100.0 * df[pd_amt_col] / df[tot_col].replace({0.0: pd.NA})
        else:
            df["_dealer"] = pd.NA

    df["_btc"] = df[btc_col] if btc_col else pd.NA

    # Only keep rows that have at least one computed metric
    has_data = df["_tail"].notna() | df["_dealer"].notna()
    if "_btc" in df.columns:
        has_data = has_data | pd.to_numeric(df["_btc"], errors="coerce").notna()
    df = df[has_data]

    df = df.sort_values(date_col, ascending=False).head(n)

    rows: list[dict[str, object]] = []
    for _, r in df.iterrows():
        term_label = str(r.get(term_col, "")) if term_col else ""
        rows.append({
            "date": str(pd.to_datetime(r[date_col]).date()),
            "term": term_label,
            "tenor": _classify_auction_tenor(term_label),
            "tail_bps": round(float(r["_tail"]), 1) if pd.notna(r["_tail"]) else None,
            "dealer_take_pct": round(float(r["_dealer"]), 1) if pd.notna(r["_dealer"]) else None,
            "btc": round(float(r["_btc"]), 2) if pd.notna(r.get("_btc")) else None,
        })
    return rows


def _fetch_treasury_auction_results(*, start_date: str, refresh: bool) -> pd.DataFrame:
    fd = FiscalDataClient()
    params = {
        # Match the dataset’s typical ordering: newest auctions first.
        "sort": "-auction_date,-issue_date,maturity_date",
        # Keep server-side filter to reduce payload; if it fails, we’ll fall back.
        "filter": f"record_date:gte:{start_date}",
    }
    try:
        df = fd.fetch(
            endpoint=_AUCTION_ENDPOINT,
            params=params,
            # Single cache file is simpler for users to find.
            cache_key="auctions_query",
            refresh=refresh,
        )
    except Exception:
        # Fallback: no filter (some environments/schemas may differ).
        df = fd.fetch(
            endpoint=_AUCTION_ENDPOINT,
            params={"sort": "-auction_date,-issue_date,maturity_date"},
            cache_key="auctions_query",
            refresh=refresh,
        )
    if df is None or df.empty:
        return pd.DataFrame()
    if "auction_date" not in df.columns:
        return df
    df = df.copy()
    df["auction_date"] = pd.to_datetime(df["auction_date"], errors="coerce")
    start_ts = pd.to_datetime(start_date, errors="coerce")
    if pd.isna(start_ts):
        return df
    return df[df["auction_date"] >= start_ts].reset_index(drop=True)


def _classify_bucket(
    *,
    record_date: pd.Timestamp,
    maturity_date: pd.Timestamp | None,
    security_type: str | None,
) -> str | None:
    """Return one of {'BILLS','COUPONS','LONG'}.

    Primary rule (preferred): classify by *remaining maturity*.
    Fallback rule: classify by security type text.

    Buckets (simple + robust):
    - BILLS: remaining maturity <= 1 year
    - COUPONS: > 1 year and < 10 years
    - LONG: >= 10 years
    """
    if maturity_date is not None and pd.notna(maturity_date):
        years = (maturity_date - record_date).days / 365.25
        if years <= 1.0:
            return "BILLS"
        if years < 10.0:
            return "COUPONS"
        return "LONG"

    s = (security_type or "").lower()
    if "bill" in s or "cmb" in s or "cash management" in s:
        return "BILLS"
    # Treat TIPS/FRNs as coupons unless maturity-based classification is available.
    if "bond" in s or "30-year" in s or "20-year" in s:
        return "LONG"
    if "note" in s or "tip" in s or "inflation" in s or "frn" in s or "floating" in s:
        return "COUPONS"
    return None


def _fetch_mspd_marketable_outstanding_buckets(
    *,
    start_date: str,
    refresh: bool,
) -> pd.DataFrame:
    """Fetch MSPD marketable detail and aggregate outstanding by remaining-maturity buckets.

    Returns monthly rows with columns:
      - date
      - OUT_BILLS, OUT_COUPONS, OUT_LONG

    Notes:
    - MSPD is monthly; this function intentionally returns monthly frequency.
    - We compute remaining maturity from (maturity_date - record_date) when available.
    """
    fd = FiscalDataClient()
    df = fd.fetch(
        endpoint=_MSPD_MARKET_ENDPOINT,
        params={
            "sort": "record_date",
            # fiscaldata uses `filter` syntax; keep it simple and server-side.
            "filter": f"record_date:gte:{start_date}",
            # pull all fields; the client may already request everything.
        },
        cache_key=f"mspd_table_3_market_{start_date}",
        refresh=refresh,
    )
    if df.empty:
        return pd.DataFrame(columns=["date", "OUT_BILLS", "OUT_COUPONS", "OUT_LONG"])

    date_col = _first_existing_col(df, _MSPD_DATE_COLS)
    amt_col = _first_existing_col(df, _MSPD_AMT_COLS)
    if date_col is None or amt_col is None:
        return pd.DataFrame(columns=["date", "OUT_BILLS", "OUT_COUPONS", "OUT_LONG"])

    mat_col = _first_existing_col(df, _MSPD_MATURITY_COLS)
    typ_col = _first_existing_col(df, _MSPD_TYPE_COLS)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty:
        return pd.DataFrame(columns=["date", "OUT_BILLS", "OUT_COUPONS", "OUT_LONG"])

    if mat_col is not None:
        df[mat_col] = pd.to_datetime(df[mat_col], errors="coerce")

    if typ_col is not None:
        df[typ_col] = df[typ_col].astype(str)

    df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce")
    df = df.dropna(subset=[amt_col])
    if df.empty:
        return pd.DataFrame(columns=["date", "OUT_BILLS", "OUT_COUPONS", "OUT_LONG"])

    # Classify each row into a maturity bucket
    def _bucket_row(r: pd.Series) -> str | None:
        rd = pd.to_datetime(r[date_col])
        md = pd.to_datetime(r[mat_col]) if mat_col is not None and pd.notna(r.get(mat_col)) else None
        st = str(r.get(typ_col)) if typ_col is not None else None
        return _classify_bucket(record_date=rd, maturity_date=md, security_type=st)

    df["_bucket"] = df.apply(_bucket_row, axis=1)
    df = df.dropna(subset=["_bucket"])
    if df.empty:
        return pd.DataFrame(columns=["date", "OUT_BILLS", "OUT_COUPONS", "OUT_LONG"])

    out = (
        df.groupby([date_col, "_bucket"], as_index=False)[amt_col]
        .sum()
        .pivot(index=date_col, columns="_bucket", values=amt_col)
        .reset_index()
        .rename(columns={date_col: "date", "BILLS": "OUT_BILLS", "COUPONS": "OUT_COUPONS", "LONG": "OUT_LONG"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    for c in ("OUT_BILLS", "OUT_COUPONS", "OUT_LONG"):
        if c not in out.columns:
            out[c] = 0.0

    return out[["date", "OUT_BILLS", "OUT_COUPONS", "OUT_LONG"]]


def _compute_net_issuance_from_outstanding(out: pd.DataFrame) -> pd.DataFrame:
    """Compute net issuance as Δ outstanding (monthly)."""
    if out.empty:
        return pd.DataFrame(columns=["date", "NET_ISS_BILLS", "NET_ISS_COUPONS", "NET_ISS_LONG", "LONG_DURATION_ISS_SHARE"])

    out = out.sort_values("date").reset_index(drop=True)
    out["NET_ISS_BILLS"] = out["OUT_BILLS"].diff(1)
    out["NET_ISS_COUPONS"] = out["OUT_COUPONS"].diff(1)
    out["NET_ISS_LONG"] = out["OUT_LONG"].diff(1)

    denom = out[["NET_ISS_BILLS", "NET_ISS_COUPONS", "NET_ISS_LONG"]].sum(axis=1)
    out["LONG_DURATION_ISS_SHARE"] = out["NET_ISS_LONG"] / denom.replace({0.0: pd.NA})

    return out[["date", "NET_ISS_BILLS", "NET_ISS_COUPONS", "NET_ISS_LONG", "LONG_DURATION_ISS_SHARE"]]


def _net_issuance_by_original_term(
    *,
    mspd: pd.DataFrame,
    target_years: float,
    tol_years: float = 0.6,
) -> pd.DataFrame:
    """
    Compute net issuance for a specific *original-term* sector (e.g. 2Y, 10Y) from MSPD detail.

    Definition:
    - outstanding_sector(t) = sum(outstanding_amt for CUSIPs where (maturity_date - issue_date) ≈ target_years)
    - net_issuance_sector(t) = outstanding_sector(t) - outstanding_sector(t-1)

    This avoids needing auction plumbing and matches the "true net issuance = Δ outstanding" intent.
    """
    if mspd.empty:
        return pd.DataFrame(columns=["date", "OUT", "NET_ISS"])

    # Required columns in MSPD table 3 market
    for c in ("record_date", "issue_date", "maturity_date", "outstanding_amt"):
        if c not in mspd.columns:
            return pd.DataFrame(columns=["date", "OUT", "NET_ISS"])

    df = mspd.copy()
    df["record_date"] = pd.to_datetime(df["record_date"], errors="coerce")
    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], errors="coerce")
    df["outstanding_amt"] = pd.to_numeric(df["outstanding_amt"], errors="coerce")
    df = df.dropna(subset=["record_date", "issue_date", "maturity_date", "outstanding_amt"])
    if df.empty:
        return pd.DataFrame(columns=["date", "OUT", "NET_ISS"])

    # Original term in years
    df["orig_term_years"] = (df["maturity_date"] - df["issue_date"]).dt.days / 365.25
    lo = float(target_years) - float(tol_years)
    hi = float(target_years) + float(tol_years)
    df = df[(df["orig_term_years"] >= lo) & (df["orig_term_years"] <= hi)]
    if df.empty:
        return pd.DataFrame(columns=["date", "OUT", "NET_ISS"])

    out = (
        df.groupby("record_date", as_index=False)["outstanding_amt"]
        .sum()
        .rename(columns={"record_date": "date", "outstanding_amt": "OUT"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    out["NET_ISS"] = out["OUT"].diff(1)
    return out


def _fetch_optional(*, fred: FredClient, series_id: str, start_date: str, refresh: bool) -> pd.DataFrame | None:
    try:
        return fred.fetch_series(series_id=series_id, start_date=start_date, refresh=refresh)
    except Exception:
        return None


def _weighted_score(row: pd.Series, weights: Dict[str, float]) -> float | None:
    """
    Weighted mean over available (non-NaN) components. Normalized by sum(abs(w)).
    """
    num = 0.0
    denom = 0.0
    for col, w in weights.items():
        v = row.get(col, None)
        if v is None or pd.isna(v):
            continue
        num += float(w) * float(v)
        denom += abs(float(w))
    if denom <= 0:
        return None
    return num / denom


def _rolling_12m_sum_monthly(flow: pd.Series) -> pd.Series:
    """
    12-month rolling sum for monthly flows.

    IMPORTANT: do this on the *monthly* series, not on a daily forward-filled grid.
    """
    return flow.rolling(12).sum()


def _yoy_pct_change(series: pd.Series, periods: int) -> pd.Series:
    """
    Percent change vs `periods` observations ago (e.g. 12 for monthly, 4 for quarterly).
    """
    return series.pct_change(periods) * 100.0


def _tga_behavior_metrics(
    *,
    fred: FredClient,
    start_date: str,
    refresh: bool,
    z_window_weeks: int = 104,
) -> dict[str, float | str | None]:
    """
    Lean TGA behavior features from WTREGEN (weekly, level).

    Returns:
    - tga_asof
    - tga_level
    - tga_d_4w
    - tga_d_13w
    - tga_z_d_4w (z-score of 4w change vs recent history)
    - tga_z_level (z-score of level vs recent history)
    """
    df = fred.fetch_series(series_id=FISCAL_FRED_SERIES["TGA"], start_date=start_date, refresh=refresh)
    if df.empty:
        return {
            "tga_asof": None,
            "tga_level": None,
            "tga_d_4w": None,
            "tga_d_13w": None,
            "tga_z_d_4w": None,
            "tga_z_level": None,
        }

    df = df.sort_values("date").reset_index(drop=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    if df.empty:
        return {
            "tga_asof": None,
            "tga_level": None,
            "tga_d_4w": None,
            "tga_d_13w": None,
            "tga_z_d_4w": None,
            "tga_z_level": None,
        }

    # Weekly -> use observation shifts
    df["D_4W"] = df["value"] - df["value"].shift(4)
    df["D_13W"] = df["value"] - df["value"].shift(13)

    mu = df["D_4W"].rolling(int(z_window_weeks)).mean()
    sd = df["D_4W"].rolling(int(z_window_weeks)).std(ddof=0)
    df["Z_D_4W"] = (df["D_4W"] - mu) / sd

    mu_lvl = df["value"].rolling(int(z_window_weeks)).mean()
    sd_lvl = df["value"].rolling(int(z_window_weeks)).std(ddof=0)
    df["Z_LEVEL"] = (df["value"] - mu_lvl) / sd_lvl

    last = df.iloc[-1]
    return {
        "tga_asof": str(pd.to_datetime(last["date"]).date()),
        "tga_level": float(last["value"]) if pd.notna(last["value"]) else None,
        "tga_d_4w": float(last["D_4W"]) if pd.notna(last["D_4W"]) else None,
        "tga_d_13w": float(last["D_13W"]) if pd.notna(last["D_13W"]) else None,
        "tga_z_d_4w": float(last["Z_D_4W"]) if pd.notna(last["Z_D_4W"]) else None,
        "tga_z_level": float(last["Z_LEVEL"]) if pd.notna(last["Z_LEVEL"]) else None,
    }


def _gdp_asof(
    *,
    fred: FredClient,
    asof: pd.Timestamp,
    start_date: str,
    refresh: bool,
) -> dict[str, object | None]:
    """
    Best-effort GDP as-of (quarterly).

    Prefer nominal GDP (series_id = "GDP"). Fall back to real GDP ("GDPC1") if needed.
    Returns GDP in USD *millions* to align with MTSDS133FMS units.
    """
    for sid in ("GDP", "GDPC1"):
        try:
            g = fred.fetch_series(series_id=sid, start_date=start_date, refresh=refresh)
        except Exception:
            continue
        if g is None or g.empty:
            continue
        g = g.sort_values("date").reset_index(drop=True)
        g["date"] = pd.to_datetime(g["date"], errors="coerce")
        g["value"] = pd.to_numeric(g["value"], errors="coerce")
        g = g.dropna(subset=["date", "value"])
        if g.empty:
            continue
        sub = g[g["date"] <= asof]
        if sub.empty:
            continue
        last = sub.iloc[-1]
        # FRED GDP / GDPC1 are typically in USD billions (SAAR). Convert to millions for ratio computations.
        return {
            "series": sid,
            "asof": str(pd.to_datetime(last["date"]).date()),
            "gdp_millions": float(last["value"]) * 1000.0,
        }
    return {"series": None, "asof": None, "gdp_millions": None}




def build_fiscal_dataset(settings: Settings, start_date: str = "2011-01-01", refresh: bool = False) -> pd.DataFrame:
    """
    Build a daily-grid fiscal dataset with best-effort features for regime labeling.

    MVP columns you can rely on:
    - DEFICIT_12M (rolling 12m deficit, positive = worse)
    - TGA (level) and TGA_CHG_28D
    - INTEREST_EXPENSE_YOY + INTEREST_EXPENSE_YOY_ACCEL (if series available)

    Optional placeholders (may be NaN until you wire a Treasury issuance/auction data source):
    - NET_ISS_* (bills/coupons/long) and LONG_DURATION_ISS_SHARE (best-effort via MSPD Δ outstanding)
    - AUCTION_TAIL_BPS, DEALER_TAKE_PCT (placeholders until auction results are wired)
    """
    if not settings.FRED_API_KEY:
        raise RuntimeError("Missing FRED_API_KEY in environment / .env")

    fred = FredClient(api_key=settings.FRED_API_KEY)

    # Fetch raw series (native frequency) first.
    raw: Dict[str, pd.DataFrame] = {}
    for name, sid in FISCAL_FRED_SERIES.items():
        if name in _OPTIONAL_SERIES:
            df = _fetch_optional(fred=fred, series_id=sid, start_date=start_date, refresh=refresh)
            if df is None or df.empty:
                continue
        else:
            df = fred.fetch_series(series_id=sid, start_date=start_date, refresh=refresh)
        raw[name] = df.sort_values("date").reset_index(drop=True)

    if "SURPLUS_DEFICIT" not in raw or raw["SURPLUS_DEFICIT"].empty:
        raise RuntimeError("Failed to load fiscal deficit series (FRED FYFSD).")

    # ---------------------------------------------------------------------
    # Derived monthly/quarterly features computed on native frequency, then merged to daily grid.
    # ---------------------------------------------------------------------
    derived_frames: Dict[str, pd.DataFrame] = {}

    # Rolling 12m deficit (positive = larger deficit)
    sd = raw["SURPLUS_DEFICIT"].copy()
    sd = sd.rename(columns={"value": "SURPLUS_DEFICIT"})
    sd["DEFICIT"] = -sd["SURPLUS_DEFICIT"]  # FYFSD: negative values => deficit
    sd["DEFICIT_12M"] = _rolling_12m_sum_monthly(sd["DEFICIT"])
    # Deficit context: YoY change, recent trend, and a simple trend extrapolation.
    sd["DEFICIT_12M_YOY_ABS"] = sd["DEFICIT_12M"] - sd["DEFICIT_12M"].shift(12)
    sd["DEFICIT_12M_YOY_PCT"] = (sd["DEFICIT_12M"] / sd["DEFICIT_12M"].shift(12) - 1.0) * 100.0
    sd["DEFICIT_12M_CHG_6M"] = sd["DEFICIT_12M"] - sd["DEFICIT_12M"].shift(6)
    sd["DEFICIT_12M_PROJ_6M"] = sd["DEFICIT_12M"] + sd["DEFICIT_12M_CHG_6M"]
    derived_frames["DEFICIT_12M"] = sd[["date", "DEFICIT_12M"]]
    # Deficit trend slope (12m OLS slope) for momentum sub-score
    if len(sd) >= 12:
        x = np.arange(12, dtype=float)
        def _ols_slope(y: pd.Series) -> float:
            if y.isna().all() or len(y) < 12:
                return np.nan
            yv = y.tail(12).values.astype(float)
            if np.any(np.isnan(yv)):
                return np.nan
            xm, ym = x.mean(), yv.mean()
            slope = np.sum((x - xm) * (yv - ym)) / (np.sum((x - xm) ** 2) + 1e-12)
            return float(slope)
        sd["DEFICIT_TREND_SLOPE"] = sd["DEFICIT_12M"].rolling(12).apply(_ols_slope, raw=False)
        derived_frames["DEFICIT_TREND_SLOPE"] = sd[["date", "DEFICIT_TREND_SLOPE"]]

    # Federal receipts (deficit sustainability)
    if "TAX_RECEIPTS" in raw and not raw["TAX_RECEIPTS"].empty:
        rec = raw["TAX_RECEIPTS"].copy().rename(columns={"value": "TAX_RECEIPTS"})
        rec["RECEIPTS_12M"] = _rolling_12m_sum_monthly(rec["TAX_RECEIPTS"])
        sd_dates = sd[["date", "DEFICIT_12M"]].dropna(subset=["DEFICIT_12M"])
        rec = rec.merge(sd_dates, on="date", how="inner")
        rec["DEFICIT_PCT_RECEIPTS"] = np.where(
            rec["RECEIPTS_12M"].abs() > 1e-6,
            rec["DEFICIT_12M"] / rec["RECEIPTS_12M"],
            np.nan,
        )
        derived_frames["DEFICIT_PCT_RECEIPTS"] = rec[["date", "DEFICIT_PCT_RECEIPTS"]]

    # Interest expense YoY (optional, likely quarterly)
    if "INTEREST_EXPENSE" in raw and not raw["INTEREST_EXPENSE"].empty:
        ie = raw["INTEREST_EXPENSE"].copy().rename(columns={"value": "INTEREST_EXPENSE"})
        # This particular series is quarterly; YoY is 4 periods.
        ie["INTEREST_EXPENSE_YOY"] = _yoy_pct_change(ie["INTEREST_EXPENSE"], periods=4)
        ie["INTEREST_EXPENSE_YOY_ACCEL"] = ie["INTEREST_EXPENSE_YOY"] - ie["INTEREST_EXPENSE_YOY"].shift(4)
        derived_frames["INTEREST_EXPENSE_YOY"] = ie[["date", "INTEREST_EXPENSE_YOY"]]
        derived_frames["INTEREST_EXPENSE_YOY_ACCEL"] = ie[["date", "INTEREST_EXPENSE_YOY_ACCEL"]]

    # True net issuance by maturity bucket via MSPD (Δ outstanding). Best-effort; may be empty if endpoint unavailable.
    try:
        mspd_out = _fetch_mspd_marketable_outstanding_buckets(start_date=start_date, refresh=refresh)
        mspd_net = _compute_net_issuance_from_outstanding(mspd_out)
        if not mspd_net.empty:
            derived_frames["NET_ISS_BILLS"] = mspd_net[["date", "NET_ISS_BILLS"]]
            derived_frames["NET_ISS_COUPONS"] = mspd_net[["date", "NET_ISS_COUPONS"]]
            derived_frames["NET_ISS_LONG"] = mspd_net[["date", "NET_ISS_LONG"]]
            derived_frames["LONG_DURATION_ISS_SHARE"] = mspd_net[["date", "LONG_DURATION_ISS_SHARE"]]
    except Exception:
        # Keep placeholders as NaN if MSPD pull fails.
        pass

    # Auction quality / absorption (TreasuryDirect auctions via FiscalData)
    try:
        a = _fetch_treasury_auction_results(start_date=start_date, refresh=refresh)
        a_monthly = _compute_auction_tail_and_dealer_take(a)
        if not a_monthly.empty:
            derived_frames["AUCTION_TAIL_BPS"] = a_monthly[["date", "AUCTION_TAIL_BPS"]]
            derived_frames["DEALER_TAKE_PCT"] = a_monthly[["date", "DEALER_TAKE_PCT"]]
    except Exception:
        # Keep placeholders as NaN if auctions pull fails.
        pass

    # ---------------------------------------------------------------------
    # Daily grid: merge levels (TGA) + derived features (ffill).
    # ---------------------------------------------------------------------
    max_date = max(df["date"].max() for df in raw.values())
    base = pd.DataFrame(
        {"date": pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(max_date), freq="D")}
    )

    # Merge level series onto daily grid (ffill)
    level_series: Dict[str, pd.DataFrame] = {}
    if "TGA" in raw:
        level_series["TGA"] = raw["TGA"].rename(columns={"value": "TGA"})[["date", "TGA"]]
    if "FOREIGN_HOLDINGS" in raw:
        level_series["FOREIGN_HOLDINGS"] = raw["FOREIGN_HOLDINGS"].rename(columns={"value": "FOREIGN_HOLDINGS"})[["date", "FOREIGN_HOLDINGS"]]
    if "CUSTODY" in raw:
        level_series["CUSTODY"] = raw["CUSTODY"].rename(columns={"value": "CUSTODY"})[["date", "CUSTODY"]]

    # merge_series_daily expects frames with `value` column; build compatible frames
    merge_map: Dict[str, pd.DataFrame] = {}
    for k, df in level_series.items():
        tmp = df.rename(columns={k: "value"})
        merge_map[k] = tmp[["date", "value"]]
    merged = merge_series_daily(base, merge_map, ffill=True)

    # Merge derived series with ffill
    for col, df in derived_frames.items():
        merged = merged.merge(df, on="date", how="left")
        # Avoid pandas future downcasting warnings by coercing to numeric before ffill.
        merged[col] = pd.to_numeric(merged[col], errors="coerce").ffill()

    # TGA dynamics
    if "TGA" in merged.columns:
        merged["TGA_CHG_28D"] = merged["TGA"] - merged["TGA"].shift(28)
    # Foreign holdings 6m change (demand structure)
    if "FOREIGN_HOLDINGS" in merged.columns:
        merged["FOREIGN_HOLDINGS_CHG_6M"] = merged["FOREIGN_HOLDINGS"] - merged["FOREIGN_HOLDINGS"].shift(126)
    # Custody 4w change (Fed custody proxy for foreign CB demand)
    if "CUSTODY" in merged.columns:
        merged["CUSTODY_CHG_4W"] = merged["CUSTODY"] - merged["CUSTODY"].shift(28)

    # ---------------------------------------------------------------------
    # Ensure stable columns exist even if some upstream sources are unavailable.
    for c in [
        "NET_ISS_BILLS",
        "NET_ISS_COUPONS",
        "NET_ISS_LONG",
        "LONG_DURATION_ISS_SHARE",
        "AUCTION_TAIL_BPS",
        "DEALER_TAKE_PCT",
        "DEFICIT_PCT_RECEIPTS",
        "DEFICIT_TREND_SLOPE",
        "FOREIGN_HOLDINGS_CHG_6M",
        "CUSTODY_CHG_4W",
    ]:
        if c not in merged.columns:
            merged[c] = float("nan")

    # ---------------------------------------------------------------------
    # Standardize (z-scores) on the daily grid as a pragmatic baseline.
    # Note: For monthly/quarterly features, forward-fill means "daily" z-scores are still usable
    # as a *relative* positioning metric, but we keep windows long to reduce step artifacts.
    # ---------------------------------------------------------------------
    merged["Z_DEFICIT_12M"] = zscore(merged["DEFICIT_12M"], window=252 * 3)
    if "TGA_CHG_28D" in merged.columns:
        merged["Z_TGA_CHG_28D"] = zscore(merged["TGA_CHG_28D"], window=252 * 3)
    if "INTEREST_EXPENSE_YOY" in merged.columns:
        merged["Z_INTEREST_EXPENSE_YOY"] = zscore(merged["INTEREST_EXPENSE_YOY"], window=252 * 3)
    merged["Z_LONG_DURATION_ISS_SHARE"] = zscore(
        pd.to_numeric(merged["LONG_DURATION_ISS_SHARE"], errors="coerce"),
        window=252 * 3,
    )
    merged["Z_AUCTION_TAIL_BPS"] = zscore(
        pd.to_numeric(merged["AUCTION_TAIL_BPS"], errors="coerce"),
        window=252 * 3,
    )
    merged["Z_DEALER_TAKE_PCT"] = zscore(
        pd.to_numeric(merged["DEALER_TAKE_PCT"], errors="coerce"),
        window=252 * 3,
    )

    # Composite fiscal pressure score (positive = more pressure)
    weights: Dict[str, float] = {
        "Z_DEFICIT_12M": 0.30,
        "Z_LONG_DURATION_ISS_SHARE": 0.20,
        "Z_AUCTION_TAIL_BPS": 0.20,
        "Z_DEALER_TAKE_PCT": 0.20,
        "Z_INTEREST_EXPENSE_YOY": 0.10,
    }
    merged["FISCAL_PRESSURE_SCORE"] = merged.apply(lambda r: _weighted_score(r, weights), axis=1)

    return merged


def build_fiscal_deficit_page_data(
    settings: Settings,
    lookback_years: int = 5,
    refresh: bool = False,
) -> dict[str, object]:
    """
    Minimal fiscal "page" data:
    - asof date
    - rolling 12m deficit (positive = larger deficit)

    This intentionally avoids pulling optional/non-core fiscal features so you can
    understand the plumbing with a single signal first.
    """
    if not settings.FRED_API_KEY:
        raise RuntimeError("Missing FRED_API_KEY in environment / .env")

    # We only need enough history to compute a 12-month rolling sum.
    # Use a modest lookback window so "current snapshot" doesn't require the user to think about start dates.
    start_date = str((pd.Timestamp.today().normalize() - pd.DateOffset(years=int(lookback_years))).date())

    fred = FredClient(api_key=settings.FRED_API_KEY)
    sd = fred.fetch_series(
        series_id=FISCAL_FRED_SERIES["SURPLUS_DEFICIT"],
        start_date=start_date,
        refresh=refresh,
    )
    if sd.empty:
        raise RuntimeError("No data returned for fiscal deficit series (FRED MTSDS133FMS).")

    sd = sd.sort_values("date").reset_index(drop=True).rename(columns={"value": "SURPLUS_DEFICIT"})
    sd["DEFICIT"] = -sd["SURPLUS_DEFICIT"]
    sd["DEFICIT_12M"] = _rolling_12m_sum_monthly(sd["DEFICIT"])
    roll = sd.dropna(subset=["DEFICIT_12M"])
    if roll.empty:
        raise RuntimeError(
            "Not enough observations to compute rolling 12-month deficit. "
            "Try increasing --lookback-years (e.g. 10)."
        )
    last = roll.iloc[-1]

    # Lean TGA behavior metrics (weekly series, reported separately from deficit as-of)
    tga = _tga_behavior_metrics(fred=fred, start_date=start_date, refresh=refresh)

    # Minimal deficit context: rolling 12m deficit now vs ~30d ago vs ~1y ago.
    # Since the underlying series is monthly, we take the last observation on/before the target date.
    def _deficit_12m_asof(target_date: pd.Timestamp) -> dict[str, object | None]:
        sub = roll[roll["date"] <= target_date]
        if sub.empty:
            return {"asof": None, "deficit_12m": None}
        r = sub.iloc[-1]
        return {
            "asof": str(pd.to_datetime(r["date"]).date()),
            "deficit_12m": float(r["DEFICIT_12M"]) if pd.notna(r["DEFICIT_12M"]) else None,
        }

    asof_now = pd.to_datetime(last["date"])
    d_30 = _deficit_12m_asof(asof_now - pd.Timedelta(days=30))
    d_1y = _deficit_12m_asof(asof_now - pd.Timedelta(days=365))

    # Deficit pace (lean):
    # 1) Δ rolling 12m deficit (YoY): D12m(t) - D12m(t-12)
    # 2) Impulse as % GDP: ΔD12m / GDP
    roll2 = roll.copy()
    roll2["DEFICIT_12M_LAG12"] = roll2["DEFICIT_12M"].shift(12)
    last2 = roll2.iloc[-1]
    delta_yoy = None
    impulse_pct_gdp = None
    gdp = _gdp_asof(fred=fred, asof=asof_now, start_date=start_date, refresh=refresh)
    if pd.notna(last2.get("DEFICIT_12M_LAG12")):
        delta_yoy = float(last2["DEFICIT_12M"] - last2["DEFICIT_12M_LAG12"])
        gdp_m = gdp.get("gdp_millions")
        if isinstance(gdp_m, (int, float)) and float(gdp_m) != 0.0:
            impulse_pct_gdp = 100.0 * float(delta_yoy) / float(gdp_m)

    deficit_pct_gdp = None
    gdp_m2 = gdp.get("gdp_millions")
    if isinstance(gdp_m2, (int, float)) and float(gdp_m2) != 0.0:
        deficit_pct_gdp = 100.0 * float(last["DEFICIT_12M"]) / float(gdp_m2)

    # True net issuance (Δ outstanding) from MSPD.
    # We expose both:
    # - maturity buckets (Bills/Coupons/Long + long-duration share)
    # - specific original-term sectors (2Y/10Y) for convenience
    net_iss = None
    net_iss_2y = None
    net_iss_10y = None
    try:
        mspd_out = _fetch_mspd_marketable_outstanding_buckets(start_date=start_date, refresh=refresh)
        mspd_bucket_net = _compute_net_issuance_from_outstanding(mspd_out).dropna(
            subset=["NET_ISS_BILLS", "NET_ISS_COUPONS", "NET_ISS_LONG"],
            how="all",
        )
        if not mspd_bucket_net.empty:
            last_bucket = mspd_bucket_net.iloc[-1]
            net_iss = {
                "asof": str(pd.to_datetime(last_bucket["date"]).date()),
                "bills": float(last_bucket["NET_ISS_BILLS"]) if pd.notna(last_bucket["NET_ISS_BILLS"]) else None,
                "coupons": float(last_bucket["NET_ISS_COUPONS"]) if pd.notna(last_bucket["NET_ISS_COUPONS"]) else None,
                "long": float(last_bucket["NET_ISS_LONG"]) if pd.notna(last_bucket["NET_ISS_LONG"]) else None,
                "long_duration_share": float(last_bucket["LONG_DURATION_ISS_SHARE"])
                if pd.notna(last_bucket["LONG_DURATION_ISS_SHARE"])
                else None,
                "notes": "True net issuance computed as month-over-month Δ outstanding from MSPD table 3 (marketable), bucketed by remaining maturity.",
            }

        # Use the raw MSPD table (already cached) for original-term computations.
        mspd_raw = FiscalDataClient().fetch(
            endpoint=_MSPD_MARKET_ENDPOINT,
            params={"sort": "record_date", "filter": f"record_date:gte:{start_date}"},
            cache_key=f"mspd_table_3_market_{start_date}",
            refresh=refresh,
        )

        s2 = _net_issuance_by_original_term(mspd=mspd_raw, target_years=2.0)
        if not s2.empty and s2.dropna(subset=["NET_ISS"]).shape[0] > 0:
            net_iss_2y = float(s2.dropna(subset=["NET_ISS"]).iloc[-1]["NET_ISS"])

        s10 = _net_issuance_by_original_term(mspd=mspd_raw, target_years=10.0)
        if not s10.empty and s10.dropna(subset=["NET_ISS"]).shape[0] > 0:
            net_iss_10y = float(s10.dropna(subset=["NET_ISS"]).iloc[-1]["NET_ISS"])
    except Exception:
        net_iss = None
        net_iss_2y = None
        net_iss_10y = None

    # Auctions (best-effort): monthly aggregated + per-tenor + recent individual
    auctions = None
    try:
        a = _fetch_treasury_auction_results(start_date=start_date, refresh=refresh)
        a_monthly = _compute_auction_tail_and_dealer_take(a)
        by_tenor = _compute_auction_detail_by_tenor(a)
        recent_list = _compute_recent_auctions_list(a, n=20)
        if not a_monthly.empty:
            last_a = a_monthly.dropna(how="all", subset=["AUCTION_TAIL_BPS", "DEALER_TAKE_PCT"]).iloc[-1]
            auctions = {
                "asof": str(pd.to_datetime(last_a["date"]).date()),
                "tail_bps": float(last_a["AUCTION_TAIL_BPS"]) if pd.notna(last_a["AUCTION_TAIL_BPS"]) else None,
                "dealer_take_pct": float(last_a["DEALER_TAKE_PCT"]) if pd.notna(last_a["DEALER_TAKE_PCT"]) else None,
                "by_tenor": by_tenor,
                "recent": recent_list,
                "notes": (
                    "Tail is proxied as (high_yield - median_yield) in bps — NOT true tail vs when-issued. "
                    "Dealer take is primary dealer takedown as % of total accepted. "
                    "by_tenor splits front-end (2Y-5Y) vs back-end (7Y-30Y) over last 3 months."
                ),
            }
    except Exception:
        auctions = None

    return {
        "asof": str(pd.to_datetime(last["date"]).date()),
        "deficit_12m": float(last["DEFICIT_12M"]),
        "deficit_12m_30d_ago": d_30,
        "deficit_12m_1y_ago": d_1y,
        "deficit_12m_delta_yoy": delta_yoy,
        "deficit_impulse_pct_gdp": impulse_pct_gdp,
        "deficit_pct_gdp": deficit_pct_gdp,
        "gdp": gdp,
        "units": "FRED units (MTSDS133FMS is typically USD millions; rolling 12m sum; sign-adjusted so positive=deficit)",
        "lookback_years": int(lookback_years),
        "tga": tga,
        # MSPD units are typically USD millions; treat as such in display formatting.
        "net_issuance": net_iss,
        "net_issuance_2y": net_iss_2y,
        "net_issuance_10y": net_iss_10y,
        "net_issuance_units": "USD millions (MSPD table 3)",
        "net_issuance_notes": (
            "Net issuance is computed as month-over-month Δ outstanding from MSPD (marketable). "
            "Buckets are by remaining maturity; 2Y/10Y are by original term (maturity_date - issue_date)."
        ),
        "auctions": auctions,
        "series_used": {
            "fred": sorted(
                {
                    FISCAL_FRED_SERIES["SURPLUS_DEFICIT"],
                    FISCAL_FRED_SERIES["TGA"],
                    str(gdp.get("series")) if gdp.get("series") else "",
                }
                - {""}
            ),
            "fiscaldata": ["mspd_table_3_market", "auctions_query"],
        },
    }


def build_fiscal_state(settings: Settings, start_date: str = "2011-01-01", refresh: bool = False) -> FiscalState:
    df = build_fiscal_dataset(settings=settings, start_date=start_date, refresh=refresh)

    # Require core series
    last = df.dropna(subset=["DEFICIT_12M"]).iloc[-1]

    score = float(last["FISCAL_PRESSURE_SCORE"]) if pd.notna(last["FISCAL_PRESSURE_SCORE"]) else None

    inputs = FiscalInputs(
        deficit_12m=float(last["DEFICIT_12M"]) if pd.notna(last["DEFICIT_12M"]) else None,
        tga_level=float(last["TGA"]) if "TGA" in last and pd.notna(last["TGA"]) else None,
        tga_chg_28d=float(last["TGA_CHG_28D"]) if "TGA_CHG_28D" in last and pd.notna(last["TGA_CHG_28D"]) else None,
        interest_expense_yoy=float(last["INTEREST_EXPENSE_YOY"])
        if "INTEREST_EXPENSE_YOY" in last and pd.notna(last["INTEREST_EXPENSE_YOY"])
        else None,
        interest_expense_yoy_accel=float(last["INTEREST_EXPENSE_YOY_ACCEL"])
        if "INTEREST_EXPENSE_YOY_ACCEL" in last and pd.notna(last["INTEREST_EXPENSE_YOY_ACCEL"])
        else None,
        net_issuance_bills=float(last["NET_ISS_BILLS"]) if pd.notna(last["NET_ISS_BILLS"]) else None,
        net_issuance_coupons=float(last["NET_ISS_COUPONS"]) if pd.notna(last["NET_ISS_COUPONS"]) else None,
        net_issuance_long=float(last["NET_ISS_LONG"]) if pd.notna(last["NET_ISS_LONG"]) else None,
        long_duration_issuance_share=float(last["LONG_DURATION_ISS_SHARE"])
        if pd.notna(last["LONG_DURATION_ISS_SHARE"])
        else None,
        auction_tail_bps=float(last["AUCTION_TAIL_BPS"]) if pd.notna(last["AUCTION_TAIL_BPS"]) else None,
        dealer_take_pct=float(last["DEALER_TAKE_PCT"]) if pd.notna(last["DEALER_TAKE_PCT"]) else None,
        bid_to_cover_avg=None,  # TODO: extract from FiscalData auction response when schema available
        deficit_pct_receipts=float(last["DEFICIT_PCT_RECEIPTS"]) if "DEFICIT_PCT_RECEIPTS" in last and pd.notna(last["DEFICIT_PCT_RECEIPTS"]) else None,
        foreign_holdings_pct=None,  # TODO: require total debt series for share
        foreign_holdings_chg_6m=float(last["FOREIGN_HOLDINGS_CHG_6M"]) if "FOREIGN_HOLDINGS_CHG_6M" in last and pd.notna(last["FOREIGN_HOLDINGS_CHG_6M"]) else None,
        custody_holdings_chg_4w=float(last["CUSTODY_CHG_4W"]) if "CUSTODY_CHG_4W" in last and pd.notna(last["CUSTODY_CHG_4W"]) else None,
        wam_years=None,  # TODO: compute from MSPD maturity buckets
        wam_chg_12m=None,
        deficit_trend_slope=float(last["DEFICIT_TREND_SLOPE"]) if "DEFICIT_TREND_SLOPE" in last and pd.notna(last["DEFICIT_TREND_SLOPE"]) else None,
        move_index=None,  # TODO: FMP ^MOVE
        move_index_z=None,
        z_deficit_12m=float(last["Z_DEFICIT_12M"]) if pd.notna(last["Z_DEFICIT_12M"]) else None,
        z_tga_chg_28d=float(last["Z_TGA_CHG_28D"]) if "Z_TGA_CHG_28D" in last and pd.notna(last["Z_TGA_CHG_28D"]) else None,
        z_interest_expense_yoy=float(last["Z_INTEREST_EXPENSE_YOY"])
        if "Z_INTEREST_EXPENSE_YOY" in last and pd.notna(last["Z_INTEREST_EXPENSE_YOY"])
        else None,
        z_long_duration_issuance_share=float(last["Z_LONG_DURATION_ISS_SHARE"])
        if pd.notna(last["Z_LONG_DURATION_ISS_SHARE"])
        else None,
        z_auction_tail_bps=float(last["Z_AUCTION_TAIL_BPS"]) if pd.notna(last["Z_AUCTION_TAIL_BPS"]) else None,
        z_dealer_take_pct=float(last["Z_DEALER_TAKE_PCT"]) if pd.notna(last["Z_DEALER_TAKE_PCT"]) else None,
        fiscal_pressure_score=score,
        components={
            "w_z_deficit_12m": 0.30,
            "w_z_long_duration_iss_share": 0.20,
            "w_z_auction_tail_bps": 0.20,
            "w_z_dealer_take_pct": 0.20,
            "w_z_interest_expense_yoy": 0.10,
        },
    )

    return FiscalState(
        asof=str(pd.to_datetime(last["date"]).date()),
        start_date=start_date,
        inputs=inputs,
        notes=(
            "Fiscal regime snapshot built from rolling 12m deficit (FYFSD), TGA (WTREGEN), "
            "and an optional interest expense proxy (A091RC1Q027SBEA). "
            "Issuance mix + auction quality fields are placeholders until a Treasury auctions/issuance source is wired."
        ),
    )



"""
Hyperscaler capex tracker (v1).

Pulls quarterly cash-flow-statement data from FMP for the AI capex spenders
and reduces it to a handful of bubble-relevant numbers:

  - Latest quarter capex ($B)
  - YoY capex growth (%) — bubble fuel
  - Capex / operating cash flow ratio — when this approaches 100% they're
    burning cash flow on AI infra; >100% means they're funding it with debt
  - Aggregate TTM capex YoY across the group

Quarterly fundamentals change rarely, so we cache for 7 days.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import requests

from lox.altdata.cache import cache_path, read_cache, write_cache
from lox.config import Settings


@dataclass
class CapexQuarter:
    symbol: str
    period_end: str           # e.g. "2026-03-31"
    capex_usd: float          # positive number, in dollars (FMP reports negative; we flip)
    operating_cash_flow: float | None
    revenue: float | None


@dataclass
class CapexStat:
    symbol: str
    latest_period: str
    latest_capex_bn: float | None       # $B
    capex_yoy_pct: float | None         # latest q vs same q prior year (%)
    capex_to_ocf_pct: float | None      # capex / OCF, ttm (%)
    ttm_capex_bn: float | None          # trailing 4 quarters ($B)
    ttm_revenue_bn: float | None


@dataclass
class CapexPanel:
    per_name: list[CapexStat]
    aggregate_ttm_capex_bn: float | None
    aggregate_ttm_capex_yoy_pct: float | None
    aggregate_capex_to_ocf_pct: float | None


def _fetch_quarterly_cashflow(settings: Settings, symbol: str, refresh: bool) -> list[CapexQuarter]:
    key = f"ai_capex/{symbol.upper()}_qcf"
    path = cache_path(key)
    if not refresh:
        cached = read_cache(path, max_age=timedelta(days=7))
        if cached is not None:
            return [CapexQuarter(**r) for r in cached]

    if not settings.fmp_api_key:
        return []

    url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}"
    try:
        resp = requests.get(
            url,
            params={"apikey": settings.fmp_api_key, "period": "quarter", "limit": 8},
            timeout=20,
        )
        resp.raise_for_status()
        cf_rows = resp.json() if resp.ok else []
    except Exception:
        return []
    if not isinstance(cf_rows, list):
        return []

    # Pull revenue for the same quarters
    rev_by_period: dict[str, float] = {}
    try:
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}"
        resp = requests.get(
            url,
            params={"apikey": settings.fmp_api_key, "period": "quarter", "limit": 8},
            timeout=20,
        )
        if resp.ok:
            for r in resp.json() or []:
                if isinstance(r, dict) and r.get("date") and r.get("revenue") is not None:
                    rev_by_period[str(r["date"])] = float(r["revenue"])
    except Exception:
        pass

    out: list[CapexQuarter] = []
    for r in cf_rows:
        if not isinstance(r, dict):
            continue
        date = r.get("date")
        capex = r.get("capitalExpenditure")
        ocf = r.get("operatingCashFlow") or r.get("netCashProvidedByOperatingActivities")
        if date is None or capex is None:
            continue
        # FMP reports capex as a negative number — flip to positive magnitude
        capex_pos = abs(float(capex))
        out.append(CapexQuarter(
            symbol=symbol.upper(),
            period_end=str(date),
            capex_usd=capex_pos,
            operating_cash_flow=float(ocf) if ocf is not None else None,
            revenue=rev_by_period.get(str(date)),
        ))

    out.sort(key=lambda q: q.period_end, reverse=True)
    write_cache(path, [q.__dict__ for q in out])
    return out


def _stat_for_symbol(quarters: list[CapexQuarter]) -> CapexStat | None:
    if not quarters:
        return None
    latest = quarters[0]
    latest_capex_bn = latest.capex_usd / 1e9

    # YoY (find quarter ~365 days earlier — assume index 4 if available)
    yoy = None
    if len(quarters) >= 5:
        prior = quarters[4]
        if prior.capex_usd:
            yoy = (latest.capex_usd / prior.capex_usd - 1.0) * 100.0

    # TTM aggregates (last 4 quarters)
    ttm_quarters = quarters[:4]
    ttm_capex = sum(q.capex_usd for q in ttm_quarters) if len(ttm_quarters) == 4 else None
    ttm_ocf_vals = [q.operating_cash_flow for q in ttm_quarters if q.operating_cash_flow is not None]
    ttm_ocf = sum(ttm_ocf_vals) if len(ttm_ocf_vals) == 4 else None
    ttm_rev_vals = [q.revenue for q in ttm_quarters if q.revenue is not None]
    ttm_rev = sum(ttm_rev_vals) if len(ttm_rev_vals) == 4 else None

    capex_to_ocf = (100.0 * ttm_capex / ttm_ocf) if (ttm_capex is not None and ttm_ocf and ttm_ocf > 0) else None

    return CapexStat(
        symbol=latest.symbol,
        latest_period=latest.period_end,
        latest_capex_bn=latest_capex_bn,
        capex_yoy_pct=yoy,
        capex_to_ocf_pct=capex_to_ocf,
        ttm_capex_bn=(ttm_capex / 1e9) if ttm_capex is not None else None,
        ttm_revenue_bn=(ttm_rev / 1e9) if ttm_rev is not None else None,
    )


def fetch_capex_panel(*, settings: Settings, symbols: list[str], refresh: bool = False) -> CapexPanel:
    """Pull capex stats for the given symbols and aggregate."""
    per_name: list[CapexStat] = []
    quarters_by_symbol: dict[str, list[CapexQuarter]] = {}
    for sym in symbols:
        qs = _fetch_quarterly_cashflow(settings, sym, refresh=refresh)
        quarters_by_symbol[sym] = qs
        st = _stat_for_symbol(qs)
        if st is not None:
            per_name.append(st)

    # ── Aggregate TTM capex and prior-year TTM capex for YoY ─────────────
    def _ttm_capex(qs: list[CapexQuarter], offset: int) -> float | None:
        slice_ = qs[offset:offset + 4]
        if len(slice_) < 4:
            return None
        return sum(q.capex_usd for q in slice_)

    ttm_now_vals: list[float] = []
    ttm_prior_vals: list[float] = []
    ttm_ocf_vals: list[float] = []
    for sym, qs in quarters_by_symbol.items():
        now = _ttm_capex(qs, 0)
        prior = _ttm_capex(qs, 4)
        if now is not None:
            ttm_now_vals.append(now)
        if prior is not None:
            ttm_prior_vals.append(prior)
        ttm_ocf_subset = [q.operating_cash_flow for q in qs[:4] if q.operating_cash_flow is not None]
        if len(ttm_ocf_subset) == 4:
            ttm_ocf_vals.append(sum(ttm_ocf_subset))

    agg_ttm_capex = sum(ttm_now_vals) if ttm_now_vals else None
    agg_ttm_capex_yoy = None
    if agg_ttm_capex is not None and ttm_prior_vals and sum(ttm_prior_vals) > 0:
        agg_ttm_capex_yoy = (agg_ttm_capex / sum(ttm_prior_vals) - 1.0) * 100.0
    agg_ttm_ocf = sum(ttm_ocf_vals) if ttm_ocf_vals else None
    agg_capex_to_ocf = (100.0 * agg_ttm_capex / agg_ttm_ocf) if (
        agg_ttm_capex is not None and agg_ttm_ocf and agg_ttm_ocf > 0
    ) else None

    return CapexPanel(
        per_name=per_name,
        aggregate_ttm_capex_bn=(agg_ttm_capex / 1e9) if agg_ttm_capex is not None else None,
        aggregate_ttm_capex_yoy_pct=agg_ttm_capex_yoy,
        aggregate_capex_to_ocf_pct=agg_capex_to_ocf,
    )

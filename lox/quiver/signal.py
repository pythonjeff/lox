"""
Cross-source signal scanner — congress + trump trades.

Surfaces tickers where government insiders have bet in a direction
but the stock hasn't yet moved to confirm it. That gap is the edge.

Demoted per the Quiver Edge Expansion plan (Stage 3, see
data/cache/quiver_research/ and scripts/quiver_*_backtest.py): a standalone
backtest of congress buy disclosures showed only a weak, inconsistent effect
(mean/median divergence and sub-50% hit rate at most horizons), so the bar
for a standalone congress/trump signal is intentionally higher than before.
This module is a candidate for a convergence bonus with a stronger family
later, not a primary signal on its own.

Signal filter (any one qualifies):
  - cluster_count >= 3     (multiple officials, same ticker, same side)
  - amount >= $500K + lag <= 14d

Score weights:
  35% cluster size  |  30% freshness (1 − lag/45)  |
  25% alpha gap     |  10% cross-source bonus (congress AND trump)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from statistics import mean
from typing import Optional

import pandas as pd


# ─── unified trade record ─────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    ticker: str
    company: str
    official: str
    source: str             # "congress" | "trump"
    side: str               # "Buy" | "Sell"
    traded: Optional[date]
    filed: Optional[date]
    lag_days: int
    excess_return: Optional[float]   # stock α vs SPY since trade (pct)
    price_change: Optional[float]    # raw stock pct change since trade
    amount_mid_usd: float


@dataclass
class QuiverSignal:
    ticker: str
    company: str
    side: str
    sources: list[str]       # ["congress"] / ["trump"] / both
    officials: list[str]
    cluster_count: int
    avg_lag_days: float
    alpha_vs_spy: Optional[float]
    price_change: Optional[float]
    total_usd: float
    score: float
    suggested_action: str
    rationale: str
    committee_aligned: bool = False  # trader sits on oversight committee for this sector
    committee_note: str = ""         # e.g. "Armed Services — defense contractor oversight"


# ─── parse helpers ───────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return None if (f != f) else f
    except (TypeError, ValueError):
        return None


def _safe_date(v) -> Optional[date]:
    if v is None or str(v) in ("nan", "None", ""):
        return None
    try:
        return pd.to_datetime(str(v), errors="coerce").date()
    except Exception:
        return None


def _parse_amount(v) -> float:
    if v is None:
        return 0.0
    s = str(v).strip()
    try:
        return max(0.0, float(s.replace(",", "").replace("$", "")))
    except ValueError:
        pass
    if "-" in s:
        parts = [p.strip().lstrip("$").replace(",", "") for p in s.split("-")]
        try:
            lo = float(parts[0]) if parts[0] else 0.0
            hi = float(parts[1]) if len(parts) > 1 and parts[1] else lo
            return (lo + hi) / 2.0
        except ValueError:
            pass
    return 0.0


def _classify_side(raw) -> str:
    if raw is None:
        return ""
    s = str(raw).strip().lower()
    if "purchase" in s or s == "buy" or "acquisition" in s:
        return "Buy"
    if "sale" in s or s == "sell" or "disposition" in s:
        return "Sell"
    return ""


# ─── ingest ──────────────────────────────────────────────────────────────────

_STOCK_TYPES = {"st", "stock", ""}   # exclude "op" (options contracts)


def records_from_congress(df: pd.DataFrame) -> list[TradeRecord]:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    records: list[TradeRecord] = []
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "") or "").strip().upper()
        if not ticker or ticker in ("NAN", "NONE"):
            continue
        ticker_type = str(row.get("tickertype", "")).strip().lower()
        if ticker_type not in _STOCK_TYPES:
            continue

        traded = _safe_date(row.get("transactiondate") or row.get("traded"))
        filed = _safe_date(row.get("reportdate") or row.get("filed"))
        lag = (filed - traded).days if traded and filed and filed >= traded else 30

        side = _classify_side(row.get("transaction"))
        if not side:
            continue

        amount_mid = _parse_amount(row.get("amount")) or _parse_amount(row.get("range"))

        _desc = row.get("description")
        _company = str(_desc) if _desc and str(_desc).lower() not in ("nan", "none", "") else ticker
        records.append(TradeRecord(
            ticker=ticker,
            company=_company,
            official=str(row.get("representative") or "").strip(),
            source="congress",
            side=side,
            traded=traded,
            filed=filed,
            lag_days=max(0, lag),
            excess_return=_safe_float(row.get("excessreturn")),
            price_change=_safe_float(row.get("pricechange")),
            amount_mid_usd=amount_mid,
        ))
    return records


def records_from_trump(df: pd.DataFrame) -> list[TradeRecord]:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    records: list[TradeRecord] = []
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "") or "").strip().upper()
        if not ticker or ticker in ("NAN", "NONE"):
            continue

        traded = _safe_date(row.get("traded"))
        filed = _safe_date(row.get("filed"))
        lag = (filed - traded).days if traded and filed and filed >= traded else 30

        side = _classify_side(row.get("transaction"))
        if not side:
            continue

        amount_mid = _parse_amount(row.get("amount")) or _parse_amount(row.get("range"))

        records.append(TradeRecord(
            ticker=ticker,
            company=str(row.get("company") or ticker),
            official="Trump",
            source="trump",
            side=side,
            traded=traded,
            filed=filed,
            lag_days=max(0, lag),
            excess_return=_safe_float(row.get("excessreturn")),
            price_change=_safe_float(row.get("pricechange")),
            amount_mid_usd=amount_mid,
        ))
    return records


# ─── scoring ─────────────────────────────────────────────────────────────────

_MAX_LAG = 45.0


def _score(records: list[TradeRecord], side: str) -> float:
    n_officials = len({r.official for r in records})
    n_sources = len({r.source for r in records})

    cluster_score = min(n_officials / 3.0, 1.0)
    lag_score = max(0.0, 1.0 - mean(r.lag_days for r in records) / _MAX_LAG)
    cross_bonus = 0.10 if n_sources > 1 else 0.0

    alphas = [r.excess_return for r in records if r.excess_return is not None]
    if alphas:
        avg_alpha = mean(alphas)
        if side == "Buy":
            # negative alpha = stock hasn't moved up yet = full opportunity
            alpha_score = max(0.0, min(1.0, (-avg_alpha + 5.0) / 25.0))
        else:
            # positive alpha = stock hasn't fallen yet = full opportunity
            alpha_score = max(0.0, min(1.0, (avg_alpha + 5.0) / 25.0))
    else:
        alpha_score = 0.3

    raw = 0.35 * cluster_score + 0.30 * lag_score + 0.25 * alpha_score + cross_bonus
    return round(min(raw, 1.0), 3)


_PLAYED_OUT_THRESHOLD = 20.0  # α % beyond which the trade is already done

def _is_notable(records: list[TradeRecord], side: str, score: float) -> bool:
    if score < 0.30:
        return False

    # Drop signals where the trade has already strongly played out
    alphas = [r.excess_return for r in records if r.excess_return is not None]
    if alphas:
        avg_alpha = mean(alphas)
        if side == "Buy" and avg_alpha > _PLAYED_OUT_THRESHOLD:
            return False
        if side == "Sell" and avg_alpha < -_PLAYED_OUT_THRESHOLD:
            return False

    n = len({r.official for r in records})
    max_amt = max((r.amount_mid_usd for r in records), default=0)
    avg_lag = mean(r.lag_days for r in records)

    # Demoted per the Quiver Edge Expansion plan (Stage 3): backtesting congress
    # buy disclosures (scripts/quiver_congress_backtest.py) showed only a weak,
    # inconsistent effect (mean/median divergence, sub-50% hit rate at most
    # horizons), and the standalone Trump-any-trade carve-out below was the
    # loosest of the old thresholds. Standalone congress/trump signals now need
    # a materially larger cluster or a materially larger, still-fresh trade —
    # single small disclosures are no longer enough on their own.
    if n >= 3:
        return True
    if max_amt >= 500_000 and avg_lag <= 14:
        return True
    return False


def _suggest(side: str, score: float, alpha: Optional[float]) -> tuple[str, str]:
    if side == "Buy":
        action = "Long call — 45-60 DTE, ATM or 5% OTM"
        if score >= 0.70:
            rationale = "Cluster buy + fresh disclosure + stock hasn't moved. High conviction."
        elif score >= 0.50:
            rationale = "Multiple signals align. Consider a call spread to reduce premium risk."
        else:
            rationale = "Single official. Size small or wait for price confirmation."
    else:
        action = "Long put — 45-60 DTE, ATM or 5% OTM"
        if score >= 0.70:
            rationale = "Cluster sell + fresh + stock still elevated. High conviction."
        elif score >= 0.50:
            rationale = "Multiple signals align. Consider a put spread to finance the position."
        else:
            rationale = "Single official. Size small or wait for downside confirmation."
    return action, rationale


# ─── main entry ──────────────────────────────────────────────────────────────

def build_signals(
    congress_df: Optional[pd.DataFrame],
    trump_df: Optional[pd.DataFrame],
    max_lag_days: int = 30,
    min_score: float = 0.0,
    congress_api_key: Optional[str] = None,
) -> list[QuiverSignal]:
    all_records: list[TradeRecord] = []
    if congress_df is not None and not congress_df.empty:
        all_records.extend(records_from_congress(congress_df))
    if trump_df is not None and not trump_df.empty:
        all_records.extend(records_from_trump(trump_df))

    all_records = [r for r in all_records if r.lag_days <= max_lag_days]
    if not all_records:
        return []

    # Build bioguide_map and pre-warm committee cache for alignment checks.
    bioguide_map: dict[str, str] = {}
    if congress_df is not None and not congress_df.empty:
        rep_col = next((c for c in congress_df.columns if c.lower() == "representative"), None)
        bio_col = next((c for c in congress_df.columns if c.lower() == "bioguideid"), None)
        if rep_col and bio_col:
            for _, row in congress_df.iterrows():
                name = str(row[rep_col]).strip()
                bio = str(row[bio_col]).strip()
                if name and bio and bio.lower() not in ("nan", "none", ""):
                    bioguide_map[name] = bio

    if bioguide_map:
        from lox.quiver.committees import get_member_committees, is_committee_aligned, alignment_note as _alignment_note
        if congress_api_key:
            for bio_id in set(bioguide_map.values()):
                if bio_id:
                    get_member_committees(bio_id, congress_api_key)
    else:
        from lox.quiver.committees import is_committee_aligned, alignment_note as _alignment_note

    groups: dict[tuple[str, str], list[TradeRecord]] = defaultdict(list)
    for r in all_records:
        groups[(r.ticker, r.side)].append(r)

    signals: list[QuiverSignal] = []
    for (ticker, side), records in groups.items():
        score = _score(records, side)
        if not _is_notable(records, side, score):
            continue

        alphas = [r.excess_return for r in records if r.excess_return is not None]
        price_changes = [r.price_change for r in records if r.price_change is not None]
        avg_alpha = round(mean(alphas), 1) if alphas else None
        action, rationale = _suggest(side, score, avg_alpha)

        # Committee alignment: True if ANY congress official is oversight-aligned.
        aligned = False
        matched_committee = ""
        for r in records:
            if r.source == "congress":
                bio = bioguide_map.get(r.official, "")
                is_alg, committee = is_committee_aligned(bio, ticker)
                if is_alg:
                    aligned = True
                    matched_committee = committee
                    break

        signals.append(QuiverSignal(
            ticker=ticker,
            company=records[0].company,
            side=side,
            sources=sorted({r.source for r in records}),
            officials=sorted({r.official for r in records}),
            cluster_count=len({r.official for r in records}),
            avg_lag_days=round(mean(r.lag_days for r in records), 1),
            alpha_vs_spy=avg_alpha,
            price_change=round(mean(price_changes), 1) if price_changes else None,
            total_usd=sum(r.amount_mid_usd for r in records),
            score=score,
            suggested_action=action,
            rationale=rationale,
            committee_aligned=aligned,
            committee_note=_alignment_note(matched_committee) if matched_committee else "",
        ))

    signals.sort(key=lambda s: (s.avg_lag_days, -s.total_usd))
    if min_score > 0:
        signals = [s for s in signals if s.score >= min_score]
    return signals

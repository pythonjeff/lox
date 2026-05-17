"""
Speculation pillar — leveraged ETF AUM + put/call ratio.

Two reads that historically light up before busts:

  1. Levered ETF AUM.  Total assets in 3× long ETFs (TQQQ, SOXL, UPRO, FAS)
     compared to the inverse 3× products (SQQQ, SOXS, SPXU, FAZ).  Bull-bias
     dollars stacked into daily-reset leveraged products is one of the
     cleanest reads on retail/quant-trend speculation: these products bleed
     under volatility and only attract net flows when traders are convinced
     the trend continues.

  2. Equity put/call ratio.  Best-effort SPY-options PCR via yfinance.  Low
     PCR (<0.7) = call-buying froth; very low (<0.55) = euphoric.
     If the options fetch fails this stays None and the classifier degrades.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from lox.altdata.fmp import fetch_profile
from lox.config import Settings


LEVERED_LONGS: tuple[str, ...] = ("TQQQ", "SOXL", "UPRO", "FAS")
LEVERED_SHORTS: tuple[str, ...] = ("SQQQ", "SOXS", "SPXU", "FAZ")


@dataclass
class SpeculationSnapshot:
    asof: str
    levered_long_aum_bn: float | None
    levered_short_aum_bn: float | None
    long_to_short_ratio: float | None              # >5 euphoric, <1 panic, ~2-3 normal
    per_etf_aum_bn: dict[str, float] = field(default_factory=dict)
    put_call_ratio: float | None = None
    pcr_source: str = ""


def _aum_of(settings: Settings, ticker: str) -> float | None:
    """Market cap of a levered ETF ≈ AUM (shares × NAV)."""
    try:
        prof = fetch_profile(settings=settings, ticker=ticker)
    except Exception:
        return None
    if prof is None or prof.market_cap is None:
        return None
    return float(prof.market_cap) / 1_000_000_000.0  # → $ billions


def _best_effort_spy_pcr(settings: Settings) -> tuple[float | None, str]:
    """Light-weight SPY PCR via yfinance options. Returns (pcr, source_label)."""
    try:
        from lox.data.yfinance_options import fetch_options_chain_yfinance
        from lox.positioning.data import compute_pcr_from_chain
        candidates = fetch_options_chain_yfinance("SPY", max_expiries=4)
        if not candidates:
            return None, ""
        pcr = compute_pcr_from_chain(candidates)
        if pcr is None:
            return None, ""
        return float(pcr), "yfinance SPY chain (OI-weighted)"
    except Exception:
        return None, ""


def fetch_speculation_snapshot(*, settings: Settings, refresh: bool = False) -> SpeculationSnapshot:
    long_aums: dict[str, float] = {}
    short_aums: dict[str, float] = {}

    for t in LEVERED_LONGS:
        v = _aum_of(settings, t)
        if v is not None:
            long_aums[t] = v
    for t in LEVERED_SHORTS:
        v = _aum_of(settings, t)
        if v is not None:
            short_aums[t] = v

    long_total = sum(long_aums.values()) if long_aums else None
    short_total = sum(short_aums.values()) if short_aums else None

    ratio: float | None = None
    if long_total is not None and short_total is not None and short_total > 0:
        ratio = float(long_total / short_total)

    pcr, pcr_src = _best_effort_spy_pcr(settings)

    return SpeculationSnapshot(
        asof=datetime.utcnow().strftime("%Y-%m-%d"),
        levered_long_aum_bn=long_total,
        levered_short_aum_bn=short_total,
        long_to_short_ratio=ratio,
        per_etf_aum_bn={**long_aums, **short_aums},
        put_call_ratio=pcr,
        pcr_source=pcr_src,
    )

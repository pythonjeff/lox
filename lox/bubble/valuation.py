"""
Valuation pillar — Buffett indicator (market cap of corporate equities / GDP).

The original 2001 Fortune-article definition uses the market value of all US
corporate equities (Z.1 Flow of Funds table L.223) divided by GDP.  On FRED
that's `NCBEILQ027S` ("Nonfinancial Corporate Business; Corporate Equities;
Liability, Level"), quarterly, $ millions.  GDP is `GDP`, quarterly, $ billions.

The legacy Wilshire 5000 / GDP variant is unavailable — all `WILL5000*` series
return 400s on FRED — so we use the Z.1 series directly.  Same cadence (both
quarterly) means a clean alignment without forward-fill.

Returned dataclass also includes the historical percentile so the regime
classifier can compare today's reading against the full series rather than
hard-coded thresholds that go stale.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from lox.config import Settings
from lox.data.fred import FredClient


# Z.1 corporate-equities market value (Buffett's 2001 definition), $ millions, quarterly.
EQUITY_MARKETCAP_SERIES_ID = "NCBEILQ027S"


@dataclass
class BuffettSnapshot:
    asof: str
    market_cap_bn: float | None         # corporate equities market value, $ billions
    gdp_bn: float | None                # GDP (latest quarter), $ billions
    ratio_pct: float | None             # 100 * mkt cap / GDP
    percentile_full: float | None       # percentile over the entire available history
    percentile_10y: float | None        # percentile over the trailing 10 years
    series_id: str = EQUITY_MARKETCAP_SERIES_ID


def _empty_snapshot() -> BuffettSnapshot:
    return BuffettSnapshot(asof="", market_cap_bn=None, gdp_bn=None,
                           ratio_pct=None, percentile_full=None, percentile_10y=None)


def fetch_buffett_snapshot(*, settings: Settings, refresh: bool = False) -> BuffettSnapshot:
    """Pull corporate-equities market cap + GDP from FRED and return the snapshot."""
    fred = FredClient(api_key=settings.FRED_API_KEY)

    try:
        mkt = fred.fetch_series(EQUITY_MARKETCAP_SERIES_ID,
                                start_date="1990-01-01", refresh=refresh)
    except Exception:
        mkt = pd.DataFrame()

    try:
        gdp = fred.fetch_series("GDP", start_date="1990-01-01", refresh=refresh)
    except Exception:
        gdp = pd.DataFrame()

    if mkt.empty or gdp.empty:
        return _empty_snapshot()

    # Z.1 is $millions, GDP is $billions — convert to a common $billions basis.
    m = mkt.set_index("date")["value"].sort_index() / 1_000.0
    g = gdp.set_index("date")["value"].sort_index()

    aligned = pd.DataFrame({"mkt": m, "gdp": g}).dropna()
    if aligned.empty:
        return _empty_snapshot()

    aligned["ratio"] = 100.0 * aligned["mkt"] / aligned["gdp"]

    last = aligned.iloc[-1]
    ratio = float(last["ratio"])

    pct_full = float((aligned["ratio"] <= ratio).mean() * 100.0)

    cutoff = aligned.index[-1] - pd.Timedelta(days=365 * 10)
    win10 = aligned.loc[aligned.index >= cutoff, "ratio"]
    pct_10 = float((win10 <= ratio).mean() * 100.0) if not win10.empty else None

    return BuffettSnapshot(
        asof=str(aligned.index[-1].date()),
        market_cap_bn=float(last["mkt"]),
        gdp_bn=float(last["gdp"]),
        ratio_pct=ratio,
        percentile_full=pct_full,
        percentile_10y=pct_10,
    )

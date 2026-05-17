"""
Margin debt pillar — borrowing to buy stocks.

FINRA publishes monthly margin statistics, but the data isn't freely available
in a stable form: their public API endpoint at api.finra.org requires
authentication, and they don't host a downloadable CSV at a fixed URL.  The
closest free proxy is the Z.1 Flow of Funds series `BOGZ1FL663067003Q`
("Security brokers and dealers; security credit; asset") on FRED, which is
quarterly and tracks the same underlying behaviour: dealer-extended credit to
investors collateralized by securities.

We compute level, YoY growth, and historical percentile.  We also flag a
rollover-from-peak: margin debt typically peaks ahead of equity peaks, so the
combination of "high level + rolling over" is a classic late-cycle signal.

To upgrade to true monthly FINRA cadence later, plug a FINRA API key (or a
paid feed like Nasdaq Data Link) into a new fetcher and update
`fetch_margin_snapshot` to prefer it.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from lox.config import Settings
from lox.data.fred import FredClient


MARGIN_SERIES_ID = "BOGZ1FL663067003Q"  # Z.1 brokers/dealers security credit, $millions, quarterly


@dataclass
class MarginSnapshot:
    asof: str
    series_id: str
    level_bn: float | None             # latest, $ billions (nominal)
    yoy_pct: float | None              # YoY change (level)
    percentile_full: float | None      # historical percentile of level
    percentile_yoy_full: float | None  # historical percentile of YoY (froth check)
    rolling_over: bool                 # True if level < 6m-ago AND was in top-quartile recently
    # Inflation/GDP-normalized series — the right way to compare across decades.
    pct_of_gdp: float | None = None
    pct_of_gdp_percentile_full: float | None = None


def _percentile(series: pd.Series, value: float) -> float | None:
    if series is None or series.empty or value is None:
        return None
    return float((series <= value).mean() * 100.0)


def fetch_margin_snapshot(*, settings: Settings, refresh: bool = False) -> MarginSnapshot:
    fred = FredClient(api_key=settings.FRED_API_KEY)

    try:
        df = fred.fetch_series(MARGIN_SERIES_ID, start_date="1990-01-01", refresh=refresh)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        return MarginSnapshot(asof="", series_id=MARGIN_SERIES_ID, level_bn=None,
                              yoy_pct=None, percentile_full=None,
                              percentile_yoy_full=None, rolling_over=False)

    s = df.set_index("date")["value"].sort_index()
    # Z.1 is in $millions — convert to $bn for display.
    s = s / 1_000.0

    yoy = s.pct_change(4) * 100.0  # quarterly series → YoY = 4-step

    last_val = float(s.iloc[-1])
    last_yoy = float(yoy.iloc[-1]) if pd.notna(yoy.iloc[-1]) else None

    pct_full = _percentile(s, last_val)
    pct_yoy = _percentile(yoy.dropna(), last_yoy) if last_yoy is not None else None

    # ── Margin debt as % of GDP (inflation/growth-normalized) ──────────────
    pct_of_gdp: float | None = None
    pct_of_gdp_pctl: float | None = None
    try:
        gdp_df = fred.fetch_series("GDP", start_date="1990-01-01", refresh=refresh)
        if not gdp_df.empty:
            g = gdp_df.set_index("date")["value"].sort_index()
            # Align quarterly margin level to quarterly GDP on date.
            aligned = pd.DataFrame({"margin": s, "gdp": g}).dropna()
            if not aligned.empty:
                ratio = 100.0 * aligned["margin"] / aligned["gdp"]
                pct_of_gdp = float(ratio.iloc[-1])
                pct_of_gdp_pctl = float((ratio <= pct_of_gdp).mean() * 100.0)
    except Exception:
        pass

    # Rolling-over: latest reading is meaningfully below value of ~2q ago AND
    # the level was in the top quartile within the trailing 2 years.
    rolling_over = False
    if len(s) >= 3:
        recent_peak = s.iloc[-8:].max() if len(s) >= 8 else s.max()
        prev = float(s.iloc[-3])  # ~6 months ago given quarterly cadence
        # 2y top-quartile threshold
        win = s.iloc[-8:] if len(s) >= 8 else s
        in_top_quartile = last_val >= win.quantile(0.75) or recent_peak == last_val
        rolling_over = bool(last_val < prev * 0.97 and not in_top_quartile and
                            recent_peak > last_val * 1.05)

    return MarginSnapshot(
        asof=str(s.index[-1].date()),
        series_id=MARGIN_SERIES_ID,
        level_bn=last_val,
        yoy_pct=last_yoy,
        percentile_full=pct_full,
        percentile_yoy_full=pct_yoy,
        rolling_over=rolling_over,
        pct_of_gdp=pct_of_gdp,
        pct_of_gdp_percentile_full=pct_of_gdp_pctl,
    )

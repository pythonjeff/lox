"""
Policy / geopolitical uncertainty data layer.

Fetches and computes raw inputs for the policy regime classifier.
No classification logic lives here — just data retrieval and normalization.

Data sources (all existing APIs, zero new keys):
  - FRED USEPUINDXD: Economic Policy Uncertainty Index (daily)
  - FRED IR: Import Price Index (monthly, for tariff pass-through)
  - FRED VIXCLS / DTWEXBGS: VIX and DXY for amplifier cross-signals
  - Alpaca + FMP news: keyword-filtered policy article counts

Author: Lox Capital Research
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from lox.config import Settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Policy news keywords
# ─────────────────────────────────────────────────────────────────────────────

POLICY_NEWS_KEYWORDS: list[str] = [
    "tariff",
    "sanction",
    "trade war",
    "executive order",
    "trade policy",
    "import duty",
    "export ban",
    "trade deal",
    "trade agreement",
    "embargo",
    "retaliatory tariff",
    "section 301",
    "section 232",
    "trade deficit",
    "trade representative",
    "geopolitical",
    "trade restriction",
    "trade tension",
    "customs duty",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PolicyInputs:
    """Raw inputs for the policy regime classifier."""

    # Layer 1: Base signals
    epu_level: float | None = None               # USEPUINDXD latest value
    epu_1y_percentile: float | None = None        # Current EPU percentile in trailing 252 days
    epu_30d_change: float | None = None           # 30-day absolute change
    epu_series_90d: tuple[float, ...] | None = None  # Last 90 daily values for sparkline

    news_article_count_7d: int | None = None      # Policy keyword articles in last 7 days
    news_article_count_30d: int | None = None      # Policy keyword articles in last 30 days
    news_top_headlines: tuple[dict, ...] | None = None  # Top 5 headlines

    import_price_yoy: float | None = None          # Import Price Index YoY %
    import_price_mom_accel: float | None = None    # MoM acceleration (2nd derivative)

    # Layer 2: Amplifier inputs
    vix_level: float | None = None
    dxy_level: float | None = None
    dxy_20d_chg: float | None = None

    # Metadata
    asof: str = ""
    error: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_pct_change(series, periods: int) -> float | None:
    """Compute percent change over N periods, returning None if insufficient data."""
    if series is None or len(series) < periods + 1:
        return None
    try:
        old = float(series.iloc[-(periods + 1)])
        new = float(series.iloc[-1])
        if old == 0:
            return None
        return (new - old) / abs(old) * 100
    except Exception:
        return None


def _percentile_rank(series, value: float) -> float:
    """Percentile rank (0-100) of value within a pandas Series."""
    try:
        return float((series < value).sum() / len(series) * 100)
    except Exception:
        return 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Main data fetch
# ─────────────────────────────────────────────────────────────────────────────

def compute_policy_inputs(
    *,
    settings: Settings,
    start_date: str = "2020-01-01",
    refresh: bool = False,
) -> PolicyInputs:
    """Fetch all policy regime inputs from FRED + News APIs.

    Returns a PolicyInputs dataclass with all raw data for the classifier.
    Gracefully handles missing data — classifier will skip sub-scores with None.
    """
    import pandas as pd
    from lox.data.fred import FredClient

    fred_key = settings.fred_api_key
    if not fred_key:
        return PolicyInputs(error="No FRED API key configured", asof=_now_str())

    fred = FredClient(api_key=fred_key)
    asof = _now_str()

    # ── EPU Index (USEPUINDXD) ───────────────────────────────────────────
    epu_level: float | None = None
    epu_1y_pctl: float | None = None
    epu_30d_chg: float | None = None
    epu_series_90d: tuple[float, ...] | None = None

    try:
        epu_df = fred.fetch_series("USEPUINDXD", start_date=start_date, refresh=refresh)
        if epu_df is not None and not epu_df.empty:
            epu_df = epu_df.sort_values("date").reset_index(drop=True)
            epu_level = float(epu_df["value"].iloc[-1])

            # 1Y percentile (trailing 252 business days)
            window = min(252, len(epu_df))
            trailing = epu_df["value"].iloc[-window:]
            epu_1y_pctl = _percentile_rank(trailing, epu_level)

            # 30-day change (22 business days)
            if len(epu_df) > 22:
                old_val = float(epu_df["value"].iloc[-23])
                epu_30d_chg = epu_level - old_val

            # 90-day sparkline series
            spark_window = min(90, len(epu_df))
            epu_series_90d = tuple(epu_df["value"].iloc[-spark_window:].tolist())
    except Exception as e:
        logger.warning(f"Failed to fetch EPU index: {e}")

    # ── Import Price Index (IR) ──────────────────────────────────────────
    import_price_yoy: float | None = None
    import_price_mom_accel: float | None = None

    try:
        ir_df = fred.fetch_series("IR", start_date="2018-01-01", refresh=refresh)
        if ir_df is not None and not ir_df.empty:
            ir_df = ir_df.sort_values("date").reset_index(drop=True)

            # YoY: compare latest to 12 months ago (≈12 observations for monthly)
            if len(ir_df) >= 13:
                latest = float(ir_df["value"].iloc[-1])
                yr_ago = float(ir_df["value"].iloc[-13])
                if yr_ago != 0:
                    import_price_yoy = (latest / yr_ago - 1) * 100

            # MoM acceleration: (latest MoM) - (3-month-ago MoM)
            if len(ir_df) >= 5:
                vals = ir_df["value"].iloc[-5:].values
                mom_latest = (float(vals[-1]) / float(vals[-2]) - 1) * 100 if vals[-2] != 0 else 0
                mom_3m_ago = (float(vals[-3]) / float(vals[-4]) - 1) * 100 if vals[-4] != 0 else 0
                import_price_mom_accel = mom_latest - mom_3m_ago
    except Exception as e:
        logger.warning(f"Failed to fetch Import Price Index: {e}")

    # ── VIX (VIXCLS) ────────────────────────────────────────────────────
    vix_level: float | None = None
    try:
        vix_df = fred.fetch_series("VIXCLS", start_date="2024-01-01", refresh=refresh)
        if vix_df is not None and not vix_df.empty:
            vix_level = float(vix_df.sort_values("date")["value"].iloc[-1])
    except Exception as e:
        logger.warning(f"Failed to fetch VIX: {e}")

    # ── DXY (DTWEXBGS) ──────────────────────────────────────────────────
    dxy_level: float | None = None
    dxy_20d_chg: float | None = None
    try:
        dxy_df = fred.fetch_series("DTWEXBGS", start_date="2024-01-01", refresh=refresh)
        if dxy_df is not None and not dxy_df.empty:
            dxy_df = dxy_df.sort_values("date").reset_index(drop=True)
            dxy_level = float(dxy_df["value"].iloc[-1])
            # 20-day % change
            if len(dxy_df) > 20:
                old_dxy = float(dxy_df["value"].iloc[-21])
                if old_dxy != 0:
                    dxy_20d_chg = (dxy_level / old_dxy - 1) * 100
    except Exception as e:
        logger.warning(f"Failed to fetch DXY: {e}")

    # ── Policy News ──────────────────────────────────────────────────────
    # fetch_unified_news uses keywords only for relevance boosting, not
    # filtering.  We need hard filtering: only articles whose title or
    # snippet actually contains at least one policy keyword.
    news_7d: int | None = None
    news_30d: int | None = None
    top_headlines: tuple[dict, ...] | None = None

    try:
        from lox.altdata.news import fetch_unified_news

        raw_articles = fetch_unified_news(
            settings=settings,
            keywords=POLICY_NEWS_KEYWORDS,
            lookback_days=30,
            limit=100,  # fetch more so we have enough after filtering
            include_content=False,  # only need titles + metadata
        )

        # Hard-filter: article must contain at least one policy keyword.
        # Use word-boundary matching to avoid false positives like
        # "industries" matching "USTR" or "transaction" matching "sanction".
        import re
        kw_patterns = [
            re.compile(r"\b" + re.escape(kw.lower()) + r"s?\b")
            for kw in POLICY_NEWS_KEYWORDS
        ]
        articles = []
        for a in raw_articles:
            text = f"{a.title} {a.snippet or ''}".lower()
            if any(pat.search(text) for pat in kw_patterns):
                articles.append(a)

        if articles:
            now = datetime.now(timezone.utc)
            cutoff_7d = now - timedelta(days=7)

            # Count by recency
            count_7d = 0
            count_30d = len(articles)
            for a in articles:
                try:
                    pub = pd.to_datetime(a.published_at)
                    if pub.tzinfo is None:
                        pub = pub.tz_localize("UTC")
                    if pub >= cutoff_7d:
                        count_7d += 1
                except Exception:
                    pass

            news_7d = count_7d
            news_30d = count_30d

            # Top 5 headlines (most recent, sorted by date desc)
            top_5 = articles[:5]
            top_headlines = tuple(
                {
                    "title": a.title[:120],
                    "source": a.source or a.provider,
                    "date": a.published_at[:10] if a.published_at else "",
                }
                for a in top_5
            )
    except Exception as e:
        logger.warning(f"Failed to fetch policy news: {e}")

    return PolicyInputs(
        epu_level=epu_level,
        epu_1y_percentile=epu_1y_pctl,
        epu_30d_change=epu_30d_chg,
        epu_series_90d=epu_series_90d,
        news_article_count_7d=news_7d,
        news_article_count_30d=news_30d,
        news_top_headlines=top_headlines,
        import_price_yoy=import_price_yoy,
        import_price_mom_accel=import_price_mom_accel,
        vix_level=vix_level,
        dxy_level=dxy_level,
        dxy_20d_chg=dxy_20d_chg,
        asof=asof,
    )


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

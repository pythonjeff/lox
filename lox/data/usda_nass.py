"""
USDA NASS QuickStats API client.

Fetches crop reports from the National Agricultural Statistics Service:
  - Prospective Plantings (March 31) — intended acreage
  - Crop Progress — weekly % planted / % emerged / % harvested
  - Crop Condition — weekly % good/excellent ratings
  - Grain Stocks — quarterly actual inventories

Free API key: https://quickstats.nass.usda.gov/api
Set USDA_NASS_API_KEY in .env to enable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

NASS_API_BASE = "https://quickstats.nass.usda.gov/api/api_GET/"

# 5-year average planted acres (million acres) for context
HIST_PLANTED_ACRES: dict[str, float] = {
    "CORN": 90.8,
    "SOYBEANS": 87.5,
    "WHEAT": 47.5,
}


@dataclass(frozen=True)
class PlantingData:
    commodity: str
    year: int
    planted_acres_m: float | None = None   # million acres
    prior_year_acres_m: float | None = None
    five_yr_avg_acres_m: float | None = None
    yoy_change_pct: float | None = None


@dataclass(frozen=True)
class CropProgress:
    commodity: str
    year: int
    week_ending: str = ""
    pct_planted: float | None = None
    pct_emerged: float | None = None
    pct_harvested: float | None = None
    prior_year_pct_planted: float | None = None
    five_yr_avg_pct_planted: float | None = None
    pct_silking: float | None = None
    prior_year_pct_emerged: float | None = None
    five_yr_avg_pct_emerged: float | None = None


@dataclass(frozen=True)
class CropCondition:
    commodity: str
    year: int
    week_ending: str = ""
    pct_good: float | None = None
    pct_excellent: float | None = None
    pct_good_excellent: float | None = None
    prior_year_ge: float | None = None
    five_yr_avg_ge: float | None = None


@dataclass
class CropReportData:
    """Aggregated NASS crop report data for all tracked commodities."""
    plantings: dict[str, PlantingData] = field(default_factory=dict)
    progress: dict[str, CropProgress] = field(default_factory=dict)
    condition: dict[str, CropCondition] = field(default_factory=dict)
    asof: str = ""


def _nass_query(
    api_key: str,
    params: dict[str, str],
    cache_hours: int = 12,
) -> list[dict[str, Any]]:
    """Execute a NASS QuickStats API query with caching.

    The NASS API returns HTTP 400 when no records match the query (e.g. data
    not yet published for the requested year).  We treat that as an empty
    result and cache it briefly so we don't hammer the endpoint.
    """
    from lox.altdata.cache import cache_path, read_cache, write_cache
    import hashlib
    import requests

    param_hash = hashlib.md5(str(sorted(params.items())).encode()).hexdigest()[:12]
    cache_key = f"usda_nass_{param_hash}"
    p = cache_path(cache_key)
    cached = read_cache(p, max_age=timedelta(hours=cache_hours))
    if isinstance(cached, list):
        return cached

    query_params = {**params, "key": api_key, "format": "JSON"}

    try:
        resp = requests.get(NASS_API_BASE, params=query_params, timeout=30)

        if resp.status_code == 400:
            # 400 = "bad request" from NASS means no matching records.
            # Cache the empty result so we don't retry for a while.
            write_cache(p, [])
            logger.debug(
                "NASS: no data for %s/%s/%s",
                params.get("commodity_desc", "?"),
                params.get("statisticcat_desc", "?"),
                params.get("year", "?"),
            )
            return []

        resp.raise_for_status()
        data = resp.json()
        rows = data.get("data", [])
        if isinstance(rows, list):
            write_cache(p, rows)
            return rows
    except requests.exceptions.HTTPError as e:
        logger.warning("NASS HTTP error: %s", e)
    except Exception as e:
        logger.warning("NASS query error: %s", e)
    return []


def _parse_nass_value(val: str | None) -> float | None:
    if not val or val in ("", " ", "(D)", "(Z)", "(NA)"):
        return None
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _planting_query(api_key: str, commodity: str, year: int, cache_hours: int = 24) -> float | None:
    """Fetch planted acreage for a single commodity+year, returning million acres or None."""
    rows = _nass_query(api_key, {
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "commodity_desc": commodity.upper(),
        "statisticcat_desc": "AREA PLANTED",
        "prodn_practice_desc": "ALL PRODUCTION PRACTICES",
        "agg_level_desc": "NATIONAL",
        "year": str(year),
        "reference_period_desc": "YEAR",
    }, cache_hours=cache_hours)
    for r in rows:
        val = _parse_nass_value(r.get("Value"))
        if val is not None:
            return val / 1_000_000
    return None


def fetch_prospective_plantings(
    api_key: str,
    commodity: str,
    year: int | None = None,
) -> PlantingData | None:
    """Fetch Prospective Plantings for a commodity.

    If current-year data isn't available yet (report hasn't been released),
    falls back to showing last two years so we at least have acreage context.
    """
    if year is None:
        year = datetime.now(timezone.utc).year

    current_acres = _planting_query(api_key, commodity, year, cache_hours=12)
    prior_acres = _planting_query(api_key, commodity, year - 1, cache_hours=168)

    # If current year not published, try to show prior vs year-before-prior
    if current_acres is None and prior_acres is not None:
        two_yr_ago = _planting_query(api_key, commodity, year - 2, cache_hours=168)
        yoy = None
        if two_yr_ago is not None and two_yr_ago > 0:
            yoy = ((prior_acres - two_yr_ago) / two_yr_ago) * 100
        return PlantingData(
            commodity=commodity.upper(),
            year=year - 1,
            planted_acres_m=prior_acres,
            prior_year_acres_m=two_yr_ago,
            five_yr_avg_acres_m=HIST_PLANTED_ACRES.get(commodity.upper()),
            yoy_change_pct=yoy,
        )

    if current_acres is None and prior_acres is None:
        return None

    yoy = None
    if current_acres is not None and prior_acres is not None and prior_acres > 0:
        yoy = ((current_acres - prior_acres) / prior_acres) * 100

    return PlantingData(
        commodity=commodity.upper(),
        year=year,
        planted_acres_m=current_acres,
        prior_year_acres_m=prior_acres,
        five_yr_avg_acres_m=HIST_PLANTED_ACRES.get(commodity.upper()),
        yoy_change_pct=yoy,
    )


def fetch_crop_progress(
    api_key: str,
    commodity: str,
    year: int | None = None,
) -> CropProgress | None:
    """Fetch latest weekly Crop Progress report.

    If the current year hasn't started reporting yet (progress reports begin
    in April), returns None — the CLI will note that progress tracking begins
    soon via the report calendar.
    """
    if year is None:
        year = datetime.now(timezone.utc).year

    rows = _nass_query(api_key, {
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "commodity_desc": commodity.upper(),
        "statisticcat_desc": "PROGRESS",
        "unit_desc": "PCT PLANTED",
        "agg_level_desc": "NATIONAL",
        "year": str(year),
    }, cache_hours=12)

    if not rows:
        return None

    rows.sort(key=lambda r: str(r.get("week_ending") or r.get("end_code") or ""), reverse=True)
    latest = rows[0]
    pct_planted = _parse_nass_value(latest.get("Value"))
    week_ending = str(latest.get("week_ending") or "")

    # Prior year same-ish week for comparison
    prior_pct = None
    prior_rows = _nass_query(api_key, {
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "commodity_desc": commodity.upper(),
        "statisticcat_desc": "PROGRESS",
        "unit_desc": "PCT PLANTED",
        "agg_level_desc": "NATIONAL",
        "year": str(year - 1),
    }, cache_hours=168)

    if prior_rows and week_ending:
        try:
            target_dt = datetime.strptime(week_ending, "%Y-%m-%d")
            target_doy = target_dt.timetuple().tm_yday
            best_diff = 999
            for r in prior_rows:
                we = str(r.get("week_ending") or "")
                if not we:
                    continue
                try:
                    pdoy = datetime.strptime(we, "%Y-%m-%d").timetuple().tm_yday
                    diff = abs(pdoy - target_doy)
                    if diff < best_diff:
                        best_diff = diff
                        prior_pct = _parse_nass_value(r.get("Value"))
                except ValueError:
                    pass
        except ValueError:
            pass

    # ── 5-year average % planted for the same week ────────────────────
    five_yr_avg_planted = None
    if week_ending:
        try:
            target_dt = datetime.strptime(week_ending, "%Y-%m-%d")
            target_doy = target_dt.timetuple().tm_yday
            yearly_vals = []
            for hist_yr in range(year - 5, year):
                hist_rows = _nass_query(api_key, {
                    "source_desc": "SURVEY",
                    "sector_desc": "CROPS",
                    "commodity_desc": commodity.upper(),
                    "statisticcat_desc": "PROGRESS",
                    "unit_desc": "PCT PLANTED",
                    "agg_level_desc": "NATIONAL",
                    "year": str(hist_yr),
                }, cache_hours=168)
                best_val, best_diff = None, 999
                for r in hist_rows:
                    we = str(r.get("week_ending") or "")
                    if not we:
                        continue
                    try:
                        hdoy = datetime.strptime(we, "%Y-%m-%d").timetuple().tm_yday
                        diff = abs(hdoy - target_doy)
                        if diff < best_diff:
                            best_diff = diff
                            best_val = _parse_nass_value(r.get("Value"))
                    except ValueError:
                        pass
                if best_val is not None and best_diff <= 10:
                    yearly_vals.append(best_val)
            if yearly_vals:
                five_yr_avg_planted = sum(yearly_vals) / len(yearly_vals)
        except ValueError:
            pass

    # ── % emerged ───────────────────────────────────────────────────
    pct_emerged = None
    emerged_rows = _nass_query(api_key, {
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "commodity_desc": commodity.upper(),
        "statisticcat_desc": "PROGRESS",
        "unit_desc": "PCT EMERGED",
        "agg_level_desc": "NATIONAL",
        "year": str(year),
    }, cache_hours=12)
    if emerged_rows:
        emerged_rows.sort(key=lambda r: str(r.get("week_ending") or ""), reverse=True)
        pct_emerged = _parse_nass_value(emerged_rows[0].get("Value"))

    # Prior-year % emerged for same week
    prior_pct_emerged = None
    five_yr_avg_emerged = None
    if pct_emerged is not None and week_ending:
        try:
            target_dt = datetime.strptime(week_ending, "%Y-%m-%d")
            target_doy = target_dt.timetuple().tm_yday
            yearly_emerged = []
            for hist_yr in range(year - 5, year):
                hist_rows = _nass_query(api_key, {
                    "source_desc": "SURVEY",
                    "sector_desc": "CROPS",
                    "commodity_desc": commodity.upper(),
                    "statisticcat_desc": "PROGRESS",
                    "unit_desc": "PCT EMERGED",
                    "agg_level_desc": "NATIONAL",
                    "year": str(hist_yr),
                }, cache_hours=168)
                best_val, best_diff = None, 999
                for r in hist_rows:
                    we = str(r.get("week_ending") or "")
                    if not we:
                        continue
                    try:
                        hdoy = datetime.strptime(we, "%Y-%m-%d").timetuple().tm_yday
                        diff = abs(hdoy - target_doy)
                        if diff < best_diff:
                            best_diff = diff
                            best_val = _parse_nass_value(r.get("Value"))
                    except ValueError:
                        pass
                if best_val is not None and best_diff <= 10:
                    yearly_emerged.append(best_val)
                    if hist_yr == year - 1:
                        prior_pct_emerged = best_val
            if yearly_emerged:
                five_yr_avg_emerged = sum(yearly_emerged) / len(yearly_emerged)
        except ValueError:
            pass

    # ── % silking (corn only, mid-summer) ───────────────────────────
    pct_silking = None
    if commodity.upper() == "CORN":
        silk_rows = _nass_query(api_key, {
            "source_desc": "SURVEY",
            "sector_desc": "CROPS",
            "commodity_desc": "CORN",
            "statisticcat_desc": "PROGRESS",
            "unit_desc": "PCT SILKING",
            "agg_level_desc": "NATIONAL",
            "year": str(year),
        }, cache_hours=12)
        if silk_rows:
            silk_rows.sort(key=lambda r: str(r.get("week_ending") or ""), reverse=True)
            pct_silking = _parse_nass_value(silk_rows[0].get("Value"))

    return CropProgress(
        commodity=commodity.upper(),
        year=year,
        week_ending=week_ending,
        pct_planted=pct_planted,
        pct_emerged=pct_emerged,
        prior_year_pct_planted=prior_pct,
        five_yr_avg_pct_planted=five_yr_avg_planted,
        pct_silking=pct_silking,
        prior_year_pct_emerged=prior_pct_emerged,
        five_yr_avg_pct_emerged=five_yr_avg_emerged,
    )


def fetch_crop_condition(
    api_key: str,
    commodity: str,
    year: int | None = None,
) -> CropCondition | None:
    """Fetch latest weekly Crop Condition report (% good + % excellent).

    Condition reports typically start in late May/early June.  Returns None
    if no data is available for the requested year.
    """
    if year is None:
        year = datetime.now(timezone.utc).year

    rows_good = _nass_query(api_key, {
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "commodity_desc": commodity.upper(),
        "statisticcat_desc": "CONDITION",
        "unit_desc": "PCT GOOD",
        "agg_level_desc": "NATIONAL",
        "year": str(year),
    }, cache_hours=12)

    rows_exc = _nass_query(api_key, {
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "commodity_desc": commodity.upper(),
        "statisticcat_desc": "CONDITION",
        "unit_desc": "PCT EXCELLENT",
        "agg_level_desc": "NATIONAL",
        "year": str(year),
    }, cache_hours=12)

    if not rows_good and not rows_exc:
        return None

    def _latest(rows: list) -> tuple[float | None, str]:
        if not rows:
            return None, ""
        rows.sort(key=lambda r: str(r.get("week_ending") or ""), reverse=True)
        return _parse_nass_value(rows[0].get("Value")), str(rows[0].get("week_ending") or "")

    pct_good, we1 = _latest(rows_good)
    pct_exc, we2 = _latest(rows_exc)
    week_ending = we1 or we2

    ge = None
    if pct_good is not None and pct_exc is not None:
        ge = pct_good + pct_exc
    elif pct_good is not None:
        ge = pct_good

    # ── Prior-year and 5yr avg G/E for same week ────────────────────
    prior_year_ge = None
    five_yr_avg_ge = None
    if ge is not None and week_ending:
        try:
            target_dt = datetime.strptime(week_ending, "%Y-%m-%d")
            target_doy = target_dt.timetuple().tm_yday
            yearly_ge = []
            for hist_yr in range(year - 5, year):
                hg_rows = _nass_query(api_key, {
                    "source_desc": "SURVEY",
                    "sector_desc": "CROPS",
                    "commodity_desc": commodity.upper(),
                    "statisticcat_desc": "CONDITION",
                    "unit_desc": "PCT GOOD",
                    "agg_level_desc": "NATIONAL",
                    "year": str(hist_yr),
                }, cache_hours=168)
                he_rows = _nass_query(api_key, {
                    "source_desc": "SURVEY",
                    "sector_desc": "CROPS",
                    "commodity_desc": commodity.upper(),
                    "statisticcat_desc": "CONDITION",
                    "unit_desc": "PCT EXCELLENT",
                    "agg_level_desc": "NATIONAL",
                    "year": str(hist_yr),
                }, cache_hours=168)

                def _find_closest(rows, doy):
                    best_val, best_diff = None, 999
                    for r in rows:
                        we = str(r.get("week_ending") or "")
                        if not we:
                            continue
                        try:
                            hdoy = datetime.strptime(we, "%Y-%m-%d").timetuple().tm_yday
                            diff = abs(hdoy - doy)
                            if diff < best_diff:
                                best_diff = diff
                                best_val = _parse_nass_value(r.get("Value"))
                        except ValueError:
                            pass
                    return best_val if best_diff <= 10 else None

                hg = _find_closest(hg_rows, target_doy)
                he = _find_closest(he_rows, target_doy)
                if hg is not None:
                    hist_ge = hg + (he or 0)
                    yearly_ge.append(hist_ge)
                    if hist_yr == year - 1:
                        prior_year_ge = hist_ge
            if yearly_ge:
                five_yr_avg_ge = sum(yearly_ge) / len(yearly_ge)
        except ValueError:
            pass

    return CropCondition(
        commodity=commodity.upper(),
        year=year,
        week_ending=week_ending,
        pct_good=pct_good,
        pct_excellent=pct_exc,
        pct_good_excellent=ge,
        prior_year_ge=prior_year_ge,
        five_yr_avg_ge=five_yr_avg_ge,
    )


def fetch_all_crop_reports(
    api_key: str,
    commodities: list[str] | None = None,
) -> CropReportData:
    """Fetch all available NASS crop reports for tracked commodities."""
    if not api_key:
        return CropReportData()

    if commodities is None:
        commodities = ["CORN", "SOYBEANS", "WHEAT"]

    result = CropReportData(asof=datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    for comm in commodities:
        try:
            pl = fetch_prospective_plantings(api_key, comm)
            if pl is not None:
                result.plantings[comm] = pl
        except Exception as e:
            logger.debug("Plantings fetch failed for %s: %s", comm, e)

        try:
            prog = fetch_crop_progress(api_key, comm)
            if prog is not None:
                result.progress[comm] = prog
        except Exception as e:
            logger.debug("Progress fetch failed for %s: %s", comm, e)

        try:
            cond = fetch_crop_condition(api_key, comm)
            if cond is not None:
                result.condition[comm] = cond
        except Exception as e:
            logger.debug("Condition fetch failed for %s: %s", comm, e)

    return result


# ── USDA Report Calendar ────────────────────────────────────────────────────

@dataclass(frozen=True)
class USDAReport:
    name: str
    date: date
    description: str
    impact: str  # "high" | "medium" | "low"


def upcoming_usda_reports(days_ahead: int = 30) -> list[USDAReport]:
    """Return upcoming USDA agriculture reports within the next N days."""
    today = date.today()
    end = today + timedelta(days=days_ahead)
    year = today.year

    reports: list[USDAReport] = []

    # Prospective Plantings — always March 31
    pp_date = date(year, 3, 31)
    if pp_date.weekday() >= 5:
        pp_date -= timedelta(days=pp_date.weekday() - 4)
    reports.append(USDAReport("Prospective Plantings", pp_date, "Intended planted acreage for corn, soy, wheat", "high"))

    # Grain Stocks — March 31, June 30, Sep 30
    for m, d in [(3, 31), (6, 30), (9, 30)]:
        gs = date(year, m, d)
        if gs.weekday() >= 5:
            gs -= timedelta(days=gs.weekday() - 4)
        reports.append(USDAReport("Grain Stocks", gs, "Actual grain inventories as of quarter-end", "high"))

    # Acreage report — June 30
    ac = date(year, 6, 30)
    if ac.weekday() >= 5:
        ac -= timedelta(days=ac.weekday() - 4)
    reports.append(USDAReport("Acreage (Final)", ac, "Final planted acreage — revises March intentions", "high"))

    # WASDE — approximately 10th-12th of each month
    for m in range(1, 13):
        wasde = date(year, m, 10)
        while wasde.weekday() >= 5:
            wasde += timedelta(days=1)
        reports.append(USDAReport("WASDE", wasde, "World Agricultural Supply & Demand Estimates", "high"))

    # Crop Progress — every Monday, April through November
    d = date(year, 4, 1)
    while d.weekday() != 0:
        d += timedelta(days=1)
    while d.month <= 11:
        reports.append(USDAReport("Crop Progress", d, "Weekly planting/harvest progress + condition", "medium"))
        d += timedelta(days=7)

    # Filter to window and sort
    filtered = [r for r in reports if today <= r.date <= end]
    filtered.sort(key=lambda r: r.date)

    # Deduplicate (Grain Stocks and Prospective Plantings share March 31)
    seen: set[tuple[str, date]] = set()
    deduped = []
    for r in filtered:
        key = (r.name, r.date)
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return deduped

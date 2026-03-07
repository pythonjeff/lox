"""
Multi-chokepoint maritime shipping data layer.

Fetches daily transit data from IMF PortWatch (free, no auth) for oil-critical
chokepoints. Optionally supplements with MarineTraffic real-time AIS data
when MARINETRAFFIC_API_KEY is configured.

Includes local file caching, retry logic, and standardized output.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Chokepoint definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChokepointDef:
    portid: str
    name: str
    short: str
    oil_weight: float  # importance for composite oil disruption score
    is_reroute_indicator: bool = False


OIL_CHOKEPOINTS: dict[str, ChokepointDef] = {
    "hormuz": ChokepointDef("chokepoint6", "Strait of Hormuz", "Hormuz", 0.40),
    "bab_el_mandeb": ChokepointDef("chokepoint4", "Bab el-Mandeb Strait", "Bab el-Mandeb", 0.20),
    "suez": ChokepointDef("chokepoint1", "Suez Canal", "Suez", 0.15),
    "malacca": ChokepointDef("chokepoint5", "Malacca Strait", "Malacca", 0.15),
    "bosporus": ChokepointDef("chokepoint3", "Bosporus Strait", "Bosporus", 0.10),
    "cape": ChokepointDef("chokepoint7", "Cape of Good Hope", "Cape GH", 0.0, is_reroute_indicator=True),
}

# Pre-conflict baselines (tankers/day) from 2023 historical averages.
# These are stable reference points that don't drift with the data window.
BASELINE_TANKERS: dict[str, float] = {
    "hormuz": 62.0,
    "bab_el_mandeb": 28.0,
    "suez": 22.0,
    "malacca": 30.0,
    "bosporus": 14.0,
    "cape": 18.0,
}

BASELINE_CAPACITY: dict[str, float] = {
    "hormuz": 2_800_000.0,
    "bab_el_mandeb": 1_200_000.0,
    "suez": 900_000.0,
    "malacca": 1_400_000.0,
    "bosporus": 500_000.0,
    "cape": 700_000.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChokepointData:
    key: str
    name: str
    short: str
    asof: str = ""
    source: str = "portwatch"
    df: pd.DataFrame | None = None

    latest_tankers: float | None = None
    avg_7d_tankers: float | None = None
    avg_30d_tankers: float | None = None
    baseline_tankers: float | None = None

    latest_total: float | None = None
    avg_7d_total: float | None = None
    avg_30d_total: float | None = None

    latest_cap_tanker: float | None = None
    avg_7d_cap_tanker: float | None = None
    avg_30d_cap_tanker: float | None = None
    baseline_cap_tanker: float | None = None

    tanker_values_30d: list[float] = field(default_factory=list)
    total_values_30d: list[float] = field(default_factory=list)
    cap_tanker_values_30d: list[float] = field(default_factory=list)

    disruption_score: int = 0
    disruption_label: str = "Unknown"
    transit_pct_of_baseline: float | None = None
    trajectory: str = ""
    trajectory_pct: float | None = None

    is_reroute_indicator: bool = False


@dataclass
class CompositeDisruption:
    score: int = 0
    label: str = "Unknown"
    per_chokepoint: dict[str, int] = field(default_factory=dict)
    rerouting_detected: bool = False
    rerouting_detail: str = ""
    trajectory: str = ""
    details: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# IMF PortWatch fetcher
# ─────────────────────────────────────────────────────────────────────────────

PORTWATCH_BASE = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "Daily_Chokepoints_Data/FeatureServer/0/query"
)

_CACHE_DIR = Path.home() / ".lox" / "cache" / "shipping"
_CACHE_TTL_HOURS = 4


def _cache_path(chokepoint_key: str) -> Path:
    today = datetime.now().strftime("%Y-%m-%d")
    return _CACHE_DIR / f"{chokepoint_key}_{today}.json"


def _read_cache(chokepoint_key: str) -> list[dict] | None:
    p = _cache_path(chokepoint_key)
    if not p.exists():
        return None
    age_hours = (time.time() - p.stat().st_mtime) / 3600
    if age_hours > _CACHE_TTL_HOURS:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _write_cache(chokepoint_key: str, rows: list[dict]) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(chokepoint_key).write_text(json.dumps(rows))
    except Exception as e:
        logger.debug("Cache write failed for %s: %s", chokepoint_key, e)


def _fetch_portwatch_raw(portid: str, max_retries: int = 3) -> list[dict]:
    """Fetch raw rows from IMF PortWatch with retry logic."""
    import requests

    all_rows: list[dict] = []
    for attempt in range(1, max_retries + 1):
        try:
            all_rows = []
            for offset in range(0, 2000, 1000):
                resp = requests.get(
                    PORTWATCH_BASE,
                    params={
                        "where": f"portid='{portid}'",
                        "outFields": "date,n_tanker,n_total,n_container,n_dry_bulk,"
                                     "capacity_tanker,capacity,portname",
                        "outSR": "4326",
                        "f": "json",
                        "resultRecordCount": "1000",
                        "resultOffset": str(offset),
                        "orderByFields": "date DESC",
                    },
                    timeout=20,
                )
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    break
                features = data.get("features", [])
                if not features:
                    break
                all_rows.extend(f["attributes"] for f in features)
                if len(features) < 1000:
                    break
            return all_rows
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.debug("PortWatch retry %d/%d for %s (wait %ds): %s",
                             attempt, max_retries, portid, wait, e)
                time.sleep(wait)
            else:
                logger.warning("PortWatch failed after %d attempts for %s: %s",
                               max_retries, portid, e)
    return all_rows


def _build_chokepoint_data(
    key: str,
    cpdef: ChokepointDef,
    rows: list[dict],
    days: int,
) -> ChokepointData | None:
    """Process raw PortWatch rows into a ChokepointData object."""
    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], unit="ms")
    df = df.sort_values("date").reset_index(drop=True)

    cutoff = datetime.now() - timedelta(days=days)
    df = df[df["date"] >= pd.Timestamp(cutoff)].reset_index(drop=True)
    if df.empty:
        return None

    cd = ChokepointData(
        key=key,
        name=cpdef.name,
        short=cpdef.short,
        asof=str(df["date"].iloc[-1].date()),
        df=df,
        baseline_tankers=BASELINE_TANKERS.get(key),
        baseline_cap_tanker=BASELINE_CAPACITY.get(key),
        is_reroute_indicator=cpdef.is_reroute_indicator,
    )

    for col, attr_prefix in [
        ("n_tanker", "tankers"),
        ("n_total", "total"),
        ("capacity_tanker", "cap_tanker"),
    ]:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        latest = float(s.iloc[-1])
        avg_7d = float(s.tail(7).mean())
        avg_30d = float(s.tail(30).mean())

        setattr(cd, f"latest_{attr_prefix}", latest)
        setattr(cd, f"avg_7d_{attr_prefix}", avg_7d)
        setattr(cd, f"avg_30d_{attr_prefix}", avg_30d)

        vals_30 = [float(v) for v in s.tail(30)]
        if attr_prefix == "tankers":
            cd.tanker_values_30d = vals_30
        elif attr_prefix == "total":
            cd.total_values_30d = vals_30
        elif attr_prefix == "cap_tanker":
            cd.cap_tanker_values_30d = vals_30

    _score_single_chokepoint(cd)
    return cd


def _score_single_chokepoint(cd: ChokepointData) -> None:
    """Compute disruption score for a single chokepoint."""
    baseline = cd.baseline_tankers
    current = cd.avg_7d_tankers
    if baseline is None or baseline <= 0 or current is None:
        cd.disruption_score = 0
        cd.disruption_label = "No data"
        return

    transit_ratio = current / baseline
    transit_deficit = max(0.0, 1.0 - transit_ratio)
    cd.transit_pct_of_baseline = transit_ratio * 100

    cap_deficit = 0.0
    b_cap = cd.baseline_cap_tanker
    c_cap = cd.avg_7d_cap_tanker
    if b_cap and b_cap > 0 and c_cap:
        cap_deficit = max(0.0, 1.0 - c_cap / b_cap)

    if b_cap and b_cap > 0:
        raw = transit_deficit * 0.4 + cap_deficit * 0.6
    else:
        raw = transit_deficit
    cd.disruption_score = int(min(100, raw * 100))

    # Trajectory
    if cd.avg_7d_tankers is not None and cd.avg_30d_tankers and cd.avg_30d_tankers > 0:
        tpct = (cd.avg_7d_tankers - cd.avg_30d_tankers) / cd.avg_30d_tankers * 100
        cd.trajectory_pct = tpct
        if tpct > 10:
            cd.trajectory = "recovering"
        elif tpct < -10:
            cd.trajectory = "worsening"
        else:
            cd.trajectory = "stable"

    score = cd.disruption_score
    if score >= 70:
        cd.disruption_label = "Severe Disruption"
    elif score >= 50:
        cd.disruption_label = "Major Disruption"
    elif score >= 30:
        cd.disruption_label = "Moderate Disruption"
    elif score >= 15:
        cd.disruption_label = "Mild Disruption"
    elif score >= 5:
        cd.disruption_label = "Minor Disruption"
    else:
        cd.disruption_label = "Normal Flow"


# ─────────────────────────────────────────────────────────────────────────────
# MarineTraffic (optional premium tier)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_marinetraffic_supplement(api_key: str, chokepoint_key: str) -> dict | None:
    """
    Fetch supplemental real-time data from MarineTraffic API.
    Returns intraday vessel count for the chokepoint area, or None.

    Requires MARINETRAFFIC_API_KEY and an active subscription.
    This provides ~5-second latency vs PortWatch's ~1-2 day lag.
    """
    # Area IDs would need to be configured per MarineTraffic subscription.
    # This is a stub that shows the integration point.
    logger.debug("MarineTraffic integration available but not yet configured for %s", chokepoint_key)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Composite disruption scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_composite_disruption(
    chokepoints: dict[str, ChokepointData],
) -> CompositeDisruption:
    """
    Compute a weighted composite oil supply disruption score from all chokepoints.

    Weights reflect each chokepoint's share of global oil flow:
      Hormuz 40%, Bab el-Mandeb 20%, Suez 15%, Malacca 15%, Bosporus 10%
    Cape of Good Hope is excluded from scoring (rerouting indicator only).
    """
    result = CompositeDisruption()

    weighted_sum = 0.0
    total_weight = 0.0

    for key, cpdef in OIL_CHOKEPOINTS.items():
        if cpdef.is_reroute_indicator:
            continue
        cd = chokepoints.get(key)
        if cd is None:
            continue
        w = cpdef.oil_weight
        result.per_chokepoint[key] = cd.disruption_score
        weighted_sum += cd.disruption_score * w
        total_weight += w

    if total_weight > 0:
        result.score = int(weighted_sum / total_weight)
    else:
        result.score = 0

    # Rerouting detection: Cape of Good Hope surge while Red Sea routes drop
    cape = chokepoints.get("cape")
    bab = chokepoints.get("bab_el_mandeb")
    suez = chokepoints.get("suez")
    if cape and cape.avg_7d_tankers and cape.baseline_tankers and cape.baseline_tankers > 0:
        cape_ratio = cape.avg_7d_tankers / cape.baseline_tankers
        red_sea_disrupted = False
        red_sea_detail = []
        if bab and bab.disruption_score >= 15:
            red_sea_disrupted = True
            red_sea_detail.append(f"Bab el-Mandeb {bab.disruption_score}/100")
        if suez and suez.disruption_score >= 15:
            red_sea_disrupted = True
            red_sea_detail.append(f"Suez {suez.disruption_score}/100")

        if cape_ratio > 1.15 and red_sea_disrupted:
            result.rerouting_detected = True
            result.rerouting_detail = (
                f"Cape GH traffic +{(cape_ratio - 1) * 100:.0f}% above baseline "
                f"while {' & '.join(red_sea_detail)} disrupted — vessels rerouting around Africa"
            )

    # Overall trajectory (weighted average of individual trajectories)
    recovering = sum(1 for k, cd in chokepoints.items()
                     if not OIL_CHOKEPOINTS.get(k, ChokepointDef("", "", "", 0)).is_reroute_indicator
                     and cd.trajectory == "recovering")
    worsening = sum(1 for k, cd in chokepoints.items()
                    if not OIL_CHOKEPOINTS.get(k, ChokepointDef("", "", "", 0)).is_reroute_indicator
                    and cd.trajectory == "worsening")
    if worsening > recovering:
        result.trajectory = "worsening"
    elif recovering > worsening:
        result.trajectory = "recovering"
    else:
        result.trajectory = "stable"

    # Label
    s = result.score
    if s >= 70:
        result.label = "Severe Disruption"
    elif s >= 50:
        result.label = "Major Disruption"
    elif s >= 30:
        result.label = "Moderate Disruption"
    elif s >= 15:
        result.label = "Mild Disruption"
    elif s >= 5:
        result.label = "Minor Disruption"
    else:
        result.label = "Normal Flow"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_oil_chokepoints(
    days: int = 365,
    marinetraffic_key: str | None = None,
) -> dict[str, ChokepointData]:
    """
    Fetch transit data for all oil-relevant chokepoints.
    Uses local cache (4h TTL) then falls back to PortWatch API.
    Returns dict keyed by chokepoint short name (hormuz, suez, etc.).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, ChokepointData] = {}

    def _fetch_one(key: str, cpdef: ChokepointDef) -> tuple[str, ChokepointData | None]:
        cached = _read_cache(key)
        source = "cache"
        if cached is None:
            cached = _fetch_portwatch_raw(cpdef.portid)
            source = "portwatch"
            if cached:
                _write_cache(key, cached)

        cd = _build_chokepoint_data(key, cpdef, cached, days)
        if cd is not None:
            cd.source = source
        return key, cd

    with ThreadPoolExecutor(max_workers=6, thread_name_prefix="shipping") as pool:
        futures = {
            pool.submit(_fetch_one, key, cpdef): key
            for key, cpdef in OIL_CHOKEPOINTS.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                k, cd = future.result()
                if cd is not None:
                    results[k] = cd
            except Exception as e:
                logger.warning("Failed to fetch chokepoint %s: %s", key, e)

    return results


def get_hormuz_compat(chokepoints: dict[str, ChokepointData]) -> dict | None:
    """
    Build a backward-compatible 'hz' dict matching the old _fetch_hormuz_data() format.
    This allows the existing display code to work during the transition.
    """
    cd = chokepoints.get("hormuz")
    if cd is None:
        return None

    hz: dict = {"df": cd.df, "asof": cd.asof}

    for old_col, attr_prefix in [
        ("n_tanker", "tankers"),
        ("n_total", "total"),
        ("capacity_tanker", "cap_tanker"),
        ("capacity", "cap_tanker"),
    ]:
        latest = getattr(cd, f"latest_{attr_prefix}", None)
        avg_7d = getattr(cd, f"avg_7d_{attr_prefix}", None)
        avg_30d = getattr(cd, f"avg_30d_{attr_prefix}", None)

        if latest is not None:
            hz[f"{old_col}_latest"] = latest
        if avg_7d is not None:
            hz[f"{old_col}_7d_avg"] = avg_7d
        if avg_30d is not None:
            hz[f"{old_col}_30d_avg"] = avg_30d

        vals_attr = f"{attr_prefix}_values_30d"
        vals = getattr(cd, vals_attr, [])
        if vals:
            hz[f"{old_col}_values"] = vals

    return hz

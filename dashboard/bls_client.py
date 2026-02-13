"""
BLS (Bureau of Labor Statistics) API v2 client with CSV caching.

Fetches CPI sub-index data for the Lived Inflation Index.
Modeled after the existing FredClient in lox/data/fred.py.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
CACHE_DIR = Path(__file__).parent / "data" / "cache" / "bls"


class BLSClient:
    """Fetch and cache BLS time-series data (CPI sub-indices)."""

    def __init__(self, api_key: str | None = None, cache_dir: str | Path | None = None):
        self.api_key = api_key or os.environ.get("BLS_API_KEY", "")
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, series_id: str) -> Path:
        return self.cache_dir / f"{series_id}.csv"

    def _is_cache_fresh(self, path: Path, max_age_days: int = 15) -> bool:
        """Cache is fresh if it exists and was updated within max_age_days."""
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age = (datetime.now() - mtime).days
        return age < max_age_days

    def fetch_series(
        self,
        series_ids: list[str],
        start_year: int = 2018,
        end_year: int | None = None,
        refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch multiple BLS series. Returns {series_id: DataFrame(date, value)}.

        Uses CSV cache. Only re-fetches if cache is stale or refresh=True.
        BLS API v2 supports up to 50 series per request.
        """
        if end_year is None:
            end_year = date.today().year

        result: dict[str, pd.DataFrame] = {}
        to_fetch: list[str] = []

        # Check cache first
        for sid in series_ids:
            cache_path = self._cache_path(sid)
            if not refresh and self._is_cache_fresh(cache_path):
                try:
                    df = pd.read_csv(cache_path, parse_dates=["date"])
                    result[sid] = df
                    continue
                except Exception:
                    pass
            to_fetch.append(sid)

        if not to_fetch:
            return result

        # BLS API v2: batch up to 50 series per request
        for i in range(0, len(to_fetch), 50):
            batch = to_fetch[i : i + 50]
            try:
                data = self._api_fetch(batch, start_year, end_year)
                for sid, df in data.items():
                    # Save to cache
                    cache_path = self._cache_path(sid)
                    df.to_csv(cache_path, index=False)
                    result[sid] = df
            except Exception as e:
                logger.error("BLS API fetch failed: %s", e)
                # Try loading stale cache as fallback
                for sid in batch:
                    cache_path = self._cache_path(sid)
                    if cache_path.exists():
                        try:
                            result[sid] = pd.read_csv(cache_path, parse_dates=["date"])
                        except Exception:
                            pass

        return result

    def _api_fetch(
        self, series_ids: list[str], start_year: int, end_year: int
    ) -> dict[str, pd.DataFrame]:
        """Call BLS API v2 and parse response into DataFrames."""
        # BLS API has a 20-year max range per request
        all_data: dict[str, list[dict]] = {sid: [] for sid in series_ids}

        for yr_start in range(start_year, end_year + 1, 20):
            yr_end = min(yr_start + 19, end_year)

            payload = {
                "seriesid": series_ids,
                "startyear": str(yr_start),
                "endyear": str(yr_end),
                "registrationkey": self.api_key,
            }

            resp = requests.post(BLS_API_URL, json=payload, timeout=30)
            resp.raise_for_status()
            body = resp.json()

            if body.get("status") != "REQUEST_SUCCEEDED":
                msg = body.get("message", ["Unknown error"])
                logger.warning("BLS API warning: %s", msg)

            for series in body.get("Results", {}).get("series", []):
                sid = series.get("seriesID", "")
                for item in series.get("data", []):
                    year = int(item["year"])
                    period = item["period"]  # "M01" through "M12"
                    if not period.startswith("M"):
                        continue  # skip annual averages (M13)
                    month = int(period[1:])
                    value = item.get("value", "")
                    try:
                        val = float(value)
                    except (ValueError, TypeError):
                        continue
                    all_data.setdefault(sid, []).append(
                        {"date": f"{year}-{month:02d}-01", "value": val}
                    )

        # Convert to DataFrames
        result: dict[str, pd.DataFrame] = {}
        for sid, rows in all_data.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
            result[sid] = df

        return result

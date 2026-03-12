"""
Fetch and cache Fama-French 5-factor + momentum daily returns.

Data source: Ken French Data Library (free, no API key).
Returns are stored as decimals (0.01 = 1%).

Author: Lox Capital Research
"""
from __future__ import annotations

import io
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FRENCH_5F_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
FRENCH_MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)

_CACHE_DIR = Path("data/cache/french_factors")
_5F_CACHE = _CACHE_DIR / "ff5_daily.csv"
_MOM_CACHE = _CACHE_DIR / "mom_daily.csv"

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
MOM_COL = "Mom"


def _is_stale(path: Path) -> bool:
    """True if file is missing or last modified before today."""
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return mtime < today


def _download_zip_csv(url: str) -> str:
    """Download ZIP from URL and extract the single CSV inside as text."""
    import requests

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        return zf.read(csv_name).decode("utf-8", errors="replace")


def _parse_french_csv(text: str, expected_cols: list[str]) -> pd.DataFrame:
    """Parse a Ken French daily CSV into a DataFrame.

    Strategy:
    1. Find the header row containing the first expected column name
    2. Read subsequent non-blank lines as data
    3. Stop at first blank line after data starts
    4. Dates are YYYYMMDD integers, values in percent (divide by 100)
    """
    lines = text.splitlines()

    # Find header row
    header_idx = None
    search_col = expected_cols[0]
    for i, line in enumerate(lines):
        if search_col in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not find header row containing '{search_col}'")

    # Parse header to get column positions
    header_parts = lines[header_idx].split(",")
    header_parts = [h.strip() for h in header_parts]

    # Read data rows until blank line
    data_rows: list[list[str]] = []
    for i in range(header_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line:
            break
        parts = line.split(",")
        # First field must look like a date (6-8 digits)
        first = parts[0].strip()
        if not first.isdigit() or len(first) < 6:
            break
        data_rows.append([p.strip() for p in parts])

    if not data_rows:
        raise ValueError("No data rows found in French CSV")

    # Build DataFrame
    n_cols = len(header_parts)
    df = pd.DataFrame(data_rows, columns=[f"col_{i}" for i in range(len(data_rows[0]))])

    # First column is date
    df["date"] = pd.to_datetime(df["col_0"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.set_index("date")

    # Map remaining columns to expected names
    result = pd.DataFrame(index=df.index)
    for j, col_name in enumerate(expected_cols):
        src_col = f"col_{j + 1}"
        if src_col in df.columns:
            result[col_name] = pd.to_numeric(df[src_col], errors="coerce") / 100.0

    return result.dropna().sort_index()


def _fetch_5factor(refresh: bool = False) -> pd.DataFrame:
    """Fetch 5-factor daily data, using cache if fresh."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not refresh and not _is_stale(_5F_CACHE):
        try:
            df = pd.read_csv(_5F_CACHE, index_col=0, parse_dates=True)
            if not df.empty:
                return df
        except Exception:
            pass

    try:
        text = _download_zip_csv(FRENCH_5F_URL)
        df = _parse_french_csv(text, FACTOR_COLS)
        df.to_csv(_5F_CACHE)
        logger.info(f"Cached {len(df)} rows of FF5 daily data")
        return df
    except Exception as e:
        # Fall back to stale cache
        if _5F_CACHE.exists():
            logger.warning(f"FF5 download failed ({e}), using stale cache")
            return pd.read_csv(_5F_CACHE, index_col=0, parse_dates=True)
        raise RuntimeError(
            f"Failed to download Fama-French 5-factor data and no cache exists: {e}"
        ) from e


def _fetch_momentum(refresh: bool = False) -> pd.DataFrame:
    """Fetch momentum factor daily data, using cache if fresh."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not refresh and not _is_stale(_MOM_CACHE):
        try:
            df = pd.read_csv(_MOM_CACHE, index_col=0, parse_dates=True)
            if not df.empty:
                return df
        except Exception:
            pass

    try:
        text = _download_zip_csv(FRENCH_MOM_URL)
        df = _parse_french_csv(text, [MOM_COL])
        df.to_csv(_MOM_CACHE)
        logger.info(f"Cached {len(df)} rows of momentum daily data")
        return df
    except Exception as e:
        if _MOM_CACHE.exists():
            logger.warning(f"Momentum download failed ({e}), using stale cache")
            return pd.read_csv(_MOM_CACHE, index_col=0, parse_dates=True)
        raise RuntimeError(
            f"Failed to download momentum factor data and no cache exists: {e}"
        ) from e


def fetch_french_factors(refresh: bool = False) -> pd.DataFrame:
    """Return merged DataFrame with all factor returns.

    Columns: ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'Mom']
    Index: DatetimeIndex (daily)
    Values: decimal returns (0.01 = 1%)
    """
    ff5 = _fetch_5factor(refresh=refresh)
    mom = _fetch_momentum(refresh=refresh)

    merged = ff5.join(mom, how="inner")
    return merged.sort_index()

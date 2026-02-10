"""
Unified date parsing utilities for various input formats.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional


def parse_iso_date(s: str | None) -> date | None:
    """Parse ISO date string (YYYY-MM-DD) to date."""
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def parse_ymd(s: str) -> date:
    """Parse YYYY-MM-DD string to date. Raises ValueError on failure."""
    return date.fromisoformat(s)


def parse_timestamp(s: str) -> datetime:
    """Parse ISO timestamp string to datetime. Raises ValueError on failure."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def parse_datetime_any(value: Any) -> datetime | None:
    """
    Parse various datetime formats commonly found in APIs.
    
    Handles:
    - ISO strings (with/without timezone)
    - Excel serial dates
    - Unix timestamps (seconds or milliseconds)
    - Datetime objects (passthrough)
    """
    if value is None:
        return None
    
    if isinstance(value, datetime):
        return value
    
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    
    if isinstance(value, (int, float)):
        # Check if it's an Excel serial date (roughly 1900-2100 range)
        if 1 < value < 100000:
            # Excel serial date
            return datetime(1899, 12, 30) + __import__('datetime').timedelta(days=value)
        elif value > 1e9:
            # Unix timestamp in milliseconds
            return datetime.utcfromtimestamp(value / 1000)
        else:
            # Unix timestamp in seconds
            return datetime.utcfromtimestamp(value)
    
    if isinstance(value, str):
        value = value.strip()
        
        # Try ISO format first
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%m/%d/%Y %H:%M:%S",
        ]:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        
        # Try pandas-style parsing as fallback
        try:
            import pandas as pd
            return pd.to_datetime(value).to_pydatetime()
        except Exception:
            pass
    
    return None


def format_date(d: date | datetime | None, fmt: str = "%Y-%m-%d") -> str:
    """Format date/datetime to string, returns 'N/A' if None."""
    if d is None:
        return "N/A"
    return d.strftime(fmt)


def days_between(d1: date, d2: date) -> int:
    """Return number of days between two dates."""
    return (d2 - d1).days


def utc_now_iso() -> str:
    """Return current UTC time as ISO string."""
    from datetime import timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

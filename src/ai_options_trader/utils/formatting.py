"""
Display formatting utilities for CLI output.

Provides consistent formatting for:
- Numbers (integers, floats, percentages)
- Currency values
- Tables and panels

Author: Lox Capital Research
"""
from __future__ import annotations

from typing import Optional, Union, Any


# ============================================================================
# Number Formatting
# ============================================================================

def fmt_int(x: Optional[int]) -> str:
    """Format integer with comma separators, or 'n/a'."""
    if x is None or not isinstance(x, (int, float)):
        return "n/a"
    return f"{int(x):,}"


def fmt_float(x: Optional[float], decimals: int = 2) -> str:
    """Format float with specified decimals, or 'n/a'."""
    if x is None or not isinstance(x, (int, float)):
        return "n/a"
    return f"{float(x):.{decimals}f}"


def fmt_pct(x: Optional[float], decimals: int = 1, multiply: bool = True) -> str:
    """
    Format as percentage.
    
    Args:
        x: Value to format
        decimals: Decimal places to show
        multiply: If True, multiply by 100 (i.e., 0.05 -> 5.0%)
    """
    if x is None or not isinstance(x, (int, float)):
        return "n/a"
    value = float(x) * 100.0 if multiply else float(x)
    return f"{value:.{decimals}f}%"


def fmt_signed_pct(x: Optional[float], decimals: int = 1, multiply: bool = True) -> str:
    """Format as signed percentage with + prefix for positives."""
    if x is None or not isinstance(x, (int, float)):
        return "n/a"
    value = float(x) * 100.0 if multiply else float(x)
    return f"{value:+.{decimals}f}%"


# ============================================================================
# Currency Formatting
# ============================================================================

def fmt_usd(x: Optional[float], show_cents: bool = True) -> str:
    """Format as USD currency."""
    if x is None or not isinstance(x, (int, float)):
        return "n/a"
    if show_cents:
        return f"${float(x):,.2f}"
    return f"${float(x):,.0f}"


def fmt_signed_usd(x: Optional[float], show_cents: bool = True) -> str:
    """Format as signed USD with + prefix for positives."""
    if x is None or not isinstance(x, (int, float)):
        return "n/a"
    if show_cents:
        return f"${float(x):+,.2f}"
    return f"${float(x):+,.0f}"


def fmt_compact_usd(x: Optional[float]) -> str:
    """Format large USD values with K/M/B suffixes."""
    if x is None or not isinstance(x, (int, float)):
        return "n/a"
    
    v = float(x)
    if abs(v) >= 1e12:
        return f"${v/1e12:.1f}T"
    elif abs(v) >= 1e9:
        return f"${v/1e9:.1f}B"
    elif abs(v) >= 1e6:
        return f"${v/1e6:.1f}M"
    elif abs(v) >= 1e3:
        return f"${v/1e3:.1f}K"
    else:
        return f"${v:.0f}"


# ============================================================================
# Basis Points
# ============================================================================

def fmt_bps(x: Optional[float], from_pct: bool = False) -> str:
    """
    Format as basis points.
    
    Args:
        x: Value to format
        from_pct: If True, input is already in % (multiply by 100 for bps)
    """
    if x is None or not isinstance(x, (int, float)):
        return "n/a"
    
    bps = float(x) * 100.0 if from_pct else float(x) * 10000.0
    return f"{bps:.0f}bp"


# ============================================================================
# Direction/Status Indicators
# ============================================================================

def fmt_direction(x: Optional[float], threshold: float = 0.0) -> str:
    """Return direction indicator based on sign."""
    if x is None:
        return "-"
    if float(x) > threshold:
        return "(rising)"
    elif float(x) < -threshold:
        return "(falling)"
    return "(flat)"


def fmt_status_color(status: str) -> str:
    """
    Return Rich color for status string.
    
    Common statuses: healthy, warning, error, neutral
    """
    colors = {
        "healthy": "green",
        "good": "green",
        "ok": "green",
        "warning": "yellow",
        "caution": "yellow",
        "error": "red",
        "bad": "red",
        "stop": "red",
        "neutral": "dim",
    }
    return colors.get(status.lower(), "white")


# ============================================================================
# Greek Formatting
# ============================================================================

def fmt_delta(x: Optional[float]) -> str:
    """Format delta with sign."""
    if x is None:
        return "n/a"
    return f"{float(x):+.2f}"


def fmt_theta(x: Optional[float]) -> str:
    """Format theta (typically negative for long options)."""
    if x is None:
        return "n/a"
    return f"{float(x):.3f}"


def fmt_vega(x: Optional[float]) -> str:
    """Format vega."""
    if x is None:
        return "n/a"
    return f"{float(x):.2f}"


def fmt_gamma(x: Optional[float]) -> str:
    """Format gamma."""
    if x is None:
        return "n/a"
    return f"{float(x):.4f}"


def fmt_iv(x: Optional[float]) -> str:
    """Format implied volatility as percentage."""
    if x is None:
        return "n/a"
    return f"{float(x)*100:.1f}%"


# ============================================================================
# Table Helpers
# ============================================================================

def truncate(s: str, max_len: int = 20) -> str:
    """Truncate string with ellipsis if needed."""
    if len(s) <= max_len:
        return s
    return s[:max_len-1] + "…"


def pad_right(s: str, width: int) -> str:
    """Right-pad string to specified width."""
    return s.ljust(width)


def pad_left(s: str, width: int) -> str:
    """Left-pad string to specified width."""
    return s.rjust(width)

def fmt_usd_from_millions(x) -> str:
    """Format USD from millions input (e.g., 100 -> $100M)."""
    if x is None or not isinstance(x, (int, float)):
        return "n/a"
    
    v = float(x)
    if abs(v) >= 1e6:
        return f"${v/1e6:.1f}T"
    elif abs(v) >= 1e3:
        return f"${v/1e3:.1f}B"
    else:
        return f"${v:.0f}M"

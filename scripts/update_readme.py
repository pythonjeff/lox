#!/usr/bin/env python3
"""
Update the README.md Performance section with live fund data.

Reads local CSV files (nav_sheet.csv, nav_flows.csv) for fund metrics.
Optionally fetches SPY benchmark return from FMP API if FMP_API_KEY is set.

Usage:
    python scripts/update_readme.py          # Uses .env for API keys
    FMP_API_KEY=xxx python scripts/update_readme.py  # Explicit key
    python scripts/update_readme.py --no-benchmark   # Skip SPY fetch
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Fund inception date — must match dashboard/data_fetchers.py
FUND_INCEPTION_DATE = "2026-01-09"

# Markers in README.md that delimit the dynamic performance section
PERF_START = "<!-- PERF_START -->"
PERF_END = "<!-- PERF_END -->"


def get_project_root() -> Path:
    """Return project root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def get_latest_nav_data(project_root: Path) -> dict | None:
    """Read the latest snapshot from nav_sheet.csv."""
    nav_path = project_root / "data" / "nav_sheet.csv"
    if not nav_path.exists():
        return None

    with open(nav_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    latest = rows[-1]
    return {
        "equity": float(latest["equity"]),
        "twr_cum": float(latest["twr_cum"]),
        "ts": latest["ts"],
    }


def get_total_capital(project_root: Path) -> float:
    """Sum all positive flows from nav_flows.csv."""
    flows_path = project_root / "data" / "nav_flows.csv"
    if not flows_path.exists():
        return 0.0

    with open(flows_path) as f:
        reader = csv.DictReader(f)
        return sum(float(row["amount"]) for row in reader if float(row["amount"]) > 0)


def get_spy_return() -> float | None:
    """Fetch SPY return since fund inception from FMP API."""
    try:
        import requests  # noqa: delay import so script can run without requests for local-only mode
    except ImportError:
        print("  requests not installed — skipping benchmark fetch")
        return None

    fmp_key = os.environ.get("FMP_API_KEY")
    if not fmp_key:
        # Try loading from .env manually
        env_path = get_project_root() / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("FMP_API_KEY="):
                    fmp_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if not fmp_key:
        print("  FMP_API_KEY not found — skipping benchmark fetch")
        return None

    try:
        url = "https://financialmodelingprep.com/api/v3/historical-price-full/SPY"
        resp = requests.get(
            url,
            params={"apikey": fmp_key, "from": FUND_INCEPTION_DATE},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, dict) or "historical" not in data:
            return None

        historical = data["historical"]
        if not historical or len(historical) < 2:
            return None

        # historical is sorted newest-first
        current_price = historical[0].get("close", 0)
        inception_price = historical[-1].get("close", 0)

        if inception_price <= 0:
            return None

        return ((current_price - inception_price) / inception_price) * 100

    except Exception as e:
        print(f"  SPY fetch error: {e}")
        return None


def format_date(ts_str: str) -> str:
    """Format ISO timestamp to 'Feb 7, 2026' style."""
    try:
        dt = datetime.fromisoformat(ts_str)
    except Exception:
        dt = datetime.now(tz=timezone.utc)
    # Cross-platform: use %d and strip leading zero
    day = dt.day
    return dt.strftime(f"%b {day}, %Y")


def format_pct(value: float) -> str:
    """Format a percentage with sign."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def build_performance_section(
    total_capital: float,
    equity: float,
    twr_pct: float,
    spy_return: float | None,
    last_updated: str,
) -> str:
    """Build the markdown table for the Performance section."""
    alpha = (twr_pct - spy_return) if spy_return is not None else None

    lines: list[str] = []
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append("| **Inception** | January 9, 2026 |")
    lines.append(f"| **Total Capital** | ${total_capital:,.0f} |")
    lines.append(f"| **Current NAV** | ${equity:,.0f} |")
    lines.append(f"| **TWR (Since Inception)** | **{format_pct(twr_pct)}** |")

    if spy_return is not None:
        lines.append(f"| **Benchmark (SPY)** | {format_pct(spy_return)} |")

    if alpha is not None:
        alpha_str = f"+{alpha:.1f}%" if alpha >= 0 else f"{alpha:.1f}%"
        lines.append(f"| **Alpha** | {alpha_str} |")

    lines.append("| **Strategy** | Discretionary macro with tail-risk hedging |")
    lines.append("")
    lines.append(
        f"*Last updated: {last_updated} • Live at [loxfund.com](https://loxfund.com)*"
    )

    return "\n".join(lines)


def update_readme(project_root: Path, skip_benchmark: bool = False) -> bool:
    """Update the Performance section in README.md between PERF markers."""
    readme_path = project_root / "README.md"
    if not readme_path.exists():
        print("ERROR: README.md not found")
        return False

    content = readme_path.read_text()

    if PERF_START not in content or PERF_END not in content:
        print(f"ERROR: Markers {PERF_START} / {PERF_END} not found in README.md")
        return False

    # --- Gather data ---
    print("Reading NAV data...")
    nav_data = get_latest_nav_data(project_root)
    if not nav_data:
        print("ERROR: No NAV data in nav_sheet.csv")
        return False

    total_capital = get_total_capital(project_root)
    equity = nav_data["equity"]
    twr_pct = nav_data["twr_cum"] * 100  # decimal → percentage
    last_updated = format_date(nav_data["ts"])

    print(f"  NAV: ${equity:,.2f}  |  TWR: {format_pct(twr_pct)}  |  Capital: ${total_capital:,.0f}")

    spy_return: float | None = None
    if not skip_benchmark:
        print("Fetching SPY benchmark...")
        spy_return = get_spy_return()
        if spy_return is not None:
            print(f"  SPY: {format_pct(spy_return)}")
        else:
            print("  SPY: unavailable")

    # --- Build new section ---
    new_section = build_performance_section(
        total_capital=total_capital,
        equity=equity,
        twr_pct=twr_pct,
        spy_return=spy_return,
        last_updated=last_updated,
    )

    # --- Replace between markers ---
    start_idx = content.index(PERF_START) + len(PERF_START)
    end_idx = content.index(PERF_END)

    new_content = content[:start_idx] + "\n" + new_section + "\n" + content[end_idx:]

    if new_content == content:
        print("README.md already up to date — no changes.")
        return True

    readme_path.write_text(new_content)
    print(f"README.md updated ({last_updated})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Update README.md performance section")
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip fetching SPY benchmark (no network call)",
    )
    args = parser.parse_args()

    project_root = get_project_root()
    success = update_readme(project_root, skip_benchmark=args.no_benchmark)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

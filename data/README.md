# Data Directory Structure

This directory contains persistent data files for the Lox Capital trading system.

## Directory Layout

```
data/
├── nav_sheet.csv          # NAV snapshots (tracked in git)
├── nav_flows.csv          # Fund cash flows (tracked in git)  
├── nav_investor_flows.csv # Investor-level flows (tracked in git)
├── live/                  # Live trading data (gitignored)
│   ├── actions.jsonl      # Trade action logs
│   └── ticks.jsonl        # Price tick data
├── cache/                 # API response cache (gitignored)
│   ├── fred/              # FRED macro series
│   ├── fmp_prices/        # Historical price data
│   ├── altdata/           # FMP quotes, profiles, news
│   ├── fiscaldata/        # Treasury fiscal data
│   └── ...                # Other cached data
└── tracker.sqlite3        # Recommendation tracking DB (gitignored)
```

## File Categories

### Tracked Files (version controlled)
- `nav_sheet.csv` - Daily NAV snapshots with P&L and TWR calculations
- `nav_flows.csv` - Fund-level cash contributions and withdrawals
- `nav_investor_flows.csv` - Per-investor contribution ledger

### Gitignored Files
- `cache/` - All cached API responses (regenerated as needed)
- `live/` - Real-time trading logs
- `tracker.sqlite3` - SQLite database for recommendation tracking

## Configuration

File paths can be overridden via environment variables:
- `AOT_DATA_DIR` - Data directory (default: project root `/data`; used when no per-file env is set)
- `AOT_NAV_SHEET` - Path to NAV sheet (default: `data/nav_sheet.csv`)
- `AOT_NAV_FLOWS` - Path to flows file (default: `data/nav_flows.csv`)
- `AOT_NAV_INVESTOR_FLOWS` - Path to investor flows (default: `data/nav_investor_flows.csv`)
- `AOT_CACHE_DIR` - Cache directory (default: `data/cache`)
- `AOT_TRACKER_DB` - Tracker database (default: `data/tracker.sqlite3`)

## Fund accounting

**Adding a deposit**  
Use one command so both ledgers stay in sync:

```bash
lox nav investor contribute <code> <amount> [--note "round N"]
```

This updates **nav_investor_flows.csv** (with correct NAV/unit and units) and **nav_flows.csv** (for TWR). Do not add deposits only via `lox nav investor seed` or `investor import` if you care about correct TWR in NAV snapshots.

**Total capital**  
Always derived from **nav_investor_flows.csv** (sum of positive amounts). The dashboard, CLI, and regime logic all use this. The env var `FUND_TOTAL_CAPITAL` is only a fallback when the investor flows file is missing or empty; prefer keeping the CSV up to date via `lox nav investor contribute`.

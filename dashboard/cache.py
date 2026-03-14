"""
Centralized cache registry for the LOX FUND Dashboard.

All server-side caches, locks, and TTL constants live here.
Service modules import what they need — no circular dependencies.
"""

import os
import threading

# ── Palmer Analysis Cache (30-min auto-refresh) ──
PALMER_CACHE = {
    "analysis": None,
    "regime_snapshot": None,
    "timestamp": None,
    "last_refresh": None,
    "traffic_lights": None,
    "portfolio_analysis": None,
    "prev_traffic_lights": None,
    "regime_changed": False,
    "regime_change_details": None,
}
PALMER_CACHE_LOCK = threading.Lock()
PALMER_REFRESH_INTERVAL = 30 * 60  # 30 minutes

# ── Monte Carlo Cache (1-hour auto-refresh) ──
MC_CACHE = {
    "forecast": None,
    "timestamp": None,
    "last_refresh": None,
}
MC_CACHE_LOCK = threading.Lock()
MC_REFRESH_INTERVAL = 60 * 60  # 1 hour

# ── Positions Cache (short-lived, matches frontend polling) ──
POSITIONS_CACHE = {"data": None, "timestamp": None}
POSITIONS_CACHE_LOCK = threading.Lock()
POSITIONS_CACHE_TTL = 30  # 30 seconds

# ── Investors Cache ──
INVESTORS_CACHE = {"data": None, "timestamp": None}
INVESTORS_CACHE_LOCK = threading.Lock()
INVESTORS_CACHE_TTL = 10  # 10 seconds

# ── Closed Trades Cache ──
TRADES_CACHE = {"data": None, "timestamp": None}
TRADES_CACHE_LOCK = threading.Lock()
TRADES_CACHE_TTL = 60  # 1 minute

# ── Regime Domains Cache ──
DOMAINS_CACHE = {"data": None, "timestamp": None}
DOMAINS_CACHE_LOCK = threading.Lock()
DOMAINS_CACHE_TTL = 600  # 10 minutes

# ── Market News Cache ──
NEWS_CACHE = {"data": None, "timestamp": None}
NEWS_CACHE_LOCK = threading.Lock()
NEWS_CACHE_TTL = 300  # 5 minutes

# ── Regime Context Cache ──
REGIME_CTX_CACHE = {"data": None, "timestamp": None}
REGIME_CTX_CACHE_LOCK = threading.Lock()
REGIME_CTX_CACHE_TTL = 300  # 5 minutes

# ── Benchmark Returns Cache ──
BENCHMARK_CACHE = {"data": None, "timestamp": None}
BENCHMARK_CACHE_LOCK = threading.Lock()
BENCHMARK_CACHE_TTL = 300  # 5 minutes

# ── Live Indicator Source Cache ──
INDICATOR_SOURCE_CACHE = {}
INDICATOR_SOURCE_CACHE_LOCK = threading.Lock()
INDICATOR_SOURCE_TTL = 1800  # 30 minutes

# ── Position Thesis Cache ──
THESIS_CACHE = {"data": None, "timestamp": None}
THESIS_CACHE_LOCK = threading.Lock()
THESIS_CACHE_TTL = 3600  # 1 hour — thesis doesn't change with price

# ── Admin Secret ──
ADMIN_SECRET = os.environ.get("PALMER_ADMIN_SECRET", "lox-admin-2026")

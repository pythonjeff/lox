"""
Centralized cache registry for the LOX FUND Dashboard.

Only caches for actively-used endpoints are kept here.
Frontend polls: /api/positions (30s), /api/closed-trades (3m),
/api/regime-summary (5m), /api/nav-history (once), /api/position-thesis (lazy).
"""

import os
import threading

# ── Positions Cache (short-lived, matches frontend polling) ──
POSITIONS_CACHE = {"data": None, "timestamp": None}
POSITIONS_CACHE_LOCK = threading.Lock()
POSITIONS_CACHE_TTL = 30  # 30 seconds

# ── Closed Trades Cache ──
TRADES_CACHE = {"data": None, "timestamp": None}
TRADES_CACHE_LOCK = threading.Lock()
TRADES_CACHE_TTL = 60  # 1 minute

# ── Position Thesis Cache ──
THESIS_CACHE = {"data": None, "timestamp": None}
THESIS_CACHE_LOCK = threading.Lock()
THESIS_CACHE_TTL = 3600  # 1 hour — thesis doesn't change with price

# ── Benchmark Returns Cache (used by positions performance) ──
BENCHMARK_CACHE = {"data": None, "timestamp": None}
BENCHMARK_CACHE_LOCK = threading.Lock()
BENCHMARK_CACHE_TTL = 300  # 5 minutes

# ── Regime Context Cache (used by regime-summary market pulse) ──
REGIME_CTX_CACHE = {"data": None, "timestamp": None}
REGIME_CTX_CACHE_LOCK = threading.Lock()
REGIME_CTX_CACHE_TTL = 300  # 5 minutes

# ── Investors Cache (used by /my-account page) ──
INVESTORS_CACHE = {"data": None, "timestamp": None}
INVESTORS_CACHE_LOCK = threading.Lock()
INVESTORS_CACHE_TTL = 10  # 10 seconds

# ── Palmer Analysis Cache (lazy-loaded, no background thread) ──
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

# ── Monte Carlo Cache (lazy-loaded, no background thread) ──
MC_CACHE = {
    "forecast": None,
    "timestamp": None,
    "last_refresh": None,
}
MC_CACHE_LOCK = threading.Lock()
MC_REFRESH_INTERVAL = 60 * 60  # 1 hour

# ── Admin Secret ──
ADMIN_SECRET = os.environ.get("PALMER_ADMIN_SECRET", "lox-admin-2026")

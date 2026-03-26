"""Position, thesis, and closed-trade API routes."""

from flask import Blueprint, jsonify
from datetime import datetime, timezone

from dashboard.cache import (
    TRADES_CACHE, TRADES_CACHE_LOCK, TRADES_CACHE_TTL,
)

positions_api = Blueprint("positions_api", __name__)


@positions_api.route('/api/positions')
def api_positions():
    """Positions data — LIVE updates (public)."""
    from dashboard.positions import get_positions_data
    data = get_positions_data()
    response = jsonify(data)
    response.headers['Cache-Control'] = 'no-cache'
    return response


@positions_api.route('/api/closed-trades')
def api_closed_trades():
    """Closed trades (realized P&L) with caching (public)."""
    with TRADES_CACHE_LOCK:
        if TRADES_CACHE["data"] and TRADES_CACHE["timestamp"]:
            cache_age = (datetime.now(timezone.utc) - TRADES_CACHE["timestamp"]).total_seconds()
            if cache_age < TRADES_CACHE_TTL:
                response = jsonify(TRADES_CACHE["data"])
                response.headers['Cache-Control'] = 'public, max-age=60'
                return response

    try:
        from dashboard.trades import get_closed_trades_data
        data = get_closed_trades_data()
        with TRADES_CACHE_LOCK:
            TRADES_CACHE["data"] = data
            TRADES_CACHE["timestamp"] = datetime.now(timezone.utc)
        response = jsonify(data)
        response.headers['Cache-Control'] = 'public, max-age=60'
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "trades": [], "total_pnl": 0, "win_rate": 0})

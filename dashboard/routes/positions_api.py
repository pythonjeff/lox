"""Position, thesis, and closed-trade API routes."""

from flask import Blueprint, jsonify, request
from flask_login import login_required
from datetime import datetime, timezone

from dashboard.cache import (
    THESIS_CACHE, THESIS_CACHE_LOCK, THESIS_CACHE_TTL,
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


@positions_api.route('/api/position-thesis')
@login_required
def api_position_thesis():
    """AI-generated position thesis (cached 1 hour)."""
    with THESIS_CACHE_LOCK:
        if THESIS_CACHE["data"] and THESIS_CACHE["timestamp"]:
            cache_age = (datetime.now(timezone.utc) - THESIS_CACHE["timestamp"]).total_seconds()
            if cache_age < THESIS_CACHE_TTL:
                response = jsonify(THESIS_CACHE["data"])
                response.headers['Cache-Control'] = 'public, max-age=3600'
                return response

    try:
        from dashboard.positions import get_positions_data, get_regime_context, generate_position_theory
        from lox.config import load_settings
        positions_data = get_positions_data()
        positions = positions_data.get("positions", [])
        settings = load_settings()
        regime_context = get_regime_context(settings)

        theses = {}
        for pos in positions:
            symbol = pos.get("symbol", "")
            if symbol:
                theses[symbol] = generate_position_theory(pos, regime_context, settings)

        result = {
            "theses": theses,
            "count": len(theses),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with THESIS_CACHE_LOCK:
            THESIS_CACHE["data"] = result
            THESIS_CACHE["timestamp"] = datetime.now(timezone.utc)

        response = jsonify(result)
        response.headers['Cache-Control'] = 'public, max-age=3600'
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "theses": {}})


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

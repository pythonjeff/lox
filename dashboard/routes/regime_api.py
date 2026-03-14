"""Regime analysis API routes."""

from flask import Blueprint, jsonify, request
from flask_login import login_required
from datetime import datetime, timezone

from dashboard.cache import (
    PALMER_CACHE, PALMER_CACHE_LOCK,
    DOMAINS_CACHE, DOMAINS_CACHE_LOCK, DOMAINS_CACHE_TTL,
    ADMIN_SECRET,
)

regime_api = Blueprint("regime_api", __name__)


@regime_api.route('/api/regime-domains')
@login_required
def api_regime_domains():
    """Regime domain indicators with caching."""
    with DOMAINS_CACHE_LOCK:
        if DOMAINS_CACHE["data"] and DOMAINS_CACHE["timestamp"]:
            cache_age = (datetime.now(timezone.utc) - DOMAINS_CACHE["timestamp"]).total_seconds()
            if cache_age < DOMAINS_CACHE_TTL:
                response = jsonify(DOMAINS_CACHE["data"])
                response.headers['Cache-Control'] = 'public, max-age=120'
                return response

    try:
        from lox.config import load_settings
        from dashboard.regime_utils import get_regime_domains_data
        settings = load_settings()
        data = get_regime_domains_data(settings)
        with DOMAINS_CACHE_LOCK:
            DOMAINS_CACHE["data"] = data
            DOMAINS_CACHE["timestamp"] = datetime.now(timezone.utc)
        response = jsonify(data)
        response.headers['Cache-Control'] = 'public, max-age=120'
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "domains": {}})


@regime_api.route('/api/regime/<regime_name>')
def api_regime_detail(regime_name):
    """Detailed regime data (single domain)."""
    from dashboard.regime_utils import REGIME_NAMES, get_regime_detail
    from lox.config import load_settings
    if regime_name not in REGIME_NAMES:
        return jsonify({"error": f"Unknown regime: {regime_name}"}), 404
    refresh = request.args.get("refresh", "").lower() in ("true", "1", "yes")
    try:
        settings = load_settings()
        data = get_regime_detail(settings, regime_name, refresh=refresh)
        if "error" in data and len(data) == 1:
            return jsonify(data), 500
        response = jsonify(data)
        response.headers['Cache-Control'] = 'public, max-age=300'
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@regime_api.route('/api/regime-summary')
def api_regime_summary():
    """Lightweight summary of all regime scores/labels + market pulse."""
    try:
        from lox.config import load_settings
        from dashboard.regime_utils import get_regime_summary
        from dashboard.positions import get_regime_context
        settings = load_settings()
        data = get_regime_summary(settings)
        try:
            ctx = get_regime_context(settings)
            pulse = {}
            if ctx.get("vix") and ctx["vix"] != "N/A":
                pulse["vix"] = float(ctx["vix"])
            if ctx.get("yield_10y") and ctx["yield_10y"] != "N/A":
                pulse["ten_y"] = float(str(ctx["yield_10y"]).replace("%", ""))
            if ctx.get("hy_oas_bps") and ctx["hy_oas_bps"] != "N/A":
                pulse["hy_oas"] = float(ctx["hy_oas_bps"])
            if ctx.get("dxy") and ctx["dxy"] != "N/A":
                pulse["dxy"] = float(ctx["dxy"])
            data["market_pulse"] = pulse
        except Exception as e:
            print(f"[RegimeSummary] Market pulse enrichment error: {e}")
            data["market_pulse"] = {}
        response = jsonify(data)
        response.headers['Cache-Control'] = 'public, max-age=300'
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "regimes": []}), 500


@regime_api.route('/api/regime-analysis')
def api_regime_analysis():
    """Palmer: Cached LLM analysis. Auto-refreshes every 30 min (public)."""
    with PALMER_CACHE_LOCK:
        analysis = PALMER_CACHE.get("analysis") or ""
        summary = ""
        if analysis:
            sentences = analysis.split('.')
            if sentences:
                summary = sentences[0].strip() + '.'
                if len(summary) > 150:
                    summary = summary[:147] + '...'
        return jsonify({
            "analysis": analysis,
            "summary": summary,
            "regime_snapshot": PALMER_CACHE.get("regime_snapshot"),
            "traffic_lights": PALMER_CACHE.get("traffic_lights"),
            "portfolio_analysis": PALMER_CACHE.get("portfolio_analysis"),
            "events": PALMER_CACHE.get("events"),
            "headlines": PALMER_CACHE.get("headlines"),
            "monte_carlo": PALMER_CACHE.get("monte_carlo"),
            "timestamp": PALMER_CACHE.get("timestamp"),
            "regime_changed": PALMER_CACHE.get("regime_changed", False),
            "regime_change_details": PALMER_CACHE.get("regime_change_details"),
            "error": PALMER_CACHE.get("error"),
        })


@regime_api.route('/api/regime-analysis/force-refresh')
@login_required
def api_regime_analysis_force():
    """Admin-only: Force refresh Palmer's analysis."""
    secret = request.args.get("secret", "")
    if secret != ADMIN_SECRET:
        return jsonify({"error": "Unauthorized. Admin secret required."}), 403
    from dashboard.palmer import refresh_palmer_cache
    from flask import current_app
    refresh_palmer_cache(app=current_app._get_current_object())
    return jsonify({"message": "Palmer analysis refreshed", "timestamp": datetime.now(timezone.utc).isoformat()})


@regime_api.route('/api/regime-performance')
def api_regime_performance():
    """Regime-tagged trade performance: attribution, edge summary, equity curve."""
    try:
        from dashboard.regime_history import get_regime_performance, get_edge_summary
        from dashboard.models import RegimeSnapshot
        from dashboard.trades import get_closed_trades_data
        from dashboard.positions import get_positions_data
        from dashboard.data_fetchers import get_sp500_return_since_inception
        from lox.config import load_settings
        from flask import current_app

        trades_data = get_closed_trades_data()
        closed_trades = trades_data.get("trades", [])
        app_obj = current_app._get_current_object()
        perf = get_regime_performance(app_obj, closed_trades)

        settings = load_settings()
        spy_return = get_sp500_return_since_inception(settings)
        edge = get_edge_summary(app_obj, closed_trades, spy_return=spy_return)
        bands = RegimeSnapshot.get_regime_bands()

        # Build equity curve from closed trades
        trades_by_exit = []
        for t in closed_trades:
            exit_dt = t.get("exit_date")
            if exit_dt is None:
                continue
            if isinstance(exit_dt, str):
                try:
                    exit_dt = datetime.fromisoformat(exit_dt.replace("Z", "+00:00"))
                except Exception:
                    continue
            trades_by_exit.append((exit_dt, t.get("pnl", 0)))
        trades_by_exit.sort(key=lambda x: x[0])

        equity_by_day = {}
        running_pnl = 0
        for exit_dt, pnl in trades_by_exit:
            day = exit_dt.date().isoformat() if hasattr(exit_dt, 'date') else str(exit_dt)[:10]
            running_pnl += pnl
            equity_by_day[day] = running_pnl

        try:
            positions_data = get_positions_data()
            unrealized = sum(p.get("pnl", 0) for p in positions_data.get("positions", []))
            today_str = datetime.now(timezone.utc).date().isoformat()
            equity_by_day[today_str] = running_pnl + unrealized
        except Exception:
            pass

        equity_series = [{"date": d, "pnl": equity_by_day[d]}
                         for d in sorted(equity_by_day.keys())]

        return jsonify({
            "by_regime": perf["by_regime"],
            "edge": edge,
            "regime_bands": bands,
            "equity_series": equity_series,
            "spy_return": spy_return,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "by_regime": {}, "edge": {}})

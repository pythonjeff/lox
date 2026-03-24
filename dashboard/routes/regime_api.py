"""Regime analysis API routes.

Only /api/regime-summary is actively used by the frontend.
Other endpoints are kept but load lazily (no background caches).
"""

from flask import Blueprint, jsonify, request
from flask_login import login_required
from datetime import datetime, timezone

from dashboard.cache import (
    PALMER_CACHE, PALMER_CACHE_LOCK,
    ADMIN_SECRET,
)

regime_api = Blueprint("regime_api", __name__)


@regime_api.route('/api/regime-summary')
def api_regime_summary():
    """Lightweight summary of all regime scores/labels + market pulse.

    This is the ONLY regime endpoint the frontend polls (every 5 min).
    """
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


@regime_api.route('/api/regime-domains')
@login_required
def api_regime_domains():
    """Regime domain indicators (lazy-loaded, no background cache)."""
    try:
        from lox.config import load_settings
        from dashboard.regime_utils import get_regime_domains_data
        settings = load_settings()
        data = get_regime_domains_data(settings)
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


@regime_api.route('/api/regime-analysis')
def api_regime_analysis():
    """Palmer: Cached LLM analysis (lazy — only refreshes on demand)."""
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

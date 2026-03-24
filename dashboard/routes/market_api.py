"""Market data API routes: Monte Carlo, NAV, investors, news."""

from flask import Blueprint, jsonify, request
from flask_login import login_required
from datetime import datetime, timezone, timedelta

from dashboard.cache import (
    MC_CACHE, MC_CACHE_LOCK,
    INVESTORS_CACHE, INVESTORS_CACHE_LOCK, INVESTORS_CACHE_TTL,
    ADMIN_SECRET,
)

market_api = Blueprint("market_api", __name__)


@market_api.route('/api/monte-carlo')
@login_required
def api_monte_carlo():
    """Monte Carlo simulation results. Auto-refreshes hourly."""
    with MC_CACHE_LOCK:
        forecast = MC_CACHE.get("forecast")
        timestamp = MC_CACHE.get("timestamp")
    if forecast is None:
        return jsonify({"error": "Monte Carlo simulation not ready yet. Please wait.", "timestamp": None})
    return jsonify({"forecast": forecast, "timestamp": timestamp})


@market_api.route('/api/monte-carlo/force-refresh')
@login_required
def api_monte_carlo_force():
    """Admin-only: Force refresh Monte Carlo simulation."""
    secret = request.args.get("secret", "")
    if secret != ADMIN_SECRET:
        return jsonify({"error": "Unauthorized. Admin secret required."}), 403
    from dashboard.monte_carlo import refresh_mc_cache
    refresh_mc_cache()
    return jsonify({"message": "Monte Carlo refreshed", "timestamp": datetime.now(timezone.utc).isoformat()})


@market_api.route('/api/nav-history')
def api_nav_history():
    """NAV TWR history for equity curve chart (public)."""
    try:
        from lox.nav.store import read_nav_sheet
        snapshots = read_nav_sheet()
        by_date = {}
        for snap in snapshots:
            date_str = str(snap.ts)[:10]
            by_date[date_str] = {
                "date": date_str,
                "twr_cum_pct": round(snap.twr_cum * 100, 2),
                "equity": round(snap.equity, 2),
            }
        series = [by_date[d] for d in sorted(by_date.keys())]
        response = jsonify({"series": series, "count": len(series)})
        response.headers['Cache-Control'] = 'public, max-age=300'
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "series": []})


@market_api.route('/api/investors')
@login_required
def api_investors():
    """Investor ledger — LIVE updates."""
    with INVESTORS_CACHE_LOCK:
        if INVESTORS_CACHE["data"] and INVESTORS_CACHE["timestamp"]:
            cache_age = (datetime.now(timezone.utc) - INVESTORS_CACHE["timestamp"]).total_seconds()
            if cache_age < INVESTORS_CACHE_TTL:
                response = jsonify(INVESTORS_CACHE["data"])
                response.headers['Cache-Control'] = 'no-cache'
                return response
    try:
        from dashboard.investors import get_investor_data
        data = get_investor_data()
        with INVESTORS_CACHE_LOCK:
            INVESTORS_CACHE["data"] = data
            INVESTORS_CACHE["timestamp"] = datetime.now(timezone.utc)
        response = jsonify(data)
        response.headers['Cache-Control'] = 'no-cache'
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "investors": [], "fund_return": 0, "total_capital": 0})


@market_api.route('/api/market-news')
@login_required
def api_market_news():
    """Portfolio ticker news and economic calendar."""
    try:
        data = _get_market_news_data()
        return jsonify(data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "news": [], "calendar": []})


def _get_market_news_data():
    """Fetch news for portfolio tickers and upcoming economic events."""
    import requests

    news_items = []
    calendar_items = []

    try:
        from lox.config import load_settings
        settings = load_settings()
    except Exception:
        return {"news": news_items, "calendar": calendar_items, "error": "Settings not available"}

    portfolio_tickers = set()
    try:
        from dashboard.positions import get_positions_data
        positions_data = get_positions_data()
        for pos in positions_data.get("positions", []):
            sym = pos.get("symbol", "")
            if len(sym) > 10:
                underlying = sym[:6].rstrip("0123456789 ")
                portfolio_tickers.add(underlying)
            else:
                portfolio_tickers.add(sym)
    except Exception as e:
        print(f"[News] Error getting portfolio tickers: {e}")
        portfolio_tickers = {"SPY", "QQQ"}

    if settings.FMP_API_KEY and portfolio_tickers:
        try:
            tickers_str = ",".join(list(portfolio_tickers)[:5])
            url = "https://financialmodelingprep.com/api/v3/stock_news"
            resp = requests.get(url, params={
                "tickers": tickers_str,
                "limit": 8,
                "apikey": settings.FMP_API_KEY
            }, timeout=10)
            data = resp.json()
            if isinstance(data, list):
                for item in data[:8]:
                    news_items.append({
                        "title": item.get("title", "")[:100],
                        "symbol": item.get("symbol", ""),
                        "source": item.get("site", ""),
                        "url": item.get("url", ""),
                        "time": item.get("publishedDate", "")[:16],
                    })
        except Exception as e:
            print(f"[News] FMP news error: {e}")

    if settings.FMP_API_KEY:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            url = "https://financialmodelingprep.com/api/v3/economic_calendar"
            resp = requests.get(url, params={
                "from": today,
                "to": next_week,
                "apikey": settings.FMP_API_KEY
            }, timeout=10)
            data = resp.json()
            high_impact_keywords = ["CPI", "PPI", "NFP", "FOMC", "Fed", "GDP", "Unemployment", "Retail Sales", "PCE", "Jobs"]
            if isinstance(data, list):
                for item in data:
                    event_name = item.get("event", "")
                    country = item.get("country", "")
                    impact = item.get("impact", "")
                    if country == "US" and (impact == "High" or any(kw in event_name for kw in high_impact_keywords)):
                        calendar_items.append({
                            "event": event_name[:50],
                            "date": item.get("date", "")[:10],
                            "time": item.get("date", "")[11:16] if len(item.get("date", "")) > 11 else "",
                            "previous": item.get("previous", ""),
                            "estimate": item.get("estimate", ""),
                            "impact": impact,
                        })
                calendar_items = calendar_items[:5]
        except Exception as e:
            print(f"[News] FMP calendar error: {e}")

    return {
        "news": news_items,
        "calendar": calendar_items,
        "tickers": list(portfolio_tickers),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

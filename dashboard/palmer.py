"""
Palmer's Analysis Engine for the LOX FUND Dashboard.

LLM-powered portfolio assessment, traffic light system,
regime change detection, and scenario impact analysis.
"""

import time
from datetime import datetime, timezone

from lox.config import load_settings

from dashboard.cache import (
    PALMER_CACHE, PALMER_CACHE_LOCK, PALMER_REFRESH_INTERVAL,
    MC_CACHE, MC_CACHE_LOCK,
)
from dashboard.data_fetchers import (
    get_hy_oas, get_vix, get_10y_yield,
    get_cpi_inflation, get_yield_curve_spread,
)
from dashboard.regime_utils import get_regime_label
from dashboard.news_utils import fetch_macro_headlines
from dashboard.positions import get_positions_data
from dashboard.portfolio import categorize_portfolio_positions
from dashboard.calendar_data import fetch_fed_fiscal_calendar
from dashboard.monte_carlo import calculate_monte_carlo_forecast


# ── Traffic light helpers ──

def get_regime_status(vix_val, hy_val):
    """Determine overall market regime based on VIX and HY spreads."""
    if vix_val is None:
        return "UNKNOWN", "gray"
    if vix_val > 25 or (hy_val and hy_val > 400):
        return "RISK-OFF", "red"
    elif vix_val > 18 or (hy_val and hy_val > 350):
        return "CAUTIOUS", "yellow"
    else:
        return "RISK-ON", "green"


def get_vol_status(vix_val):
    if vix_val is None:
        return "UNKNOWN", "gray"
    if vix_val > 25:
        return "ELEVATED", "red"
    elif vix_val > 18:
        return "MODERATE", "yellow"
    else:
        return "LOW", "green"


def get_credit_status(hy_val):
    if hy_val is None:
        return "UNKNOWN", "gray"
    if hy_val > 400:
        return "STRESSED", "red"
    elif hy_val > 325:
        return "WATCHING", "yellow"
    else:
        return "STABLE", "green"


def get_rates_status(yield_val):
    if yield_val is None:
        return "UNKNOWN", "gray"
    if yield_val > 4.5:
        return "HIGH", "red"
    elif yield_val > 4.0:
        return "ELEVATED", "yellow"
    elif yield_val > 3.5:
        return "MODERATE", "green"
    else:
        return "LOW", "green"


# ── Context builders ──

def build_vix_context(vix_val):
    if not vix_val:
        return ""
    if vix_val < 14:
        return f"VIX at {vix_val:.1f} (5th-15th percentile) — implied vol compressed, hedges cheap but portfolio vol positions face theta decay"
    elif vix_val < 18:
        return f"VIX at {vix_val:.1f} (25th-45th percentile) — normal regime, balanced convexity vs carry trade-off"
    elif vix_val < 22:
        return f"VIX at {vix_val:.1f} (50th-70th percentile) — elevated regime, event risk priced, vol positions approaching profitability zone"
    elif vix_val < 28:
        return f"VIX at {vix_val:.1f} (75th-90th percentile) — stressed regime, term structure likely inverted, vol positions should be delta-hedging"
    else:
        return f"VIX at {vix_val:.1f} (>90th percentile) — crisis regime, vol positions at max vega, consider rolling strikes"


def build_hy_context(hy_val):
    if not hy_val:
        return ""
    if hy_val < 300:
        return f"HY OAS at {hy_val:.0f}bp (tight) — credit risk underpriced, HYG put holders waiting for catalyst"
    elif hy_val < 350:
        return f"HY OAS at {hy_val:.0f}bp (normal) — credit benign but tightening cycle mature, watching for spread decompression"
    elif hy_val < 450:
        return f"HY OAS at {hy_val:.0f}bp (widening) — early stress signals, HYG put deltas expanding, equity-credit correlation rising"
    else:
        return f"HY OAS at {hy_val:.0f}bp (stressed) — credit dislocation, HYG puts deep ITM, contagion risk to equity"


def build_rates_context(yield_val):
    if not yield_val:
        return ""
    if yield_val < 3.5:
        return f"10Y at {yield_val:.2f}% (dovish) — duration tailwind, TLT calls profitable, growth outperforming value"
    elif yield_val < 4.2:
        return f"10Y at {yield_val:.2f}% (neutral) — Fed at terminal, duration-sensitive plays range-bound"
    elif yield_val < 4.7:
        return f"10Y at {yield_val:.2f}% (restrictive) — term premium rebuilding, growth/tech equity multiples under pressure"
    else:
        return f"10Y at {yield_val:.2f}% (hawkish extreme) — fiscal supply pressure, equity multiple compression accelerating"


def build_scenario_impacts(scenario_matrix):
    """Build scenario impact descriptions from the scenario matrix."""
    impacts = []

    risk_off_winners = scenario_matrix.get("risk_off_spike", {}).get("winners", [])
    risk_off_losers = scenario_matrix.get("risk_off_spike", {}).get("losers", [])
    if risk_off_winners or risk_off_losers:
        impact = "RISK-OFF SPIKE (VIX +10, SPX -10%): "
        parts = []
        if risk_off_winners:
            parts.append(f"Winners: {', '.join(list(set(risk_off_winners))[:3])}")
        if risk_off_losers:
            parts.append(f"Losers: {', '.join(list(set(risk_off_losers))[:3])}")
        impacts.append(impact + " | ".join(parts))

    rally_winners = scenario_matrix.get("goldilocks_rally", {}).get("winners", [])
    rally_losers = scenario_matrix.get("goldilocks_rally", {}).get("losers", [])
    if rally_winners or rally_losers:
        impact = "GOLDILOCKS RALLY (VIX -5, SPX +5%): "
        parts = []
        if rally_winners:
            parts.append(f"Winners: {', '.join(list(set(rally_winners))[:3])}")
        if rally_losers:
            parts.append(f"Losers: {', '.join(list(set(rally_losers))[:3])}")
        impacts.append(impact + " | ".join(parts))

    credit_winners = scenario_matrix.get("credit_stress", {}).get("winners", [])
    if credit_winners:
        impacts.append(f"CREDIT STRESS (HY +150bp): Winners: {', '.join(list(set(credit_winners))[:3])}")

    return impacts


# ── Regime change detection ──

def detect_regime_change(old_lights, new_lights):
    """Detect if any traffic light changed color."""
    if not old_lights or not new_lights:
        return False, None
    changes = []
    for key in ["regime", "volatility", "credit", "rates"]:
        old_color = old_lights.get(key, {}).get("color") if old_lights.get(key) else None
        new_color = new_lights.get(key, {}).get("color") if new_lights.get(key) else None
        old_label = old_lights.get(key, {}).get("label") if old_lights.get(key) else None
        new_label = new_lights.get(key, {}).get("label") if new_lights.get(key) else None
        if old_color and new_color and old_color != new_color:
            severity_order = {"green": 0, "yellow": 1, "red": 2}
            old_sev = severity_order.get(old_color, 0)
            new_sev = severity_order.get(new_color, 0)
            direction = "worsening" if new_sev > old_sev else "improving"
            changes.append({
                "indicator": key.upper(),
                "from": old_label,
                "to": new_label,
                "direction": direction,
            })
    if changes:
        return True, changes
    return False, None


# ── Core Palmer analysis ──

def generate_palmer_analysis():
    """Generate Palmer's portfolio assessment (called by cache refresh)."""
    try:
        settings = load_settings()
    except Exception as e:
        print(f"[Palmer] Settings load error: {e}")
        return {"error": f"Settings error: {e}", "analysis": None}

    if not settings or not hasattr(settings, 'openai_api_key') or not settings.openai_api_key:
        return {"error": "OpenAI API key not configured", "analysis": None}

    # Fetch macro data
    hy_oas = get_hy_oas(settings, refresh=True)
    vix = get_vix(settings)
    yield_10y = get_10y_yield(settings)
    cpi = get_cpi_inflation(settings, refresh=True)
    yield_curve = get_yield_curve_spread(settings, refresh=True)

    fed_fiscal_events, calendar_date = fetch_fed_fiscal_calendar(settings)
    positions_data = get_positions_data()

    # Portfolio tickers for headline fallback
    portfolio_tickers = []
    for p in positions_data.get("positions", []):
        opt = p.get("opt_info")
        if opt:
            portfolio_tickers.append(opt.get("underlying"))
        else:
            portfolio_tickers.append(p.get("symbol"))

    headlines = fetch_macro_headlines(settings, portfolio_tickers=portfolio_tickers, limit=3)

    regime_snapshot = {
        "hy_oas_bps": hy_oas.get("value") if hy_oas else None,
        "vix": vix.get("value") if vix else None,
        "yield_10y": yield_10y.get("value") if yield_10y else None,
        "cpi_yoy": cpi.get("value") if cpi else None,
        "yield_curve_2s10s": yield_curve.get("value") if yield_curve else None,
        "portfolio_nav": positions_data.get("nav_equity"),
        "portfolio_pnl": positions_data.get("total_pnl"),
    }

    vix_val = regime_snapshot.get("vix")
    hy_val = regime_snapshot.get("hy_oas_bps")
    yield_val = regime_snapshot.get("yield_10y")
    cpi_val = regime_snapshot.get("cpi_yoy")
    curve_val = regime_snapshot.get("yield_curve_2s10s")

    regime_label, regime_color = get_regime_status(vix_val, hy_val)
    vol_label, vol_color = get_vol_status(vix_val)
    credit_label, credit_color = get_credit_status(hy_val)
    rates_label, rates_color = get_rates_status(yield_val)

    # Format events
    events_display = {
        "date": calendar_date or datetime.now().strftime("%A, %B %d, %Y"),
        "releases": [],
    }
    for e in fed_fiscal_events[:15]:
        events_display["releases"].append({
            "time": e.get("time", ""),
            "event": e["event"],
            "actual": e.get("actual"),
            "previous": e.get("previous"),
            "estimate": e.get("estimate"),
            "surprise_direction": e.get("surprise_direction"),
            "url": e.get("url", ""),
        })

    headlines_display = []
    for h in headlines[:4]:
        headlines_display.append({
            "headline": h["headline"],
            "source": h["source"],
            "time": h["time"],
            "ticker": h.get("ticker", ""),
            "url": h.get("url", ""),
        })

    # LLM portfolio assessment
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)

        portfolio_positions = positions_data.get("positions", []) if positions_data else []
        portfolio_analysis = categorize_portfolio_positions(portfolio_positions)
        scenario_matrix = portfolio_analysis.get("scenario_matrix", {})

        vix_context = build_vix_context(vix_val)
        hy_context = build_hy_context(hy_val)
        rate_context = build_rates_context(yield_val)
        scenario_impacts = build_scenario_impacts(scenario_matrix)

        events_context = ""
        if fed_fiscal_events:
            event_details = []
            for e in fed_fiscal_events[:6]:
                evt_name = e.get("event", "")
                actual = e.get("actual")
                estimate = e.get("estimate")
                if actual is not None and estimate is not None:
                    try:
                        diff = float(actual) - float(estimate)
                        direction = "beat" if diff > 0 else "miss"
                        event_details.append(f"{evt_name}: {actual} vs {estimate} ({direction})")
                    except Exception:
                        event_details.append(f"{evt_name}: {actual}")
                elif actual is not None:
                    event_details.append(f"{evt_name}: {actual}")
                else:
                    event_details.append(f"{evt_name} (pending)")
            events_context = " | ".join(event_details)

        news_context = ""
        if headlines:
            news_items = [f"• {h.get('headline', '')}" for h in headlines[:5] if h]
            news_context = chr(10).join(news_items)

        position_lines = []
        for p in portfolio_positions:
            symbol = p.get("symbol", "")
            qty = p.get("qty", 0)
            pnl = p.get("pnl", 0)
            mv = p.get("market_value", 0)
            opt = p.get("opt_info")
            if opt:
                underlying = opt.get("underlying", "")
                opt_type = opt.get("opt_type", opt.get("type", ""))
                strike = opt.get("strike", "")
                expiry = opt.get("expiry", "")
                position_lines.append(
                    f"  {qty:+d} {underlying} {strike}{opt_type} exp {expiry} | MV ${mv:.0f} | P&L ${pnl:+.0f}"
                )
            elif symbol:
                position_lines.append(f"  {qty:+d} {symbol} | MV ${mv:.0f} | P&L ${pnl:+.0f}")
        positions_detail = chr(10).join(position_lines) if position_lines else "No positions"

        prompt = f"""You are the portfolio manager of LOX Fund, a small macro options fund.
Your job: assess whether our current book is well-positioned given today's market moves and news.
Answer the question: "Are we in a good spot or a bad spot right now, and what should we watch?"

{'='*50}
CURRENT MARKET LEVELS
{'='*50}
• {vix_context or "VIX data unavailable"}
• {hy_context or "Credit spread data unavailable"}
• {rate_context or "Rates data unavailable"}
• CPI YoY: {f"{cpi_val:.1f}%" if cpi_val else "N/A"}
• Yield Curve 2s10s: {f"{curve_val:.0f}bp" if curve_val else "N/A"}

{'='*50}
TODAY'S NEWS & EVENTS
{'='*50}
{news_context if news_context else "No headlines available"}

Calendar: {events_context if events_context else "No releases today"}

{'='*50}
OUR POSITIONS (LIVE)
{'='*50}
{positions_detail}

Total Unrealized P&L: ${positions_data.get("total_pnl", 0):+,.0f}
NAV: ${positions_data.get("nav_equity", 0):,.0f}
Cash: ${positions_data.get("cash_available", 0):,.0f}

Category summary: {portfolio_analysis.get("summary", "N/A")}

Scenario impacts:
{chr(10).join(scenario_impacts) if scenario_impacts else "N/A"}

{'='*50}
INSTRUCTIONS
{'='*50}
Write a 4-6 sentence portfolio assessment. Structure it as:

1. VERDICT: Open with a one-sentence verdict — is the book in a good or bad position today?
   Ground this in what's actually moving (news, data, VIX direction).

2. WHAT'S WORKING: Which specific positions benefit from today's market conditions?
   Reference the actual tickers, strikes, and P&L where relevant.

3. WHAT'S AT RISK: Which positions are under pressure or bleeding theta without a catalyst?

4. WATCH: Name 1-2 specific catalysts or levels that would materially shift our P&L
   this week (e.g., "CPI print Thursday", "VIX below 18 kills our vol book").

STYLE:
- Write as a PM talking to the IC, not a news anchor
- Be direct: "Our XRT puts are printing" or "The AMZN calls are dead weight"
- Reference actual position P&L when making points
- No hedging words, no sentiment fluff
- Maximum 6 sentences"""

        response = client.chat.completions.create(
            model=settings.openai_model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        insight = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Palmer] LLM error: {e}")
        insight = "Market analysis temporarily unavailable."

    # MC forecast from cache (or fallback)
    with MC_CACHE_LOCK:
        mc_forecast = MC_CACHE.get("forecast")
    if mc_forecast is None:
        mc_forecast = calculate_monte_carlo_forecast(
            vix_val, hy_val, regime_label,
            positions_data=positions_data.get("positions", []),
            cash_available=positions_data.get("cash_available", 0),
        )

    try:
        portfolio_analysis_output = categorize_portfolio_positions(positions_data.get("positions", []))
    except Exception:
        portfolio_analysis_output = {"summary": "Analysis unavailable", "scenario_matrix": {}}

    return {
        "analysis": insight,
        "regime_snapshot": regime_snapshot,
        "traffic_lights": {
            "regime": {"label": regime_label, "color": regime_color},
            "volatility": {"label": vol_label, "color": vol_color, "value": f"VIX {vix_val:.1f}" if vix_val else "N/A"},
            "credit": {"label": credit_label, "color": credit_color, "value": f"{hy_val:.0f}bp" if hy_val else "N/A"},
            "rates": {"label": rates_label, "color": rates_color, "value": f"{yield_val:.2f}%" if yield_val else "N/A"},
            "inflation": {"label": cpi.get("context") if cpi else "N/A", "color": cpi.get("color") if cpi else "gray", "value": f"{cpi_val:.1f}%" if cpi_val else "N/A"},
            "yield_curve": {"label": yield_curve.get("context") if yield_curve else "N/A", "color": yield_curve.get("color") if yield_curve else "gray", "value": f"{curve_val:.0f}bp" if curve_val else "N/A"},
        },
        "portfolio_analysis": {
            "summary": portfolio_analysis_output.get("summary", ""),
            "scenario_matrix": portfolio_analysis_output.get("scenario_matrix", {}),
        },
        "events": events_display,
        "headlines": headlines_display,
        "monte_carlo": mc_forecast,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Cache refresh ──

def refresh_palmer_cache(app=None):
    """Refresh Palmer's cached analysis."""
    print(f"[Palmer] Refreshing analysis at {datetime.now(timezone.utc).isoformat()}")
    try:
        result = generate_palmer_analysis()
        with PALMER_CACHE_LOCK:
            old_lights = PALMER_CACHE.get("traffic_lights")
            new_lights = result.get("traffic_lights")
            changed, change_details = detect_regime_change(old_lights, new_lights)

            PALMER_CACHE["prev_traffic_lights"] = old_lights
            PALMER_CACHE["analysis"] = result.get("analysis")
            PALMER_CACHE["regime_snapshot"] = result.get("regime_snapshot")
            PALMER_CACHE["traffic_lights"] = new_lights
            PALMER_CACHE["portfolio_analysis"] = result.get("portfolio_analysis")
            PALMER_CACHE["events"] = result.get("events")
            PALMER_CACHE["headlines"] = result.get("headlines")
            PALMER_CACHE["monte_carlo"] = result.get("monte_carlo")
            PALMER_CACHE["timestamp"] = result.get("timestamp")
            PALMER_CACHE["last_refresh"] = datetime.now(timezone.utc)
            PALMER_CACHE["error"] = result.get("error")
            PALMER_CACHE["regime_changed"] = changed
            PALMER_CACHE["regime_change_details"] = change_details

            if changed:
                print(f"[Palmer] REGIME CHANGE DETECTED: {change_details}")

        # Persist daily regime snapshot
        try:
            from dashboard.regime_history import snapshot_today
            rs = result.get("regime_snapshot", {})
            if app:
                snapshot_today(
                    app,
                    vix_val=rs.get("vix"),
                    hy_val=rs.get("hy_oas_bps"),
                    yield_val=rs.get("yield_10y"),
                    cpi_val=rs.get("cpi_yoy"),
                    curve_val=rs.get("yield_curve_2s10s"),
                )
        except Exception as snap_err:
            print(f"[Palmer] Regime snapshot save error: {snap_err}")

        print(f"[Palmer] Cache refreshed successfully")
    except Exception as e:
        import traceback
        traceback.print_exc()
        with PALMER_CACHE_LOCK:
            PALMER_CACHE["error"] = str(e)
        print(f"[Palmer] Cache refresh failed: {e}")


def palmer_background_refresh(app=None):
    """Background thread loop — refreshes Palmer every 30 minutes."""
    while True:
        try:
            time.sleep(PALMER_REFRESH_INTERVAL)
            refresh_palmer_cache(app=app)
        except Exception as e:
            print(f"[Palmer] Background refresh error: {e}")
            time.sleep(60)

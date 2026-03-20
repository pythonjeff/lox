"""Ticker data fetching — FMP API calls for price, fundamentals, IV, peers."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def fetch_price_data(settings, symbol: str) -> dict:
    """Fetch historical price data."""
    try:
        import requests
        from datetime import datetime, timedelta

        if not settings.fmp_api_key:
            return {}

        # Get quote
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
        resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=15)
        quote = {}
        if resp.ok:
            data = resp.json()
            if data and isinstance(data, list):
                quote = data[0]

        # Get historical
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
        resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=15)
        historical = []
        if resp.ok:
            data = resp.json()
            historical = data.get("historical", [])[:756]  # 3 years

        return {
            "symbol": symbol,
            "quote": quote,
            "historical": historical,
        }
    except Exception:
        logger.debug("Failed to fetch price data for %s", symbol, exc_info=True)
        return {}


def fetch_fundamentals(settings, symbol: str) -> dict:
    """Fetch fundamental data (auto-detects ETFs vs stocks)."""
    try:
        import requests

        if not settings.fmp_api_key:
            return {}

        result = {}

        # Company profile (works for both stocks and ETFs)
        url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
        resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=15)
        if resp.ok:
            data = resp.json()
            if data and isinstance(data, list):
                result["profile"] = data[0]

        # Check if ETF — fetch ETF-specific data instead of stock ratios
        is_etf = result.get("profile", {}).get("isEtf", False)

        if is_etf:
            # ETF info (AUM, expense ratio, holdings count, etc.)
            url = "https://financialmodelingprep.com/api/v4/etf-info"
            resp = requests.get(url, params={"symbol": symbol, "apikey": settings.fmp_api_key}, timeout=15)
            if resp.ok:
                data = resp.json()
                if data and isinstance(data, list):
                    result["etf_info"] = data[0]
        else:
            # Stock: key metrics and ratios
            url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}"
            resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=15)
            if resp.ok:
                data = resp.json()
                if data and isinstance(data, list):
                    result["metrics"] = data[0]

            url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}"
            resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=15)
            if resp.ok:
                data = resp.json()
                if data and isinstance(data, list):
                    result["ratios"] = data[0]

            # Income statement (revenue, gross profit for margin and growth)
            url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}"
            resp = requests.get(url, params={"apikey": settings.fmp_api_key, "limit": 3}, timeout=15)
            if resp.ok:
                data = resp.json()
                if data and isinstance(data, list):
                    result["income_statement"] = data

            # Cash flow (FCF)
            url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}"
            resp = requests.get(url, params={"apikey": settings.fmp_api_key, "limit": 2}, timeout=15)
            if resp.ok:
                data = resp.json()
                if data and isinstance(data, list):
                    result["cash_flow"] = data

            # Balance sheet (cash, debt)
            url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}"
            resp = requests.get(url, params={"apikey": settings.fmp_api_key, "limit": 1}, timeout=15)
            if resp.ok:
                data = resp.json()
                if data and isinstance(data, list):
                    result["balance_sheet"] = data[0]

            # Revenue growth (YoY)
            url = f"https://financialmodelingprep.com/api/v3/income-statement-growth/{symbol}"
            resp = requests.get(url, params={"apikey": settings.fmp_api_key, "limit": 2}, timeout=15)
            if resp.ok:
                data = resp.json()
                if data and isinstance(data, list):
                    result["income_growth"] = data

        return result
    except Exception:
        logger.debug("Failed to fetch fundamentals for %s", symbol, exc_info=True)
        return {}


def fetch_atm_implied_vol(settings, symbol: str, current_price: float | None) -> float | None:
    """Fetch ATM implied vol from options chain (Polygon) if available. Returns annualized IV as decimal or None."""
    if not current_price or current_price <= 0:
        return None
    try:
        from datetime import date, timedelta
        from lox.data.polygon import fetch_options_chain_polygon
        expiry_lte = (date.today() + timedelta(days=60)).isoformat()
        expiry_gte = (date.today() + timedelta(days=20)).isoformat()
        chain = fetch_options_chain_polygon(
            settings, symbol, expiration_date_gte=expiry_gte, expiration_date_lte=expiry_lte, limit=100
        )
        if not chain:
            return None
        # Near ATM: strike within ~5% of spot
        ivs = []
        for c in chain:
            if c.iv is None:
                continue
            strike = c.strike
            if strike and abs(strike - current_price) / current_price <= 0.05:
                ivs.append(c.iv)
        if not ivs:
            return None
        return float(sum(ivs) / len(ivs))
    except Exception:
        logger.debug("Failed to fetch ATM IV for %s", symbol, exc_info=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# E-mini futures depth — maps index ETFs to their CME futures
# ─────────────────────────────────────────────────────────────────────────────

FUTURES_ETF_MAP: dict[str, dict] = {
    "SPY": {"root": "ES", "name": "E-mini S&P 500", "multiplier": 50},
    "VOO": {"root": "ES", "name": "E-mini S&P 500", "multiplier": 50},
    "IVV": {"root": "ES", "name": "E-mini S&P 500", "multiplier": 50},
    "QQQ": {"root": "NQ", "name": "E-mini Nasdaq 100", "multiplier": 20},
    "TQQQ": {"root": "NQ", "name": "E-mini Nasdaq 100", "multiplier": 20},
    "IWM": {"root": "RTY", "name": "E-mini Russell 2000", "multiplier": 50},
    "DIA": {"root": "YM", "name": "E-mini Dow", "multiplier": 5},
}

_MONTH_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}


def _next_quarterly_expiry(from_date):
    """Next CME quarterly futures expiry (3rd Friday of Mar/Jun/Sep/Dec)."""
    from datetime import date, timedelta

    for year_offset in range(2):
        for month in (3, 6, 9, 12):
            year = from_date.year + year_offset
            first_day = date(year, month, 1)
            days_to_fri = (4 - first_day.weekday()) % 7
            third_friday = first_day + timedelta(days=days_to_fri + 14)
            if third_friday > from_date:
                return third_friday
    return from_date


def fetch_futures_depth(
    settings,
    symbol: str,
    etf_price: float | None,
    div_yield_pct: float | None,
    technicals: dict | None = None,
) -> dict | None:
    """Fetch E-mini futures depth: health assessment, COT, and cost-of-carry basis.

    Combines VIX-based depth proxy, roll window detection, COT crowding,
    and technical context into a composite depth health rating.
    """
    if symbol not in FUTURES_ETF_MAP:
        return None

    from datetime import date, timedelta

    technicals = technicals or {}
    futures_info = FUTURES_ETF_MAP[symbol]
    root = futures_info["root"]
    today = date.today()

    # Front and back quarterly contracts
    front_expiry = _next_quarterly_expiry(today)
    front_dte = (front_expiry - today).days
    front_code = _MONTH_CODES.get(front_expiry.month, "?")
    front_sym = f"{root}{front_code}{front_expiry.year % 100}"

    back_expiry = _next_quarterly_expiry(front_expiry + timedelta(days=1))
    back_dte = (back_expiry - today).days
    back_code = _MONTH_CODES.get(back_expiry.month, "?")
    back_sym = f"{root}{back_code}{back_expiry.year % 100}"

    # Active month shifts to back contract during roll window
    in_roll = front_dte <= 8
    active_contract = back_sym if in_roll else front_sym
    active_dte = back_dte if in_roll else front_dte

    result: dict = {
        "name": futures_info["name"],
        "root": root,
        "multiplier": futures_info["multiplier"],
        "front_contract": front_sym,
        "front_dte": front_dte,
        "back_contract": back_sym,
        "back_dte": back_dte,
        "active_contract": active_contract,
        "active_dte": active_dte,
        "in_roll": in_roll,
    }

    # ── VIX from FRED (depth proxy: ES book depth inversely tracks VIX) ──
    vix = None
    vix_pctile = None
    risk_free_rate = None

    if settings.FRED_API_KEY:
        try:
            from lox.data.fred import FredClient
            import numpy as np

            fred = FredClient(api_key=settings.FRED_API_KEY)

            # VIX
            vdf = fred.fetch_series("VIXCLS", start_date="2024-01-01")
            if vdf is not None and not vdf.empty:
                vdf = vdf.dropna(subset=["value"])
                vix = float(vdf.iloc[-1]["value"])
                result["vix"] = vix
                result["vix_date"] = str(vdf.iloc[-1]["date"])[:10]
                # 1-year percentile
                last_252 = vdf.tail(252)["value"].astype(float).values
                if len(last_252) >= 50:
                    vix_pctile = float(np.sum(last_252 <= vix) / len(last_252) * 100)
                    result["vix_pctile"] = vix_pctile

            # Fed funds rate for basis
            fdf = fred.fetch_series("DFF", start_date="2025-01-01")
            if fdf is not None and not fdf.empty:
                risk_free_rate = float(fdf.dropna(subset=["value"]).iloc[-1]["value"]) / 100
                result["risk_free_rate"] = risk_free_rate
        except Exception:
            logger.debug("Failed to fetch FRED data for futures depth", exc_info=True)

    # ── Cost-of-carry basis ──────────────────────────────────────────────
    div_yield = (div_yield_pct or 0) / 100
    result["div_yield"] = div_yield

    if etf_price and etf_price > 0 and risk_free_rate is not None:
        carry = risk_free_rate - div_yield
        result["net_carry_ann"] = carry
        result["front_basis_bps"] = carry * (front_dte / 365) * 10000
        result["back_basis_bps"] = carry * (back_dte / 365) * 10000
        result["front_fair_value"] = etf_price * (1 + carry * front_dte / 365)
        result["back_fair_value"] = etf_price * (1 + carry * back_dte / 365)

    # ── CFTC COT speculative positioning ─────────────────────────────────
    if settings.fmp_api_key:
        try:
            from lox.positioning.data import fetch_cot_data

            cot_net, cot_z, cot_dates = fetch_cot_data(settings, symbols=[root])
            result["cot_net_spec"] = cot_net.get(root)
            result["cot_z_score"] = cot_z.get(root)
            result["cot_date"] = cot_dates.get(root)
        except Exception:
            logger.debug("Failed to fetch COT data for %s", root, exc_info=True)

    # ── Depth health assessment ──────────────────────────────────────────
    # ES top-of-book depth inversely correlates with VIX (well-documented).
    # We combine VIX level, roll window, HV, COT crowding, and technical
    # weakness into a composite health rating.
    flags: list[tuple[str, str]] = []  # (severity, message)

    if vix is not None:
        if vix >= 30:
            flags.append(("critical", f"VIX at {vix:.1f} — order books severely thin"))
        elif vix >= 25:
            flags.append(("warning", f"VIX at {vix:.1f} — books thinning sharply"))
        elif vix >= 20:
            flags.append(("caution", f"VIX at {vix:.1f} — depth modestly reduced"))
        if vix_pctile is not None and vix_pctile >= 80:
            flags.append(("caution", f"VIX at {vix_pctile:.0f}th percentile (1y) — elevated regime"))

    if in_roll:
        flags.append(("warning", f"{front_sym} expires in {front_dte}d — roll window, front-month liquidity draining"))

    hv = technicals.get("volatility_30d") or technicals.get("volatility")
    if hv is not None:
        if hv > 30:
            flags.append(("warning", f"HV {hv:.0f}% ann. — market makers widen quotes"))
        elif hv > 20:
            flags.append(("caution", f"HV {hv:.0f}% ann. — above-average realized vol"))

    cot_z = result.get("cot_z_score")
    if cot_z is not None and abs(cot_z) > 1.5:
        crowd_dir = "long" if cot_z > 0 else "short"
        flags.append(("caution", f"Specs crowded {crowd_dir} (z={cot_z:+.1f}) — unwind risk if triggered"))

    rsi = technicals.get("rsi")
    if rsi is not None and rsi < 30:
        flags.append(("caution", "RSI oversold — forced selling / margin call risk elevates fragility"))

    trend_label = technicals.get("trend_label") or ""
    if "Below all" in trend_label:
        flags.append(("caution", "Below all major MAs — trend followers adding pressure"))
    elif "Below 50 SMA" in trend_label:
        flags.append(("caution", "Below 50 SMA — intermediate trend weak"))

    vol_vs_avg = technicals.get("vol_vs_avg")
    if vol_vs_avg is not None and vol_vs_avg < 0.7:
        flags.append(("caution", f"Volume {vol_vs_avg:.1f}x avg — thin participation"))

    # Composite health rating
    n_critical = sum(1 for s, _ in flags if s == "critical")
    n_warning = sum(1 for s, _ in flags if s == "warning")
    n_caution = sum(1 for s, _ in flags if s == "caution")

    if n_critical > 0:
        health = "CRITICAL"
    elif n_warning >= 2 or (n_warning >= 1 and n_caution >= 2):
        health = "FRAGILE"
    elif n_warning >= 1 or n_caution >= 3:
        health = "THINNING"
    elif n_caution >= 1:
        health = "ADEQUATE"
    else:
        health = "HEALTHY"

    result["depth_health"] = health
    result["depth_flags"] = flags

    return result


def fetch_peers(settings, symbol: str) -> list[str]:
    """Fetch peer symbols from FMP (same sector, similar cap). Returns up to 5 symbols."""
    try:
        import requests
        url = "https://financialmodelingprep.com/api/v4/stock_peers"
        resp = requests.get(url, params={"symbol": symbol, "apikey": settings.fmp_api_key}, timeout=10)
        if not resp.ok:
            return []
        data = resp.json()
        if not data or not isinstance(data, list):
            return []
        peers = data[0].get("peersList", []) if isinstance(data[0], dict) else []
        return [p for p in peers if p and p != symbol][:5]
    except Exception:
        logger.debug("Failed to fetch peers for %s", symbol, exc_info=True)
        return []

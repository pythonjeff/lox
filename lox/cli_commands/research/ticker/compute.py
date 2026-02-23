"""Ticker computation — technicals, ETF flow metrics, refinancing wall."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_technicals(price_data: dict) -> dict:
    """Compute technical indicators from price data."""
    if not price_data or not price_data.get("historical"):
        return {}

    try:
        import pandas as pd
        import numpy as np

        historical = price_data["historical"]
        df = pd.DataFrame(historical)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        # Current price
        current = closes[-1] if len(closes) > 0 else 0

        # Moving averages
        ma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else None
        ma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else None
        ma_200 = np.mean(closes[-200:]) if len(closes) >= 200 else None

        # RSI (14-day)
        rsi = None
        if len(closes) >= 15:
            deltas = np.diff(closes[-15:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

        # 52-week high/low
        high_52w = max(highs) if len(highs) > 0 else None
        low_52w = min(lows) if len(lows) > 0 else None

        # Support/Resistance (20d swing low/high) with method labels
        support = min(lows[-20:]) if len(lows) >= 20 else None
        resistance = max(highs[-20:]) if len(highs) >= 20 else None
        support_method = "20d swing low" if support is not None else None
        resistance_method = "20d swing high" if resistance is not None else None

        # Volatility: 30d for display (CFA-style), 20d for backward compat
        ret_30 = np.diff(closes[-31:]) / closes[-31:-1] if len(closes) >= 31 else np.array([])
        ret_20 = np.diff(closes[-21:]) / closes[-21:-1] if len(closes) >= 21 else np.array([])
        volatility_30d = float(np.std(ret_30) * np.sqrt(252) * 100) if len(ret_30) > 0 else None
        volatility = float(np.std(ret_20) * np.sqrt(252) * 100) if len(ret_20) > 0 else None

        # Trend: factual label from MA positions (no editorializing)
        trend = None
        trend_label = "N/A"
        if ma_50 is not None and current:
            trend = "bullish" if current > ma_50 else "bearish"
            above_20 = (ma_20 is not None and current > ma_20)
            above_50 = current > ma_50
            above_200 = (ma_200 is not None and current > ma_200)
            if above_20 and above_50 and above_200:
                trend_label = "Above all major MAs (20/50/200)"
            elif above_50 and above_200 and not above_20:
                trend_label = "Above 50/200 SMA, below 20 SMA"
            elif above_50 and not above_200 and ma_200 is not None:
                trend_label = "Above 50 SMA, below 200 SMA"
            elif not above_50 and above_200 and ma_200 is not None:
                trend_label = "Below 50 SMA, above 200 SMA"
            elif not above_50 and not above_200:
                trend_label = "Below all major MAs (20/50/200)"
            else:
                trend_label = "Above 20/50 SMA" if above_20 and above_50 else "Below 20/50 SMA"

        # 50/200 SMA crossover (Golden Cross / Death Cross)
        sma_crossover = None
        if ma_50 is not None and ma_200 is not None and len(closes) >= 200:
            if ma_50 > ma_200:
                sma_crossover = "Golden Cross (50 > 200)"
            else:
                sma_crossover = "Death Cross (50 < 200)"

        # MACD (12, 26, 9)
        macd_signal = None
        if len(closes) >= 34:
            ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean().values
            ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean().values
            macd_line = ema12 - ema26
            signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
            if macd_line[-1] > signal_line[-1]:
                macd_signal = "Bullish (MACD above signal)"
            else:
                macd_signal = "Bearish (MACD below signal)"

        # Volume context
        if "volume" in df.columns:
            vol_series = df["volume"].astype(float)
            avg_volume = float(vol_series.tail(20).mean()) if len(vol_series) >= 20 else None
            today_volume = float(vol_series.iloc[-1]) if len(vol_series) > 0 else None
            vol_vs_avg = (today_volume / avg_volume) if avg_volume and avg_volume > 0 else None
        else:
            avg_volume = today_volume = vol_vs_avg = None

        return {
            "current": current,
            "ma_20": ma_20,
            "ma_50": ma_50,
            "ma_200": ma_200,
            "rsi": rsi,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "support": support,
            "resistance": resistance,
            "support_method": support_method,
            "resistance_method": resistance_method,
            "volatility": volatility,
            "volatility_30d": volatility_30d,
            "trend": trend,
            "trend_label": trend_label,
            "sma_crossover": sma_crossover,
            "macd_signal": macd_signal,
            "avg_volume": avg_volume,
            "today_volume": today_volume,
            "vol_vs_avg": vol_vs_avg,
            "df": df,
        }
    except Exception:
        logger.debug("Failed to compute technicals", exc_info=True)
        return {}


def compute_refinancing_wall(settings, symbol: str) -> dict | None:
    """Compute refinancing wall summary for LLM context."""
    import re
    import requests
    from collections import defaultdict
    from datetime import datetime

    try:
        url = f"https://financialmodelingprep.com/api/v3/etf-holder/{symbol}"
        resp = requests.get(url, params={"apikey": settings.fmp_api_key}, timeout=20)
        if not resp.ok:
            return None
        holdings = resp.json()
    except Exception:
        return None

    date_pattern = re.compile(r"(\d{2}/\d{2}/(\d{4}))\s*$")
    by_year: dict[int, float] = defaultdict(float)
    total_mv = 0
    current_year = datetime.now().year

    for h in holdings:
        name = h.get("name", "")
        mv = h.get("marketValue", 0) or 0
        match = date_pattern.search(name)
        if match:
            year = int(match.group(2))
            by_year[year] += mv
            total_mv += mv

    if total_mv <= 0:
        return None

    near_term = sum(v for y, v in by_year.items() if y <= current_year + 2)
    mid_term = sum(v for y, v in by_year.items() if current_year + 3 <= y <= current_year + 5)

    wall = {}
    for y in sorted(by_year):
        if current_year <= y <= current_year + 10:
            wall[str(y)] = f"${by_year[y] / 1e9:.2f}B ({by_year[y] / total_mv * 100:.1f}%)"

    return {
        "maturity_by_year": wall,
        "near_term_pct": round(near_term / total_mv * 100, 1),
        "mid_term_pct": round(mid_term / total_mv * 100, 1),
        "total_bonds_parsed": sum(1 for _ in by_year),
        "total_market_value": f"${total_mv / 1e9:.1f}B",
        "peak_year": str(max(by_year, key=by_year.get)),
        "peak_year_pct": round(by_year[max(by_year, key=by_year.get)] / total_mv * 100, 1),
    }


def compute_flow_context(price_data: dict) -> dict | None:
    """Compute ETF flow metrics for LLM context."""
    import numpy as np

    historical = price_data.get("historical", [])
    if len(historical) < 21:
        return None

    hist = list(reversed(historical[:60]))
    closes = [h["close"] for h in hist]
    volumes = [h["volume"] for h in hist]
    highs = [h["high"] for h in hist]
    lows = [h["low"] for h in hist]

    # MFI
    mfi = None
    if len(hist) >= 15:
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        pos_mf = sum(typical_prices[i] * volumes[i] for i in range(-14, 0) if typical_prices[i] > typical_prices[i - 1])
        neg_mf = sum(typical_prices[i] * volumes[i] for i in range(-14, 0) if typical_prices[i] <= typical_prices[i - 1])
        mfi = 100 - (100 / (1 + pos_mf / neg_mf)) if neg_mf > 0 else (100.0 if pos_mf > 0 else 50.0)

    # Dollar volume
    dv = [c * v for c, v in zip(closes, volumes)]
    dv_5d = float(np.mean(dv[-5:])) if len(dv) >= 5 else None
    dv_20d = float(np.mean(dv[-20:])) if len(dv) >= 20 else None

    # Up/Down ratio
    if len(closes) >= 21:
        up_vol = sum(v for c1, c0, v in zip(closes[-20:], closes[-21:-1], volumes[-20:]) if c1 > c0)
        dn_vol = sum(v for c1, c0, v in zip(closes[-20:], closes[-21:-1], volumes[-20:]) if c1 <= c0)
        flow_ratio = up_vol / dn_vol if dn_vol > 0 else 10.0
    else:
        flow_ratio = 1.0

    if flow_ratio > 1.3:
        signal = "STRONG INFLOWS"
    elif flow_ratio > 1.1:
        signal = "INFLOWS"
    elif flow_ratio < 0.7:
        signal = "STRONG OUTFLOWS"
    elif flow_ratio < 0.9:
        signal = "OUTFLOWS"
    else:
        signal = "BALANCED"

    return {
        "net_flow_signal_20d": signal,
        "up_down_volume_ratio": round(flow_ratio, 2),
        "mfi_14d": round(mfi, 1) if mfi else None,
        "dollar_vol_5d_avg": f"${dv_5d / 1e6:.0f}M" if dv_5d else None,
        "dollar_vol_20d_avg": f"${dv_20d / 1e6:.0f}M" if dv_20d else None,
    }

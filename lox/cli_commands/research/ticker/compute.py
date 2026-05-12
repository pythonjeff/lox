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


def compute_earnings_outlook(
    earnings_data: dict,
    price_data: dict | None,
    spot: float | None,
    implied_vol: float | None,
    *,
    iv_dte_assumed: int = 30,
    realized_vol_fallback: float | None = None,
) -> dict:
    """
    Synthesize earnings-expectation metrics from raw FMP + price + IV inputs.

    Returns:
        next_earnings    — dict with date, dte, time ("amc"/"bmo"), fiscal_period,
                           consensus_eps, consensus_revenue, prior_year_eps,
                           prior_year_revenue (None where unavailable)
        history          — list of past prints (most recent first), each with
                           date, fiscal_period, actual_eps, est_eps, surprise_pct,
                           beat (bool), move_1d_pct
        beat_summary     — {n_quarters, beat_count, beat_rate, avg_surprise_pct,
                            avg_abs_move_1d_pct}
        implied_move     — dict with iv, dte_to_earnings, move_to_earnings_pct,
                           move_to_iv_expiry_pct, ratio_vs_avg_history
        fy_estimates     — list of {year, eps_avg, revenue_avg, num_analysts} for
                           the next 1-2 fiscal years (current FY + 1)

    Empty dicts/lists when input is missing — every key is optional.
    """
    from datetime import date, datetime, timedelta
    import math

    out: dict = {
        "next_earnings": None,
        "history": [],
        "beat_summary": None,
        "implied_move": None,
        "fy_estimates": [],
    }

    today = date.today()
    cal = earnings_data.get("earnings_calendar") or []

    # Build a date-indexed close map from price history for post-earnings move calc
    close_by_date: dict[str, float] = {}
    if price_data and price_data.get("historical"):
        for h in price_data["historical"]:
            d = h.get("date")
            c = h.get("close")
            if d and c is not None:
                close_by_date[str(d)[:10]] = float(c)

    # Sorted date list for next-business-day lookup
    sorted_dates = sorted(close_by_date.keys())

    def _next_close_pct(earn_date_str: str) -> float | None:
        """Return % change from earn-date close to next-session close."""
        if earn_date_str not in close_by_date:
            # If earnings reported AMC, the move shows up the next session anyway.
            # Otherwise we may not have that exact trading day. Find nearest <= date.
            prior = [d for d in sorted_dates if d <= earn_date_str]
            if not prior:
                return None
            earn_date_str = prior[-1]
        base = close_by_date.get(earn_date_str)
        if not base:
            return None
        # Next trading day after earn_date_str
        idx = sorted_dates.index(earn_date_str) if earn_date_str in sorted_dates else None
        if idx is None or idx + 1 >= len(sorted_dates):
            return None
        next_close = close_by_date[sorted_dates[idx + 1]]
        return (next_close - base) / base * 100.0

    # ── Split calendar into past (has actuals) and future (estimate only) ──
    past = []
    future = []
    for e in cal:
        d = e.get("date")
        if not d:
            continue
        try:
            ed = datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
        except Exception:
            continue
        if ed > today:
            future.append((ed, e))
        elif e.get("eps") is not None:  # has actuals
            past.append((ed, e))
        else:
            # Today or past but no actual reported yet — treat as upcoming
            future.append((ed, e))

    past.sort(key=lambda x: x[0], reverse=True)
    future.sort(key=lambda x: x[0])

    # ── Next earnings ─────────────────────────────────────────────────────
    if future:
        nd, ne = future[0]
        dte = (nd - today).days
        # Look up prior-year quarter for YoY context
        prior_year_eps = None
        prior_year_rev = None
        for _pd, pe in past:
            if _pd.year == nd.year - 1 and _pd.month in (nd.month - 1, nd.month, nd.month + 1):
                prior_year_eps = pe.get("eps")
                prior_year_rev = pe.get("revenue")
                break
        out["next_earnings"] = {
            "date": str(nd),
            "dte": dte,
            "time": (ne.get("time") or "").lower() or None,
            "fiscal_period": ne.get("fiscalDateEnding"),
            "consensus_eps": ne.get("epsEstimated"),
            "consensus_revenue": ne.get("revenueEstimated"),
            "prior_year_eps": prior_year_eps,
            "prior_year_revenue": prior_year_rev,
        }

    # ── Beat/miss history (last 8 quarters) ───────────────────────────────
    history = []
    for ed, e in past[:8]:
        actual = e.get("eps")
        est = e.get("epsEstimated")
        if actual is None:
            continue
        surprise_pct = None
        beat = None
        if est is not None and est != 0:
            surprise_pct = (float(actual) - float(est)) / abs(float(est)) * 100.0
            beat = float(actual) >= float(est)
        move_1d = _next_close_pct(str(ed))
        history.append({
            "date": str(ed),
            "fiscal_period": e.get("fiscalDateEnding"),
            "actual_eps": float(actual),
            "est_eps": float(est) if est is not None else None,
            "actual_revenue": e.get("revenue"),
            "est_revenue": e.get("revenueEstimated"),
            "surprise_pct": surprise_pct,
            "beat": beat,
            "move_1d_pct": move_1d,
        })
    out["history"] = history

    if history:
        n = len(history)
        beats = sum(1 for h in history if h["beat"] is True)
        surprises = [h["surprise_pct"] for h in history if h["surprise_pct"] is not None]
        moves = [abs(h["move_1d_pct"]) for h in history if h["move_1d_pct"] is not None]
        out["beat_summary"] = {
            "n_quarters": n,
            "beat_count": beats,
            "beat_rate": (beats / n) if n else None,
            "avg_surprise_pct": (sum(surprises) / len(surprises)) if surprises else None,
            "avg_abs_move_1d_pct": (sum(moves) / len(moves)) if moves else None,
        }

    # ── Implied earnings move from ATM IV (fallback: realized vol) ───────
    # We compare the vol-implied *1-day* move against the avg historical 1-day
    # post-earnings move so the ratio is apples-to-apples. IV from options
    # near earnings bakes in the event premium; HV doesn't, so a HV-based
    # ratio < 1 is expected and means "no event premium captured."
    vol_used = implied_vol if implied_vol is not None else realized_vol_fallback
    vol_source = "iv" if implied_vol is not None else ("hv" if realized_vol_fallback is not None else None)
    if vol_used is not None and out["next_earnings"]:
        dte_earn = out["next_earnings"]["dte"]
        if dte_earn is not None and dte_earn >= 0:
            # 1-σ move ≈ σ·√T  (as % of spot)
            move_1d_pct = float(vol_used) * math.sqrt(1.0 / 252.0) * 100.0
            move_to_earn_pct = float(vol_used) * math.sqrt(max(dte_earn, 1) / 365.0) * 100.0
            avg_hist = out["beat_summary"]["avg_abs_move_1d_pct"] if out["beat_summary"] else None
            ratio = (move_1d_pct / avg_hist) if (avg_hist and avg_hist > 0) else None
            out["implied_move"] = {
                "vol": float(vol_used),
                "vol_source": vol_source,
                "dte_to_earnings": dte_earn,
                "iv_dte_assumed": iv_dte_assumed,
                "move_1d_pct": move_1d_pct,
                "move_to_earnings_pct": move_to_earn_pct,
                "avg_hist_1d_move_pct": avg_hist,
                "ratio_vs_avg_history": ratio,
            }

    # ── FY estimates ──────────────────────────────────────────────────────
    # FMP returns estimates with fiscalDate ~= fiscal-year end. Keep the
    # nearest 2 future fiscal years (current + next), not the most-distant.
    fy_rows = []
    for fy in (earnings_data.get("analyst_estimates") or []):
        d = fy.get("date")
        if not d:
            continue
        try:
            year = int(str(d)[:4])
            fy_end = datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
        except Exception:
            continue
        if fy_end < today:
            continue  # already-reported FY
        fy_rows.append({
            "year": year,
            "fiscal_date": d,
            "eps_avg": fy.get("estimatedEpsAvg"),
            "eps_low": fy.get("estimatedEpsLow"),
            "eps_high": fy.get("estimatedEpsHigh"),
            "revenue_avg": fy.get("estimatedRevenueAvg"),
            "num_analysts_eps": fy.get("numberAnalystsEstimatedEps"),
            "num_analysts_revenue": fy.get("numberAnalystEstimatedRevenue"),
        })
    fy_rows.sort(key=lambda r: r["year"])
    out["fy_estimates"] = fy_rows[:2]

    return out


def detect_stacked_signals(
    *,
    symbol: str,
    is_etf: bool,
    technicals: dict | None,
    fundamentals: dict | None,
    earnings_outlook: dict | None,
    ratings_consensus: dict | None,
    price_target: dict | None,
    futures_data: dict | None,
    flow_context: dict | None,
    iv: float | None,
    current_price: float | None,
) -> list[dict]:
    """
    Detect multi-block signal stacks — combinations that compound into edge.

    Each detected stack returns:
        {
            "name": str,           — short label
            "direction": "bull" | "bear" | "caution",
            "thesis": str,         — one-sentence "why this matters"
            "signals": list[str],  — supporting evidence (1 line each)
            "trade": str | None,   — suggested structure / action
        }

    Stacks fire only when ≥2 supporting signals agree. Each stack is independent;
    multiple can fire on the same symbol (the picture is the whole list, not a
    single verdict).
    """
    from lox.utils.formatting import safe_float

    stacks: list[dict] = []
    technicals = technicals or {}
    fundamentals = fundamentals or {}
    earnings_outlook = earnings_outlook or {}

    # ── Common metric extraction ──────────────────────────────────────────
    current = current_price or technicals.get("current") or 0
    high_52w = technicals.get("high_52w")
    low_52w = technicals.get("low_52w")
    pct_52w = None
    if current and high_52w and low_52w and high_52w > low_52w:
        pct_52w = (current - low_52w) / (high_52w - low_52w) * 100.0

    rsi = technicals.get("rsi")
    trend_label = (technicals.get("trend_label") or "")
    above_all = "Above all major" in trend_label
    below_all = "Below all major" in trend_label
    crossover = technicals.get("sma_crossover") or ""
    golden = "Golden" in crossover
    death = "Death" in crossover

    hv = technicals.get("volatility_30d") or technicals.get("volatility")
    iv_hv_spread = None
    if iv is not None and hv is not None:
        iv_hv_spread = (iv * 100.0) - float(hv)  # percentage points

    # Ratings
    consensus_label = None
    pt_upside = None
    if ratings_consensus:
        consensus_label = ratings_consensus.get("consensus") or ""
    if price_target and current:
        tc = price_target.get("targetConsensus") or price_target.get("targetMedian")
        if isinstance(tc, (int, float)) and tc > 0:
            pt_upside = (tc - current) / current * 100.0

    # Earnings
    next_e = earnings_outlook.get("next_earnings") if earnings_outlook else None
    beat_summary = earnings_outlook.get("beat_summary") if earnings_outlook else None
    implied_move = earnings_outlook.get("implied_move") if earnings_outlook else None
    dte_e = next_e.get("dte") if next_e else None
    beat_rate = beat_summary.get("beat_rate") if beat_summary else None
    avg_hist_move = beat_summary.get("avg_abs_move_1d_pct") if beat_summary else None
    impl_ratio = implied_move.get("ratio_vs_avg_history") if implied_move else None
    impl_source = implied_move.get("vol_source") if implied_move else None

    # Fundamentals
    ratios = fundamentals.get("ratios", {}) or {}
    income = (fundamentals.get("income_statement") or [{}])[0]
    income_prev = (fundamentals.get("income_statement") or [None, None])[1] if len(fundamentals.get("income_statement") or []) >= 2 else None
    net_margin = safe_float(ratios.get("netProfitMarginTTM"))
    rev = safe_float(income.get("revenue"))
    rev_prev = safe_float(income_prev.get("revenue") if income_prev else None)
    rev_growth = None
    if rev and rev_prev and rev_prev != 0:
        rev_growth = (rev - rev_prev) / rev_prev * 100.0

    # ───────────────────────────────────────────────────────────────────────
    # BEAR STACKS
    # ───────────────────────────────────────────────────────────────────────

    # 1) Crowded long top — sell-side and price both pinned at the high
    if ratings_consensus and pct_52w is not None and rsi is not None:
        sb = ratings_consensus.get("strongBuy", 0) or 0
        b = ratings_consensus.get("buy", 0) or 0
        h = ratings_consensus.get("hold", 0) or 0
        s = ratings_consensus.get("sell", 0) or 0
        ss = ratings_consensus.get("strongSell", 0) or 0
        total = sb + b + h + s + ss
        bear_share = (s + ss) / total if total else 1.0
        if total >= 10 and bear_share < 0.10 and pct_52w > 85 and rsi > 68:
            stacks.append({
                "name": "Crowded long top",
                "direction": "bear",
                "thesis": "Everyone's already positioned long at the highs — no marginal buyer left, asymmetric downside on any disappointment.",
                "signals": [
                    f"{sb + b}/{total} analysts at Buy/Strong Buy, only {s + ss} bearish ({bear_share*100:.0f}%)",
                    f"Price at {pct_52w:.0f}th percentile of 52W range",
                    f"RSI {rsi:.0f} (overbought)",
                ],
                "trade": "Bear call spread or wait for first downgrade as trigger.",
            })

    # 2) Vol-crush short straddle — stretched into earnings with expensive options
    if (dte_e is not None and 0 <= dte_e <= 25
            and pct_52w is not None and pct_52w > 80
            and rsi is not None and rsi > 65
            and iv_hv_spread is not None and iv_hv_spread > 4
            and (beat_rate is None or beat_rate <= 0.70)):
        signals = [
            f"Earnings in {dte_e}d",
            f"Price at {pct_52w:.0f}th percentile of 52W range",
            f"RSI {rsi:.0f} (overbought)",
            f"IV {iv*100:.0f}% vs HV {hv:.0f}% — vol premium +{iv_hv_spread:.1f}pp",
        ]
        if beat_rate is not None:
            signals.append(f"Beat rate only {beat_rate*100:.0f}% — mixed track record")
        stacks.append({
            "name": "Vol-crush short straddle setup",
            "direction": "bear",
            "thesis": "Stretched price + expensive options + uneven beat history — straddle is overpriced for what this name actually does.",
            "signals": signals,
            "trade": "Sell straddle/iron condor through earnings; size for IV crush + mild mean reversion.",
        })

    # 3) Mechanical headwind — below the regime line, no oversold bounce
    if below_all and (death or (rsi is not None and 35 <= rsi < 55)):
        signals = [f"Below all major MAs (20/50/200)"]
        if death:
            signals.append("Death Cross (50 SMA < 200 SMA)")
        if rsi is not None:
            signals.append(f"RSI {rsi:.0f} (no oversold bounce in sight)")
        stacks.append({
            "name": "Mechanical headwind",
            "direction": "bear",
            "thesis": "CTAs are short, trend followers piling on, no technical reason for buyers to step in — fundamentals don't matter until the structure heals.",
            "signals": signals,
            "trade": "Wait for failed breakout above 50 SMA before fading shorts.",
        })

    # 4) Quality deteriorating — negative trend confirmed by negative fundamentals
    if net_margin is not None and net_margin < 0:
        signals = [f"Net margin negative ({net_margin:.1f}%)"]
        confirms = 0
        if rev_growth is not None and rev_growth < 0:
            signals.append(f"Revenue declining ({rev_growth:+.1f}% YoY)")
            confirms += 1
        if below_all or death:
            signals.append("Below all MAs" if below_all else "Death Cross")
            confirms += 1
        if consensus_label in ("Sell", "Strong Sell"):
            signals.append(f"Analyst consensus: {consensus_label}")
            confirms += 1
        if confirms >= 1:
            stacks.append({
                "name": "Quality deteriorating",
                "direction": "bear",
                "thesis": "Losing money, technically broken, and the sell-side is exiting — no clear floor.",
                "signals": signals,
                "trade": "Avoid long entries; rallies are exit liquidity for existing holders.",
            })

    # ───────────────────────────────────────────────────────────────────────
    # BULL STACKS
    # ───────────────────────────────────────────────────────────────────────

    # 5) Mean reversion candidate — washed out with sell-side support intact
    if pct_52w is not None and pct_52w < 30 and rsi is not None and rsi < 38:
        confirms_bull = []
        if pt_upside is not None and pt_upside > 15:
            confirms_bull.append(f"Analyst PT consensus implies +{pt_upside:.0f}% upside")
        if consensus_label in ("Buy", "Strong Buy"):
            confirms_bull.append(f"Analyst consensus: {consensus_label}")
        if beat_rate is not None and beat_rate >= 0.60:
            confirms_bull.append(f"Beat rate still {beat_rate*100:.0f}% — fundamentals not broken")
        if confirms_bull:
            stacks.append({
                "name": "Mean reversion candidate",
                "direction": "bull",
                "thesis": "Capitulation overshoot — washed out technically while the fundamental support is intact.",
                "signals": [
                    f"Price at {pct_52w:.0f}th percentile of 52W range",
                    f"RSI {rsi:.0f} (oversold)",
                    *confirms_bull,
                ],
                "trade": "Scale in long; consider call spreads or risk-reversals to define downside.",
            })

    # 6) Cheap earnings straddle — historically volatile name, options not pricing it
    if (dte_e is not None and 0 <= dte_e <= 25
            and beat_rate is not None and beat_rate >= 0.70
            and avg_hist_move is not None and avg_hist_move >= 3.0
            and impl_ratio is not None and impl_source == "iv" and impl_ratio <= 1.2):
        stacks.append({
            "name": "Cheap earnings straddle",
            "direction": "bull",
            "thesis": "Reliable beat history with big post-print moves, but options pricing in line — convexity is on sale.",
            "signals": [
                f"Earnings in {dte_e}d",
                f"Beat rate {beat_rate*100:.0f}% over last 8 quarters",
                f"Avg historical 1d post-print move ±{avg_hist_move:.1f}%",
                f"Market pricing only {impl_ratio:.1f}× normal — no event premium",
            ],
            "trade": "Long straddle / strangle for the print; size as a defined-risk lottery.",
        })

    # 7) Mechanical tailwind — trend up, not yet euphoric
    if above_all and golden and rsi is not None and 50 <= rsi <= 70:
        signals = [
            "Above all major MAs (20/50/200)",
            "Golden Cross (50 SMA > 200 SMA)",
            f"RSI {rsi:.0f} — trending, not overbought",
        ]
        macd = technicals.get("macd_signal") or ""
        if "Bullish" in macd:
            signals.append("MACD bullish")
        stacks.append({
            "name": "Mechanical tailwind",
            "direction": "bull",
            "thesis": "CTAs long, trend followers piling in, no overbought exhaustion — fundamental drawdowns get absorbed in this regime.",
            "signals": signals,
            "trade": "Long with trailing stop below 50 SMA; pullbacks are entries, not warnings.",
        })

    # ───────────────────────────────────────────────────────────────────────
    # CAUTION / CONFLICT STACKS
    # ───────────────────────────────────────────────────────────────────────

    # 8) Sell-side disconnect — analysts say buy, tape says sell
    if (consensus_label in ("Buy", "Strong Buy")
            and pt_upside is not None and pt_upside > 15
            and (below_all or (rsi is not None and rsi < 45))):
        signals = [
            f"Analyst consensus: {consensus_label}",
            f"PT implies +{pt_upside:.0f}% upside",
        ]
        if below_all:
            signals.append("Price below all major MAs")
        if rsi is not None and rsi < 45:
            signals.append(f"RSI {rsi:.0f} (weak momentum)")
        stacks.append({
            "name": "Sell-side disconnect",
            "direction": "caution",
            "thesis": "Tape is rejecting the analyst story — either analysts cut soon or price catches up to PT. Wait for which side blinks.",
            "signals": signals,
            "trade": "No size yet — watch for either downgrades (validates price) or a structural reclaim (validates analysts).",
        })

    # 9) Forced-selling fragility — futures liquidity broken + crowded
    if futures_data:
        health = futures_data.get("depth_health")
        cot_z = futures_data.get("cot_z_score")
        if health in ("FRAGILE", "CRITICAL") and cot_z is not None and abs(cot_z) > 1.5:
            crowd = "long" if cot_z > 0 else "short"
            stacks.append({
                "name": "Forced-selling fragility",
                "direction": "caution",
                "thesis": "Underlying futures book is thin and specs are one-sidedly crowded — one trigger and the unwind is violent.",
                "signals": [
                    f"Futures depth: {health}",
                    f"Specs crowded {crowd} (COT z={cot_z:+.1f})",
                ],
                "trade": "Reduce size; widen stops; consider OTM put hedges (cheap when nobody's hedged).",
            })

    # 10) ETF momentum + flow confirmation
    if is_etf and flow_context:
        net = flow_context.get("net_flow_signal_20d") or ""
        mfi = flow_context.get("mfi_14d")
        if "STRONG INFLOWS" in net and above_all and mfi is not None and mfi > 60:
            stacks.append({
                "name": "Flow-confirmed ETF momentum",
                "direction": "bull",
                "thesis": "Money is following price — feedback loop is active and self-reinforcing while it lasts.",
                "signals": [
                    f"20d flow signal: {net}",
                    f"MFI {mfi:.0f} (strong)",
                    "Above all major MAs",
                ],
                "trade": "Long with trailing stop; first breakdown of 20 SMA + MFI roll-over = exit.",
            })

    return stacks


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

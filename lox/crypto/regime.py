"""Crypto perps regime classifier.

Uses CCXT snapshot data (funding rates, OI, technicals, volume, price action)
to classify the current crypto market regime.  Follows the same RegimeResult
interface as all other LOX regime domains.

Score guide: 0 = strong crypto risk-on → 100 = strong crypto risk-off / stress.

Key signals:
  1. Funding rate  — leverage/positioning proxy (high positive = overleveraged longs)
  2. Technical trend — EMA alignment, RSI, MACD on the 4H timeframe
  3. Volume health — current volume vs 20-period average
  4. Momentum      — 24h change, RSI extremes
  5. Open interest  — rising OI + falling price = short pressure / liquidation risk
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CryptoRegimeResult:
    """Matches the RegimeResult interface used across the LOX regime system."""
    name: str
    label: str
    description: str
    score: float
    domain: str = "crypto"
    tags: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def classify_crypto_regime(snapshots: dict[str, Any]) -> CryptoRegimeResult:
    """Classify the overall crypto perps regime from multi-coin snapshots.

    Parameters
    ----------
    snapshots : dict
        Output of ``CryptoPerpsData.multi_snapshot()`` — keyed by coin.

    Returns
    -------
    CryptoRegimeResult
    """
    if not snapshots:
        return CryptoRegimeResult(
            name="unknown",
            label="No Data",
            description="No crypto data available.",
            score=50,
            tags=["no_data"],
        )

    # ── Collect per-coin signals ─────────────────────────────────────────
    funding_rates: list[float] = []
    rsi_14s: list[float] = []
    ema_trends: list[float] = []       # +1 bullish, -1 bearish
    volume_ratios: list[float] = []
    change_24hs: list[float] = []
    macd_signs: list[float] = []       # +1 positive, -1 negative
    prices: dict[str, float] = {}

    for coin, snap in snapshots.items():
        prices[coin] = snap.get("price", 0)

        # Funding rate
        if snap.get("funding") and snap["funding"].get("funding_rate") is not None:
            funding_rates.append(snap["funding"]["funding_rate"])

        # Long-timeframe technicals (4H by default — trend context)
        lt = snap.get("long_tf", {}).get("latest", {})
        if lt:
            if lt.get("rsi_14") is not None:
                rsi_14s.append(lt["rsi_14"])
            if lt.get("ema_20") is not None and lt.get("ema_50") is not None:
                ema_trends.append(1.0 if lt["ema_20"] > lt["ema_50"] else -1.0)
            if lt.get("volume") is not None and lt.get("volume_ma") is not None and lt["volume_ma"] > 0:
                volume_ratios.append(lt["volume"] / lt["volume_ma"])
            if lt.get("macd") is not None:
                macd_signs.append(1.0 if lt["macd"] > 0 else -1.0)

        # 24h change from ticker
        if snap.get("ticker") and snap["ticker"].get("change_pct_24h") is not None:
            change_24hs.append(snap["ticker"]["change_pct_24h"])

    # ── Aggregate signals ────────────────────────────────────────────────
    avg_funding = _safe_avg(funding_rates, 0.0)
    avg_rsi = _safe_avg(rsi_14s, 50.0)
    avg_trend = _safe_avg(ema_trends, 0.0)
    avg_vol_ratio = _safe_avg(volume_ratios, 1.0)
    avg_change_24h = _safe_avg(change_24hs, 0.0)
    avg_macd = _safe_avg(macd_signs, 0.0)

    # ── Score (0 = risk-on, 100 = risk-off) ──────────────────────────────
    score = 50  # neutral baseline

    # 1. Funding rate (annualized > 30% = very overleveraged longs)
    ann_funding = avg_funding * 3 * 365  # 8h rate → annualized
    if ann_funding > 50:
        score += 15    # extreme leverage → liquidation risk
    elif ann_funding > 30:
        score += 8
    elif ann_funding > 15:
        score += 3
    elif ann_funding < -15:
        score -= 5     # shorts paying → contrarian bullish
    elif ann_funding < -30:
        score -= 10    # heavy shorts → squeeze potential

    # 2. Technical trend (EMA alignment across coins)
    if avg_trend < -0.5:
        score += 12    # most coins bearish
    elif avg_trend < 0:
        score += 5
    elif avg_trend > 0.5:
        score -= 12    # most coins bullish
    elif avg_trend > 0:
        score -= 5

    # 3. RSI (overbought = risk of reversal, oversold = risk-off already priced)
    if avg_rsi > 75:
        score += 8     # overbought → correction risk
    elif avg_rsi > 65:
        score += 3
    elif avg_rsi < 25:
        score += 5     # deeply oversold = already in stress
    elif avg_rsi < 35:
        score += 2

    # 4. Volume health
    if avg_vol_ratio < 0.5:
        score += 5     # thin volume = fragile
    elif avg_vol_ratio > 2.0:
        score -= 3     # strong participation (momentum)

    # 5. 24h price action
    if avg_change_24h < -8:
        score += 15    # sharp selloff
    elif avg_change_24h < -4:
        score += 8
    elif avg_change_24h < -2:
        score += 3
    elif avg_change_24h > 8:
        score -= 8     # strong rally
    elif avg_change_24h > 4:
        score -= 5
    elif avg_change_24h > 2:
        score -= 2

    # 6. MACD confirmation
    if avg_macd < -0.5:
        score += 3
    elif avg_macd > 0.5:
        score -= 3

    score = max(0, min(100, score))

    # ── Label ────────────────────────────────────────────────────────────
    if score >= 75:
        label = "Crypto Capitulation"
        name = "crypto_capitulation"
    elif score >= 65:
        label = "Crypto Risk-Off"
        name = "crypto_risk_off"
    elif score >= 55:
        label = "Crypto Cautious"
        name = "crypto_cautious"
    elif score >= 45:
        label = "Crypto Neutral"
        name = "crypto_neutral"
    elif score >= 35:
        label = "Crypto Constructive"
        name = "crypto_constructive"
    elif score >= 20:
        label = "Crypto Risk-On"
        name = "crypto_risk_on"
    else:
        label = "Crypto Euphoria"
        name = "crypto_euphoria"

    # ── Tags ─────────────────────────────────────────────────────────────
    tags: list[str] = ["crypto"]
    if ann_funding > 30:
        tags.append("overleveraged_longs")
    elif ann_funding < -15:
        tags.append("overleveraged_shorts")
    if avg_rsi > 70:
        tags.append("overbought")
    elif avg_rsi < 30:
        tags.append("oversold")
    if avg_change_24h < -5:
        tags.append("selloff")
    elif avg_change_24h > 5:
        tags.append("rally")

    # ── Description ──────────────────────────────────────────────────────
    btc_price = prices.get("BTC", prices.get(list(prices.keys())[0], 0)) if prices else 0
    parts: list[str] = []
    if btc_price:
        parts.append(f"BTC: ${btc_price:,.0f}")
    parts.append(f"24h: {avg_change_24h:+.1f}%")
    if funding_rates:
        parts.append(f"Funding: {ann_funding:+.0f}% ann")
    parts.append(f"RSI(14): {avg_rsi:.0f}")
    trend_word = "Bullish" if avg_trend > 0.3 else ("Bearish" if avg_trend < -0.3 else "Mixed")
    parts.append(f"Trend: {trend_word}")
    parts.append(f"Vol: {avg_vol_ratio:.1f}x avg")

    # ── Metrics dict ─────────────────────────────────────────────────────
    metrics = {
        "btc_price": f"${btc_price:,.0f}" if btc_price else None,
        "avg_change_24h": f"{avg_change_24h:+.1f}%",
        "ann_funding": f"{ann_funding:+.0f}%",
        "avg_rsi_14": f"{avg_rsi:.0f}",
        "trend": trend_word,
        "volume_ratio": f"{avg_vol_ratio:.1f}x",
    }

    return CryptoRegimeResult(
        name=name,
        label=label,
        description=" | ".join(parts),
        score=score,
        tags=tags,
        metrics=metrics,
    )


def _safe_avg(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default

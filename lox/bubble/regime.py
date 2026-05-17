"""
Broad-market bubble regime classifier.

Five pillars, each scored 0-100, then a weighted composite.  Weights are
tilted toward leverage and speculation because valuation can stay stretched
for years — it's the late-cycle leverage/speculation behaviour that signals
the turn:

  Valuation      (15%): Buffett indicator percentile vs full history.
  Concentration  (15%): SPY top-10 weight share + cap-vs-equal-weight spread.
  Leverage       (25%): Margin-debt level percentile + YoY froth.
  Speculation    (25%): Levered-long ETF AUM + long-to-short skew + PCR.
  Sentiment      (20%): AAII bull % + VIX-vs-realized-vol gap.

"Cracks" adjustments push the composite higher when classic late-cycle
warnings appear (AI breadth narrowing, margin debt rolling over from a peak,
complacency reading on VIX).  The label progression:

  < 30 : No Bubble Signal
  30-45: Mildly Elevated
  45-60: Frothy
  60-75: Stretched
  75-90: Late-stage Bubble
  > 90 : Blow-off with Cracks   (only when score is high AND cracks present)
"""
from __future__ import annotations

from lox.regimes.base import RegimeResult


def _valuation_subscore(percentile_full: float | None) -> float | None:
    if percentile_full is None:
        return None
    return float(max(0.0, min(100.0, percentile_full)))


def _concentration_subscore(
    top10_share_pct: float | None,
    spy_minus_rsp_ytd: float | None,
    spy_minus_rsp_1y: float | None,
    ai_leadership_flag: bool,
) -> float | None:
    """Combine top-10 share (primary) with RSP-vs-SPY confirmation."""
    if top10_share_pct is None and spy_minus_rsp_ytd is None and spy_minus_rsp_1y is None:
        return None

    score: float | None = None

    if top10_share_pct is not None:
        # 2014-2019 averaged ~18%; >30% elevated; >35% extreme; >40% blow-off.
        s = top10_share_pct
        if s <= 18:
            score = 25.0
        elif s <= 24:
            score = 45.0
        elif s <= 30:
            score = 60.0
        elif s <= 35:
            score = 75.0
        elif s <= 40:
            score = 88.0
        else:
            score = 95.0
    else:
        # Fallback to RSP/SPY proxy only
        ytd = spy_minus_rsp_ytd or 0.0
        if ytd <= 0:
            score = 25.0
        elif ytd < 5:
            score = 40.0
        elif ytd < 10:
            score = 55.0
        elif ytd < 15:
            score = 70.0
        elif ytd < 20:
            score = 82.0
        else:
            score = 92.0

    # RSP/SPY confirmation amplifies if both reads agree
    if spy_minus_rsp_ytd is not None:
        if spy_minus_rsp_ytd > 10:
            score += 4.0
        elif spy_minus_rsp_ytd < -5:
            score -= 6.0

    if spy_minus_rsp_1y is not None and spy_minus_rsp_1y > 12:
        score += 3.0

    if ai_leadership_flag:
        score += 5.0

    return float(max(0.0, min(100.0, score)))


def _leverage_subscore(
    percentile_full: float | None,
    yoy_pct: float | None,
) -> float | None:
    if percentile_full is None and yoy_pct is None:
        return None
    base = float(percentile_full) if percentile_full is not None else 50.0
    if yoy_pct is not None:
        if yoy_pct > 30:
            base += 15
        elif yoy_pct > 20:
            base += 8
        elif yoy_pct > 10:
            base += 3
        elif yoy_pct < -10:
            base -= 12
        elif yoy_pct < 0:
            base -= 4
    return float(max(0.0, min(100.0, base)))


def _speculation_subscore(
    levered_long_aum_bn: float | None,
    long_to_short_ratio: float | None,
    put_call_ratio: float | None,
) -> float | None:
    """Levered-ETF AUM + long/short skew + PCR.  Higher = more speculation."""
    if (levered_long_aum_bn is None and long_to_short_ratio is None
            and put_call_ratio is None):
        return None

    components: list[float] = []

    if levered_long_aum_bn is not None:
        # Long levered ETF AUM thresholds (sum of TQQQ+SOXL+UPRO+FAS).
        # Pre-2020: <$10B; 2021 mania: ~$25B; 2024+: $40B+; >$60B = late-cycle.
        a = levered_long_aum_bn
        if a < 8:
            components.append(20.0)
        elif a < 15:
            components.append(40.0)
        elif a < 25:
            components.append(55.0)
        elif a < 40:
            components.append(70.0)
        elif a < 60:
            components.append(82.0)
        else:
            components.append(92.0)

    if long_to_short_ratio is not None:
        # 1:1 = panic, 2:1 = normal, 4:1 = euphoric, 7:1+ = blow-off.
        r = long_to_short_ratio
        if r < 1.5:
            components.append(20.0)
        elif r < 2.5:
            components.append(40.0)
        elif r < 4.0:
            components.append(60.0)
        elif r < 6.0:
            components.append(78.0)
        else:
            components.append(90.0)

    if put_call_ratio is not None:
        # SPY/index PCR baseline is ~1.5-2.0 because institutional hedging
        # (pensions, vol-targeting funds) keeps put OI structurally high.
        # Speculation/froth shows up when PCR FALLS below that baseline —
        # i.e. retail call-buying overwhelms the hedging book.
        #   >2.5  : elevated hedging / fear (low froth)
        #   1.8-2.5: normal institutional baseline
        #   1.5-1.8: call buying picking up
        #   1.2-1.5: frothy
        #   <1.2  : euphoric for an index
        p = put_call_ratio
        if p >= 2.5:
            components.append(15.0)
        elif p >= 1.8:
            components.append(40.0)
        elif p >= 1.5:
            components.append(60.0)
        elif p >= 1.2:
            components.append(78.0)
        else:
            components.append(90.0)

    return float(sum(components) / len(components))


def _sentiment_subscore(
    aaii_bull_pct: float | None,
    vix_minus_realized: float | None,
) -> float | None:
    """AAII + complacency gap. Higher = more bubble sentiment."""
    if aaii_bull_pct is None and vix_minus_realized is None:
        return None

    components: list[float] = []

    if aaii_bull_pct is not None:
        # AAII historical avg ~38%; >50% extreme bull; <25% extreme bear.
        b = aaii_bull_pct
        if b < 25:
            components.append(15.0)
        elif b < 35:
            components.append(40.0)
        elif b < 45:
            components.append(55.0)
        elif b < 55:
            components.append(75.0)
        else:
            components.append(90.0)

    if vix_minus_realized is not None:
        # Gap < 0 = complacent.  Gap > 5pp = fear premium present.
        g = vix_minus_realized
        if g >= 5:
            components.append(20.0)
        elif g >= 0:
            components.append(45.0)
        elif g >= -3:
            components.append(65.0)
        elif g >= -6:
            components.append(80.0)
        else:
            components.append(92.0)

    return float(sum(components) / len(components))


def classify_bubble(
    *,
    valuation_pct_full: float | None,
    top10_share_pct: float | None,
    spy_minus_rsp_ytd: float | None,
    spy_minus_rsp_1y: float | None,
    ai_leadership_flag: bool,
    ai_breadth_200d: float | None,
    margin_pct_full: float | None,
    margin_yoy_pct: float | None,
    margin_rolling_over: bool,
    levered_long_aum_bn: float | None,
    long_to_short_ratio: float | None,
    put_call_ratio: float | None,
    aaii_bull_pct: float | None,
    vix_minus_realized: float | None,
    complacency_flag: bool,
) -> RegimeResult:
    val_sub = _valuation_subscore(valuation_pct_full)
    con_sub = _concentration_subscore(top10_share_pct, spy_minus_rsp_ytd,
                                      spy_minus_rsp_1y, ai_leadership_flag)
    lev_sub = _leverage_subscore(margin_pct_full, margin_yoy_pct)
    spec_sub = _speculation_subscore(levered_long_aum_bn, long_to_short_ratio,
                                     put_call_ratio)
    sent_sub = _sentiment_subscore(aaii_bull_pct, vix_minus_realized)

    weights = {
        "valuation":     0.15,
        "concentration": 0.15,
        "leverage":      0.25,
        "speculation":   0.25,
        "sentiment":     0.20,
    }
    pairs = [
        (val_sub,  weights["valuation"]),
        (con_sub,  weights["concentration"]),
        (lev_sub,  weights["leverage"]),
        (spec_sub, weights["speculation"]),
        (sent_sub, weights["sentiment"]),
    ]

    weighted_sum = 0.0
    total_w = 0.0
    for sub, w in pairs:
        if sub is not None:
            weighted_sum += sub * w
            total_w += w

    base = (weighted_sum / total_w) if total_w > 0 else 50.0

    # ── Cracks: late-cycle confirmations ─────────────────────────────────
    cracks: list[str] = []
    cracks_amp = 0.0

    if margin_rolling_over and (margin_pct_full or 0) > 70:
        cracks_amp += 8
        cracks.append("margin debt rolling over from a high")

    if ai_leadership_flag and ai_breadth_200d is not None and ai_breadth_200d < 55:
        cracks_amp += 8
        cracks.append(f"AI leaders narrowing — only {ai_breadth_200d:.0f}% above 200dma")

    if complacency_flag and val_sub is not None and val_sub >= 70:
        cracks_amp += 5
        cracks.append("VIX under realized while valuations extreme")

    hot = sum(1 for x in (val_sub, con_sub, lev_sub, spec_sub, sent_sub)
              if x is not None and x >= 70)
    if hot >= 4:
        cracks_amp += 8
        cracks.append(f"{hot}/5 pillars in top quartile")
    elif hot == 3:
        cracks_amp += 4

    score = float(max(0.0, min(100.0, base + cracks_amp)))

    tags = ["bubble"]
    if val_sub is not None and val_sub >= 75:
        tags.append("valuation_stretched")
    if con_sub is not None and con_sub >= 75:
        tags.append("concentration_extreme")
    if lev_sub is not None and lev_sub >= 75:
        tags.append("leverage_extreme")
    if spec_sub is not None and spec_sub >= 75:
        tags.append("speculation_hot")
    if sent_sub is not None and sent_sub >= 75:
        tags.append("sentiment_euphoric")
    if ai_leadership_flag:
        tags.append("ai_concentration")
    if margin_rolling_over:
        tags.append("margin_rollover")
    if complacency_flag:
        tags.append("complacency")

    has_cracks = cracks_amp >= 12
    blowoff = score >= 75

    if blowoff and has_cracks:
        label = "Blow-off with Cracks"
        tags += ["blowoff", "cracks"]
    elif score >= 75:
        label = "Late-stage Bubble"
        tags.append("blowoff")
    elif score >= 60:
        label = "Stretched"
    elif score >= 45:
        label = "Frothy"
    elif score >= 30:
        label = "Mildly Elevated"
    else:
        label = "No Bubble Signal"

    parts: list[str] = []
    if val_sub is not None and valuation_pct_full is not None:
        parts.append(f"valuation: {valuation_pct_full:.0f}th pctl")
    if top10_share_pct is not None:
        parts.append(f"top-10 SPY share: {top10_share_pct:.1f}%")
    elif spy_minus_rsp_ytd is not None:
        parts.append(f"SPY-RSP YTD: {spy_minus_rsp_ytd:+.1f}pp")
    if lev_sub is not None and margin_pct_full is not None:
        yoy_str = f", YoY {margin_yoy_pct:+.0f}%" if margin_yoy_pct is not None else ""
        parts.append(f"margin: {margin_pct_full:.0f}th pctl{yoy_str}")
    if levered_long_aum_bn is not None:
        parts.append(f"levered long AUM: ${levered_long_aum_bn:.0f}B")
    if aaii_bull_pct is not None:
        parts.append(f"AAII bull: {aaii_bull_pct:.0f}%")
    if cracks:
        parts.append("cracks: " + ", ".join(cracks))

    return RegimeResult(
        name="bubble",
        label=label,
        description=" | ".join(parts) if parts else "Insufficient data",
        score=score,
        domain="bubble",
        tags=tags,
        metrics={
            "Valuation sub":     f"{val_sub:.0f}"  if val_sub  is not None else None,
            "Concentration sub": f"{con_sub:.0f}"  if con_sub  is not None else None,
            "Leverage sub":      f"{lev_sub:.0f}"  if lev_sub  is not None else None,
            "Speculation sub":   f"{spec_sub:.0f}" if spec_sub is not None else None,
            "Sentiment sub":     f"{sent_sub:.0f}" if sent_sub is not None else None,
            "Buffett pctl":      f"{valuation_pct_full:.0f}" if valuation_pct_full is not None else None,
            "Top-10 SPY":        f"{top10_share_pct:.1f}%"  if top10_share_pct is not None else None,
            "SPY-RSP YTD":       f"{spy_minus_rsp_ytd:+.1f}pp" if spy_minus_rsp_ytd is not None else None,
            "Margin pctl":       f"{margin_pct_full:.0f}"  if margin_pct_full is not None else None,
            "Margin YoY":        f"{margin_yoy_pct:+.1f}%" if margin_yoy_pct is not None else None,
            "Levered long AUM":  f"${levered_long_aum_bn:.1f}B" if levered_long_aum_bn is not None else None,
            "Long/Short ratio":  f"{long_to_short_ratio:.1f}x" if long_to_short_ratio is not None else None,
            "PCR":               f"{put_call_ratio:.2f}" if put_call_ratio is not None else None,
            "AAII bull":         f"{aaii_bull_pct:.0f}%" if aaii_bull_pct is not None else None,
            "VIX - realized":    f"{vix_minus_realized:+.1f}pp" if vix_minus_realized is not None else None,
        },
    )

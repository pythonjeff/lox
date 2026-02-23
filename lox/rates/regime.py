from __future__ import annotations

from dataclasses import dataclass

from lox.rates.models import RatesInputs

# ── Regime classification thresholds ──────────────────────────────────────
# Momentum: minimum 20d change (in %) to classify as directional
MOMENTUM_THRESHOLD_PCT = 0.02
# Z-score gates
Z_SIGNAL_GATE = 0.5        # min |z| for steepener/flattener signal
Z_SHOCK_THRESHOLD = 1.5    # 10Y z-score for rate shock classification
# Real yield attribution threshold (20d change, %)
REAL_YIELD_MOVE_THRESHOLD = 0.10
# Static curve shape (percent or z-score)
STEEP_CURVE_THRESHOLD = 1.0

# ── Regime scores (0 = benign/easing → 100 = acute stress) ───────────────
SCORE_INVERTED = 75
SCORE_BEAR_STEEPENER = 70
SCORE_BEAR_FLATTENER = 65
SCORE_BULL_FLATTENER = 55
SCORE_NEUTRAL = 45
SCORE_STEEP_CURVE = 40
SCORE_BULL_STEEPENER = 40
SCORE_SHOCK_UP = 75
SCORE_SHOCK_DOWN = 30


@dataclass(frozen=True)
class RatesRegime:
    name: str
    label: str | None = None
    description: str = ""
    tags: tuple[str, ...] = ()
    score: int = 50

    @property
    def display_label(self) -> str:
        """Label for display — falls back to name if label is None."""
        return self.label or self.name


def _real_yield_note(inputs: RatesInputs) -> str:
    """Build a parenthetical note attributing the rate move to real or breakeven."""
    ry = inputs.real_yield_10y_chg_20d
    be = inputs.breakeven_10y
    ry_level = inputs.real_yield_10y

    parts: list[str] = []
    if ry is not None:
        if abs(ry) > REAL_YIELD_MOVE_THRESHOLD:
            direction = "rising" if ry > 0 else "falling"
            parts.append(f"real yields {direction} ({ry:+.2f}% 20d)")
        else:
            parts.append("real yields flat")
    if be is not None:
        parts.append(f"10Y BE {be:.2f}%")
    if ry_level is not None:
        parts.append(f"10Y real {ry_level:.2f}%")
    return " | ".join(parts) if parts else ""


def classify_rates_regime(inputs: RatesInputs) -> RatesRegime:
    """
    Rates / yield curve regime classifier.

    Two-layer approach:
    1. **Curve level** — inversion is the strongest macro signal.
    2. **Curve dynamics** — the 4 classic steepener/flattener regimes based on
       how the front end (2Y) and long end (30Y, falling back to 10Y) are
       *moving* relative to each other over 20 trading days.
    3. **Rate shock** — unusually large 10Y moves vs recent history.
    4. **Static shape** — steep vs neutral when dynamics are quiet.

    Score guide: 0 = benign/easing → 100 = acute stress/tightening.
    """
    curve = inputs.curve_2s10s
    curve_2s30s = inputs.curve_2s30s
    z_curve = inputs.z_curve_2s10s
    z_dy = inputs.z_ust_10y_chg_20d

    d2y = inputs.ust_2y_chg_20d
    d30y = inputs.ust_30y_chg_20d
    d10y = inputs.ust_10y_chg_20d
    z_d2y = inputs.z_ust_2y_chg_20d
    z_d30y = inputs.z_ust_30y_chg_20d

    # Use 30Y momentum when available, else fall back to 10Y as the "long end".
    d_long = d30y if d30y is not None else d10y
    z_d_long = z_d30y if z_d30y is not None else z_dy

    real_note = _real_yield_note(inputs)

    # ── 1) Inversion — strongest macro signal ────────────────────────────
    if curve is not None and curve < 0:
        desc = "2s10s curve is inverted — historically a late-cycle / recession signal."
        if curve_2s30s is not None:
            desc += f" 2s30s: {curve_2s30s:+.0f}bp."
        if real_note:
            desc += f" ({real_note})"
        return RatesRegime(
            name="inverted_curve",
            label="Inverted curve (growth scare)",
            description=desc,
            tags=("rates", "curve", "growth_scare"),
            score=SCORE_INVERTED,
        )

    # ── 2) Steepener / flattener dynamics ────────────────────────────────
    # Require both front-end and long-end momentum to classify.
    # Use a significance gate: at least one z-score should be notable (|z| > 0.5)
    # to avoid classifying tiny noise as a regime.
    if d2y is not None and d_long is not None:
        has_signal = (
            (z_d2y is not None and abs(z_d2y) > Z_SIGNAL_GATE)
            or (z_d_long is not None and abs(z_d_long) > Z_SIGNAL_GATE)
        )

        if has_signal:
            # Bear steepener: front end stable/falling, long end rising
            if d2y <= MOMENTUM_THRESHOLD_PCT and d_long > MOMENTUM_THRESHOLD_PCT:
                desc = (
                    f"Front end stable/falling (2Y Δ20d {d2y:+.2f}%), long end selling off "
                    f"({'30Y' if d30y is not None else '10Y'} Δ20d {d_long:+.2f}%). "
                    "Classic fiscal supply / inflation-fear steepener — "
                    "bearish for duration, watch term premium."
                )
                if real_note:
                    desc += f" ({real_note})"
                return RatesRegime(
                    name="bear_steepener",
                    label="Bear steepener (long end selling off)",
                    description=desc,
                    tags=("rates", "curve", "bear_steepener", "duration_risk"),
                    score=SCORE_BEAR_STEEPENER,
                )

            # Bull flattener: front end stable/rising, long end falling
            if d2y >= -MOMENTUM_THRESHOLD_PCT and d_long < -MOMENTUM_THRESHOLD_PCT:
                desc = (
                    f"Long end rallying ({'30Y' if d30y is not None else '10Y'} Δ20d {d_long:+.2f}%), "
                    f"front end stable/rising (2Y Δ20d {d2y:+.2f}%). "
                    "Flight-to-safety flattener — risk-off, duration bid."
                )
                if real_note:
                    desc += f" ({real_note})"
                return RatesRegime(
                    name="bull_flattener",
                    label="Bull flattener (flight to safety)",
                    description=desc,
                    tags=("rates", "curve", "bull_flattener", "risk_off"),
                    score=SCORE_BULL_FLATTENER,
                )

            # Bear flattener: both rising, front end faster → hawkish tightening
            if d2y > MOMENTUM_THRESHOLD_PCT and d_long > MOMENTUM_THRESHOLD_PCT and d2y > d_long:
                desc = (
                    f"Both ends rising, front end faster (2Y Δ20d {d2y:+.2f}% vs "
                    f"{'30Y' if d30y is not None else '10Y'} Δ20d {d_long:+.2f}%). "
                    "Hawkish repricing — Fed tightening expectations flattening the curve."
                )
                if real_note:
                    desc += f" ({real_note})"
                return RatesRegime(
                    name="bear_flattener",
                    label="Bear flattener (hawkish tightening)",
                    description=desc,
                    tags=("rates", "curve", "bear_flattener", "tightening"),
                    score=SCORE_BEAR_FLATTENER,
                )

            # Bull steepener: both falling, front end faster → easing cycle
            if d2y < -MOMENTUM_THRESHOLD_PCT and d_long < -MOMENTUM_THRESHOLD_PCT and d2y < d_long:
                desc = (
                    f"Both ends falling, front end faster (2Y Δ20d {d2y:+.2f}% vs "
                    f"{'30Y' if d30y is not None else '10Y'} Δ20d {d_long:+.2f}%). "
                    "Easing cycle — market pricing rate cuts, curve steepening."
                )
                if real_note:
                    desc += f" ({real_note})"
                return RatesRegime(
                    name="bull_steepener",
                    label="Bull steepener (easing cycle)",
                    description=desc,
                    tags=("rates", "curve", "bull_steepener", "easing"),
                    score=SCORE_BULL_STEEPENER,
                )

    # ── 3) Rate shock (large 10Y move vs history) ────────────────────────
    if z_dy is not None and z_dy > Z_SHOCK_THRESHOLD:
        desc = "10Y yields have risen sharply vs recent history (20d). Duration headwind."
        if real_note:
            desc += f" ({real_note})"
        return RatesRegime(
            name="rates_shock_up",
            label="Rates shock higher (duration headwind)",
            description=desc,
            tags=("rates", "duration", "tightening"),
            score=SCORE_SHOCK_UP,
        )
    if z_dy is not None and z_dy < -Z_SHOCK_THRESHOLD:
        desc = "10Y yields have fallen sharply vs recent history (20d). Duration tailwind."
        if real_note:
            desc += f" ({real_note})"
        return RatesRegime(
            name="rates_shock_down",
            label="Rates shock lower (duration tailwind)",
            description=desc,
            tags=("rates", "duration", "easing"),
            score=SCORE_SHOCK_DOWN,
        )

    # ── 4) Static curve shape ────────────────────────────────────────────
    if (curve is not None and curve > STEEP_CURVE_THRESHOLD) or (z_curve is not None and z_curve > STEEP_CURVE_THRESHOLD):
        desc = "Curve is steep — consistent with expansion / reflation."
        if real_note:
            desc += f" ({real_note})"
        return RatesRegime(
            name="steep_curve",
            label="Steep curve (risk-on / reflation)",
            description=desc,
            tags=("rates", "curve", "expansion"),
            score=SCORE_STEEP_CURVE,
        )

    # ── 5) Neutral ───────────────────────────────────────────────────────
    desc = "No strong signal from curve dynamics or rate shocks."
    if real_note:
        desc += f" ({real_note})"
    return RatesRegime(
        name="neutral",
        label="Neutral rates backdrop",
        description=desc,
        tags=("rates",),
        score=SCORE_NEUTRAL,
    )

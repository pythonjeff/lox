from __future__ import annotations

from dataclasses import dataclass

from lox.rates.models import RatesInputs


@dataclass(frozen=True)
class RatesRegime:
    name: str
    label: str | None = None
    description: str = ""
    tags: tuple[str, ...] = ()


def classify_rates_regime(inputs: RatesInputs) -> RatesRegime:
    """
    Rates / yield curve regime classifier (MVP).

    Intuition:
    - Curve level (inversion vs steep) is a macro growth/stress signal.
    - 10y yield change is the "duration shock" channel most relevant to equities.
    - Z-scores are vs recent history; they are NOT a claim about absolute rarity.
    """
    curve = inputs.curve_2s10s
    z_curve = inputs.z_curve_2s10s
    z_dy = inputs.z_ust_10y_chg_20d

    # 1) Growth scare / inversion
    if curve is not None and curve < 0:
        return RatesRegime(
            name="inverted_curve",
            label="Inverted curve (growth scare)",
            description="2s10s curve is inverted; this often coincides with late-cycle / growth-scare conditions.",
            tags=("rates", "curve", "growth_scare"),
        )

    # 2) Big duration shock (bearish for long-duration equities)
    if z_dy is not None and z_dy > 1.5:
        return RatesRegime(
            name="rates_shock_up",
            label="Rates shock higher (duration headwind)",
            description="10y yields have risen quickly vs recent history (20d change). This is a headwind for long-duration assets.",
            tags=("rates", "duration", "tightening"),
        )
    if z_dy is not None and z_dy < -1.5:
        return RatesRegime(
            name="rates_shock_down",
            label="Rates shock lower (duration tailwind)",
            description="10y yields have fallen quickly vs recent history (20d change). This is supportive for long-duration assets.",
            tags=("rates", "duration", "easing"),
        )

    # 3) Steep / normal curve
    if (curve is not None and curve > 1.0) or (z_curve is not None and z_curve > 1.0):
        return RatesRegime(
            name="steep_curve",
            label="Steep curve (risk-on / reflation-ish)",
            description="Curve is relatively steep; historically this is more consistent with expansions than late-cycle inversions.",
            tags=("rates", "curve", "expansion"),
        )

    return RatesRegime(
        name="neutral",
        label="Neutral rates backdrop",
        description="No strong signal from curve level or recent 10y yield shock.",
        tags=("rates",),
    )



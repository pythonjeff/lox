from __future__ import annotations

from dataclasses import dataclass

from lox.monetary.models import MonetaryInputs


@dataclass
class MonetaryRegime:
    name: str
    description: str
    label: str | None = None


MONETARY_REGIME_CHOICES = (
    "abundant_reserves",
    "neutral",
    "thinning_buffers_transitional",
    "qt_biting",
)


def classify_monetary_regime(inputs: MonetaryInputs) -> MonetaryRegime:
    """
    Lean monetary regime classifier (MVP) using only:
    1) EFFR (DFF)
    2) Total reserves (TOTRESNS)
    3) Fed balance sheet size Δ (WALCL Δ)
    4) ON RRP usage (RRPONTSYD)

    Goal:
    - Distinguish abundant vs constrained reserves
    - Identify when QT is likely "biting" (assets shrinking + reserves weak + RRP low)
    """
    z_res = inputs.z_total_reserves
    z_rrp = inputs.z_on_rrp
    z_qt = inputs.z_fed_assets_chg_13w

    # Heuristics (tune over time):
    # - Low reserves: z < -0.75
    # - High reserves: z > +0.75
    # - Low RRP: z < -0.75 (facility mostly drained)
    # - QT biting: balance sheet shrinking unusually fast (z_qt < -0.75) + low reserves + low RRP
    low_res = z_res is not None and z_res < -0.75
    high_res = z_res is not None and z_res > 0.75
    low_rrp = z_rrp is not None and z_rrp < -0.75

    qt_biting = (z_qt is not None and z_qt < -0.75) and low_res and low_rrp
    if qt_biting:
        return MonetaryRegime(
            name="qt_biting",
            label="QT Biting (reserves constrained)",
            description="Fed assets are shrinking quickly while reserves look low and ON RRP is drained—plumbing constraints more likely to show up.",
        )

    if high_res:
        return MonetaryRegime(
            name="abundant_reserves",
            label="Abundant Reserves",
            description="Reserves look high vs recent history; monetary plumbing appears ample.",
        )

    if low_res and low_rrp:
        return MonetaryRegime(
            name="thinning_buffers_transitional",
            label="Thinning Buffers / Transitional",
            description=(
                "Reserves look low vs recent history and ON RRP appears drained—buffers are thinner. "
                "Hold off on strong conclusions until funding stress signals (repo spreads) or SRF/discount-window usage are added."
            ),
        )

    return MonetaryRegime(
        name="neutral",
        label="Neutral / Normal",
        description="No clear signal of constrained reserves or acute QT bite from the MVP plumbing metrics.",
    )



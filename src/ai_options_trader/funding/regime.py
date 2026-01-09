from __future__ import annotations

from dataclasses import dataclass

from ai_options_trader.funding.models import FundingInputs


@dataclass
class FundingRegime:
    name: str
    description: str
    label: str | None = None


FUNDING_REGIME_CHOICES = (
    "normal_funding",
    "tightening_balance_sheet_constraint",
    "funding_stress",
)


def classify_funding_regime(inputs: FundingInputs) -> FundingRegime:
    """
    Lean funding regime classifier (MVP).

    Key idea:
    - Avoid fixed levels. Use distributional baselines computed from the last ~2–3y:
      baseline = median, stress = baseline + k·std (k in [1.5, 2.0] by default in signals).
    - Distinguish event vs regime via:
      - spike indicator (rolling 5d max)
      - persistence (share of last 20d above stress threshold)
      - spread volatility (rolling 20d std)
    """
    s = inputs.spread_corridor_bps
    spike = inputs.spike_5d_bps
    p = inputs.persistence_20d
    vol = inputs.vol_20d_bps

    tight = inputs.tight_threshold_bps
    stress = inputs.stress_threshold_bps
    p_tight = inputs.persistence_tight
    p_stress = inputs.persistence_stress
    vol_tight = inputs.vol_tight_bps
    vol_stress = inputs.vol_stress_bps

    # If we have no corridor spread, we can't classify meaningfully.
    if s is None:
        return FundingRegime(
            name="normal_funding",
            label="Normal funding",
            description="Insufficient data to compute corridor dislocation; defaulting to Normal funding.",
        )

    # Funding stress: persistent above stress threshold, or spike+vol both elevated.
    if (
        (p is not None and p_stress is not None and p >= p_stress)
        or (spike is not None and stress is not None and spike >= stress and vol is not None and vol_stress is not None and vol >= vol_stress)
    ):
        return FundingRegime(
            name="funding_stress",
            label="Funding stress",
            description="Corridor dislocations are wide and persistent with elevated spread volatility; liquidity is not elastic.",
        )

    # Tightening: above tight threshold more often, volatility rising, or occasional spikes.
    if (
        (p is not None and p_tight is not None and p >= p_tight)
        or (vol is not None and vol_tight is not None and vol >= vol_tight)
        or (spike is not None and stress is not None and spike >= stress)
        or (s is not None and tight is not None and s >= tight)
    ):
        return FundingRegime(
            name="tightening_balance_sheet_constraint",
            label="Tightening / balance-sheet constraint",
            description="Corridor dislocations are more common and/or spread volatility is rising; stress events possible but not a persistent blowout.",
        )

    return FundingRegime(
        name="normal_funding",
        label="Normal funding",
        description="Spreads near baseline with low volatility and limited spike/persistence.",
    )



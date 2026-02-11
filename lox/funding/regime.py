from __future__ import annotations

from dataclasses import dataclass, field

from lox.funding.models import FundingInputs


@dataclass
class FundingRegime:
    name: str
    description: str
    label: str | None = None
    score: int = 50


FUNDING_REGIME_CHOICES = (
    "normal_funding",
    "tightening_balance_sheet_constraint",
    "structural_tightening",
    "funding_stress",
)


def classify_funding_regime(inputs: FundingInputs) -> FundingRegime:
    """
    Funding regime classifier with continuous scoring.

    Two-layer approach:
    1. **Corridor dynamics** (base score): SOFR-IORB spread, spike, persistence,
       volatility — assessed against distributional baselines (median + k*std).
    2. **Structural liquidity amplifiers**: ON RRP buffer depletion, bank reserves
       proximity to stress, TGA drain pace.  These shift the base score up/down
       because the *same* corridor spread reading is more dangerous when the
       system has no buffer (ON RRP near zero, reserves near minimum comfort).

    Score guide: 0 = ample liquidity (risk-on) → 100 = severe funding stress (risk-off).
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
            score=50,
            description="Insufficient data to compute corridor dislocation; defaulting to Normal funding.",
        )

    # ── Layer 1: Base score from corridor dynamics ───────────────────────
    base_score = 40  # start at "normal"

    # Funding stress: persistent above stress threshold, or spike+vol both elevated.
    is_stress = (
        (p is not None and p_stress is not None and p >= p_stress)
        or (spike is not None and stress is not None and spike >= stress
            and vol is not None and vol_stress is not None and vol >= vol_stress)
    )
    # Tightening: above tight threshold more often, volatility rising, or occasional spikes.
    is_tightening = (
        (p is not None and p_tight is not None and p >= p_tight)
        or (vol is not None and vol_tight is not None and vol >= vol_tight)
        or (spike is not None and stress is not None and spike >= stress)
        or (s is not None and tight is not None and s >= tight)
    )

    if is_stress:
        base_score = 78
    elif is_tightening:
        base_score = 58

    # Fine-tune within bands based on how far spread is from thresholds
    if tight is not None and stress is not None and s is not None:
        if is_tightening and not is_stress:
            # Scale 58-74 based on distance toward stress threshold
            ratio = min(1.0, max(0.0, (s - tight) / (stress - tight))) if stress > tight else 0.0
            base_score = 58 + int(ratio * 16)
        elif not is_tightening and not is_stress:
            # Scale 25-40 based on distance from zero to tight
            if tight > 0 and s > 0:
                ratio = min(1.0, max(0.0, s / tight))
                base_score = 25 + int(ratio * 15)

    # ── Layer 2: Structural liquidity amplifiers ─────────────────────────
    structural_adj = 0
    structural_tags: list[str] = []

    # ON RRP buffer depletion — when RRP is near zero, settlement days hit
    # risk assets directly because there's no cushion to absorb the drain.
    # RRP values in FundingInputs are in millions (displayed as /1000 for billions).
    rrp = inputs.on_rrp_usd_bn
    if rrp is not None:
        rrp_bn = rrp / 1000.0  # convert to billions for threshold checks
        if rrp_bn < 100:
            structural_adj += 8    # buffer essentially gone
            structural_tags.append("rrp_depleted")
        elif rrp_bn < 200:
            structural_adj += 5    # buffer thin
            structural_tags.append("rrp_thin")
        elif rrp_bn > 500:
            structural_adj -= 5    # ample buffer

    # Bank reserves proximity to stress — the "lowest comfortable level of
    # reserves" (LCLoR) is estimated around $3.0-3.2T.  Below that, banks
    # start hoarding and repo markets seize up.
    # Reserves in FundingInputs are in millions.
    reserves = inputs.bank_reserves_usd_bn
    if reserves is not None:
        reserves_tn = reserves / 1_000_000.0  # convert to trillions
        if reserves_tn < 3.0:
            structural_adj += 8    # below minimum comfort zone
            structural_tags.append("reserves_scarce")
        elif reserves_tn < 3.3:
            structural_adj += 5    # approaching stress zone
            structural_tags.append("reserves_thin")
        elif reserves_tn > 4.0:
            structural_adj -= 3    # ample reserves

    # TGA drain — large TGA inflows (settlement weeks) pull cash from
    # the private financial system.  TGA increasing rapidly = active drain.
    tga_z = inputs.z_tga_chg_4w
    tga_chg = inputs.tga_chg_4w
    if tga_z is not None and tga_z > 1.5:
        structural_adj += 5        # unusually large TGA build (heavy settlements)
        structural_tags.append("tga_drain")
    elif tga_chg is not None and tga_chg > 100_000:
        structural_adj += 5        # > $100B 4-week TGA inflow
        structural_tags.append("tga_drain")
    elif tga_chg is not None and tga_chg < -50_000:
        structural_adj -= 3        # TGA spending down → adding liquidity

    # ── Combine and clamp ────────────────────────────────────────────────
    score = max(0, min(100, base_score + structural_adj))

    # ── Label ────────────────────────────────────────────────────────────
    if score >= 75:
        name = "funding_stress"
        label = "Funding stress"
    elif score >= 65:
        name = "structural_tightening"
        label = "Structural tightening"
    elif score >= 55:
        name = "tightening_balance_sheet_constraint"
        label = "Tightening / balance-sheet constraint"
    elif score >= 45:
        name = "normal_funding"
        label = "Normal funding"
    elif score >= 30:
        name = "normal_funding"
        label = "Ample funding"
    else:
        name = "normal_funding"
        label = "Flush liquidity"

    # ── Description ──────────────────────────────────────────────────────
    parts: list[str] = []
    if s is not None:
        corridor_name = inputs.spread_corridor_name or "SOFR-IORB"
        parts.append(f"{corridor_name}: {s:+.1f}bp")
    if rrp is not None:
        rrp_bn = rrp / 1000.0
        if rrp_bn >= 1000:
            parts.append(f"ON RRP: ${rrp_bn / 1000:.1f}T")
        else:
            parts.append(f"ON RRP: ${rrp_bn:.0f}B")
    if reserves is not None:
        parts.append(f"Reserves: ${reserves / 1_000_000:.1f}T")
    if structural_tags:
        parts.append(f"⚠️ {', '.join(structural_tags)}")

    description = " | ".join(parts) if parts else label

    return FundingRegime(
        name=name,
        label=label,
        score=score,
        description=description,
    )



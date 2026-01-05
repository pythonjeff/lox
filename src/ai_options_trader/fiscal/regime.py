from __future__ import annotations

from dataclasses import dataclass

from ai_options_trader.fiscal.models import FiscalInputs


@dataclass
class FiscalRegime:
    name: str
    description: str
    # Optional display-friendly label (keep `name` stable for programmatic use / tests).
    label: str | None = None


FISCAL_REGIME_CHOICES = (
    "benign_funding",
    "heavy_funding",
    "stress_building",
    "fiscal_dominance_risk",
)


def classify_fiscal_regime(inputs: FiscalInputs) -> FiscalRegime:
    """
    Rule-based fiscal regime classifier (explainable / low noise).

    Mapping (user spec):
    - benign_funding: low deficits, bill-heavy issuance, strong auctions
    - heavy_funding: large deficits, mixed issuance, stable auctions
    - stress_building: large deficits, long issuance, rising dealer take
    - fiscal_dominance_risk: accelerating interest expense + weak auctions

    Implementation detail:
    - Prefer standardized z-scores when available.
    - Fall back to raw values when z-scores aren't present.
    - Missing optional issuance/auction series should *not* break classification;
      we classify using the available core signals.
    """
    z_def = inputs.z_deficit_12m
    z_long = inputs.z_long_duration_issuance_share
    z_tail = inputs.z_auction_tail_bps
    z_dealer = inputs.z_dealer_take_pct
    z_ie = inputs.z_interest_expense_yoy

    # Weak auctions: tails and/or high dealer take
    weak_auctions = False
    if z_tail is not None and z_tail > 0.75:
        weak_auctions = True
    if z_dealer is not None and z_dealer > 0.75:
        weak_auctions = True

    # Long-duration tilt (issuance mix)
    long_tilt = z_long is not None and z_long > 0.50

    # "Large deficits" as relative positioning
    large_deficits = z_def is not None and z_def > 0.50
    very_large_deficits = z_def is not None and z_def > 1.25

    # Interest expense growth pressure
    accelerating_interest = (
        (inputs.interest_expense_yoy_accel is not None and inputs.interest_expense_yoy_accel > 0.0)
        or (z_ie is not None and z_ie > 1.0)
    )

    # 1) Fiscal dominance risk: accelerating interest expense + weak auctions
    if accelerating_interest and weak_auctions:
        return FiscalRegime(
            name="fiscal_dominance_risk",
            description="Interest expense accelerating and auctions weakening (tails and/or dealer take elevated).",
        )

    # 2) Stress building: large deficits + long issuance tilt + rising dealer take/auction weakness
    if (large_deficits or very_large_deficits) and long_tilt and (weak_auctions or (z_dealer is not None and z_dealer > 0.25)):
        return FiscalRegime(
            name="stress_building",
            description="Large deficits alongside longer-duration issuance tilt and rising intermediation burden (dealer take / auction quality).",
        )

    # 3) Heavy funding: large deficits, but auctions stable / not clearly long-tilted
    if large_deficits or very_large_deficits:
        return FiscalRegime(
            name="heavy_funding",
            description="Large deficits imply heavy funding needs; issuance/auctions appear broadly stable (no clear stress signal yet).",
        )

    # 4) Benign: default when funding pressure is low
    return FiscalRegime(
        name="benign_funding",
        description="Funding environment appears benign: deficits not elevated; no clear auction stress or long-issuance tilt.",
    )


def classify_fiscal_regime_snapshot(
    *,
    deficit_pct_gdp: float | None,
    deficit_impulse_pct_gdp: float | None,
    long_duration_issuance_share: float | None,
    tga_z_d_4w: float | None,
    auction_tail_bps: float | None = None,
    dealer_take_pct: float | None = None,
) -> FiscalRegime:
    """
    Snapshot-focused fiscal regime labeler.

    Goal:
    - Always return a concrete, human-friendly regime label even when many optional series are missing.
    - Use only the signals available in the `lox fiscal snapshot` panel today.

    Inputs are best-effort:
    - deficit_pct_gdp: rolling-12m deficit as % of GDP (positive = deficit)
    - deficit_impulse_pct_gdp: YoY change in rolling-12m deficit as % of GDP (negative = improving)
    - long_duration_issuance_share: share of net issuance in >=10y bucket (0..1, best-effort)
    - tga_z_d_4w: z-score of 4-week change in TGA (positive = tightening/drain, negative = easing/injection)
    - auction_tail_bps: tail proxy (higher = weaker demand / worse clearing)
    - dealer_take_pct: primary dealer takedown as % of accepted (higher = weaker non-dealer demand)
    """

    # Directional context for the label.
    direction = "stable"
    if isinstance(deficit_impulse_pct_gdp, (int, float)):
        if float(deficit_impulse_pct_gdp) <= -0.75:
            direction = "improving"
        elif float(deficit_impulse_pct_gdp) >= 0.75:
            direction = "deteriorating"

    # Funding pressure level: coarse buckets by deficit size (as % GDP).
    # These are intentionally simple and should be tuned as you gather intuition.
    pressure = "unknown"
    if isinstance(deficit_pct_gdp, (int, float)):
        d = float(deficit_pct_gdp)
        if d < 3.0:
            pressure = "low"
        elif d < 6.0:
            pressure = "moderate"
        else:
            pressure = "high"

    # Duration tilt: are we leaning long in net issuance?
    long_tilt = False
    if isinstance(long_duration_issuance_share, (int, float)):
        # If most net issuance is coming from the long bucket, duration absorption risk rises.
        long_tilt = float(long_duration_issuance_share) >= 0.40

    # Auction absorption (best-effort): these are the highest-signal "market absorption" fields once wired.
    weak_auctions = False
    if isinstance(auction_tail_bps, (int, float)) and float(auction_tail_bps) >= 5.0:
        weak_auctions = True
    if isinstance(dealer_take_pct, (int, float)) and float(dealer_take_pct) >= 35.0:
        weak_auctions = True

    # Liquidity pulse from TGA: bias the stress label one notch.
    # (TGA down sharply = easing; TGA up sharply = tightening)
    liq_bias = 0
    if isinstance(tga_z_d_4w, (int, float)):
        z = float(tga_z_d_4w)
        if z <= -0.75:
            liq_bias = -1
        elif z >= 0.75:
            liq_bias = +1

    # Base regime selection.
    if pressure == "low":
        base = "benign_funding"
        desc = "Deficit level is low relative to GDP, with no obvious duration/auction stress in the snapshot signals."
    elif pressure == "moderate":
        base = "heavy_funding"
        desc = "Deficits imply meaningful funding needs, but snapshot signals don’t show clear market-absorption stress."
    else:
        # high or unknown -> assume heavy funding at minimum
        base = "heavy_funding"
        desc = "Deficits imply heavy funding needs; watch issuance mix (duration) and liquidity conditions for stress escalation."

    # Escalate to stress_building if duration tilt is high under moderate/high pressure.
    if long_tilt and pressure in {"moderate", "high", "unknown"}:
        base = "stress_building"
        desc = "Funding needs look heavy and issuance mix appears long-tilted, increasing duration absorption stress risk."

    # Escalate to stress_building if auctions look weak (even if issuance mix isn't long-tilted).
    if weak_auctions and pressure in {"moderate", "high", "unknown"}:
        base = "stress_building"
        desc = "Auction absorption looks weak (tails and/or dealer take elevated), raising funding/financial-conditions stress risk."

    # Liquidity bias adjustment (don’t jump straight to dominance risk in snapshot mode).
    if base == "stress_building" and liq_bias == -1:
        base = "heavy_funding"
        desc = "Funding needs are heavy, but near-term liquidity looks supportive (TGA down sharply), reducing immediate stress risk."
    elif base == "heavy_funding" and liq_bias == +1 and pressure in {"high", "unknown"}:
        base = "stress_building"
        desc = "Funding needs are heavy and near-term liquidity looks restrictive (TGA up sharply), increasing stress risk."

    label = base.replace("_", " ").title()
    if direction != "stable":
        label = f"{label} ({direction})"

    # Add a compact “why” line with the core numbers when available.
    bits: list[str] = []
    if isinstance(deficit_pct_gdp, (int, float)):
        bits.append(f"deficit≈{float(deficit_pct_gdp):.1f}% GDP")
    if isinstance(deficit_impulse_pct_gdp, (int, float)):
        bits.append(f"impulse={float(deficit_impulse_pct_gdp):+.2f}% GDP")
    if isinstance(long_duration_issuance_share, (int, float)):
        bits.append(f"long share={100.0*float(long_duration_issuance_share):.1f}%")
    if isinstance(tga_z_d_4w, (int, float)):
        bits.append(f"TGA z(Δ4w)={float(tga_z_d_4w):+.2f}")
    if isinstance(auction_tail_bps, (int, float)):
        bits.append(f"tail≈{float(auction_tail_bps):.1f}bp")
    if isinstance(dealer_take_pct, (int, float)):
        bits.append(f"dealer take≈{float(dealer_take_pct):.1f}%")

    detail = ""
    if bits:
        detail = " (" + ", ".join(bits) + ")"

    return FiscalRegime(name=base, label=label, description=f"{desc}{detail}")


def classify_fiscal_regime_skeleton(
    *,
    deficit_12m: float | None,
    gdp_millions: float | None,
    deficit_impulse_pct_gdp: float | None,
    long_duration_issuance_share: float | None,
    tga_z_d_4w: float | None,
    auction_tail_bps: float | None = None,
    dealer_take_pct: float | None = None,
) -> FiscalRegime:
    """
    Skeleton classifier for learning/iteration.

    This function is used by the default `lox fiscal snapshot` panel path.
    It intentionally uses only the snapshot signals so it can always return an answer.
    """
    deficit_pct_gdp = None
    if isinstance(deficit_12m, (int, float)) and isinstance(gdp_millions, (int, float)) and float(gdp_millions) != 0.0:
        deficit_pct_gdp = 100.0 * float(deficit_12m) / float(gdp_millions)

    return classify_fiscal_regime_snapshot(
        deficit_pct_gdp=deficit_pct_gdp,
        deficit_impulse_pct_gdp=deficit_impulse_pct_gdp,
        long_duration_issuance_share=long_duration_issuance_share,
        tga_z_d_4w=tga_z_d_4w,
        auction_tail_bps=auction_tail_bps,
        dealer_take_pct=dealer_take_pct,
    )



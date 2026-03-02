from __future__ import annotations

from dataclasses import dataclass

from lox.fiscal.models import FiscalInputs


@dataclass
class FiscalRegime:
    name: str
    description: str
    # Optional display-friendly label (keep `name` stable for programmatic use / tests).
    label: str | None = None


FISCAL_REGIME_CHOICES = (
    "fiscal_contraction",
    "moderate_fiscal_support",
    "strong_fiscal_stimulus",
    "fiscal_dominance_risk",
)


def classify_fiscal_regime(inputs: FiscalInputs) -> FiscalRegime:
    """
    Rule-based fiscal regime classifier (MMT sectoral balance framework).

    In MMT, government deficit = private sector surplus (net of foreign sector).
    Smaller deficits are contractionary; larger deficits support private balance sheets.

    Mapping:
    - fiscal_contraction: low deficits → private sector starved of NFA
    - moderate_fiscal_support: moderate deficits, mixed signals
    - strong_fiscal_stimulus: large deficits → strong NFA injection
    - fiscal_dominance_risk: accelerating interest expense + weak auctions
    """
    z_def = inputs.z_deficit_12m
    z_long = inputs.z_long_duration_issuance_share
    z_tail = inputs.z_auction_tail_bps
    z_dealer = inputs.z_dealer_take_pct
    z_ie = inputs.z_interest_expense_yoy

    weak_auctions = False
    if z_tail is not None and z_tail > 0.75:
        weak_auctions = True
    if z_dealer is not None and z_dealer > 0.75:
        weak_auctions = True

    long_tilt = z_long is not None and z_long > 0.50

    # MMT inversion: lower z_def = smaller deficit = less private sector support
    small_deficits = z_def is not None and z_def < -0.50
    large_deficits = z_def is not None and z_def > 0.50

    accelerating_interest = (
        (inputs.interest_expense_yoy_accel is not None and inputs.interest_expense_yoy_accel > 0.0)
        or (z_ie is not None and z_ie > 1.0)
    )

    # 1) Fiscal dominance risk: accelerating interest expense + weak auctions
    if accelerating_interest and weak_auctions:
        return FiscalRegime(
            name="fiscal_dominance_risk",
            description="Interest expense accelerating and auctions weakening — debt dynamics risk.",
        )

    # 2) Fiscal contraction: small deficits = private sector squeeze
    if small_deficits:
        desc = "Deficit below historical norm — private sector NFA accumulation insufficient."
        if weak_auctions:
            desc += " Auction stress compounds the contractionary signal."
        return FiscalRegime(name="fiscal_contraction", description=desc)

    # 3) Strong fiscal stimulus: large deficits = strong NFA injection
    if large_deficits and not weak_auctions:
        return FiscalRegime(
            name="strong_fiscal_stimulus",
            description="Large deficit injecting significant NFA into private sector. Supportive for risk assets.",
        )

    # 4) Moderate support: default
    desc = "Deficit provides moderate private sector NFA flow."
    if weak_auctions:
        desc += " Auction absorption showing strain (tails/dealer take elevated)."
    if long_tilt:
        desc += " Long-duration issuance tilt adds duration absorption risk."
    return FiscalRegime(name="moderate_fiscal_support", description=desc)


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
    MMT-oriented fiscal regime labeler (sectoral balance framework).

    In MMT: government deficit = private sector surplus (net of foreign sector).
    A shrinking deficit is contractionary — private sector NFA accumulation declines.

    Inputs are best-effort:
    - deficit_pct_gdp: rolling-12m deficit as % of GDP (positive = deficit = private NFA injection)
    - deficit_impulse_pct_gdp: YoY change in deficit as % of GDP
      (negative = fiscal drag / private surplus shrinking)
    - long_duration_issuance_share: share of net issuance in >=10y bucket (0..1)
    - tga_z_d_4w: z-score of 4-week TGA change (positive = NFA drain from private sector)
    - auction_tail_bps: tail proxy (higher = weaker demand / worse clearing)
    - dealer_take_pct: primary dealer takedown as % of accepted
    """

    # Directional context: MMT polarity — negative impulse = fiscal drag (bearish).
    direction = "stable"
    if isinstance(deficit_impulse_pct_gdp, (int, float)):
        if float(deficit_impulse_pct_gdp) <= -0.75:
            direction = "contracting"
        elif float(deficit_impulse_pct_gdp) >= 0.75:
            direction = "expanding"

    # NFA injection level: larger deficit = more private sector support.
    nfa_level = "unknown"
    if isinstance(deficit_pct_gdp, (int, float)):
        d = float(deficit_pct_gdp)
        if d < 3.0:
            nfa_level = "low"
        elif d < 6.0:
            nfa_level = "moderate"
        else:
            nfa_level = "high"

    # Duration tilt: long-tilted issuance = more duration absorption risk.
    long_tilt = False
    if isinstance(long_duration_issuance_share, (int, float)):
        long_tilt = float(long_duration_issuance_share) >= 0.40

    # Auction absorption stress (framework-agnostic market mechanics).
    weak_auctions = False
    if isinstance(auction_tail_bps, (int, float)) and float(auction_tail_bps) >= 3.0:
        weak_auctions = True
    if isinstance(dealer_take_pct, (int, float)) and float(dealer_take_pct) >= 25.0:
        weak_auctions = True

    # TGA reserve drain: positive z = TGA rising = draining private sector reserves.
    drain_bias = 0
    if isinstance(tga_z_d_4w, (int, float)):
        z = float(tga_z_d_4w)
        if z <= -0.75:
            drain_bias = -1  # TGA drawdown = reserve injection (supportive)
        elif z >= 0.75:
            drain_bias = +1  # TGA build = reserve drain (contractionary)

    # ── Base regime: determined by NFA injection level ──────────────────
    # MMT inversion: low deficit = private sector squeeze (bearish for risk assets).
    if nfa_level == "low":
        base = "fiscal_contraction"
        desc = "Deficit too small to sustain private sector NFA accumulation. Contractionary for risk assets."
    elif nfa_level == "moderate":
        base = "moderate_fiscal_support"
        desc = "Moderate deficit provides some private sector NFA flow, but impulse direction matters."
    else:
        base = "strong_fiscal_stimulus"
        desc = "Large deficit injects significant NFA into the private sector. Supportive for risk assets."

    # ── Impulse override: shrinking deficit = drag even if level is still high ──
    if isinstance(deficit_impulse_pct_gdp, (int, float)):
        imp = float(deficit_impulse_pct_gdp)
        if imp <= -1.0 and nfa_level != "high":
            base = "fiscal_contraction"
            desc = "Fiscal impulse sharply negative — private sector surplus shrinking fast. Bearish for risk assets."
        elif imp <= -1.0 and nfa_level == "high":
            base = "moderate_fiscal_support"
            desc = "Deficit level still high, but impulse sharply negative — NFA injection decelerating."

    # ── TGA drain amplifier ─────────────────────────────────────────────
    if drain_bias == +1 and base != "fiscal_contraction":
        if base == "strong_fiscal_stimulus":
            base = "moderate_fiscal_support"
            desc = "Deficit is large but TGA build-up is draining reserves from private sector, offsetting NFA injection."
        elif base == "moderate_fiscal_support":
            base = "fiscal_contraction"
            desc = "Moderate deficit combined with TGA reserve drain — net effect is contractionary for private sector."
    elif drain_bias == -1 and base == "fiscal_contraction":
        base = "moderate_fiscal_support"
        desc = "Deficit is small, but TGA drawdown is injecting reserves back into private sector (near-term relief)."

    # ── Auction stress overlay (market mechanics, framework-agnostic) ───
    if weak_auctions:
        desc += " Auction absorption weak (tails/dealer take elevated)."
    if long_tilt:
        desc += " Long-duration issuance tilt adds duration absorption risk."

    label = base.replace("_", " ").title()
    if direction != "stable":
        label = f"{label} ({direction})"

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

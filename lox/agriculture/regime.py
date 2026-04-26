from __future__ import annotations

from dataclasses import dataclass

from lox.agriculture.models import AgricultureInputs


@dataclass(frozen=True)
class AgricultureRegime:
    name: str
    label: str | None = None
    description: str = ""
    tags: tuple[str, ...] = ()


def _positioning_context(inputs: AgricultureInputs) -> str:
    """Build a positioning summary string for regime descriptions."""
    parts = []
    if inputs.cot_corn_z is not None:
        if inputs.cot_corn_z > 1.5:
            parts.append("specs crowded long corn")
        elif inputs.cot_corn_z < -1.5:
            parts.append("specs heavily short corn")
    if inputs.wasde_corn_stu_pct is not None and inputs.wasde_corn_stu_pct < 10:
        parts.append(f"tight S/U ({inputs.wasde_corn_stu_pct:.1f}%)")
    return "; ".join(parts)


def _acreage_context(inputs: AgricultureInputs) -> str:
    """Build acreage & crop condition context for regime descriptions."""
    parts = []
    if inputs.corn_planted_yoy_pct is not None:
        direction = "up" if inputs.corn_planted_yoy_pct > 0 else "down"
        parts.append(f"corn acres {direction} {abs(inputs.corn_planted_yoy_pct):.1f}% YoY")
    if inputs.soy_planted_yoy_pct is not None:
        direction = "up" if inputs.soy_planted_yoy_pct > 0 else "down"
        parts.append(f"soy acres {direction} {abs(inputs.soy_planted_yoy_pct):.1f}%")
    # Multi-crop condition context
    for label, ge in [
        ("corn", inputs.corn_condition_ge),
        ("soy", inputs.soy_condition_ge),
        ("wheat", inputs.wheat_condition_ge),
    ]:
        if ge is not None:
            if ge >= 70:
                parts.append(f"{label} condition strong ({ge:.0f}% G/E)")
            elif ge < 55:
                parts.append(f"{label} condition poor ({ge:.0f}% G/E)")
    if inputs.crop_condition_composite is not None:
        parts.append(f"composite G/E {inputs.crop_condition_composite:.0f}%")
    if inputs.corn_pct_planted is not None and inputs.corn_pct_planted_vs_avg is not None:
        if inputs.corn_pct_planted_vs_avg < -10:
            parts.append(f"corn planting delayed ({inputs.corn_pct_planted:.0f}% vs avg)")
    if inputs.planting_delay_count is not None and inputs.planting_delay_count >= 2:
        parts.append(f"{inputs.planting_delay_count} crops behind planting schedule")
    return "; ".join(parts)


def classify_agriculture_regime(inputs: AgricultureInputs) -> AgricultureRegime:
    """
    Agriculture & food-inflation regime classifier.

    Classifies the current state of agricultural markets based on:
    - Input costs (natural gas, fertilizer, diesel)
    - Crop price momentum (corn, wheat, soybeans)
    - Divergences between input costs and crop prices
    - CFTC speculative positioning (crowding / capitulation)
    - WASDE supply/demand (stocks-to-use tightness)
    - NASS crop reports (acreage shifts, planting progress, crop condition)

    Regimes (ordered by severity):
    1. ag_input_shock     — Broad input cost spike (nat gas, fertilizer, diesel).
    2. food_inflation     — Both inputs AND crops rising — active food inflation.
    3. crop_supply_risk   — Acreage down or crop condition poor — supply risk.
    4. crop_surge         — Crop prices spiking (weather, demand, geopolitics).
    5. cost_pass_through  — Input costs elevated but crops haven't followed yet.
    6. ag_disinflation    — Input costs and/or crop prices falling.
    7. neutral            — No strong directional signal.
    """
    input_score = inputs.input_cost_score
    crop_score = inputs.crop_momentum_score
    food_score = inputs.food_inflation_score
    pos_ctx = _positioning_context(inputs)
    acre_ctx = _acreage_context(inputs)

    # 1. Input shock — broad input cost spike
    if inputs.input_shock:
        desc = (
            "Natural gas, fertilizer, and/or diesel costs are spiking. "
            "This historically leads to higher crop prices and food inflation with a lag. "
            "Watch for cost pass-through into corn, wheat, and food CPI."
        )
        if pos_ctx:
            desc += f" Positioning: {pos_ctx}."
        return AgricultureRegime(
            name="ag_input_shock",
            label="Agriculture input shock",
            description=desc,
            tags=("agriculture", "input_costs", "inflation", "food"),
        )

    # 1b. Broad food inflation — consumer CPI breadth + level
    if inputs.broad_food_inflation:
        desc = (
            f"Food CPI running at {inputs.cpi_food_yoy:+.1f}% YoY with {inputs.food_breadth_count}/5 "
            "sub-categories above their 3yr medians — broad-based consumer food inflation."
        )
        if inputs.cpi_food_accel is not None and inputs.cpi_food_accel > 0:
            desc += f" Momentum accelerating ({inputs.cpi_food_3m_ann:+.1f}% 3m ann vs {inputs.cpi_food_yoy:+.1f}% YoY)."
        if inputs.protein_spike:
            desc += " Protein prices spiking — headline risk."
        if inputs.grocery_shock:
            desc += f" Grocery prices outpacing restaurants by {inputs.grocery_restaurant_gap:+.1f}pp — consumer wallet shock."
        if pos_ctx:
            desc += f" Positioning: {pos_ctx}."
        return AgricultureRegime(
            name="broad_food_inflation",
            label="Broad food inflation",
            description=desc,
            tags=("agriculture", "inflation", "food", "consumer", "cpi"),
        )

    # 2. Food inflation — both inputs and crops rising
    if (
        food_score is not None
        and food_score > 1.25
        and input_score is not None
        and input_score > 0.75
        and crop_score is not None
        and crop_score > 0.75
    ):
        desc = (
            "Both input costs and crop prices are elevated vs recent history. "
            "Active food-inflation pressure — consistent with rising food CPI, "
            "margin compression for food producers, and consumer spending pressure."
        )
        if inputs.cpi_food_yoy is not None:
            desc += f" CPI Food at {inputs.cpi_food_yoy:+.1f}% YoY."
        if inputs.protein_spike:
            desc += " Protein prices spiking — headline risk."
        if inputs.grocery_shock:
            desc += f" Grocery prices outpacing restaurants by {inputs.grocery_restaurant_gap:+.1f}pp."
        if pos_ctx:
            desc += f" Positioning: {pos_ctx}."
        return AgricultureRegime(
            name="food_inflation",
            label="Food inflation impulse",
            description=desc,
            tags=("agriculture", "inflation", "food", "consumer"),
        )

    # 3. Crop supply risk — acreage shift or poor condition signals supply tightening
    _corn_acres_down = (
        inputs.corn_planted_yoy_pct is not None and inputs.corn_planted_yoy_pct < -3.0
    )
    _soy_acres_down = (
        inputs.soy_planted_yoy_pct is not None and inputs.soy_planted_yoy_pct < -3.0
    )
    _poor_condition = any(
        ge is not None and ge < 55
        for ge in [inputs.corn_condition_ge, inputs.soy_condition_ge, inputs.wheat_condition_ge]
    )
    _composite_poor = (
        inputs.crop_condition_composite is not None and inputs.crop_condition_composite < 55
    )
    _planting_delayed = (
        inputs.corn_pct_planted_vs_avg is not None and inputs.corn_pct_planted_vs_avg < -10
    )
    _multi_planting_delay = (
        inputs.planting_delay_count is not None and inputs.planting_delay_count >= 2
    )

    if (_corn_acres_down or _soy_acres_down) and (
        (inputs.wasde_corn_stu_pct is not None and inputs.wasde_corn_stu_pct < 12)
        or crop_score is not None and crop_score > 0.5
    ):
        desc = "Prospective Plantings show reduced acreage"
        if _corn_acres_down:
            desc += f" — corn down {abs(inputs.corn_planted_yoy_pct):.1f}% YoY"
        if _soy_acres_down:
            desc += f", soy down {abs(inputs.soy_planted_yoy_pct):.1f}%"
        desc += ". Lower planted area tightens supply outlook for the marketing year."
        if inputs.wasde_corn_stu_pct is not None and inputs.wasde_corn_stu_pct < 12:
            desc += f" Already-tight S/U ({inputs.wasde_corn_stu_pct:.1f}%) amplifies the risk."
        if pos_ctx:
            desc += f" Positioning: {pos_ctx}."
        return AgricultureRegime(
            name="crop_supply_risk",
            label="Crop supply risk — acreage contraction",
            description=desc,
            tags=("agriculture", "supply", "acreage", "plantings"),
        )

    if _poor_condition or _composite_poor:
        weak_crops = []
        for label, ge in [("corn", inputs.corn_condition_ge), ("soy", inputs.soy_condition_ge), ("wheat", inputs.wheat_condition_ge)]:
            if ge is not None and ge < 55:
                weak_crops.append(f"{label} G/E {ge:.0f}%")
        if weak_crops:
            desc = f"Crop condition ratings are weak — {', '.join(weak_crops)}. "
        elif _composite_poor:
            desc = f"Composite crop condition at {inputs.crop_condition_composite:.0f}% G/E — below stress threshold. "
        else:
            desc = "Crop conditions deteriorating. "
        desc += "Below-average conditions point to yield risk and potential supply shortfalls."
        if acre_ctx:
            desc += f" {acre_ctx}."
        if pos_ctx:
            desc += f" Positioning: {pos_ctx}."
        return AgricultureRegime(
            name="crop_supply_risk",
            label="Crop supply risk — poor conditions",
            description=desc,
            tags=("agriculture", "supply", "condition", "yield"),
        )

    if _multi_planting_delay:
        desc = (
            f"{inputs.planting_delay_count} major crops are behind planting schedule (>5pp vs 5yr avg). "
            "Broad planting delays raise aggregate supply risk and can trigger weather-premium rallies."
        )
        if acre_ctx:
            desc += f" {acre_ctx}."
        if pos_ctx:
            desc += f" Positioning: {pos_ctx}."
        return AgricultureRegime(
            name="crop_supply_risk",
            label="Crop supply risk — multi-crop planting delay",
            description=desc,
            tags=("agriculture", "supply", "progress", "weather"),
        )

    if _planting_delayed:
        desc = (
            f"Corn planting significantly behind schedule ({inputs.corn_pct_planted:.0f}% planted vs "
            f"{inputs.corn_pct_planted + abs(inputs.corn_pct_planted_vs_avg):.0f}% 5yr avg). "
            "Late planting reduces yield potential and can trigger weather-premium rallies."
        )
        if pos_ctx:
            desc += f" Positioning: {pos_ctx}."
        return AgricultureRegime(
            name="crop_supply_risk",
            label="Crop supply risk — planting delay",
            description=desc,
            tags=("agriculture", "supply", "progress", "weather"),
        )

    # 4. Crop surge — crops spiking independently
    if inputs.crop_surge:
        desc = (
            "Crop prices are rising sharply (weather, supply disruption, or demand shock). "
            "Directly feeds into food inflation and consumer price pressure."
        )
        # Tight S/U reinforces the surge signal
        if inputs.wasde_corn_stu_pct is not None and inputs.wasde_corn_stu_pct < 10:
            desc += (
                f" WASDE corn S/U at {inputs.wasde_corn_stu_pct:.1f}% reinforces "
                "supply tightness — limited buffer to absorb demand shock."
            )
        if inputs.grocery_shock:
            desc += f" Grocery shelf prices already outpacing restaurants by {inputs.grocery_restaurant_gap:+.1f}pp."
        if acre_ctx:
            desc += f" Crop reports: {acre_ctx}."
        return AgricultureRegime(
            name="crop_surge",
            label="Crop price surge",
            description=desc,
            tags=("agriculture", "crops", "inflation"),
        )

    # 5. Cost pass-through lag — inputs up, crops haven't followed
    if inputs.cost_pass_through_lag:
        desc = (
            "Fertilizer and energy input costs are rising faster than crop prices. "
            "Historically, crop prices catch up with a lag (weeks to months). "
            "Bullish signal for agriculture commodities."
        )
        if inputs.cot_corn_z is not None and inputs.cot_corn_z < -0.5:
            desc += " Specs are underweight — positioning supports catch-up rally."
        return AgricultureRegime(
            name="cost_pass_through",
            label="Input cost pass-through building",
            description=desc,
            tags=("agriculture", "input_costs", "divergence"),
        )

    # 6. Cost reflation (milder)
    if input_score is not None and input_score > 1.0:
        return AgricultureRegime(
            name="ag_cost_reflation",
            label="Agriculture cost reflation",
            description=(
                "Input costs are building above recent norms. Not yet at shock levels "
                "but consistent with gradually rising agricultural commodity prices."
            ),
            tags=("agriculture", "input_costs", "reflation"),
        )

    # 7. Disinflation
    if food_score is not None and food_score < -1.0:
        return AgricultureRegime(
            name="ag_disinflation",
            label="Agriculture disinflation",
            description=(
                "Input costs and/or crop prices are falling vs recent history. "
                "Easing food-inflation pressure — positive for consumer spending, "
                "negative for agriculture commodity longs."
            ),
            tags=("agriculture", "disinflation"),
        )

    if input_score is not None and input_score < -1.25:
        return AgricultureRegime(
            name="ag_disinflation",
            label="Agriculture input costs falling",
            description=(
                "Input costs (natural gas, fertilizer, diesel) are declining. "
                "Should feed through to lower crop production costs and easing food prices."
            ),
            tags=("agriculture", "input_costs", "disinflation"),
        )

    # 8. Neutral
    return AgricultureRegime(
        name="neutral",
        label="Neutral agriculture backdrop",
        description=(
            "No strong signal in crop prices or input costs vs recent history. "
            "Score uses 20-60 day returns z-scored vs 3-year rolling window."
        ),
        tags=("agriculture",),
    )

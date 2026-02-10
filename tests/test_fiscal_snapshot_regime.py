from lox.fiscal.regime import classify_fiscal_regime_skeleton


def test_fiscal_snapshot_regime_always_classifies_even_with_missing_inputs():
    r = classify_fiscal_regime_skeleton(
        deficit_12m=None,
        gdp_millions=None,
        deficit_impulse_pct_gdp=None,
        long_duration_issuance_share=None,
        tga_z_d_4w=None,
    )
    assert r.name in {"benign_funding", "heavy_funding", "stress_building", "fiscal_dominance_risk"}
    assert r.label  # display label should be present


def test_fiscal_snapshot_regime_improving_flag_in_label():
    r = classify_fiscal_regime_skeleton(
        deficit_12m=1_600_000.0,
        gdp_millions=27_000_000.0,
        deficit_impulse_pct_gdp=-1.0,
        long_duration_issuance_share=0.2,
        tga_z_d_4w=0.0,
    )
    assert "improving" in (r.label or "").lower()


def test_fiscal_snapshot_regime_duration_tilt_escalates_to_stress_building():
    r = classify_fiscal_regime_skeleton(
        deficit_12m=1_600_000.0,
        gdp_millions=27_000_000.0,
        deficit_impulse_pct_gdp=0.0,
        long_duration_issuance_share=0.6,
        tga_z_d_4w=0.0,
    )
    assert r.name == "stress_building"


def test_fiscal_snapshot_regime_weak_auctions_escalates_to_stress_building():
    r = classify_fiscal_regime_skeleton(
        deficit_12m=1_600_000.0,
        gdp_millions=27_000_000.0,
        deficit_impulse_pct_gdp=0.0,
        long_duration_issuance_share=0.2,
        tga_z_d_4w=0.0,
        auction_tail_bps=6.0,
        dealer_take_pct=40.0,
    )
    assert r.name == "stress_building"



from lox.fiscal.regime import classify_fiscal_regime_skeleton


def test_fiscal_snapshot_regime_always_classifies_even_with_missing_inputs():
    r = classify_fiscal_regime_skeleton(
        deficit_12m=None,
        gdp_millions=None,
        deficit_impulse_pct_gdp=None,
        long_duration_issuance_share=None,
        tga_z_d_4w=None,
    )
    assert r.name in {
        "fiscal_contraction", "moderate_fiscal_support",
        "strong_fiscal_stimulus", "fiscal_dominance_risk",
    }
    assert r.label


def test_fiscal_snapshot_regime_contracting_flag_in_label():
    """Negative impulse → 'contracting' direction in MMT framing."""
    r = classify_fiscal_regime_skeleton(
        deficit_12m=1_600_000.0,
        gdp_millions=27_000_000.0,
        deficit_impulse_pct_gdp=-1.0,
        long_duration_issuance_share=0.2,
        tga_z_d_4w=0.0,
    )
    assert "contracting" in (r.label or "").lower()


def test_fiscal_snapshot_regime_low_deficit_is_contraction():
    """Low deficit (< 3% GDP) → fiscal contraction (private sector squeeze)."""
    r = classify_fiscal_regime_skeleton(
        deficit_12m=500_000.0,
        gdp_millions=27_000_000.0,  # ~1.9% GDP
        deficit_impulse_pct_gdp=0.0,
        long_duration_issuance_share=0.2,
        tga_z_d_4w=0.0,
    )
    assert r.name == "fiscal_contraction"


def test_fiscal_snapshot_regime_high_deficit_is_stimulus():
    """High deficit (> 6% GDP) → strong fiscal stimulus."""
    r = classify_fiscal_regime_skeleton(
        deficit_12m=2_000_000.0,
        gdp_millions=27_000_000.0,  # ~7.4% GDP
        deficit_impulse_pct_gdp=0.0,
        long_duration_issuance_share=0.2,
        tga_z_d_4w=0.0,
    )
    assert r.name == "strong_fiscal_stimulus"


def test_fiscal_snapshot_regime_tga_drain_escalates():
    """TGA rising sharply drains reserves → escalates toward contraction."""
    r = classify_fiscal_regime_skeleton(
        deficit_12m=1_200_000.0,
        gdp_millions=27_000_000.0,  # ~4.4% GDP → moderate
        deficit_impulse_pct_gdp=0.0,
        long_duration_issuance_share=0.2,
        tga_z_d_4w=1.5,  # strong TGA drain
    )
    assert r.name == "fiscal_contraction"


def test_fiscal_snapshot_regime_sharp_negative_impulse_overrides():
    """Sharp negative impulse overrides even moderate deficit level."""
    r = classify_fiscal_regime_skeleton(
        deficit_12m=1_200_000.0,
        gdp_millions=27_000_000.0,  # ~4.4% GDP → moderate
        deficit_impulse_pct_gdp=-1.5,  # strong fiscal drag
        long_duration_issuance_share=0.2,
        tga_z_d_4w=0.0,
    )
    assert r.name == "fiscal_contraction"

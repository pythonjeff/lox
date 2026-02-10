"""
Unified regime feature extraction for ML and Monte Carlo.

Pulls all regime classifications into a single ML-friendly feature vector.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from lox.config import load_settings
from lox.regimes.base import categorize_regime, RegimeResult

logger = logging.getLogger(__name__)


@dataclass
class UnifiedRegimeState:
    """
    Unified view of all regime classifications.
    
    Provides a consistent interface for ML feature extraction and Monte Carlo.
    """
    asof: str
    
    # Core 4 pillars (used in Monte Carlo)
    macro: Optional[RegimeResult] = None
    volatility: Optional[RegimeResult] = None
    rates: Optional[RegimeResult] = None
    funding: Optional[RegimeResult] = None
    
    # Extended regimes (for overlay/context)
    fiscal: Optional[RegimeResult] = None
    commodities: Optional[RegimeResult] = None
    housing: Optional[RegimeResult] = None
    monetary: Optional[RegimeResult] = None
    usd: Optional[RegimeResult] = None
    crypto: Optional[RegimeResult] = None
    
    # Aggregate risk assessment
    overall_risk_score: float = 50.0
    overall_category: str = "cautious"
    
    def to_feature_dict(self) -> dict:
        """Convert all regimes to flat ML feature dictionary."""
        features = {
            "asof": self.asof,
            "overall_risk_score": self.overall_risk_score,
            "overall_is_risk_on": 1.0 if self.overall_category == "risk_on" else 0.0,
            "overall_is_risk_off": 1.0 if self.overall_category == "risk_off" else 0.0,
        }
        
        # Add features from each regime
        for domain in ["macro", "volatility", "rates", "funding", "fiscal", 
                       "commodities", "housing", "monetary", "usd", "crypto"]:
            regime = getattr(self, domain, None)
            if regime:
                features[f"{domain}_regime"] = regime.name
                features[f"{domain}_score"] = regime.score
                features[f"{domain}_category"] = categorize_regime(regime.name)
            else:
                features[f"{domain}_regime"] = "unknown"
                features[f"{domain}_score"] = 50.0
                features[f"{domain}_category"] = "cautious"
        
        return features
    
    def to_monte_carlo_params(self) -> dict:
        """
        Convert regime state to Monte Carlo scenario parameters.
        
        Returns dict with:
        - equity_drift, equity_vol adjustments
        - iv_drift, iv_vol adjustments  
        - rate_drift, spread_drift adjustments
        - jump_prob, jump_size adjustments
        """
        params = {
            "equity_drift_adj": 0.0,
            "equity_vol_adj": 1.0,
            "iv_drift_adj": 0.0,
            "iv_vol_adj": 1.0,
            "rate_drift_adj": 0.0,
            "spread_drift_adj": 0.0,
            "jump_prob_adj": 1.0,
            "jump_size_adj": 1.0,
        }
        
        # Macro regime adjustments
        if self.macro:
            if "stagflation" in self.macro.name:
                params["equity_drift_adj"] -= 0.05  # -5% annual drift
                params["equity_vol_adj"] *= 1.3
                params["rate_drift_adj"] += 0.005  # +50bps
                params["jump_prob_adj"] *= 1.5
            elif "goldilocks" in self.macro.name:
                params["equity_drift_adj"] += 0.03
                params["equity_vol_adj"] *= 0.85
        
        # Volatility regime adjustments
        if self.volatility:
            if "shock" in self.volatility.name:
                params["iv_drift_adj"] += 0.10  # IV elevated
                params["iv_vol_adj"] *= 1.5
                params["jump_prob_adj"] *= 2.0
                params["jump_size_adj"] *= 1.3
            elif "elevated" in self.volatility.name:
                params["iv_drift_adj"] += 0.03
                params["iv_vol_adj"] *= 1.2
            elif "complacent" in self.volatility.name or "normal" in self.volatility.name:
                params["iv_drift_adj"] -= 0.02
                params["iv_vol_adj"] *= 0.9
        
        # Funding regime adjustments
        if self.funding:
            if "stress" in self.funding.name:
                params["spread_drift_adj"] += 0.01  # +100bps spreads
                params["equity_drift_adj"] -= 0.03
                params["jump_prob_adj"] *= 1.5
            elif "tightening" in self.funding.name:
                params["spread_drift_adj"] += 0.003  # +30bps
        
        # Rates regime adjustments
        if self.rates:
            if "shock_up" in self.rates.name:
                params["rate_drift_adj"] += 0.01
                params["equity_drift_adj"] -= 0.02  # Growth stocks hurt
            elif "shock_down" in self.rates.name:
                params["rate_drift_adj"] -= 0.01
                params["equity_drift_adj"] += 0.02
            elif "inverted" in self.rates.name:
                params["equity_drift_adj"] -= 0.02  # Recession signal
                params["jump_prob_adj"] *= 1.3
        
        # USD regime adjustments (affects EM, commodities)
        if self.usd:
            if "surge" in self.usd.name or "strong" in self.usd.name:
                # Strong USD = headwind for EM, commodities
                params["equity_drift_adj"] -= 0.01  # Modest headwind
            elif "plunge" in self.usd.name or "weak" in self.usd.name:
                # Weak USD = tailwind for risk assets
                params["equity_drift_adj"] += 0.01
        
        # Fiscal regime adjustments
        if self.fiscal:
            if "dominance" in self.fiscal.name or "stress" in self.fiscal.name:
                params["rate_drift_adj"] += 0.005  # Term premium
                params["spread_drift_adj"] += 0.002
        
        return params


def build_unified_regime_state(
    settings=None,
    start_date: str = "2020-01-01",
    refresh: bool = False,
) -> UnifiedRegimeState:
    """
    Build unified regime state by running all regime classifiers.
    
    Args:
        settings: Config settings (loaded if None)
        start_date: Historical start date for data
        refresh: Force refresh of cached data
    
    Returns:
        UnifiedRegimeState with all regime classifications
    """
    if settings is None:
        settings = load_settings()
    
    asof = datetime.now().strftime("%Y-%m-%d")
    state = UnifiedRegimeState(asof=asof)
    
    risk_scores = []
    
    # --- Macro Regime ---
    try:
        from lox.macro.signals import build_macro_state
        from lox.macro.regime import classify_macro_regime_from_state
        
        macro_state = build_macro_state(settings=settings, start_date=start_date, refresh=refresh)
        macro_regime = classify_macro_regime_from_state(
            cpi_yoy=macro_state.inputs.cpi_yoy,
            payrolls_3m_annualized=macro_state.inputs.payrolls_3m_annualized,
            inflation_momentum_minus_be5y=macro_state.inputs.inflation_momentum_minus_be5y,
            real_yield_proxy_10y=macro_state.inputs.real_yield_proxy_10y,
            z_inflation_momentum_minus_be5y=macro_state.inputs.components.get("z_infl_mom_minus_be5y") if macro_state.inputs.components else None,
            z_real_yield_proxy_10y=macro_state.inputs.components.get("z_real_yield_proxy_10y") if macro_state.inputs.components else None,
            use_zscores=True,
        )
        
        # Map to RegimeResult
        score = 70 if "stagflation" in macro_regime.name else (30 if "goldilocks" in macro_regime.name else 50)
        inp = macro_state.inputs
        state.macro = RegimeResult(
            name=macro_regime.name,
            label=macro_regime.name.replace("_", " ").title(),
            description=macro_regime.description,
            score=score,
            domain="macro",
            tags=["risk_off"] if "stagflation" in macro_regime.name else [],
            metrics={
                "CPI YoY": f"{inp.cpi_yoy:.1f}%" if inp.cpi_yoy is not None else None,
                "Core CPI": f"{inp.core_cpi_yoy:.1f}%" if inp.core_cpi_yoy is not None else None,
                "Payrolls 3m": f"{inp.payrolls_3m_annualized:.1f}%" if inp.payrolls_3m_annualized is not None else None,
                "Unemp": f"{inp.unemployment_rate:.1f}%" if inp.unemployment_rate is not None else None,
                "HY OAS": f"{inp.hy_oas * 100:.0f}bp" if inp.hy_oas is not None else None,
                "10Y": f"{inp.ust_10y:.2f}%" if inp.ust_10y is not None else None,
                "2s10s": f"{inp.curve_2s10s * 100:.0f}bp" if inp.curve_2s10s is not None else None,
                "VIX": f"{inp.vix:.1f}" if inp.vix is not None else None,
            },
        )
        risk_scores.append(score)
    except Exception as e:
        logger.warning(f"Failed to build macro regime: {e}")
    
    # --- Volatility Regime ---
    try:
        from lox.volatility.signals import build_volatility_state
        from lox.volatility.regime import classify_volatility_regime
        
        vol_state = build_volatility_state(settings=settings, start_date=start_date, refresh=refresh)
        vol_regime = classify_volatility_regime(vol_state.inputs)
        
        inp = vol_state.inputs
        state.volatility = RegimeResult(
            name=vol_regime.name,
            label=vol_regime.label,
            description=vol_regime.description,
            score=vol_regime.score if hasattr(vol_regime, 'score') else (80 if "shock" in vol_regime.name else 50),
            domain="volatility",
            tags=list(vol_regime.tags) if hasattr(vol_regime, 'tags') else [],
            metrics={
                "VIX": f"{inp.vix:.1f}" if inp.vix is not None else None,
                "VIX z": f"{inp.z_vix:+.1f}" if inp.z_vix is not None else None,
                "VIX 5d Chg": f"{inp.vix_chg_5d_pct:+.1f}%" if inp.vix_chg_5d_pct is not None else None,
                "Spike 20d": f"{inp.spike_20d_pct:.0f}%" if inp.spike_20d_pct is not None else None,
                "Term Spread": f"{inp.vix_term_spread:+.2f}" if inp.vix_term_spread is not None else None,
            },
        )
        risk_scores.append(state.volatility.score)
    except Exception as e:
        logger.warning(f"Failed to build volatility regime: {e}")
    
    # --- Rates Regime ---
    try:
        from lox.rates.signals import build_rates_state
        from lox.rates.regime import classify_rates_regime
        
        rates_state = build_rates_state(settings=settings, start_date=start_date, refresh=refresh)
        rates_regime = classify_rates_regime(rates_state.inputs)
        
        inp = rates_state.inputs
        state.rates = RegimeResult(
            name=rates_regime.name,
            label=rates_regime.label,
            description=rates_regime.description,
            score=rates_regime.score if hasattr(rates_regime, 'score') else 50,
            domain="rates",
            tags=list(rates_regime.tags) if hasattr(rates_regime, 'tags') else [],
            metrics={
                "10Y": f"{inp.ust_10y:.2f}%" if inp.ust_10y is not None else None,
                "2Y": f"{inp.ust_2y:.2f}%" if inp.ust_2y is not None else None,
                "3M": f"{inp.ust_3m:.2f}%" if inp.ust_3m is not None else None,
                "2s10s": f"{inp.curve_2s10s * 100:+.0f}bp" if inp.curve_2s10s is not None else None,
                "10Y 20d Chg": f"{inp.ust_10y_chg_20d * 100:+.0f}bp" if inp.ust_10y_chg_20d is not None else None,
                "2s10s z": f"{inp.z_curve_2s10s:+.1f}" if inp.z_curve_2s10s is not None else None,
            },
        )
        risk_scores.append(state.rates.score)
    except Exception as e:
        logger.warning(f"Failed to build rates regime: {e}")
    
    # --- Funding Regime ---
    try:
        from lox.funding.signals import build_funding_state
        from lox.funding.regime import classify_funding_regime
        
        funding_state = build_funding_state(settings=settings, start_date=start_date, refresh=refresh)
        funding_regime = classify_funding_regime(funding_state.inputs)
        
        score = 80 if "stress" in funding_regime.name else (60 if "tightening" in funding_regime.name else 40)
        inp = funding_state.inputs
        state.funding = RegimeResult(
            name=funding_regime.name,
            label=funding_regime.label,
            description=funding_regime.description if hasattr(funding_regime, 'description') else "",
            score=score,
            domain="funding",
            tags=["risk_off"] if "stress" in funding_regime.name else [],
            metrics={
                "SOFR": f"{inp.sofr:.2f}%" if inp.sofr is not None else None,
                "EFFR": f"{inp.effr:.2f}%" if inp.effr is not None else None,
                "IORB": f"{inp.iorb:.2f}%" if inp.iorb is not None else None,
                "Corridor": f"{inp.spread_corridor_bps:+.1f}bp" if inp.spread_corridor_bps is not None else None,
                "RRP": f"${inp.on_rrp_usd_bn / 1000:.0f}B" if inp.on_rrp_usd_bn is not None else None,
                "Reserves": f"${inp.bank_reserves_usd_bn / 1e6:.1f}T" if inp.bank_reserves_usd_bn is not None else None,
                "TGA": f"${inp.tga_usd_bn / 1000:.0f}B" if inp.tga_usd_bn is not None else None,
            },
        )
        risk_scores.append(score)
    except Exception as e:
        logger.warning(f"Failed to build funding regime: {e}")
    
    # --- USD Regime ---
    try:
        from lox.usd.signals import build_usd_state
        from lox.usd.regime import classify_usd_regime_from_state
        
        usd_state = build_usd_state(settings=settings, start_date=start_date, refresh=refresh)
        usd_regime = classify_usd_regime_from_state(usd_state)
        
        inp = usd_state.inputs
        state.usd = RegimeResult(
            name=usd_regime.name,
            label=usd_regime.label,
            description=usd_regime.description,
            score=usd_regime.score,
            domain="usd",
            tags=list(usd_regime.tags),
            metrics={
                "DXY": f"{inp.usd_index_broad:.1f}" if inp.usd_index_broad is not None else None,
                "20d Chg": f"{inp.usd_chg_20d_pct:+.1f}%" if inp.usd_chg_20d_pct is not None else None,
                "60d Chg": f"{inp.usd_chg_60d_pct:+.1f}%" if inp.usd_chg_60d_pct is not None else None,
                "DXY z": f"{inp.z_usd_level:+.1f}" if inp.z_usd_level is not None else None,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to build USD regime: {e}")
    
    # --- Fiscal Regime ---
    try:
        from lox.fiscal.signals import build_fiscal_deficit_page_data
        from lox.fiscal.regime import classify_fiscal_regime_snapshot
        
        fiscal_data = build_fiscal_deficit_page_data(settings=settings, lookback_years=5, refresh=refresh)
        net = fiscal_data.get("net_issuance_stats", {})
        tga = fiscal_data.get("tga_stats", {})
        
        fiscal_regime = classify_fiscal_regime_snapshot(
            deficit_pct_gdp=fiscal_data.get("deficit_pct_gdp"),
            deficit_impulse_pct_gdp=fiscal_data.get("deficit_impulse_pct_gdp"),
            long_duration_issuance_share=net.get("long_share") if net else None,
            tga_z_d_4w=tga.get("z_d_4w") if tga else None,
        )
        
        score = 80 if "dominance" in fiscal_regime.name else (60 if "stress" in fiscal_regime.name else 40)
        state.fiscal = RegimeResult(
            name=fiscal_regime.name,
            label=fiscal_regime.label,
            description=fiscal_regime.description,
            score=score,
            domain="fiscal",
            tags=[],
            metrics={
                "Deficit/GDP": f"{fiscal_data.get('deficit_pct_gdp', 0):.1f}%" if fiscal_data.get('deficit_pct_gdp') else None,
                "Deficit 12m": f"${fiscal_data.get('deficit_12m', 0) / 1e6:.1f}T" if fiscal_data.get('deficit_12m') else None,
                "Impulse/GDP": f"{fiscal_data.get('deficit_impulse_pct_gdp', 0):+.1f}%" if fiscal_data.get('deficit_impulse_pct_gdp') is not None else None,
                "TGA": f"${tga.get('tga_level', 0) / 1000:.0f}B" if tga.get('tga_level') else None,
                "Long Share": f"{net.get('long_share', 0) * 100:.0f}%" if net.get('long_share') else None,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to build fiscal regime: {e}")
    
    # --- Commodities Regime ---
    try:
        from lox.commodities.signals import build_commodities_state
        from lox.commodities.regime import classify_commodities_regime
        
        comm_state = build_commodities_state(settings=settings, start_date=start_date, refresh=refresh)
        comm_regime = classify_commodities_regime(comm_state.inputs)
        
        inp = comm_state.inputs
        state.commodities = RegimeResult(
            name=comm_regime.name,
            label=comm_regime.label,
            description=comm_regime.description,
            score=comm_regime.score if hasattr(comm_regime, 'score') else 50,
            domain="commodities",
            tags=list(comm_regime.tags) if hasattr(comm_regime, 'tags') else [],
            metrics={
                "Gold": f"${inp.gold:.0f}" if inp.gold is not None else None,
                "Gold 20d": f"{inp.gold_ret_20d_pct:+.1f}%" if inp.gold_ret_20d_pct is not None else None,
                "WTI": f"${inp.wti:.1f}" if inp.wti is not None else None,
                "Copper": f"${inp.copper:.1f}" if inp.copper is not None else None,
                "Broad 60d": f"{inp.broad_ret_60d_pct:+.1f}%" if inp.broad_ret_60d_pct is not None else None,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to build commodities regime: {e}")
    
    # --- Housing Regime ---
    try:
        from lox.housing.signals import build_housing_state
        from lox.housing.regime import classify_housing_regime
        
        housing_state = build_housing_state(settings=settings, start_date=start_date, refresh=refresh)
        housing_regime = classify_housing_regime(housing_state.inputs)
        
        score = 70 if "stress" in housing_regime.label.lower() else (30 if "easing" in housing_regime.label.lower() else 50)
        inp = housing_state.inputs
        state.housing = RegimeResult(
            name=housing_regime.label.lower().replace(" ", "_"),
            label=housing_regime.label,
            description=housing_regime.description,
            score=score,
            domain="housing",
            tags=[],
            metrics={
                "30Y Mtg": f"{inp.mortgage_30y:.2f}%" if inp.mortgage_30y is not None else None,
                "Mtg Spread": f"{inp.mortgage_spread * 100:.0f}bp" if inp.mortgage_spread is not None else None,
                "10Y": f"{inp.ust_10y:.2f}%" if inp.ust_10y is not None else None,
                "Mtg Spd z": f"{inp.z_mortgage_spread:+.1f}" if inp.z_mortgage_spread is not None else None,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to build housing regime: {e}")
    
    # --- Monetary Regime ---
    try:
        from lox.monetary.signals import build_monetary_state
        from lox.monetary.regime import classify_monetary_regime
        
        monetary_state = build_monetary_state(settings=settings, start_date=start_date, refresh=refresh)
        monetary_regime = classify_monetary_regime(monetary_state.inputs)
        
        score = 70 if "qt_biting" in monetary_regime.name else (30 if "abundant" in monetary_regime.name else 50)
        inp = monetary_state.inputs
        state.monetary = RegimeResult(
            name=monetary_regime.name,
            label=monetary_regime.label,
            description=monetary_regime.description,
            score=score,
            domain="monetary",
            tags=[],
            metrics={
                "Fed Assets": f"${inp.fed_assets / 1e6:.1f}T" if inp.fed_assets is not None else None,
                "Reserves": f"${inp.total_reserves / 1e6:.1f}T" if inp.total_reserves is not None else None,
                "Reserves z": f"{inp.z_total_reserves:+.1f}" if inp.z_total_reserves is not None else None,
                "RRP": f"${inp.on_rrp / 1000:.0f}B" if inp.on_rrp is not None else None,
                "EFFR": f"{inp.effr:.2f}%" if inp.effr is not None else None,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to build monetary regime: {e}")
    
    # --- Calculate Overall Risk Score ---
    if risk_scores:
        state.overall_risk_score = sum(risk_scores) / len(risk_scores)
    
    # Categorize overall state
    if state.overall_risk_score >= 65:
        state.overall_category = "risk_off"
    elif state.overall_risk_score <= 35:
        state.overall_category = "risk_on"
    else:
        state.overall_category = "cautious"
    
    return state


def extract_ml_features(
    settings=None,
    start_date: str = "2020-01-01",
    refresh: bool = False,
) -> dict:
    """
    Extract flat ML feature dictionary from all regimes.
    
    This is the main entry point for ML training pipelines.
    """
    state = build_unified_regime_state(settings=settings, start_date=start_date, refresh=refresh)
    return state.to_feature_dict()

"""
Stress Testing Framework for Portfolio Risk Analysis.

Provides deterministic scenario analysis across predefined market stress events.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class StressScenario(str, Enum):
    """Predefined stress scenarios calibrated to historical events."""
    
    EQUITY_CRASH = "equity_crash"          # COVID-style drawdown
    RATES_SHOCK_UP = "rates_shock_up"      # 2022-style rate shock
    RATES_SHOCK_DOWN = "rates_shock_down"  # Flight to quality
    CREDIT_EVENT = "credit_event"          # HY spread blowout
    FLASH_CRASH = "flash_crash"            # Aug 2024-style unwind
    STAGFLATION = "stagflation"            # Inflation + growth scare
    VOL_CRUSH = "vol_crush"                # Complacency/low vol regime
    CUSTOM = "custom"


@dataclass
class StressParameters:
    """Parameters for a stress scenario."""
    
    name: str
    description: str
    
    # Market moves
    spx_change_pct: float        # S&P 500 return
    vix_change_pts: float        # VIX absolute change
    rate_10y_change_bps: float   # 10Y yield change in bps
    hy_oas_change_bps: float     # HY spread change in bps
    
    # Time horizon
    horizon_days: int = 1        # Days over which stress occurs
    
    # Historical analog
    historical_analog: Optional[str] = None
    
    @classmethod
    def get_scenario(cls, scenario: StressScenario) -> "StressParameters":
        """Get predefined scenario parameters."""
        
        scenarios = {
            StressScenario.EQUITY_CRASH: cls(
                name="Equity Crash",
                description="Sharp equity drawdown with vol spike and flight to quality",
                spx_change_pct=-0.20,
                vix_change_pts=25.0,
                rate_10y_change_bps=-50.0,
                hy_oas_change_bps=200.0,
                horizon_days=5,
                historical_analog="COVID Mar 2020",
            ),
            StressScenario.RATES_SHOCK_UP: cls(
                name="Rates Shock (+50bp)",
                description="Aggressive Fed repricing, duration selloff",
                spx_change_pct=-0.05,
                vix_change_pts=5.0,
                rate_10y_change_bps=50.0,
                hy_oas_change_bps=50.0,
                horizon_days=10,
                historical_analog="Oct 2022",
            ),
            StressScenario.RATES_SHOCK_DOWN: cls(
                name="Rates Shock (-50bp)",
                description="Flight to quality, dovish Fed pivot",
                spx_change_pct=-0.03,
                vix_change_pts=8.0,
                rate_10y_change_bps=-50.0,
                hy_oas_change_bps=30.0,
                horizon_days=5,
                historical_analog="SVB Mar 2023",
            ),
            StressScenario.CREDIT_EVENT: cls(
                name="Credit Event",
                description="HY spread blowout, credit stress",
                spx_change_pct=-0.10,
                vix_change_pts=15.0,
                rate_10y_change_bps=-25.0,
                hy_oas_change_bps=250.0,
                horizon_days=5,
                historical_analog="GFC-lite",
            ),
            StressScenario.FLASH_CRASH: cls(
                name="Flash Crash",
                description="Rapid deleveraging with VIX spike",
                spx_change_pct=-0.10,
                vix_change_pts=30.0,
                rate_10y_change_bps=-50.0,
                hy_oas_change_bps=100.0,
                horizon_days=1,
                historical_analog="Aug 2024 unwind",
            ),
            StressScenario.STAGFLATION: cls(
                name="Stagflation",
                description="Inflation persistent + growth slowing",
                spx_change_pct=-0.15,
                vix_change_pts=10.0,
                rate_10y_change_bps=100.0,
                hy_oas_change_bps=150.0,
                horizon_days=20,
                historical_analog="1970s analog",
            ),
            StressScenario.VOL_CRUSH: cls(
                name="Vol Crush",
                description="Complacency regime, IV collapses",
                spx_change_pct=0.05,
                vix_change_pts=-8.0,
                rate_10y_change_bps=20.0,
                hy_oas_change_bps=-30.0,
                horizon_days=30,
                historical_analog="2017 low vol regime",
            ),
        }
        
        return scenarios.get(scenario)
    
    @classmethod
    def custom(
        cls,
        spx_pct: float,
        vix_pts: float,
        rate_bps: float,
        hy_bps: float,
        horizon_days: int = 1,
    ) -> "StressParameters":
        """Create custom stress scenario."""
        return cls(
            name="Custom Scenario",
            description=f"SPX {spx_pct*100:+.0f}%, VIX {vix_pts:+.0f}pts, 10Y {rate_bps:+.0f}bp, HY {hy_bps:+.0f}bp",
            spx_change_pct=spx_pct,
            vix_change_pts=vix_pts,
            rate_10y_change_bps=rate_bps,
            hy_oas_change_bps=hy_bps,
            horizon_days=horizon_days,
        )


@dataclass
class StressResult:
    """Results from a stress test."""
    
    scenario: StressParameters
    
    # Total P&L
    total_pnl_usd: float
    total_pnl_pct: float
    
    # Greek attribution
    delta_pnl: float
    vega_pnl: float
    theta_pnl: float
    gamma_pnl: float
    residual_pnl: float
    
    # Position-level breakdown
    position_pnls: Dict[str, float]
    
    # Risk metrics
    pct_of_nav: float
    pct_of_var95: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "scenario_name": self.scenario.name,
            "scenario_description": self.scenario.description,
            "historical_analog": self.scenario.historical_analog,
            "market_moves": {
                "spx_change_pct": self.scenario.spx_change_pct,
                "vix_change_pts": self.scenario.vix_change_pts,
                "rate_10y_change_bps": self.scenario.rate_10y_change_bps,
                "hy_oas_change_bps": self.scenario.hy_oas_change_bps,
            },
            "pnl": {
                "total_usd": self.total_pnl_usd,
                "total_pct": self.total_pnl_pct,
                "delta_contrib": self.delta_pnl,
                "vega_contrib": self.vega_pnl,
                "theta_contrib": self.theta_pnl,
                "gamma_contrib": self.gamma_pnl,
                "residual": self.residual_pnl,
            },
            "position_pnls": self.position_pnls,
        }


def run_stress_test(
    portfolio,  # Portfolio object from positions.py
    scenario: StressScenario | StressParameters,
) -> StressResult:
    """
    Run a deterministic stress test on the portfolio.
    
    Args:
        portfolio: Portfolio object with positions
        scenario: StressScenario enum or custom StressParameters
    
    Returns:
        StressResult with P&L breakdown
    """
    # Get scenario parameters
    if isinstance(scenario, StressScenario):
        params = StressParameters.get_scenario(scenario)
    else:
        params = scenario
    
    if params is None:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    nav = portfolio.nav
    
    # Calculate P&L by Greek using position's estimate_pnl method (correct math)
    delta_pnl = 0.0
    vega_pnl = 0.0
    theta_pnl = 0.0
    gamma_pnl = 0.0
    position_pnls = {}
    
    for pos in portfolio.positions:
        ticker = pos.ticker
        
        # Simplified: apply SPX move to all equity positions
        underlying_return = params.spx_change_pct
        iv_change = params.vix_change_pts  # Simplified: use VIX as proxy for IV
        days = params.horizon_days
        
        # Use position's built-in P&L estimate (has correct math with bounds)
        pos_total_pnl = pos.estimate_pnl(
            underlying_change_pct=underlying_return,
            iv_change_pts=iv_change,
            days_elapsed=days,
        )
        
        # Approximate Greek attribution (for reporting)
        # These are rough estimates - the total from estimate_pnl is the source of truth
        if pos.is_option:
            underlying_px = pos.entry_underlying_price
            delta_s = underlying_px * underlying_return
            
            # Delta P&L: Δ × ΔS per share × multiplier × qty
            pos_delta_pnl = pos.quantity * 100 * pos.delta * delta_s
            # Vega P&L: vega × Δσ per share × multiplier × qty
            pos_vega_pnl = pos.quantity * 100 * pos.vega * iv_change
            # Theta P&L
            pos_theta_pnl = pos.quantity * 100 * pos.theta * min(days, pos.dte)
            # Gamma P&L (residual)
            pos_gamma_pnl = pos_total_pnl - pos_delta_pnl - pos_vega_pnl - pos_theta_pnl
        else:
            # Stock/ETF: just delta
            pos_delta_pnl = pos_total_pnl
            pos_vega_pnl = 0.0
            pos_theta_pnl = 0.0
            pos_gamma_pnl = 0.0
        
        delta_pnl += pos_delta_pnl
        vega_pnl += pos_vega_pnl
        theta_pnl += pos_theta_pnl
        gamma_pnl += pos_gamma_pnl
        position_pnls[ticker] = pos_total_pnl
    
    total_pnl = sum(position_pnls.values())
    
    return StressResult(
        scenario=params,
        total_pnl_usd=total_pnl,
        total_pnl_pct=total_pnl / nav if nav > 0 else 0,
        delta_pnl=delta_pnl,
        vega_pnl=vega_pnl,
        theta_pnl=theta_pnl,
        gamma_pnl=gamma_pnl,
        residual_pnl=0.0,  # Placeholder for model residual
        position_pnls=position_pnls,
        pct_of_nav=total_pnl / nav * 100 if nav > 0 else 0,
    )


def run_all_stress_tests(portfolio) -> Dict[str, StressResult]:
    """
    Run all predefined stress scenarios.
    
    Returns dict mapping scenario name to StressResult.
    """
    results = {}
    
    for scenario in StressScenario:
        if scenario == StressScenario.CUSTOM:
            continue
        
        result = run_stress_test(portfolio, scenario)
        results[scenario.value] = result
    
    return results


@dataclass
class AttributionResult:
    """P&L attribution breakdown."""
    
    # Period
    start_date: str
    end_date: str
    
    # Total P&L
    total_pnl_usd: float
    total_pnl_pct: float
    
    # Attribution by Greek
    delta_attribution: float
    vega_attribution: float
    theta_attribution: float
    gamma_attribution: float
    carry_attribution: float
    residual: float
    
    # Attribution by position
    position_attributions: Dict[str, Dict[str, float]]
    
    # Factor exposures
    factor_exposures: Dict[str, float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "period": {"start": self.start_date, "end": self.end_date},
            "total_pnl": {"usd": self.total_pnl_usd, "pct": self.total_pnl_pct},
            "greek_attribution": {
                "delta": self.delta_attribution,
                "vega": self.vega_attribution,
                "theta": self.theta_attribution,
                "gamma": self.gamma_attribution,
                "carry": self.carry_attribution,
                "residual": self.residual,
            },
            "position_attributions": self.position_attributions,
            "factor_exposures": self.factor_exposures,
        }


def calculate_pnl_attribution(
    portfolio,
    spx_return: float,
    vix_change: float,
    days_elapsed: int,
    rate_change_bps: float = 0.0,
) -> AttributionResult:
    """
    Calculate P&L attribution given market moves.
    
    This provides ex-post attribution of realized P&L.
    
    Args:
        spx_return: decimal return (e.g., -0.20 for -20%)
        vix_change: VIX points change (e.g., 10 for +10pts)
        days_elapsed: number of days
    """
    nav = portfolio.nav
    summary = portfolio.summary()
    
    # Use the portfolio's estimate_pnl for total (it has correct bounded math)
    underlying_changes = {pos.ticker.split('/')[0] if '/' in pos.ticker else pos.ticker: spx_return 
                         for pos in portfolio.positions}
    iv_changes = {pos.ticker.split('/')[0] if '/' in pos.ticker else pos.ticker: vix_change 
                  for pos in portfolio.positions}
    
    total_pnl_usd, position_pnl_map = portfolio.estimate_pnl(underlying_changes, iv_changes, days_elapsed)
    
    # Approximate Greek attribution for reporting
    delta_pnl = 0.0
    vega_pnl = 0.0
    theta_pnl = 0.0
    gamma_pnl = 0.0
    
    position_attributions = {}
    for pos in portfolio.positions:
        ticker = pos.ticker
        pos_total = position_pnl_map.get(ticker, 0.0)
        
        if pos.is_option:
            underlying_px = pos.entry_underlying_price
            delta_s = underlying_px * spx_return
            
            # Delta P&L: Δ × ΔS per share × multiplier × qty
            pos_delta = pos.quantity * 100 * pos.delta * delta_s
            # Vega P&L
            pos_vega = pos.quantity * 100 * pos.vega * vix_change
            # Theta P&L
            pos_theta = pos.quantity * 100 * pos.theta * min(days_elapsed, pos.dte)
            # Gamma as residual
            pos_gamma = pos_total - pos_delta - pos_vega - pos_theta
        else:
            pos_delta = pos_total  # Stock P&L is all delta
            pos_vega = 0.0
            pos_theta = 0.0
            pos_gamma = 0.0
        
        delta_pnl += pos_delta
        vega_pnl += pos_vega
        theta_pnl += pos_theta
        gamma_pnl += pos_gamma
        
        position_attributions[ticker] = {
            "delta": pos_delta,
            "vega": pos_vega,
            "theta": pos_theta,
            "gamma": pos_gamma,
            "total": pos_total,
        }
    
    carry_pnl = 0.0  # Placeholder for dividend/financing carry
    
    # Factor exposures
    factor_exposures = {
        "equity_beta": summary["net_delta_pct"],
        "vega_exposure": summary["net_vega"],
        "theta_carry_annual": summary["net_theta_per_day"] * 252 / nav * 100 if nav > 0 else 0,
        "duration_proxy": -summary["net_vega"] * 2,  # Rough proxy
    }
    
    from datetime import datetime
    
    return AttributionResult(
        start_date=datetime.now().strftime("%Y-%m-%d"),
        end_date=datetime.now().strftime("%Y-%m-%d"),
        total_pnl_usd=total_pnl_usd,
        total_pnl_pct=total_pnl_usd / nav * 100 if nav > 0 else 0,
        delta_attribution=delta_pnl,
        vega_attribution=vega_pnl,
        theta_attribution=theta_pnl,
        gamma_attribution=gamma_pnl,
        carry_attribution=carry_pnl,
        residual=0.0,  # Placeholder
        position_attributions=position_attributions,
        factor_exposures=factor_exposures,
    )

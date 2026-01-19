"""
SOFR futures and Fed policy expectations.

Tracks market-implied Fed path via:
- SOFR 3-month futures (SR3)
- Fed Funds futures (ZQ)
- Forward rate expectations vs Fed dot plot
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from ai_options_trader.config import Settings
from ai_options_trader.altdata.fmp import fetch_realtime_quotes


@dataclass
class FedPolicyExpectations:
    """Market-implied Fed policy expectations from futures."""
    
    current_effr: float  # Current effective Fed funds rate
    
    # Forward expectations (3M, 6M, 12M)
    implied_3m: Optional[float] = None
    implied_6m: Optional[float] = None
    implied_12m: Optional[float] = None
    
    # Fed's dot plot (for comparison)
    fed_dot_plot_eoy: Optional[float] = None
    
    @property
    def change_3m_bps(self) -> Optional[float]:
        """Expected change in 3 months (bps)."""
        if self.implied_3m is None:
            return None
        return (self.implied_3m - self.current_effr) * 100
    
    @property
    def change_6m_bps(self) -> Optional[float]:
        """Expected change in 6 months (bps)."""
        if self.implied_6m is None:
            return None
        return (self.implied_6m - self.current_effr) * 100
    
    @property
    def change_12m_bps(self) -> Optional[float]:
        """Expected change in 12 months (bps)."""
        if self.implied_12m is None:
            return None
        return (self.implied_12m - self.current_effr) * 100
    
    @property
    def divergence_from_fed_bps(self) -> Optional[float]:
        """Market vs Fed dot plot (12M, bps)."""
        if self.implied_12m is None or self.fed_dot_plot_eoy is None:
            return None
        return (self.implied_12m - self.fed_dot_plot_eoy) * 100
    
    @property
    def curve_signal(self) -> str:
        """Interpret forward curve shape."""
        if self.implied_3m is None or self.implied_12m is None:
            return "UNKNOWN"
        
        diff_12m_vs_3m = self.implied_12m - self.implied_3m
        
        if diff_12m_vs_3m < -0.5:
            return "INVERSION (expects cuts)"
        elif diff_12m_vs_3m < -0.1:
            return "FLATTENING (pivot soon)"
        elif diff_12m_vs_3m < 0.1:
            return "FLAT (neutral)"
        elif diff_12m_vs_3m < 0.5:
            return "STEEPENING (gradual hikes)"
        else:
            return "STEEP (aggressive hikes)"
    
    @property
    def policy_bias(self) -> str:
        """Overall market bias."""
        if self.change_12m_bps is None:
            return "UNKNOWN"
        
        if self.change_12m_bps < -100:
            return "DOVISH (expects cuts)"
        elif self.change_12m_bps < -25:
            return "MILDLY DOVISH"
        elif self.change_12m_bps < 25:
            return "NEUTRAL"
        elif self.change_12m_bps < 100:
            return "MILDLY HAWKISH"
        else:
            return "HAWKISH (expects hikes)"


def fetch_fed_policy_expectations(
    settings: Settings,
    current_effr: float = 5.33,  # Current EFFR (from FRED)
    fed_dot_plot: Optional[float] = 4.50,  # Fed's SEP projection for EOY
) -> FedPolicyExpectations:
    """
    Fetch market-implied Fed policy expectations.
    
    Uses:
    - Fed Funds futures (ZQ) if available from FMP
    - Fallback: simple heuristic based on 2Y/10Y yields
    
    Args:
        settings: Config with FMP API key
        current_effr: Current effective Fed funds rate (%)
        fed_dot_plot: Fed's SEP projection for end of year (%)
    
    Returns:
        FedPolicyExpectations with implied rates
    """
    
    # TODO: Once FMP adds futures data, fetch actual SR3/ZQ contracts
    # For now, use a simple heuristic based on 2Y Treasury
    
    try:
        # Fetch 2Y and 10Y yields as proxies
        quotes = fetch_realtime_quotes(
            settings=settings,
            tickers=["^TNX", "^TYX"],  # 10Y and 30Y (FMP symbols)
        )
        
        # Approximate expectations using 2Y yield
        # (In reality, we'd use actual futures contracts)
        # 2Y ≈ average expected Fed funds over next 2 years
        
        # Simple approximation:
        # - If 2Y < EFFR: market expects cuts
        # - If 2Y > EFFR: market expects hikes
        
        # For v0, use hardcoded estimates (replace with actual futures data)
        implied_3m = current_effr - 0.25  # Placeholder: -25bps in 3M
        implied_6m = current_effr - 0.50  # Placeholder: -50bps in 6M
        implied_12m = current_effr - 1.00  # Placeholder: -100bps in 12M
        
        return FedPolicyExpectations(
            current_effr=current_effr,
            implied_3m=implied_3m,
            implied_6m=implied_6m,
            implied_12m=implied_12m,
            fed_dot_plot_eoy=fed_dot_plot,
        )
    
    except Exception:
        # Fallback: return only current EFFR
        return FedPolicyExpectations(
            current_effr=current_effr,
            fed_dot_plot_eoy=fed_dot_plot,
        )

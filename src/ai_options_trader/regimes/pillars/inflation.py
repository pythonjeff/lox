"""
Inflation Pillar - Comprehensive price pressure analysis.

Metrics:
- Headline CPI (YoY)
- Core CPI (YoY)
- Median CPI (stickiness proxy)
- PCE (Fed's preferred)
- Breakevens (5Y, 5Y5Y)
- Inflation expectations gap

Author: Lox Capital Research
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Any
from datetime import date

from ai_options_trader.regimes.core import (
    RegimePillar, Metric, RegimeLevel, register_pillar
)


@register_pillar("inflation")
@dataclass
class InflationPillar(RegimePillar):
    """
    Inflation regime pillar.
    
    Tracks price pressures across:
    - Realized inflation (CPI, PCE)
    - Expected inflation (breakevens)
    - Stickiness (median CPI, supercore)
    """
    name: str = "Inflation"
    description: str = "Price pressure and inflation expectations"
    metrics: List[Metric] = field(default_factory=list)
    
    # Thresholds
    TARGET: float = 2.0      # Fed target
    ELEVATED: float = 3.0    # Elevated threshold
    HIGH: float = 4.0        # High inflation
    
    def compute(self, settings: Any) -> None:
        """Fetch inflation data and compute metrics."""
        from ai_options_trader.data.fred import FredClient
        
        if not settings.FRED_API_KEY:
            return
        
        fred = FredClient(api_key=settings.FRED_API_KEY)
        
        # Headline CPI YoY
        try:
            cpi_df = fred.fetch_series("CPIAUCSL", start_date="2020-01-01")
            if cpi_df is not None and len(cpi_df) > 12:
                cpi_df = cpi_df.sort_values("date")
                latest = cpi_df["value"].iloc[-1]
                year_ago = cpi_df["value"].iloc[-13] if len(cpi_df) > 13 else cpi_df["value"].iloc[0]
                cpi_yoy = (latest / year_ago - 1) * 100
                
                # 3-month momentum
                if len(cpi_df) > 3:
                    three_mo_ago = cpi_df["value"].iloc[-4]
                    cpi_3m = (latest / three_mo_ago - 1) * 100 * 4  # Annualized
                else:
                    cpi_3m = None
                
                self.metrics.append(Metric(
                    name="CPI YoY",
                    value=cpi_yoy,
                    unit="%",
                    delta_3m=(cpi_3m - cpi_yoy) if cpi_3m else None,
                    threshold_low=self.TARGET,
                    threshold_high=self.ELEVATED,
                    source="FRED:CPIAUCSL",
                    asof=cpi_df["date"].iloc[-1].date() if hasattr(cpi_df["date"].iloc[-1], "date") else None,
                ))
        except Exception:
            pass
        
        # Core CPI YoY
        try:
            core_df = fred.fetch_series("CPILFESL", start_date="2020-01-01")
            if core_df is not None and len(core_df) > 12:
                core_df = core_df.sort_values("date")
                latest = core_df["value"].iloc[-1]
                year_ago = core_df["value"].iloc[-13] if len(core_df) > 13 else core_df["value"].iloc[0]
                core_yoy = (latest / year_ago - 1) * 100
                
                self.metrics.append(Metric(
                    name="Core CPI YoY",
                    value=core_yoy,
                    unit="%",
                    threshold_low=self.TARGET,
                    threshold_high=self.ELEVATED,
                    source="FRED:CPILFESL",
                ))
        except Exception:
            pass
        
        # Median CPI (Cleveland Fed - stickiness proxy)
        try:
            med_df = fred.fetch_series("MEDCPIM158SFRBCLE", start_date="2020-01-01")
            if med_df is not None and not med_df.empty:
                med_df = med_df.sort_values("date")
                med_cpi = med_df["value"].iloc[-1]
                
                # 3-month change
                if len(med_df) > 3:
                    delta_3m = med_cpi - med_df["value"].iloc[-4]
                else:
                    delta_3m = None
                
                self.metrics.append(Metric(
                    name="Median CPI YoY",
                    value=med_cpi,
                    unit="%",
                    delta_3m=delta_3m,
                    threshold_low=self.TARGET,
                    threshold_high=self.ELEVATED,
                    source="FRED:MEDCPIM158SFRBCLE",
                ))
        except Exception:
            pass
        
        # 5Y Breakeven
        try:
            be5_df = fred.fetch_series("T5YIE", start_date="2020-01-01")
            if be5_df is not None and not be5_df.empty:
                be5_df = be5_df.sort_values("date")
                be5 = be5_df["value"].iloc[-1]
                
                self.metrics.append(Metric(
                    name="5Y Breakeven",
                    value=be5,
                    unit="%",
                    threshold_low=1.8,
                    threshold_high=2.8,
                    source="FRED:T5YIE",
                ))
        except Exception:
            pass
        
        # 5Y5Y Forward Breakeven
        try:
            fwd_df = fred.fetch_series("T5YIFR", start_date="2020-01-01")
            if fwd_df is not None and not fwd_df.empty:
                fwd_df = fwd_df.sort_values("date")
                fwd = fwd_df["value"].iloc[-1]
                
                self.metrics.append(Metric(
                    name="5Y5Y Forward",
                    value=fwd,
                    unit="%",
                    threshold_low=2.0,
                    threshold_high=2.5,
                    source="FRED:T5YIFR",
                ))
        except Exception:
            pass
        
        # Compute composite score
        self._compute_score()
    
    def _compute_score(self) -> None:
        """Compute composite inflation score (0-100)."""
        scores = []
        
        cpi = self.get_metric("CPI YoY")
        if cpi and cpi.value is not None:
            # Score based on distance from target
            if cpi.value <= 2.0:
                scores.append(20)
            elif cpi.value <= 2.5:
                scores.append(30)
            elif cpi.value <= 3.0:
                scores.append(50)
            elif cpi.value <= 4.0:
                scores.append(70)
            else:
                scores.append(90)
        
        med_cpi = self.get_metric("Median CPI YoY")
        if med_cpi and med_cpi.value is not None:
            # Stickiness premium
            if med_cpi.value > 3.5:
                scores.append(80)
            elif med_cpi.value > 3.0:
                scores.append(60)
            else:
                scores.append(40)
        
        be5 = self.get_metric("5Y Breakeven")
        if be5 and be5.value is not None:
            # Expectations anchoring
            if be5.value > 2.8:
                scores.append(70)
            elif be5.value > 2.3:
                scores.append(50)
            else:
                scores.append(30)
        
        if scores:
            self.composite_score = sum(scores) / len(scores)
        
        self.regime = self.classify()
    
    def classify(self) -> str:
        """Classify inflation regime."""
        if self.composite_score is None:
            return "UNKNOWN"
        
        score = self.composite_score
        if score >= 70:
            return "HIGH_INFLATION"
        elif score >= 50:
            return "ELEVATED"
        elif score >= 30:
            return "ANCHORED"
        else:
            return "LOW_INFLATION"
    
    def deep_dive(self) -> dict:
        """
        Hedge-fund style deep dive analysis.
        
        Returns structured analysis for display or LLM.
        """
        cpi = self.get_metric("CPI YoY")
        core = self.get_metric("Core CPI YoY")
        med = self.get_metric("Median CPI YoY")
        be5 = self.get_metric("5Y Breakeven")
        fwd = self.get_metric("5Y5Y Forward")
        
        # Compute spreads
        realized_vs_expected = None
        if cpi and cpi.value and be5 and be5.value:
            realized_vs_expected = cpi.value - be5.value
        
        # Stickiness analysis
        sticky_vs_headline = None
        if cpi and cpi.value and med and med.value:
            sticky_vs_headline = med.value - cpi.value
        
        return {
            "regime": self.regime,
            "score": self.composite_score,
            "metrics": {
                "headline_cpi": cpi.value if cpi else None,
                "core_cpi": core.value if core else None,
                "median_cpi": med.value if med else None,
                "breakeven_5y": be5.value if be5 else None,
                "forward_5y5y": fwd.value if fwd else None,
            },
            "analysis": {
                "realized_vs_expected": realized_vs_expected,
                "sticky_vs_headline": sticky_vs_headline,
                "expectations_anchored": fwd.value < 2.5 if fwd and fwd.value else None,
            },
            "interpretation": self._interpret(),
        }
    
    def _interpret(self) -> str:
        """Generate interpretation text."""
        cpi = self.get_metric("CPI YoY")
        med = self.get_metric("Median CPI YoY")
        
        if not cpi or cpi.value is None:
            return "Insufficient data for interpretation."
        
        parts = []
        
        # Level assessment
        if cpi.value > 4.0:
            parts.append(f"Headline inflation at {cpi.value:.1f}% remains significantly above target.")
        elif cpi.value > 3.0:
            parts.append(f"Headline inflation at {cpi.value:.1f}% is elevated but moderating.")
        elif cpi.value > 2.5:
            parts.append(f"Headline inflation at {cpi.value:.1f}% is approaching target range.")
        else:
            parts.append(f"Headline inflation at {cpi.value:.1f}% is near target.")
        
        # Stickiness
        if med and med.value is not None:
            if med.value > cpi.value + 0.3:
                parts.append(f"Median CPI ({med.value:.1f}%) running above headline suggests broad-based price pressure.")
            elif med.value < cpi.value - 0.3:
                parts.append(f"Median CPI ({med.value:.1f}%) below headline suggests volatility-driven prints.")
        
        # Momentum
        if cpi.delta_3m is not None:
            if cpi.delta_3m > 0.5:
                parts.append("3-month momentum is re-accelerating — hawkish signal.")
            elif cpi.delta_3m < -0.5:
                parts.append("3-month momentum cooling — supportive of easing bias.")
        
        return " ".join(parts)

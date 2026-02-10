"""
Growth Pillar - Economic activity and labor market analysis.

Metrics:
- Payrolls (monthly change, 3m avg)
- Unemployment rate
- Initial jobless claims
- GDP growth
- ISM Manufacturing/Services

Author: Lox Capital Research
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Any

from lox.regimes.core import RegimePillar, Metric, register_pillar


@register_pillar("growth")
@dataclass
class GrowthPillar(RegimePillar):
    """
    Growth regime pillar.
    
    Tracks economic activity via:
    - Labor market (payrolls, unemployment, claims)
    - Output (GDP, industrial production)
    - Leading indicators (ISM)
    """
    name: str = "Growth"
    description: str = "Economic activity and labor market"
    metrics: List[Metric] = field(default_factory=list)
    
    def compute(self, settings: Any) -> None:
        """Fetch growth data and compute metrics."""
        from lox.data.fred import FredClient
        
        if not settings.FRED_API_KEY:
            return
        
        fred = FredClient(api_key=settings.FRED_API_KEY)
        
        # Payrolls (PAYEMS) - monthly change
        try:
            pay_df = fred.fetch_series("PAYEMS", start_date="2020-01-01")
            if pay_df is not None and len(pay_df) > 3:
                pay_df = pay_df.sort_values("date")
                
                # Monthly change (thousands)
                latest = pay_df["value"].iloc[-1]
                prev = pay_df["value"].iloc[-2]
                monthly_chg = latest - prev
                
                # 3-month annualized
                if len(pay_df) > 4:
                    three_ago = pay_df["value"].iloc[-4]
                    ann_3m = ((latest / three_ago) ** 4 - 1) * 100
                else:
                    ann_3m = None
                
                self.metrics.append(Metric(
                    name="Payrolls MoM",
                    value=monthly_chg,
                    unit="K",
                    delta_3m=ann_3m,
                    threshold_low=0,
                    threshold_high=200,
                    source="FRED:PAYEMS",
                ))
                
                self.metrics.append(Metric(
                    name="Payrolls 3m Ann",
                    value=ann_3m,
                    unit="%",
                    threshold_low=0,
                    threshold_high=2.0,
                    source="FRED:PAYEMS",
                ))
        except Exception:
            pass
        
        # Unemployment Rate
        try:
            ur_df = fred.fetch_series("UNRATE", start_date="2020-01-01")
            if ur_df is not None and not ur_df.empty:
                ur_df = ur_df.sort_values("date")
                ur = ur_df["value"].iloc[-1]
                
                # 3-month change
                if len(ur_df) > 3:
                    delta_3m = ur - ur_df["value"].iloc[-4]
                else:
                    delta_3m = None
                
                self.metrics.append(Metric(
                    name="Unemployment Rate",
                    value=ur,
                    unit="%",
                    delta_3m=delta_3m,
                    threshold_low=4.0,
                    threshold_high=5.5,
                    source="FRED:UNRATE",
                ))
        except Exception:
            pass
        
        # Initial Jobless Claims (4-week avg)
        try:
            claims_df = fred.fetch_series("IC4WSA", start_date="2020-01-01")
            if claims_df is not None and not claims_df.empty:
                claims_df = claims_df.sort_values("date")
                claims = claims_df["value"].iloc[-1] / 1000  # Convert to thousands
                
                # 4-week change
                if len(claims_df) > 4:
                    delta_4w = claims - claims_df["value"].iloc[-5] / 1000
                else:
                    delta_4w = None
                
                self.metrics.append(Metric(
                    name="Initial Claims 4w",
                    value=claims,
                    unit="K",
                    delta_1m=delta_4w,
                    threshold_low=200,
                    threshold_high=300,
                    source="FRED:IC4WSA",
                ))
        except Exception:
            pass
        
        # ISM Manufacturing PMI (if available)
        try:
            ism_df = fred.fetch_series("MANEMP", start_date="2020-01-01")  # Manufacturing employment as proxy
            if ism_df is not None and not ism_df.empty:
                # Note: Direct ISM data requires subscription
                pass
        except Exception:
            pass
        
        self._compute_score()
    
    def _compute_score(self) -> None:
        """Compute composite growth score (0-100)."""
        scores = []
        
        payrolls = self.get_metric("Payrolls 3m Ann")
        if payrolls and payrolls.value is not None:
            if payrolls.value < 0:
                scores.append(20)  # Contraction
            elif payrolls.value < 1.0:
                scores.append(40)  # Weak
            elif payrolls.value < 2.0:
                scores.append(60)  # Moderate
            else:
                scores.append(80)  # Strong
        
        ur = self.get_metric("Unemployment Rate")
        if ur and ur.value is not None:
            if ur.value < 4.0:
                scores.append(80)  # Tight labor market
            elif ur.value < 5.0:
                scores.append(60)  # Normal
            elif ur.value < 6.0:
                scores.append(40)  # Softening
            else:
                scores.append(20)  # Weak
        
        claims = self.get_metric("Initial Claims 4w")
        if claims and claims.value is not None:
            if claims.value < 220:
                scores.append(80)
            elif claims.value < 260:
                scores.append(60)
            elif claims.value < 300:
                scores.append(40)
            else:
                scores.append(20)
        
        if scores:
            self.composite_score = sum(scores) / len(scores)
        
        self.regime = self.classify()
    
    def classify(self) -> str:
        """Classify growth regime."""
        if self.composite_score is None:
            return "UNKNOWN"
        
        # Also check for stagflation conditions
        payrolls = self.get_metric("Payrolls 3m Ann")
        
        score = self.composite_score
        if score >= 70:
            return "EXPANSION"
        elif score >= 50:
            return "MODERATE"
        elif score >= 30:
            if payrolls and payrolls.value is not None and payrolls.value < 0:
                return "CONTRACTION"
            return "SOFTENING"
        else:
            return "RECESSION_RISK"
    
    def deep_dive(self) -> dict:
        """Hedge-fund style deep dive."""
        return {
            "regime": self.regime,
            "score": self.composite_score,
            "metrics": {m.name: m.value for m in self.metrics},
            "labor_market": self._labor_analysis(),
            "interpretation": self._interpret(),
        }
    
    def _labor_analysis(self) -> dict:
        """Detailed labor market analysis."""
        ur = self.get_metric("Unemployment Rate")
        claims = self.get_metric("Initial Claims 4w")
        payrolls = self.get_metric("Payrolls MoM")
        
        analysis = {
            "state": "UNKNOWN",
            "trend": "stable",
            "risks": [],
        }
        
        if ur and ur.value is not None:
            if ur.value < 4.0:
                analysis["state"] = "TIGHT"
            elif ur.value < 5.0:
                analysis["state"] = "BALANCED"
            else:
                analysis["state"] = "SLACK"
            
            if ur.delta_3m is not None and ur.delta_3m > 0.3:
                analysis["trend"] = "deteriorating"
                analysis["risks"].append("Rising unemployment")
        
        if claims and claims.value is not None and claims.value > 280:
            analysis["risks"].append("Elevated layoffs")
        
        return analysis
    
    def _interpret(self) -> str:
        """Generate interpretation."""
        payrolls = self.get_metric("Payrolls 3m Ann")
        ur = self.get_metric("Unemployment Rate")
        
        parts = []
        
        if payrolls and payrolls.value is not None:
            if payrolls.value < 0:
                parts.append(f"Labor market contracting (3m ann: {payrolls.value:.1f}%).")
            elif payrolls.value < 1.0:
                parts.append(f"Job growth slowing (3m ann: {payrolls.value:.1f}%).")
            else:
                parts.append(f"Job growth solid (3m ann: {payrolls.value:.1f}%).")
        
        if ur and ur.value is not None:
            parts.append(f"Unemployment at {ur.value:.1f}%.")
            if ur.delta_3m is not None:
                if ur.delta_3m > 0.3:
                    parts.append("Rising trend — softening signal.")
                elif ur.delta_3m < -0.2:
                    parts.append("Declining — tightening.")
        
        return " ".join(parts) if parts else "Insufficient data."
